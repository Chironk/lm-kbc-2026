#!/usr/bin/env python3
"""Macro-aligned hierarchical row selector over a frozen candidate graph.

The graph and six-challenger supply are fixed.  This module changes only the
selector and makes its training topology match deployment:

1. Canonicalize and deduplicate complete outputs; construction history is not
   a prediction identity.
2. Rank challengers *within a row* with a pairwise conditional-logit model.
3. Train a separate one-example-per-row gate to choose KEEP or the top-ranked
   challenger.
4. Give every relation equal total training weight, matching macro-F1.
5. Add strongly regularized relation-by-signal deviations on top of one
   shared ranker instead of fitting six unrelated relation selectors.
6. Combine frozen binary challenger-vs-KEEP review and frozen multiway review.

Every reported decision is produced in an outer subject fold.  Ranker
regularization is fixed; gate regularization and its probability threshold are
selected only by an inner subject-grouped cross-fit.  Validation is absent.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.baseline_conditioned_action_review import (
    _json,
    _validated_responses as _validated_binary_responses,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.crossfit_action_utility_selector import (
    DEFAULT_GOLD,
    DEFAULT_SOURCE,
    REVIEW_FEATURE_NAMES,
    TRUTH_FEATURE_NAMES,
    ActionExample,
    RowExample,
    _control_predictions,
    _feature_names,
    _features,
    _key,
    _write_json,
    load_examples,
)
from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.multi_challenger_graph_decoder import (
    EPSILON,
)
from experiments.heterogeneous_agents.row_grouped_action_ranker import (
    BASE_FEATURE_NAMES,
)
from experiments.heterogeneous_agents.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
)
from experiments.heterogeneous_agents.truth_calibrated_action_decoder import (
    StandardizedLinear,
)
from experiments.heterogeneous_agents.unified_memory_action_graph import (
    RELATIONS,
)


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_OUTPUT = RUNS / "coherent_row_selector_20260727_v2"

ARMS = (
    "shared_macro",
    "hierarchical_macro",
    "hierarchical_stable_gate",
)
PRIMARY_ARM = "hierarchical_stable_gate"
RANK_L2 = 10.0
RELATION_SHRINKAGE = 25.0
GATE_L2_GRID = (1.0, 10.0, 100.0)
GATE_THRESHOLD_GRID = (
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.60, 0.70,
)
MIN_PROMOTION_DELTA = 0.010
MIN_WINNING_FOLDS = 3
MIN_FOLD_DELTA = -0.010
MIN_RELATION_DELTA = -0.015

BINARY_FEATURE_NAMES = (
    "binary_qwen_margin",
    "binary_gemma_margin",
    "binary_qwen_use",
    "binary_gemma_use",
    "binary_qwen_uncertain",
    "binary_gemma_uncertain",
    "binary_mean_margin",
    "binary_minimum_margin",
    "binary_direction_agreement",
)
SHARED_FEATURE_NAMES = (
    BASE_FEATURE_NAMES + REVIEW_FEATURE_NAMES + BINARY_FEATURE_NAMES)
INTERACTION_SIGNAL_NAMES = (
    "action_empty", "action_replace", "action_add", "action_drop",
    "size_delta", "changed_component_count",
    "added_joint_support", "removed_joint_support",
    "numeric_log_distance",
    "binary_qwen_margin", "binary_gemma_margin",
    "qwen_within_row_rank", "gemma_within_row_rank",
)
HIERARCHICAL_FEATURE_NAMES = SHARED_FEATURE_NAMES + tuple(
    f"relation:{relation}*{name}"
    for relation in RELATIONS for name in INTERACTION_SIGNAL_NAMES
)
TRUTH_HIERARCHICAL_FEATURE_NAMES = (
    SHARED_FEATURE_NAMES
    + TRUTH_FEATURE_NAMES
    + tuple(
        f"relation:{relation}*{name}"
        for relation in RELATIONS for name in INTERACTION_SIGNAL_NAMES
    )
)
GATE_FEATURE_NAMES = (
    SHARED_FEATURE_NAMES
    + tuple(f"relation:{relation}" for relation in RELATIONS)
    + ("rank_top_score", "rank_gap", "row_alternative_count")
)

_SHARED_INDEX = {
    name: index for index, name in enumerate(SHARED_FEATURE_NAMES)}


@dataclass(frozen=True)
class CoherentAction:
    source: ActionExample
    shared_features: tuple[float, ...]
    canonical_output: tuple[str, ...]

    @property
    def delta(self) -> float:
        return float(self.source.delta)


@dataclass(frozen=True)
class CoherentRow:
    source: RowExample
    alternatives: tuple[CoherentAction, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.source.key

    @property
    def relation(self) -> str:
        return self.source.relation

    @property
    def fold(self) -> int:
        return self.source.fold


class HierarchicalPairwiseRanker:
    """Weighted conditional-logit ranker with shrinkage penalties."""

    def __init__(
        self,
        *,
        hierarchical: bool,
        include_truth: bool = False,
        utility_weighted: bool = False,
        objective: str = "ordinal",
    ):
        if objective not in {"ordinal", "helpful_contrast", "utility_regression"}:
            raise ValueError(f"unknown rank objective: {objective}")
        self.hierarchical = bool(hierarchical)
        self.include_truth = bool(include_truth)
        self.utility_weighted = bool(utility_weighted)
        self.objective = objective
        self.feature_names = _rank_feature_names(
            hierarchical=self.hierarchical,
            include_truth=self.include_truth,
        )
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    @property
    def parameter_count(self) -> int:
        return len(self.feature_names)

    def fit(self, rows: Sequence[CoherentRow]) -> "HierarchicalPairwiseRanker":
        row_pairs = {
            row.key: _ranking_pairs(row, objective=self.objective)
            for row in rows
        }
        relation_rows = Counter(
            row.relation for row in rows if row_pairs[row.key])
        matrix_rows, targets, weights = [], [], []
        for row in rows:
            pairs = row_pairs[row.key]
            if not pairs:
                continue
            pair_masses = _pair_masses(
                row, pairs, utility_weighted=self.utility_weighted)
            features = [
                _rank_features(
                    row,
                    action,
                    self.hierarchical,
                    include_truth=self.include_truth,
                )
                for action in row.alternatives
            ]
            row_mass = (
                1.0 / len(RELATIONS)
                / relation_rows[row.relation]
            )
            for (left, right), pair_mass in zip(
                pairs, pair_masses, strict=True,
            ):
                left_delta = row.alternatives[left].delta
                right_delta = row.alternatives[right].delta
                winner, loser = (
                    (left, right) if left_delta > right_delta
                    else (right, left)
                )
                difference = (
                    np.asarray(features[winner], dtype=np.float64)
                    - np.asarray(features[loser], dtype=np.float64)
                )
                matrix_rows.extend(
                    [difference.tolist(), (-difference).tolist()])
                if self.objective == "utility_regression":
                    gap = abs(left_delta - right_delta)
                    targets.extend([gap, -gap])
                else:
                    targets.extend([1.0, 0.0])
                symmetric_weight = row_mass * pair_mass / 2.0
                weights.extend([symmetric_weight, symmetric_weight])
        matrix = np.asarray(matrix_rows, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(self.feature_names)
            or matrix.shape[0] != len(target)
            or (
                self.objective != "utility_regression"
                and set(np.unique(target)) != {0.0, 1.0}
            )
        ):
            raise ValueError("invalid hierarchical pairwise arrays")
        weight *= len(weight) / weight.sum()
        variance = np.average(matrix**2, axis=0, weights=weight)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = matrix / self.scale
        beta = np.zeros(design.shape[1], dtype=np.float64)
        penalty_values = np.full(
            design.shape[1], RANK_L2, dtype=np.float64)
        if self.hierarchical:
            shared_width = (
                len(SHARED_FEATURE_NAMES)
                + (len(TRUTH_FEATURE_NAMES) if self.include_truth else 0)
            )
            penalty_values[shared_width:] *= RELATION_SHRINKAGE
        penalty = np.diag(penalty_values)
        if self.objective == "utility_regression":
            hessian = design.T @ (design * weight[:, None]) + penalty
            beta = np.linalg.solve(
                hessian, design.T @ (weight * target))
        else:
            for _ in range(100):
                logits = np.clip(design @ beta, -30.0, 30.0)
                probability = 1.0 / (1.0 + np.exp(-logits))
                curvature = np.maximum(
                    probability * (1.0 - probability), 1e-8)
                gradient = (
                    design.T @ (weight * (probability - target))
                    + penalty @ beta)
                hessian = (
                    design.T @ (
                        design * (weight * curvature)[:, None])
                    + penalty)
                step = np.linalg.solve(hessian, gradient)
                beta -= step
                if float(np.max(np.abs(step))) < 1e-9:
                    break
        self.coef = beta
        return self

    def scores(
        self, row: CoherentRow,
    ) -> list[float]:
        if self.scale is None or self.coef is None:
            raise RuntimeError("hierarchical pairwise ranker is not fitted")
        if not row.alternatives:
            return []
        matrix = np.asarray([
            _rank_features(
                row,
                action,
                self.hierarchical,
                include_truth=self.include_truth,
            )
            for action in row.alternatives
        ], dtype=np.float64)
        return list(map(float, (matrix / self.scale) @ self.coef))

    def to_dict(self) -> dict[str, Any]:
        if self.scale is None or self.coef is None:
            raise RuntimeError("hierarchical pairwise ranker is not fitted")
        return {
            "schema": "macro-hierarchical-pairwise-ranker-v1",
            "hierarchical": self.hierarchical,
            "include_truth": self.include_truth,
            "utility_weighted": self.utility_weighted,
            "objective": self.objective,
            "feature_names": list(self.feature_names),
            "l2": RANK_L2,
            "relation_shrinkage": RELATION_SHRINKAGE,
            "scale": self.scale.tolist(),
            "coefficients": self.coef.tolist(),
            "parameter_count": self.parameter_count,
            "weighting": (
                "relation_then_row_then_absolute_utility_gap"
                if self.utility_weighted
                else "relation_then_row_then_equal_pair"
            ),
        }


def _log_margin(evidence: Mapping[str, float]) -> float:
    return max(-12.0, min(12.0, (
        math.log(max(float(evidence["USE_ALTERNATIVE"]), EPSILON))
        - math.log(max(float(evidence["KEEP_CURRENT"]), EPSILON))
    ))) / 12.0


def _canonical_output(
    objects: Sequence[str], relation: str,
) -> tuple[str, ...]:
    return tuple(sorted({
        canonical_key(str(item), relation) for item in objects}))


def _binary_features(
    row: RowExample,
    action: ActionExample,
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float],
    ],
) -> list[float]:
    action_id = str(action.action["id"])
    q = evidence[(row.key, action_id, QWEN)]
    g = evidence[(row.key, action_id, GEMMA)]
    q_margin, g_margin = _log_margin(q), _log_margin(g)
    return [
        q_margin, g_margin,
        float(q["USE_ALTERNATIVE"]),
        float(g["USE_ALTERNATIVE"]),
        float(q["UNCERTAIN"]),
        float(g["UNCERTAIN"]),
        (q_margin + g_margin) / 2.0,
        min(q_margin, g_margin),
        float((q_margin > 0.0) == (g_margin > 0.0)),
    ]


def load_coherent_rows(
    source: Path, gold_path: Path,
) -> tuple[list[CoherentRow], dict[str, Any]]:
    rows, provenance = load_examples(source, gold_path)
    review_plan_path = Path(provenance["review_plan"])
    review_plan = _json(review_plan_path)
    binary = _validated_binary_responses(review_plan)
    coherent = []
    removed_collapse = removed_keep_equivalent = duplicate_outputs = 0
    for row in rows:
        keep_identity = _canonical_output(
            row.keep["objects"], row.relation)
        by_output: dict[tuple[str, ...], CoherentAction] = {}
        for action in row.alternatives:
            if str(action.action["action_type"]) == "COLLAPSE":
                removed_collapse += 1
                continue
            identity = _canonical_output(
                action.action["objects"], row.relation)
            if identity == keep_identity:
                removed_keep_equivalent += 1
                continue
            shared = [
                *_features(action, "graph_review"),
                *_binary_features(row, action, binary),
            ]
            if len(shared) != len(SHARED_FEATURE_NAMES):
                raise AssertionError("coherent shared feature schema drift")
            candidate = CoherentAction(
                source=action,
                shared_features=tuple(shared),
                canonical_output=identity,
            )
            if identity in by_output:
                duplicate_outputs += 1
                # Same prediction has the same utility. Preserve the
                # representation with stronger average comparative evidence;
                # construction labels never create two training examples.
                old = by_output[identity]
                review_start = len(BASE_FEATURE_NAMES)
                if np.mean(shared[review_start:]) > np.mean(
                    old.shared_features[review_start:]
                ):
                    by_output[identity] = candidate
            else:
                by_output[identity] = candidate
        coherent.append(CoherentRow(
            source=row,
            alternatives=tuple(by_output[key] for key in sorted(by_output)),
        ))
    return coherent, {
        **provenance,
        "binary_review_plan": str(review_plan_path),
        "binary_review_plan_sha256": sha256(review_plan_path),
        "removed_collapse_actions": removed_collapse,
        "removed_keep_equivalent_actions": removed_keep_equivalent,
        "deduplicated_outputs": duplicate_outputs,
        "retained_unique_alternatives": sum(
            len(row.alternatives) for row in coherent),
    }


def _rank_feature_names(
    *, hierarchical: bool, include_truth: bool,
) -> tuple[str, ...]:
    shared = (
        SHARED_FEATURE_NAMES + TRUTH_FEATURE_NAMES
        if include_truth else SHARED_FEATURE_NAMES
    )
    if not hierarchical:
        return shared
    return shared + tuple(
        f"relation:{relation}*{name}"
        for relation in RELATIONS for name in INTERACTION_SIGNAL_NAMES
    )


def _pair_masses(
    row: CoherentRow,
    pairs: Sequence[tuple[int, int]],
    *,
    utility_weighted: bool,
) -> list[float]:
    """Return normalized within-row pair mass without changing row weight.

    Equal weighting optimizes pair accuracy. Utility weighting instead spends
    more of the row's fixed mass on mistakes that lose more macro-F1. Keeping
    the total at one ensures rows and relations remain balanced.
    """
    if not pairs:
        return []
    if utility_weighted:
        raw = [
            abs(
                row.alternatives[left].delta
                - row.alternatives[right].delta
            )
            for left, right in pairs
        ]
    else:
        raw = [1.0] * len(pairs)
    total = sum(raw)
    if total <= EPSILON:
        return [1.0 / len(pairs)] * len(pairs)
    return [value / total for value in raw]


def _ranking_pairs(
    row: CoherentRow,
    *,
    objective: str,
) -> list[tuple[int, int]]:
    pairs = []
    for left, right in itertools.combinations(
        range(len(row.alternatives)), 2,
    ):
        left_delta = row.alternatives[left].delta
        right_delta = row.alternatives[right].delta
        if objective == "helpful_contrast":
            informative = (
                (left_delta > EPSILON)
                != (right_delta > EPSILON)
            )
        else:
            informative = abs(left_delta - right_delta) > EPSILON
        if informative:
            pairs.append((left, right))
    return pairs


def _rank_features(
    row: CoherentRow,
    action: CoherentAction,
    hierarchical: bool,
    *,
    include_truth: bool = False,
) -> list[float]:
    shared = list(action.shared_features)
    feature_values = [
        *shared,
        *(action.source.truth_features if include_truth else ()),
    ]
    if not hierarchical:
        return feature_values
    signals = [
        shared[_SHARED_INDEX[name]] for name in INTERACTION_SIGNAL_NAMES]
    interactions = []
    for relation in RELATIONS:
        active = float(row.relation == relation)
        interactions.extend(active * value for value in signals)
    values = [*feature_values, *interactions]
    expected = _rank_feature_names(
        hierarchical=hierarchical, include_truth=include_truth)
    if len(values) != len(expected):
        raise AssertionError("hierarchical rank feature schema drift")
    return values


def _top_action(
    model: HierarchicalPairwiseRanker,
    row: CoherentRow,
) -> tuple[CoherentAction | None, float, float]:
    values = model.scores(row)
    if not values:
        return None, 0.0, 0.0
    order = sorted(
        range(len(values)),
        key=lambda index: (
            values[index],
            row.alternatives[index].canonical_output,
        ),
        reverse=True,
    )
    top = order[0]
    gap = (
        float(values[top] - values[order[1]])
        if len(order) > 1 else abs(float(values[top]))
    )
    return row.alternatives[top], float(values[top]), gap


def _oof_top_actions(
    rows: Sequence[CoherentRow],
    *,
    hierarchical: bool,
    include_truth: bool = False,
    utility_weighted: bool = False,
) -> dict[tuple[str, str], tuple[CoherentAction | None, float, float]]:
    output = {}
    for fold in sorted({row.fold for row in rows}):
        fit = [row for row in rows if row.fold != fold]
        hold = [row for row in rows if row.fold == fold]
        ranker = HierarchicalPairwiseRanker(
            hierarchical=hierarchical,
            include_truth=include_truth,
            utility_weighted=utility_weighted,
        ).fit(fit)
        for row in hold:
            output[row.key] = _top_action(ranker, row)
    if set(output) != {row.key for row in rows}:
        raise ContractError("incomplete inner OOF challenger ranking")
    return output


def _gate_features(
    row: CoherentRow,
    selection: tuple[CoherentAction | None, float, float],
) -> list[float] | None:
    action, top_score, gap = selection
    if action is None:
        return None
    values = [
        *action.shared_features,
        *(float(row.relation == relation) for relation in RELATIONS),
        max(-5.0, min(5.0, top_score)) / 5.0,
        max(0.0, min(5.0, gap)) / 5.0,
        min(len(row.alternatives), 6) / 6.0,
    ]
    if len(values) != len(GATE_FEATURE_NAMES):
        raise AssertionError("row gate feature schema drift")
    return values


def _macro_row_weights(rows: Sequence[CoherentRow]) -> list[float]:
    counts = Counter(row.relation for row in rows)
    return [
        1.0 / len(RELATIONS) / counts[row.relation] for row in rows]


def _fit_gate(
    rows: Sequence[CoherentRow],
    selections: Mapping[
        tuple[str, str], tuple[CoherentAction | None, float, float],
    ],
    l2: float,
) -> StandardizedLinear:
    usable = [
        row for row in rows if selections[row.key][0] is not None]
    x = [_gate_features(row, selections[row.key]) for row in usable]
    y = [
        float(selections[row.key][0].delta > EPSILON)  # type: ignore[union-attr]
        for row in usable
    ]
    return StandardizedLinear(
        GATE_FEATURE_NAMES, l2, logistic=True).fit(
            x, y, _macro_row_weights(usable))  # type: ignore[arg-type]


def _gate_probabilities(
    model: StandardizedLinear,
    rows: Sequence[CoherentRow],
    selections: Mapping[
        tuple[str, str], tuple[CoherentAction | None, float, float],
    ],
) -> dict[tuple[str, str], float]:
    usable = [
        row for row in rows if selections[row.key][0] is not None]
    features = [_gate_features(row, selections[row.key]) for row in usable]
    probabilities = model.predict(features)  # type: ignore[arg-type]
    output = {row.key: 0.0 for row in rows}
    output.update({
        row.key: float(value)
        for row, value in zip(usable, probabilities, strict=True)})
    return output


def _decode(
    rows: Sequence[CoherentRow],
    selections: Mapping[
        tuple[str, str], tuple[CoherentAction | None, float, float],
    ],
    probabilities: Mapping[tuple[str, str], float],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics = [], []
    for row in rows:
        action, rank_score, rank_gap = selections[row.key]
        changed = action is not None and probabilities[row.key] > threshold
        objects = (
            action.source.action["objects"] if changed
            else row.source.keep["objects"]
        )
        delta = action.delta if changed and action is not None else 0.0
        predictions.append({
            "SubjectEntity": row.key[0],
            "Relation": row.key[1],
            "ObjectEntities": list(objects),
        })
        diagnostics.append({
            "SubjectEntity": row.key[0],
            "Relation": row.key[1],
            "fold": row.fold,
            "changed": changed,
            "selected_action_id": (
                str(action.source.action["id"])
                if changed and action is not None else str(row.source.keep["id"])
            ),
            "top_challenger_action_id": (
                str(action.source.action["id"]) if action is not None else None
            ),
            "top_challenger_delta": (
                float(action.delta) if action is not None else 0.0),
            "actual_delta": float(delta),
            "gate_probability": float(probabilities[row.key]),
            "gate_threshold": float(threshold),
            "rank_score": float(rank_score),
            "rank_gap": float(rank_gap),
            "shortlist_oracle_delta": max([
                0.0, *(item.delta for item in row.alternatives)]),
        })
    return predictions, diagnostics


def _quality(
    predictions: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    rows: Sequence[CoherentRow],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    selected = score(
        list(predictions), [gold_by[row.key] for row in rows])
    control = score(
        _control_predictions([row.source for row in rows]),
        [gold_by[row.key] for row in rows],
    )
    changed = [row for row in diagnostics if row["changed"]]
    return {
        "score": selected["*** All Relations ***"],
        "control": control["*** All Relations ***"],
        "delta": (
            selected["*** All Relations ***"]
            - control["*** All Relations ***"]),
        "relation_deltas": {
            relation: selected[relation] - control[relation]
            for relation in RELATIONS
        },
        "changed": len(changed),
        "helped": sum(row["actual_delta"] > EPSILON for row in changed),
        "harmed": sum(row["actual_delta"] < -EPSILON for row in changed),
        "neutral": sum(
            abs(row["actual_delta"]) <= EPSILON for row in changed),
        "top_challenger_helpful": sum(
            row["top_challenger_delta"] > EPSILON for row in diagnostics),
        "headroom_rows": sum(
            row["shortlist_oracle_delta"] > EPSILON for row in diagnostics),
    }


def _select_gate(
    rows: Sequence[CoherentRow],
    selections: Mapping[
        tuple[str, str], tuple[CoherentAction | None, float, float],
    ],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    stable: bool,
) -> tuple[float, float, dict[str, Any]]:
    folds = sorted({row.fold for row in rows})
    candidates = {}
    for l2 in GATE_L2_GRID:
        probabilities = {}
        for fold in folds:
            fit = [row for row in rows if row.fold != fold]
            hold = [row for row in rows if row.fold == fold]
            model = _fit_gate(fit, selections, l2)
            probabilities.update(
                _gate_probabilities(model, hold, selections))
        for threshold in GATE_THRESHOLD_GRID:
            predictions, diagnostics = _decode(
                rows, selections, probabilities, threshold)
            aggregate = _quality(
                predictions, diagnostics, rows, gold_by)
            fold_deltas = []
            for fold in folds:
                hold = [row for row in rows if row.fold == fold]
                fold_predictions, fold_diagnostics = _decode(
                    hold, selections, probabilities, threshold)
                fold_deltas.append(_quality(
                    fold_predictions, fold_diagnostics, hold, gold_by
                )["delta"])
            candidates[(l2, threshold)] = {
                **aggregate,
                "fold_deltas": fold_deltas,
                "winning_folds": sum(value > 0.0 for value in fold_deltas),
                "nonnegative_folds": sum(
                    value >= -EPSILON for value in fold_deltas),
                "worst_fold": min(fold_deltas),
            }
    eligible = list(candidates)
    if stable:
        stable_candidates = [
            item for item, quality in candidates.items()
            if (
                quality["nonnegative_folds"] >= len(folds) - 1
                and quality["worst_fold"] >= -0.005
                and quality["helped"] >= quality["harmed"]
            )
        ]
        if stable_candidates:
            eligible = stable_candidates
    selected = max(
        eligible,
        key=lambda item: (
            candidates[item]["score"],
            -candidates[item]["harmed"],
            -candidates[item]["changed"],
            item[1],
            item[0],
        ),
    )
    return selected[0], selected[1], {
        "selected_l2": selected[0],
        "selected_threshold": selected[1],
        "stability_constrained": stable,
        "eligible_candidates": len(eligible),
        "quality": candidates[selected],
        "grid": [
            {"l2": l2, "threshold": threshold, **candidates[(l2, threshold)]}
            for l2 in GATE_L2_GRID
            for threshold in GATE_THRESHOLD_GRID
        ],
    }


def _cross_fit(
    rows: Sequence[CoherentRow],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    *, hierarchical: bool, stable_gate: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics, fold_reports = [], [], []
    for outer in sorted({row.fold for row in rows}):
        fit = [row for row in rows if row.fold != outer]
        hold = [row for row in rows if row.fold == outer]
        # The row gate is trained on challengers selected by rankers that did
        # not see those rows. This prevents the gate from learning against
        # artificially optimistic in-sample challenger identities.
        fit_oof_selections = _oof_top_actions(
            fit, hierarchical=hierarchical)
        gate_l2, threshold, inner = _select_gate(
            fit, fit_oof_selections, gold_by, stable=stable_gate)
        gate = _fit_gate(fit, fit_oof_selections, gate_l2)
        ranker = HierarchicalPairwiseRanker(
            hierarchical=hierarchical).fit(fit)
        hold_selections = {
            row.key: _top_action(ranker, row) for row in hold}
        probabilities = _gate_probabilities(
            gate, hold, hold_selections)
        fold_predictions, fold_diagnostics = _decode(
            hold, hold_selections, probabilities, threshold)
        quality = _quality(
            fold_predictions, fold_diagnostics, hold, gold_by)
        predictions.extend(fold_predictions)
        diagnostics.extend(fold_diagnostics)
        fold_reports.append({
            "fold": outer,
            "fit_rows": len(fit),
            "hold_rows": len(hold),
            "inner_gate_selection": inner,
            "holdout": quality,
            "ranker": ranker.to_dict(),
            "gate": gate.to_dict(),
        })
    predictions_by = {_key(row): row for row in predictions}
    diagnostics_by = {_key(row): row for row in diagnostics}
    return (
        [predictions_by[row.key] for row in rows],
        [diagnostics_by[row.key] for row in rows],
        fold_reports,
    )


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    source = Path(args.source_run).resolve()
    gold_path = Path(args.gold).resolve()
    rows, provenance = load_coherent_rows(source, gold_path)
    gold_rows = read_jsonl(gold_path)
    gold_by = {_key(row): row for row in gold_rows}
    artifacts, results = {}, {}
    for arm in ARMS:
        print(f"coherent selector OOF arm={arm}", flush=True)
        hierarchical = arm != "shared_macro"
        predictions, diagnostics, folds = _cross_fit(
            rows,
            gold_by,
            hierarchical=hierarchical,
            stable_gate=(arm == "hierarchical_stable_gate"),
        )
        prediction_path = output / f"{arm}/TRAIN_OOF_PREDICTIONS.jsonl"
        diagnostic_path = output / f"{arm}/TRAIN_OOF_DIAGNOSTICS.jsonl"
        fold_path = output / f"{arm}/FOLDS.json"
        write_jsonl_atomic(prediction_path, predictions)
        write_jsonl_atomic(diagnostic_path, diagnostics)
        _write_json(fold_path, {"arm": arm, "folds": folds})
        quality = _quality(
            predictions, diagnostics, rows, gold_by)
        fold_deltas = [item["holdout"]["delta"] for item in folds]
        results[arm] = {
            **quality,
            "fold_deltas": fold_deltas,
            "winning_folds": sum(value > 0.0 for value in fold_deltas),
            "worst_fold": min(fold_deltas),
            "ranker_parameter_count": (
                len(HIERARCHICAL_FEATURE_NAMES)
                if hierarchical
                else len(SHARED_FEATURE_NAMES)
            ),
            "gate_parameter_count": len(GATE_FEATURE_NAMES) + 1,
        }
        artifacts[arm] = {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
            "folds": str(fold_path),
            "folds_sha256": sha256(fold_path),
        }

    primary = results[PRIMARY_ARM]
    checks = {
        "minimum_delta": primary["delta"] >= MIN_PROMOTION_DELTA,
        "minimum_winning_folds": (
            primary["winning_folds"] >= MIN_WINNING_FOLDS),
        "minimum_fold_delta": primary["worst_fold"] >= MIN_FOLD_DELTA,
        "minimum_relation_delta": (
            min(primary["relation_deltas"].values()) >= MIN_RELATION_DELTA),
        "beats_shared_ablation": (
            primary["delta"] > results["shared_macro"]["delta"]),
    }
    result = {
        "schema": "coherent-row-selector-result-v1",
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "primary_arm": PRIMARY_ARM,
        "arms": results,
        "artifacts": artifacts,
        "promotion_gate": {"passed": all(checks.values()), "checks": checks},
        "failure_ledger": {
            "headroom_rows": primary["headroom_rows"],
            "top_challenger_helpful": primary["top_challenger_helpful"],
            "helpful_selected": primary["helped"],
            "harmful_selected": primary["harmed"],
            "ranking_loss": (
                primary["headroom_rows"]
                - primary["top_challenger_helpful"]),
            "gate_loss": (
                primary["top_challenger_helpful"] - primary["helped"]),
        },
        "provenance": provenance,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "development_only": True,
        "deployable": False,
        "next_stage": (
            "freeze_and_prepare_one_validation_confirmation"
            if all(checks.values())
            else "reject_or_revise_coherent_selector"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "RESULT.json"
    _write_json(result_path, result)
    lines = [
        "# Coherent macro-aligned row selector",
        "",
        "Nested subject-grouped train OOF; validation was not opened.",
        "",
        f"- Primary delta: `{primary['delta']:+.9f}`",
        f"- Changed/helped/harmed/neutral: "
        f"`{primary['changed']}/{primary['helped']}/"
        f"{primary['harmed']}/{primary['neutral']}`",
        f"- Helpful top challenger/headroom rows: "
        f"`{primary['top_challenger_helpful']}/{primary['headroom_rows']}`",
        f"- Winning folds: `{primary['winning_folds']}/5`",
        f"- Gate: `{'PASS' if all(checks.values()) else 'FAIL'}`",
        "",
        "| Arm | Delta | Top helpful | Changed | Helped | Harmed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = results[arm]
        lines.append(
            f"| {arm} | {item['delta']:+.6f} | "
            f"{item['top_challenger_helpful']} | {item['changed']} | "
            f"{item['helped']} | {item['harmed']} |")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "primary": primary,
        "arms": {
            arm: {
                "delta": item["delta"],
                "top_challenger_helpful": item["top_challenger_helpful"],
                "changed": item["changed"],
                "helped": item["helped"],
                "harmed": item["harmed"],
            } for arm, item in results.items()
        },
        "promotion_gate": result["promotion_gate"],
        "failure_ledger": result["failure_ledger"],
        "result": str(result_path),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-run", default=str(DEFAULT_SOURCE))
    result.add_argument("--gold", default=str(DEFAULT_GOLD))
    result.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    result.set_defaults(function=analyze)
    return result


def main() -> int:
    arguments = parser().parse_args()
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
