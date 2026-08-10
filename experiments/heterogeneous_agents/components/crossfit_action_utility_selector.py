#!/usr/bin/env python3
"""Nested cross-fitted selector for the multi-challenger action graph.

This is the missing translation layer between the high-recall graph action
shortlist and a deployable output.  It deliberately does not treat either
reviewer's choice probabilities as calibrated probabilities.  Instead it:

* represents every shortlisted edit relative to the exact registered KEEP;
* adds reviewer margins, within-row ranks, and row-centred margins;
* adds candidate-truth evidence for components added and removed by the edit;
* predicts official row-F1 delta, not candidate correctness;
* selects regularization and a conservative switch guard only in nested,
  subject-grouped training folds.

The outer-fold result is the only promotion statistic.  Validation is absent
from this module and cannot influence architecture or hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.components.component_expected_f1_set_decoder import (
    _validated_truth_evidence,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.multi_challenger_graph_decoder import (
    EPSILON,
    _validated_plan,
    _validated_responses,
)
from experiments.heterogeneous_agents.components.row_grouped_action_ranker import (
    BASE_FEATURE_NAMES,
    compact_action_features,
    _utility,
)
from experiments.heterogeneous_agents.components.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    validate_registered_predictions,
)
from experiments.heterogeneous_agents.components.truth_calibrated_action_decoder import (
    StandardizedLinear,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import _key


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_SOURCE = RUNS / "multi_challenger_graph_decoder_20260727_v1"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_OUTPUT = RUNS / "crossfit_action_utility_selector_20260727_v1"

ARMS = ("graph_only", "graph_review", "graph_review_truth")
PRIMARY_ARM = "graph_review_truth"
L2_GRID = (0.1, 1.0, 10.0, 100.0)
GUARD_GRID = (-0.05, -0.02, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
MIN_PROMOTION_DELTA = 0.010
MIN_WINNING_FOLDS = 3
MIN_FOLD_DELTA = -0.010
MIN_RELATION_DELTA = -0.015

REVIEW_FEATURE_NAMES = (
    "qwen_margin",
    "gemma_margin",
    "qwen_centered_margin",
    "gemma_centered_margin",
    "qwen_within_row_rank",
    "gemma_within_row_rank",
    "mean_centered_margin",
    "minimum_centered_margin",
    "centered_margin_agreement",
    "reviewer_rank_agreement",
    "qwen_positive",
    "gemma_positive",
)
TRUTH_FEATURE_NAMES = (
    "truth_added_mean",
    "truth_added_minimum",
    "truth_added_maximum",
    "truth_removed_mean",
    "truth_removed_minimum",
    "truth_removed_maximum",
    "truth_added_qwen_maximum",
    "truth_added_gemma_maximum",
    "truth_removed_qwen_maximum",
    "truth_removed_gemma_maximum",
    "truth_added_model_disagreement",
    "truth_removed_model_disagreement",
    "truth_edit_expected_correctness",
    "truth_edit_advantage",
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _canonical_objects(
    objects: Sequence[str], relation: str,
) -> set[str]:
    return {canonical_key(str(item), relation) for item in objects}


def _percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Return deterministic within-row ranks in [-1, 1]."""
    ordered = sorted(
        values, key=lambda item: (float(values[item]), str(item)))
    if len(ordered) <= 1:
        return {item: 0.0 for item in ordered}
    return {
        item: -1.0 + 2.0 * index / (len(ordered) - 1)
        for index, item in enumerate(ordered)
    }


def _surface_truth(
    graph: Mapping[str, Any],
    evidence: Mapping[tuple[str, str, str], Mapping[str, float]],
) -> dict[str, Mapping[str, float]]:
    relation = str(graph["Relation"])
    output: dict[str, Mapping[str, float]] = {}
    for node in graph["nodes"]:
        if node.get("node_type") != "candidate_component":
            continue
        identity = (*_key(graph), str(node["id"]))
        if identity not in evidence:
            raise ContractError(f"missing component truth evidence: {identity}")
        value = evidence[identity]
        for surface in [
            *node.get("member_items", []), node["representative"],
        ]:
            canonical = canonical_key(str(surface), relation)
            if canonical in output and output[canonical] != value:
                raise ContractError(
                    f"{_key(graph)}: ambiguous truth surface {surface}")
            output[canonical] = value
    return output


def _summary(values: Sequence[float], default: float) -> tuple[float, float, float]:
    if not values:
        return default, default, default
    return statistics.mean(values), min(values), max(values)


def _truth_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    keep: Mapping[str, Any],
    evidence: Mapping[tuple[str, str, str], Mapping[str, float]],
) -> list[float]:
    relation = str(graph["Relation"])
    before = _canonical_objects(keep["objects"], relation)
    after = _canonical_objects(action["objects"], relation)
    added, removed = after - before, before - after
    by_surface = _surface_truth(graph, evidence)

    def record(surface: str) -> Mapping[str, float]:
        # Incumbent-only surfaces can be absent from the candidate inventory.
        # Neutral evidence is safer than silently treating absence as false.
        return by_surface.get(surface, {
            QWEN: 0.5, GEMMA: 0.5, "mean": 0.5,
            "minimum": 0.5, "agreement": 0.0,
        })

    added_records = [record(item) for item in sorted(added)]
    removed_records = [record(item) for item in sorted(removed)]
    added_mean = [float(item["mean"]) for item in added_records]
    removed_mean = [float(item["mean"]) for item in removed_records]
    add_mean, add_min, add_max = _summary(added_mean, 0.5)
    rem_mean, rem_min, rem_max = _summary(removed_mean, 0.5)
    added_disagreement = [
        abs(float(item[QWEN]) - float(item[GEMMA]))
        for item in added_records]
    removed_disagreement = [
        abs(float(item[QWEN]) - float(item[GEMMA]))
        for item in removed_records]
    changed_count = len(added_records) + len(removed_records)
    expected_correct = (
        (
            sum(added_mean)
            + sum(1.0 - value for value in removed_mean)
        ) / changed_count
        if changed_count else 0.5
    )
    advantage = (
        sum(value - 0.5 for value in added_mean)
        + sum(0.5 - value for value in removed_mean)
    ) / max(1, changed_count)
    return [
        add_mean, add_min, add_max,
        rem_mean, rem_min, rem_max,
        max((float(item[QWEN]) for item in added_records), default=0.5),
        max((float(item[GEMMA]) for item in added_records), default=0.5),
        max((float(item[QWEN]) for item in removed_records), default=0.5),
        max((float(item[GEMMA]) for item in removed_records), default=0.5),
        statistics.mean(added_disagreement) if added_disagreement else 0.0,
        statistics.mean(removed_disagreement) if removed_disagreement else 0.0,
        expected_correct,
        advantage,
    ]


@dataclass(frozen=True)
class ActionExample:
    action: Mapping[str, Any]
    graph_features: tuple[float, ...]
    review_features: tuple[float, ...]
    truth_features: tuple[float, ...]
    delta: float


@dataclass(frozen=True)
class RowExample:
    key: tuple[str, str]
    relation: str
    graph: Mapping[str, Any]
    keep: Mapping[str, Any]
    alternatives: tuple[ActionExample, ...]
    fold: int


def _feature_names(arm: str) -> tuple[str, ...]:
    if arm == "graph_only":
        return BASE_FEATURE_NAMES
    if arm == "graph_review":
        return BASE_FEATURE_NAMES + REVIEW_FEATURE_NAMES
    if arm == "graph_review_truth":
        return BASE_FEATURE_NAMES + REVIEW_FEATURE_NAMES + TRUTH_FEATURE_NAMES
    raise ContractError(f"unknown feature arm: {arm}")


def _features(example: ActionExample, arm: str) -> list[float]:
    values = list(example.graph_features)
    if arm in {"graph_review", "graph_review_truth"}:
        values.extend(example.review_features)
    if arm == "graph_review_truth":
        values.extend(example.truth_features)
    if len(values) != len(_feature_names(arm)):
        raise AssertionError("action utility feature schema drift")
    return values


def _margins(
    key: tuple[str, str],
    shortlist: Mapping[str, Any],
    responses: Mapping[
        tuple[tuple[str, str], str, int], Mapping[str, float],
    ],
) -> tuple[dict[str, float], dict[str, float]]:
    keep_id = str(shortlist["keep_action_id"])
    output = {QWEN: {}, GEMMA: {}}
    for group_index, group in enumerate(shortlist["groups"]):
        for agent in (QWEN, GEMMA):
            probabilities = responses[(key, agent, group_index)]
            for action_id in map(str, group):
                output[agent][action_id] = (
                    math.log(max(float(probabilities[action_id]), EPSILON))
                    - math.log(max(float(probabilities[keep_id]), EPSILON))
                )
    expected = set(map(str, shortlist["challenger_action_ids"]))
    if set(output[QWEN]) != expected or set(output[GEMMA]) != expected:
        raise ContractError(f"{key}: incomplete reviewer margins")
    return output[QWEN], output[GEMMA]


def load_examples(
    source: Path, gold_path: Path,
) -> tuple[list[RowExample], dict[str, Any]]:
    plan_path, plan = _validated_plan(source)
    responses = _validated_responses(plan)
    review_plan_path = Path(plan["review_run"]) / "plan/PLAN.json"
    review_plan = json.loads(review_plan_path.read_text())
    graphs = read_jsonl(Path(plan["action_registry"]))
    shortlists = read_jsonl(Path(plan["shortlists"]))
    gold_rows = read_jsonl(gold_path)
    gold_by = {_key(row): row for row in gold_rows}
    folds_path = Path(review_plan["folds"])
    if sha256(folds_path) != review_plan["folds_sha256"]:
        raise ContractError("stale subject-grouped folds")
    folds = {_key(row): int(row["fold"]) for row in read_jsonl(folds_path)}
    truth, truth_plan = _validated_truth_evidence(
        Path(plan["truth_run"]), str(review_plan["source_graph_sha256"]))
    control_path = Path(review_plan["starting_predictions"])
    validate_registered_predictions(
        control_path, pipeline_id=COMPETITION_PIPELINE_ID, split="train")
    control = read_jsonl(control_path)
    control_by = {_key(row): row for row in control}
    if (
        len(graphs) != len(shortlists)
        or {_key(row) for row in graphs} != set(gold_by)
        or set(control_by) != set(gold_by)
        or set(folds) != set(gold_by)
    ):
        raise ContractError("selector source coverage mismatch")

    rows = []
    for graph, shortlist in zip(graphs, shortlists, strict=True):
        key = _key(graph)
        if key != _key(shortlist):
            raise ContractError("graph/shortlist order mismatch")
        actions = {str(action["id"]): action for action in graph["actions"]}
        keep = actions[str(shortlist["keep_action_id"])]
        if _canonical_objects(
            keep["objects"], key[1],
        ) != _canonical_objects(control_by[key]["ObjectEntities"], key[1]):
            raise ContractError(f"{key}: KEEP differs from registered SOTA")
        qwen, gemma = _margins(key, shortlist, responses)
        q_median = statistics.median(qwen.values()) if qwen else 0.0
        g_median = statistics.median(gemma.values()) if gemma else 0.0
        q_rank = _percentile_ranks(qwen)
        g_rank = _percentile_ranks(gemma)
        keep_utility = _utility(graph, keep, gold_by)
        alternatives = []
        for action_id in map(str, shortlist["challenger_action_ids"]):
            action = actions[action_id]
            q_center = max(-24.0, min(24.0, qwen[action_id] - q_median))
            g_center = max(-24.0, min(24.0, gemma[action_id] - g_median))
            review = [
                max(-24.0, min(24.0, qwen[action_id])) / 24.0,
                max(-24.0, min(24.0, gemma[action_id])) / 24.0,
                q_center / 24.0,
                g_center / 24.0,
                q_rank[action_id],
                g_rank[action_id],
                (q_center + g_center) / 48.0,
                min(q_center, g_center) / 24.0,
                float(q_center * g_center > 0.0),
                1.0 - abs(q_rank[action_id] - g_rank[action_id]) / 2.0,
                float(qwen[action_id] > 0.0),
                float(gemma[action_id] > 0.0),
            ]
            graph_values = compact_action_features(
                graph, action, {}, include_review=False)
            # compact_action_features always emits the three legacy review
            # slots; graph-only uses exactly the first 24 legal graph values.
            graph_values = graph_values[:len(BASE_FEATURE_NAMES)]
            alternatives.append(ActionExample(
                action=action,
                graph_features=tuple(graph_values),
                review_features=tuple(review),
                truth_features=tuple(_truth_features(
                    graph, action, keep, truth)),
                delta=_utility(graph, action, gold_by) - keep_utility,
            ))
        rows.append(RowExample(
            key=key,
            relation=key[1],
            graph=graph,
            keep=keep,
            alternatives=tuple(alternatives),
            fold=folds[key],
        ))
    return rows, {
        "source_plan": str(plan_path),
        "source_plan_sha256": sha256(plan_path),
        "review_plan": str(review_plan_path),
        "review_plan_sha256": sha256(review_plan_path),
        "truth_plan": str(Path(plan["truth_run"]) / "plan/PLAN.json"),
        "truth_plan_sha256": sha256(
            Path(plan["truth_run"]) / "plan/PLAN.json"),
        "folds": str(folds_path),
        "folds_sha256": sha256(folds_path),
        "control": str(control_path),
        "control_sha256": sha256(control_path),
        "gold": str(gold_path),
        "gold_sha256": sha256(gold_path),
        "truth_components": len(truth),
        "truth_schema": truth_plan["schema"],
        "rows": len(rows),
    }


def _training_arrays(
    rows: Sequence[RowExample], arm: str,
) -> tuple[list[list[float]], list[float], list[float]]:
    x, y, weights = [], [], []
    for row in rows:
        if not row.alternatives:
            continue
        row_weight = 1.0 / len(row.alternatives)
        for alternative in row.alternatives:
            x.append(_features(alternative, arm))
            y.append(float(alternative.delta))
            weights.append(row_weight)
    return x, y, weights


def _fit(
    rows: Sequence[RowExample], arm: str, l2: float,
) -> StandardizedLinear:
    x, y, weights = _training_arrays(rows, arm)
    return StandardizedLinear(
        _feature_names(arm), l2, logistic=False).fit(x, y, weights)


def _row_prediction(
    model: StandardizedLinear,
    row: RowExample,
) -> list[tuple[ActionExample, float]]:
    if not row.alternatives:
        return []
    values = model.predict([
        _features(item, _current_arm(model)) for item in row.alternatives])
    return list(zip(row.alternatives, map(float, values), strict=True))


def _current_arm(model: StandardizedLinear) -> str:
    names = tuple(model.feature_names)
    for arm in ARMS:
        if names == _feature_names(arm):
            return arm
    raise ContractError("unknown model feature schema")


def _decode_rows(
    rows: Sequence[RowExample],
    scores: Mapping[tuple[str, str], Sequence[tuple[ActionExample, float]]],
    guard: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics = [], []
    for row in rows:
        ranked = list(scores[row.key])
        best = max(
            ranked, key=lambda item: (item[1], str(item[0].action["id"])),
            default=None,
        )
        selected = (
            best[0] if best is not None and best[1] > guard else None)
        action = selected.action if selected is not None else row.keep
        actual_delta = selected.delta if selected is not None else 0.0
        predictions.append({
            "SubjectEntity": row.key[0],
            "Relation": row.key[1],
            "ObjectEntities": list(action["objects"]),
        })
        diagnostics.append({
            "SubjectEntity": row.key[0],
            "Relation": row.key[1],
            "fold": row.fold,
            "keep_action_id": str(row.keep["id"]),
            "selected_action_id": str(action["id"]),
            "changed": selected is not None,
            "predicted_delta": float(best[1]) if best is not None else 0.0,
            "actual_delta": float(actual_delta),
            "shortlist_oracle_delta": max(
                [0.0, *(item.delta for item in row.alternatives)]),
        })
    return predictions, diagnostics


def _control_predictions(rows: Sequence[RowExample]) -> list[dict[str, Any]]:
    return [{
        "SubjectEntity": row.key[0],
        "Relation": row.key[1],
        "ObjectEntities": list(row.keep["objects"]),
    } for row in rows]


def _relation_deltas(
    selected: Mapping[str, float], control: Mapping[str, float],
) -> dict[str, float]:
    return {
        relation: float(selected[relation]) - float(control[relation])
        for relation in control if relation != "*** All Relations ***"
    }


def _candidate_quality(
    predictions: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_scores = score(list(predictions), list(gold))
    control_scores = score(list(control), list(gold))
    changed = [row for row in diagnostics if row["changed"]]
    return {
        "score": selected_scores["*** All Relations ***"],
        "delta": (
            selected_scores["*** All Relations ***"]
            - control_scores["*** All Relations ***"]),
        "relation_deltas": _relation_deltas(selected_scores, control_scores),
        "changed": len(changed),
        "helped": sum(row["actual_delta"] > EPSILON for row in changed),
        "harmed": sum(row["actual_delta"] < -EPSILON for row in changed),
        "neutral": sum(
            abs(row["actual_delta"]) <= EPSILON for row in changed),
    }


def _inner_select(
    rows: Sequence[RowExample],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    arm: str,
) -> tuple[float, float, dict[str, Any]]:
    folds = sorted({row.fold for row in rows})
    combinations = {}
    for l2 in L2_GRID:
        oof_scores: dict[
            tuple[str, str], Sequence[tuple[ActionExample, float]],
        ] = {}
        for fold in folds:
            fit = [row for row in rows if row.fold != fold]
            hold = [row for row in rows if row.fold == fold]
            model = _fit(fit, arm, l2)
            for row in hold:
                oof_scores[row.key] = _row_prediction(model, row)
        gold = [gold_by[row.key] for row in rows]
        control = _control_predictions(rows)
        for guard in GUARD_GRID:
            predictions, diagnostics = _decode_rows(
                rows, oof_scores, guard)
            quality = _candidate_quality(
                predictions, diagnostics, control, gold)
            combinations[(l2, guard)] = quality
    # Metric-aligned nested selection. Ties choose fewer changes, the larger
    # guard, and stronger regularization in that order.
    selected = max(
        combinations,
        key=lambda item: (
            combinations[item]["score"],
            -combinations[item]["harmed"],
            -combinations[item]["changed"],
            item[1],
            item[0],
        ),
    )
    return selected[0], selected[1], {
        "l2": selected[0],
        "guard": selected[1],
        "quality": combinations[selected],
        "grid": [
            {"l2": l2, "guard": guard, **combinations[(l2, guard)]}
            for l2 in L2_GRID for guard in GUARD_GRID
        ],
    }


def _cross_fit(
    rows: Sequence[RowExample],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics, reports = [], [], []
    for outer in sorted({row.fold for row in rows}):
        fit = [row for row in rows if row.fold != outer]
        hold = [row for row in rows if row.fold == outer]
        l2, guard, selection = _inner_select(fit, gold_by, arm)
        model = _fit(fit, arm, l2)
        hold_scores = {
            row.key: _row_prediction(model, row) for row in hold}
        fold_predictions, fold_diagnostics = _decode_rows(
            hold, hold_scores, guard)
        fold_quality = _candidate_quality(
            fold_predictions,
            fold_diagnostics,
            _control_predictions(hold),
            [gold_by[row.key] for row in hold],
        )
        predictions.extend(fold_predictions)
        diagnostics.extend(fold_diagnostics)
        reports.append({
            "fold": outer,
            "fit_rows": len(fit),
            "hold_rows": len(hold),
            "selected_l2": l2,
            "selected_guard": guard,
            "inner_selection": selection,
            "holdout": fold_quality,
            "model": model.to_dict(),
        })
    by_key = {_key(row): row for row in predictions}
    diagnostics_by = {_key(row): row for row in diagnostics}
    return (
        [by_key[row.key] for row in rows],
        [diagnostics_by[row.key] for row in rows],
        reports,
    )


def analyze(args: argparse.Namespace) -> int:
    source = Path(args.source_run).resolve()
    gold_path = Path(args.gold).resolve()
    output = Path(args.output_dir).resolve()
    rows, provenance = load_examples(source, gold_path)
    gold_rows = read_jsonl(gold_path)
    gold_by = {_key(row): row for row in gold_rows}
    control = _control_predictions(rows)
    control_scores = score(control, gold_rows)
    artifacts, arm_results = {}, {}
    for arm in ARMS:
        predictions, diagnostics, folds = _cross_fit(rows, gold_by, arm)
        prediction_path = output / f"{arm}/TRAIN_OOF_PREDICTIONS.jsonl"
        diagnostic_path = output / f"{arm}/TRAIN_OOF_DIAGNOSTICS.jsonl"
        fold_path = output / f"{arm}/FOLDS.json"
        write_jsonl_atomic(prediction_path, predictions)
        write_jsonl_atomic(diagnostic_path, diagnostics)
        _write_json(fold_path, {"arm": arm, "folds": folds})
        quality = _candidate_quality(
            predictions, diagnostics, control, gold_rows)
        fold_deltas = [
            float(item["holdout"]["delta"]) for item in folds]
        artifacts[arm] = {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
            "folds": str(fold_path),
            "folds_sha256": sha256(fold_path),
        }
        arm_results[arm] = {
            **quality,
            "feature_names": list(_feature_names(arm)),
            "parameter_count": len(_feature_names(arm)) + 1,
            "fold_deltas": fold_deltas,
            "winning_folds": sum(value > 0.0 for value in fold_deltas),
            "worst_fold": min(fold_deltas),
        }

    primary = arm_results[PRIMARY_ARM]
    gate_checks = {
        "minimum_delta": primary["delta"] >= MIN_PROMOTION_DELTA,
        "minimum_winning_folds": (
            primary["winning_folds"] >= MIN_WINNING_FOLDS),
        "minimum_fold_delta": primary["worst_fold"] >= MIN_FOLD_DELTA,
        "minimum_relation_delta": (
            min(primary["relation_deltas"].values()) >= MIN_RELATION_DELTA),
        "beats_graph_only": (
            primary["delta"] > arm_results["graph_only"]["delta"]),
    }
    result = {
        "schema": "crossfit-action-utility-selector-result-v1",
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "primary_arm": PRIMARY_ARM,
        "control_scores": control_scores,
        "arms": arm_results,
        "artifacts": artifacts,
        "promotion_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
            "minimum_delta": MIN_PROMOTION_DELTA,
            "minimum_winning_folds": MIN_WINNING_FOLDS,
            "minimum_fold_delta": MIN_FOLD_DELTA,
            "minimum_relation_delta": MIN_RELATION_DELTA,
        },
        "failure_ledger": {
            "rows": len(rows),
            "shortlist_oracle_rows": sum(
                max([0.0, *(item.delta for item in row.alternatives)])
                > EPSILON for row in rows),
            "helpful_selected": primary["helped"],
            "harmful_selected": primary["harmed"],
            "helpful_shortlist_not_recovered": sum(
                row["shortlist_oracle_delta"] > EPSILON
                and row["actual_delta"] <= EPSILON
                for row in read_jsonl(
                    Path(artifacts[PRIMARY_ARM]["diagnostics"]))
            ),
        },
        "provenance": provenance,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "development_only": True,
        "deployable": False,
        "next_stage": (
            "freeze_full_train_model_and_prepare_validation"
            if all(gate_checks.values())
            else "reject_or_revise_action_utility_selector"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "RESULT.json"
    _write_json(result_path, result)
    markdown = [
        "# Cross-fitted action-utility selector",
        "",
        "Train-only subject-grouped nested cross-fitting; validation was not opened.",
        "",
        f"- Control: `{control_scores['*** All Relations ***']:.9f}`",
        f"- Primary: `{primary['score']:.9f}`",
        f"- Delta: `{primary['delta']:+.9f}`",
        f"- Changed/helped/harmed/neutral: "
        f"`{primary['changed']}/{primary['helped']}/"
        f"{primary['harmed']}/{primary['neutral']}`",
        f"- Winning folds: `{primary['winning_folds']}/5`",
        f"- Promotion gate: "
        f"`{'PASS' if result['promotion_gate']['passed'] else 'FAIL'}`",
        "",
        "## Arms",
        "",
        "| Arm | Delta | Changed | Helped | Harmed |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = arm_results[arm]
        markdown.append(
            f"| {arm} | {item['delta']:+.6f} | {item['changed']} | "
            f"{item['helped']} | {item['harmed']} |")
    markdown.extend([
        "",
        "## Failure ledger",
        "",
        "```json",
        json.dumps(result["failure_ledger"], indent=2, sort_keys=True),
        "```",
        "",
    ])
    (output / "RESULT.md").write_text("\n".join(markdown))
    print(json.dumps({
        "control": control_scores["*** All Relations ***"],
        "primary": primary,
        "promotion_gate": result["promotion_gate"],
        "arms": {
            arm: {
                "delta": item["delta"],
                "changed": item["changed"],
                "helped": item["helped"],
                "harmed": item["harmed"],
            }
            for arm, item in arm_results.items()
        },
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
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
