#!/usr/bin/env python3
"""Low-capacity row-grouped ranking of complete heterogeneous-memory actions.

This experiment targets one failure mode: the existing selector treats many
correlated actions as if they were independent examples and then converts
candidate probabilities into complete outputs with a separately fitted gate.
Here, one subject-relation row is one weighted training group.  Every legal
complete output, including the exact SOTA incumbent as an explicit KEEP
action, is ranked jointly using pairwise official row-F1 preferences.

The primary model has one fixed regularization value and 27 coefficients.
There is no learned intercept outside the action representation, no tuned
switching threshold, no per-relation hyperparameter search, and no fallback
chosen after seeing held-out labels.  Validation is never opened.  The only
promotion result is subject-grouped out-of-fold improvement over the exact
registered SOTA train predictions.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.baseline_conditioned_action_review import (
    _json,
    _validated_responses,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    validate_registered_predictions,
)
from experiments.heterogeneous_agents.unified_memory_action_graph import (
    ACTION_TYPES,
    FEATURE_NAMES,
    RELATIONS,
    _key,
    _row_f1,
)


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_REVIEW_RUN = (
    RUNS / "baseline_conditioned_action_review_20260727_v2")
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_OUTPUT = RUNS / "row_grouped_action_ranker_20260727_v2"

# Fixed before the OOF result is produced.  A graph-only arm is reported as a
# diagnostic ablation but can never replace the predeclared primary arm.
PRIMARY_ARM = "graph_plus_review"
ARMS = ("graph_only", PRIMARY_ARM)
FIXED_L2 = 10.0
EPSILON = 1e-8

MIN_POOLED_DELTA = 0.005
MIN_POSITIVE_FOLDS = 4
MIN_RELATION_DELTA = -0.010

BASE_FEATURE_NAMES = (
    # Complete action identity, including KEEP.
    "action_keep", "action_collapse", "action_empty",
    "action_replace", "action_add", "action_drop",
    # A low-capacity family-specific cost of leaving the incumbent.
    "switch_numeric", "switch_single", "switch_list",
    # State transition geometry.
    "size_delta", "changed_component_count",
    # Contrastive memory evidence.
    "joint_support_delta",
    "qwen_self_consistency_delta", "qwen_system2_delta",
    "added_joint_support", "removed_joint_support",
    "added_cross_memory", "removed_cross_memory",
    # Numeric and cardinality messages.
    "numeric_log_distance",
    "empty_null_support", "empty_nonnull_conflict",
    "add_many_support", "drop_zero_support", "replace_one_support",
)
REVIEW_FEATURE_NAMES = (
    "qwen_review_log_odds",
    "gemma_review_log_odds",
    "mean_review_uncertainty",
)
COMPACT_FEATURE_NAMES = BASE_FEATURE_NAMES + REVIEW_FEATURE_NAMES

if len(COMPACT_FEATURE_NAMES) != 27:
    raise AssertionError("row-grouped ranker must remain a 27-parameter model")

_FULL_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _probability_log_odds(value: Mapping[str, float]) -> float:
    use = max(float(value["USE_ALTERNATIVE"]), EPSILON)
    keep = max(float(value["KEEP_CURRENT"]), EPSILON)
    return max(-12.0, min(12.0, math.log(use) - math.log(keep))) / 12.0


def compact_action_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    *,
    include_review: bool,
) -> list[float]:
    """Return the fixed compact representation of one complete output."""
    full = list(map(float, action["_inference_features"]))
    if len(full) != len(FEATURE_NAMES):
        raise ContractError("row-grouped source feature schema drift")

    def feature(name: str) -> float:
        return full[_FULL_INDEX[name]]

    action_type = str(action["action_type"])
    alternative = float(action_type != "KEEP")
    family = str(graph["relation_family"])
    q_odds = g_odds = uncertainty = 0.0
    if include_review and alternative:
        key = _key(graph), str(action["id"])
        q = evidence[key + (QWEN,)]
        g = evidence[key + (GEMMA,)]
        q_odds = _probability_log_odds(q)
        g_odds = _probability_log_odds(g)
        uncertainty = (
            float(q["UNCERTAIN"]) + float(g["UNCERTAIN"])) / 2.0

    values = [
        *(float(action_type == item) for item in ACTION_TYPES),
        alternative * float(family == "numeric"),
        alternative * float(family == "single"),
        alternative * float(family == "list"),
        feature("size_delta"),
        feature("changed_component_count"),
        feature("joint_support_delta"),
        feature("qwen_self_consistency_delta"),
        feature("qwen_system2_delta"),
        feature("added_qwen_support") + feature("added_gemma_support"),
        feature("removed_qwen_support") + feature("removed_gemma_support"),
        feature("added_cross_memory"),
        feature("removed_cross_memory"),
        feature("numeric_log_distance"),
        feature("empty_null_support"),
        feature("empty_nonnull_conflict"),
        feature("add_many_support"),
        feature("drop_zero_support"),
        feature("replace_one_support"),
        q_odds,
        g_odds,
        uncertainty,
    ]
    if len(values) != len(COMPACT_FEATURE_NAMES):
        raise AssertionError("compact action feature schema drift")
    if not all(math.isfinite(value) for value in values):
        raise ContractError("non-finite compact action feature")
    return values


def oriented_pair_examples(
    features: Sequence[Sequence[float]],
    utilities: Sequence[float],
    *,
    keep_index: int,
) -> tuple[list[list[float]], list[float]]:
    """Create symmetric pairwise examples with KEEP winning utility ties.

    Symmetry removes the need for a free logistic intercept.  Ties between two
    alternatives carry no ranking information.  A tie between KEEP and an
    alternative is deliberately resolved in favor of KEEP, preventing
    score-neutral churn.
    """
    if (
        len(features) != len(utilities)
        or not 0 <= keep_index < len(features)
        or any(len(row) != len(COMPACT_FEATURE_NAMES) for row in features)
    ):
        raise ValueError("invalid grouped pairwise arrays")
    output_x: list[list[float]] = []
    output_y: list[float] = []
    for left, right in itertools.combinations(range(len(features)), 2):
        delta = float(utilities[left]) - float(utilities[right])
        if abs(delta) <= 1e-12:
            if keep_index not in {left, right}:
                continue
            winner = keep_index
            loser = right if winner == left else left
        else:
            winner, loser = (
                (left, right) if delta > 0.0 else (right, left))
        difference = (
            np.asarray(features[winner], dtype=np.float64)
            - np.asarray(features[loser], dtype=np.float64))
        output_x.append(difference.tolist())
        output_y.append(1.0)
        output_x.append((-difference).tolist())
        output_y.append(0.0)
    return output_x, output_y


class PairwiseRanker:
    """Deterministic regularized conditional-logit action ranker."""

    def __init__(self, l2: float = FIXED_L2):
        self.l2 = float(l2)
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    @property
    def parameter_count(self) -> int:
        return len(COMPACT_FEATURE_NAMES)

    def fit(
        self,
        groups: Sequence[tuple[Sequence[Sequence[float]], Sequence[float], int]],
    ) -> "PairwiseRanker":
        # The action registry contains many more rows where KEEP is optimal
        # than rows with a beneficial switch. Balance these two kinds of
        # independent row groups, not their correlated action pairs.
        improvable = [
            max(map(float, utilities))
            > float(utilities[keep_index]) + 1e-12
            for _, utilities, keep_index in groups]
        class_counts = {
            value: sum(item == value for item in improvable)
            for value in (False, True)}
        if min(class_counts.values()) <= 0:
            raise ValueError("row-grouped ranker needs both row classes")
        matrix_rows: list[list[float]] = []
        targets: list[float] = []
        weights: list[float] = []
        for (features, utilities, keep_index), positive in zip(
            groups, improvable, strict=True,
        ):
            x, y = oriented_pair_examples(
                features, utilities, keep_index=keep_index)
            if not x:
                continue
            # Improvable and keep-optimal rows each contribute half of the
            # loss. Within each class, every subject-relation row is equal.
            row_weight = (
                0.5 / class_counts[positive] / len(x))
            matrix_rows.extend(x)
            targets.extend(y)
            weights.extend([row_weight] * len(x))
        matrix = np.asarray(matrix_rows, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(COMPACT_FEATURE_NAMES)
            or matrix.shape[0] != len(target)
            or weight.shape != target.shape
            or set(np.unique(target)) != {0.0, 1.0}
        ):
            raise ValueError("invalid row-grouped ranking data")
        weight *= len(weight) / weight.sum()
        # Symmetric pair construction has exactly zero weighted mean.
        variance = np.average(matrix**2, axis=0, weights=weight)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = matrix / self.scale
        beta = np.zeros(design.shape[1], dtype=np.float64)
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.l2
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

    def scores(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        if self.scale is None or self.coef is None:
            raise RuntimeError("row-grouped ranker is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        return (matrix / self.scale) @ self.coef

    def to_dict(self) -> dict[str, Any]:
        if self.scale is None or self.coef is None:
            raise RuntimeError("row-grouped ranker is not fitted")
        return {
            "schema": "row-grouped-complete-action-ranker-v1",
            "l2": self.l2,
            "feature_names": list(COMPACT_FEATURE_NAMES),
            "scale": self.scale.tolist(),
            "coefficients": self.coef.tolist(),
            "parameter_count": self.parameter_count,
            "explicit_keep_action": True,
            "row_equal_weighting": True,
            "improvable_keep_row_class_balance": True,
            "neutral_action_preference": "KEEP",
        }


def _utility(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    return _row_f1(
        action["objects"], gold_by[_key(graph)], str(graph["Relation"]))


def _groups(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    arm: str,
) -> list[tuple[list[list[float]], list[float], int]]:
    include_review = arm == PRIMARY_ARM
    result = []
    for graph in graphs:
        actions = list(graph["actions"])
        keep = [
            index for index, action in enumerate(actions)
            if action["action_type"] == "KEEP"]
        if len(keep) != 1:
            raise ContractError("row must contain exactly one KEEP action")
        result.append((
            [
                compact_action_features(
                    graph, action, evidence,
                    include_review=include_review)
                for action in actions
            ],
            [_utility(graph, action, gold_by) for action in actions],
            keep[0],
        ))
    return result


def _decode(
    model: PairwiseRanker,
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    include_review = arm == PRIMARY_ARM
    predictions, diagnostics = [], []
    for graph in graphs:
        actions = list(graph["actions"])
        features = [
            compact_action_features(
                graph, action, evidence, include_review=include_review)
            for action in actions]
        values = model.scores(features)
        keep_index = next(
            index for index, action in enumerate(actions)
            if action["action_type"] == "KEEP")
        selected = max(
            range(len(actions)),
            key=lambda index: (
                float(values[index]),
                index == keep_index,
                -len(actions[index]["objects"]),
                -index,
            ))
        action = actions[selected]
        before = _utility(graph, actions[keep_index], gold_by)
        after = _utility(graph, action, gold_by)
        best = max(_utility(graph, item, gold_by) for item in actions)
        predictions.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": list(action["objects"]),
        })
        diagnostics.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "selected_action": action["id"],
            "selected_action_type": action["action_type"],
            "changed": selected != keep_index,
            "selected_score": float(values[selected]),
            "keep_score": float(values[keep_index]),
            "utility_delta": after - before,
            "oracle_regret": best - after,
            "selected_is_optimal": after >= best - 1e-12,
            "row_has_beneficial_action": best > before + 1e-12,
        })
    return predictions, diagnostics


def _relation_deltas(
    selected: Mapping[str, float], control: Mapping[str, float],
) -> dict[str, float]:
    return {
        relation: float(selected[relation]) - float(control[relation])
        for relation in RELATIONS}


def _load(
    review_run: Path, gold_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], int],
    Mapping[tuple[tuple[str, str], str, str], Mapping[str, float]],
]:
    plan_path = review_run / "plan/PLAN.json"
    plan = _json(plan_path)
    registry = Path(plan["registry"])
    control_path = Path(plan["starting_predictions"])
    folds_path = Path(plan["folds"])
    if (
        plan.get("schema") != "baseline-conditioned-action-review-plan-v1"
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("starting_pipeline_id") != COMPETITION_PIPELINE_ID
        or sha256(registry) != plan["registry_sha256"]
        or sha256(control_path) != plan["starting_predictions_sha256"]
        or sha256(folds_path) != plan["folds_sha256"]
    ):
        raise ContractError("invalid source action-review plan")
    validate_registered_predictions(
        control_path, pipeline_id=COMPETITION_PIPELINE_ID, split="train")
    evidence = _validated_responses(plan)
    graphs = read_jsonl(registry)
    control_rows = read_jsonl(control_path)
    control = {_key(row): row for row in control_rows}
    gold_rows = read_jsonl(gold_path)
    gold_by = {_key(row): row for row in gold_rows}
    folds = {
        _key(row): int(row["fold"]) for row in read_jsonl(folds_path)}
    keys = {_key(row) for row in graphs}
    if (
        len(graphs) != int(plan["rows"])
        or keys != set(control) or keys != set(gold_by) or keys != set(folds)
        or sum(len(row["actions"]) for row in graphs)
        != int(plan["actions"])
    ):
        raise ContractError("row-grouped source coverage mismatch")
    return plan, graphs, control_rows, gold_by, folds, evidence


def _evaluate_arm(
    arm: str,
    graphs: Sequence[Mapping[str, Any]],
    control_by: Mapping[tuple[str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    folds: Mapping[tuple[str, str], int],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    oof: dict[tuple[str, str], dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    fold_results = []
    for outer in sorted(set(folds.values())):
        fit = [row for row in graphs if folds[_key(row)] != outer]
        hold = [row for row in graphs if folds[_key(row)] == outer]
        model = PairwiseRanker().fit(
            _groups(fit, gold_by, evidence, arm))
        predictions, detail = _decode(
            model, hold, gold_by, evidence, arm)
        for row in predictions:
            if _key(row) in oof:
                raise ContractError("duplicate row-grouped OOF prediction")
            oof[_key(row)] = row
        diagnostics.extend(
            {**item, "outer_fold": outer, "arm": arm}
            for item in detail)
        hold_gold = [gold_by[_key(row)] for row in hold]
        hold_control = [control_by[_key(row)] for row in hold]
        control_score = score(
            hold_control, hold_gold)["*** All Relations ***"]
        selected_score = score(
            predictions, hold_gold)["*** All Relations ***"]
        fold_results.append({
            "fold": outer,
            "fit_rows": len(fit),
            "hold_rows": len(hold),
            "control": control_score,
            "selected": selected_score,
            "delta": selected_score - control_score,
            "changed": sum(item["changed"] for item in detail),
            "helped": sum(item["utility_delta"] > 1e-12 for item in detail),
            "harmed": sum(item["utility_delta"] < -1e-12 for item in detail),
            "neutral": sum(
                abs(item["utility_delta"]) <= 1e-12
                and item["changed"] for item in detail),
            "mean_oracle_regret": float(np.mean([
                item["oracle_regret"] for item in detail])),
        })
    if set(oof) != {_key(row) for row in graphs}:
        raise ContractError("row-grouped OOF coverage failure")
    ordered = [oof[_key(row)] for row in graphs]
    ordered_gold = [gold_by[_key(row)] for row in graphs]
    ordered_control = [control_by[_key(row)] for row in graphs]
    control_scores = score(ordered_control, ordered_gold)
    selected_scores = score(ordered, ordered_gold)
    changed = [item for item in diagnostics if item["changed"]]
    result = {
        "arm": arm,
        "control_scores": control_scores,
        "selected_scores": selected_scores,
        "incremental_delta": (
            selected_scores["*** All Relations ***"]
            - control_scores["*** All Relations ***"]),
        "relation_deltas": _relation_deltas(
            selected_scores, control_scores),
        "changed_rows": len(changed),
        "helped_rows": sum(
            item["utility_delta"] > 1e-12 for item in changed),
        "harmed_rows": sum(
            item["utility_delta"] < -1e-12 for item in changed),
        "neutral_changed_rows": sum(
            abs(item["utility_delta"]) <= 1e-12 for item in changed),
        "positive_folds": sum(item["delta"] > 0.0 for item in fold_results),
        "folds": fold_results,
        "mean_oracle_regret": float(np.mean([
            item["oracle_regret"] for item in diagnostics])),
        "optimal_action_rate": float(np.mean([
            item["selected_is_optimal"] for item in diagnostics])),
        "beneficial_row_recall": (
            sum(
                item["row_has_beneficial_action"]
                and item["utility_delta"] > 1e-12
                for item in diagnostics)
            / max(1, sum(
                item["row_has_beneficial_action"]
                for item in diagnostics))
        ),
    }
    return result, ordered, diagnostics


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    review_run = Path(args.review_run).resolve()
    gold_path = Path(args.gold).resolve()
    plan, graphs, control_rows, gold_by, folds, evidence = _load(
        review_run, gold_path)
    control_by = {_key(row): row for row in control_rows}

    arm_results = {}
    for arm in ARMS:
        print(f"row-grouped OOF arm={arm}", flush=True)
        result, predictions, diagnostics = _evaluate_arm(
            arm, graphs, control_by, gold_by, folds, evidence)
        prediction_path = output / f"analysis/{arm}_OOF_PREDICTIONS.jsonl"
        diagnostic_path = output / f"analysis/{arm}_OOF_DIAGNOSTICS.jsonl"
        write_jsonl_atomic(prediction_path, predictions)
        write_jsonl_atomic(diagnostic_path, diagnostics)
        result["artifacts"] = {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
        }
        arm_results[arm] = result

    primary = arm_results[PRIMARY_ARM]
    passed = (
        primary["incremental_delta"] >= MIN_POOLED_DELTA
        and primary["positive_folds"] >= MIN_POSITIVE_FOLDS
        and min(primary["relation_deltas"].values())
        >= MIN_RELATION_DELTA
        and primary["helped_rows"] > primary["harmed_rows"]
    )

    final_model = PairwiseRanker().fit(
        _groups(graphs, gold_by, evidence, PRIMARY_ARM))
    model_path = output / "analysis/TRAIN_FIT_MODEL.json"
    _write_json(model_path, {
        **final_model.to_dict(),
        "development_only": True,
        "contains_labels": True,
        "gold_aware": True,
        "validation_deployable": False,
        "promotion_gate_passed": passed,
        "source_review_plan": str(review_run / "plan/PLAN.json"),
        "source_review_plan_sha256": sha256(
            review_run / "plan/PLAN.json"),
    })

    result = {
        "schema": "row-grouped-action-ranker-result-v1",
        "development_only": True,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "independent_rows": len(graphs),
        "actions": sum(len(row["actions"]) for row in graphs),
        "parameter_count": len(COMPACT_FEATURE_NAMES),
        "fixed_l2": FIXED_L2,
        "primary_arm": PRIMARY_ARM,
        "arms": arm_results,
        "deployment_gate": {
            "passed": passed,
            "minimum_pooled_delta": MIN_POOLED_DELTA,
            "minimum_positive_folds": MIN_POSITIVE_FOLDS,
            "minimum_relation_delta": MIN_RELATION_DELTA,
            "requires_helped_gt_harmed": True,
        },
        "sources": {
            "review_run": str(review_run),
            "review_plan_sha256": sha256(
                review_run / "plan/PLAN.json"),
            "registry_sha256": plan["registry_sha256"],
            "gold": str(gold_path),
            "gold_sha256": sha256(gold_path),
        },
        "artifacts": {
            "model": str(model_path),
            "model_sha256": sha256(model_path),
        },
    }
    result_path = output / "analysis/RESULT.json"
    _write_json(result_path, result)

    lines = [
        "# Row-grouped complete-action ranker", "",
        "Train-only, exact-SOTA, subject-grouped OOF audit. Validation was "
        "not opened.", "",
        "| arm | control | selected | delta | changed | helped | harmed | "
        "positive folds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = arm_results[arm]
        lines.append(
            f"| {arm} | "
            f"{item['control_scores']['*** All Relations ***']:.6f} | "
            f"{item['selected_scores']['*** All Relations ***']:.6f} | "
            f"{item['incremental_delta']:+.6f} | "
            f"{item['changed_rows']} | {item['helped_rows']} | "
            f"{item['harmed_rows']} | {item['positive_folds']}/5 |")
    lines.extend([
        "",
        f"Primary arm: **{PRIMARY_ARM}**. Fixed L2: **{FIXED_L2}**. "
        f"Parameters: **{len(COMPACT_FEATURE_NAMES)}**.",
        f"Predeclared broad promotion gate: **{passed}**.",
        "",
        "The graph-only arm is diagnostic and was not eligible to replace the "
        "predeclared primary arm after evaluation.",
    ])
    (output / "analysis/RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "primary_delta": primary["incremental_delta"],
        "primary_positive_folds": primary["positive_folds"],
        "primary_helped": primary["helped_rows"],
        "primary_harmed": primary["harmed_rows"],
        "graph_only_delta": arm_results["graph_only"]["incremental_delta"],
        "gate_passed": passed,
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--review-run", default=str(DEFAULT_REVIEW_RUN))
    value.add_argument("--gold", default=str(DEFAULT_GOLD))
    value.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return value


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
