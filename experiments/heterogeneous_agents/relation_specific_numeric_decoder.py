#!/usr/bin/env python3
"""Relation-specific risk decoding for heterogeneous numeric candidates.

This is a CPU-only development ablation over already frozen, label-free
candidate graphs.  It deliberately separates the leakage boundary:

* graph loading verifies label exclusion and hashes;
* numeric option models and guard margins are selected with five-fold train
  out-of-fold predictions;
* validation graphs are decoded before validation labels are opened;
* validation labels are used only for the final confirmation report.

The decoder treats every legal numeric output as an option.  Options include
the frozen Qwen baseline, raw Qwen/Gemma candidates, and robust representatives
of local 5%-tolerance clusters.  A separate small logistic model is fit for
``hasArea`` and ``hasCapacity`` using inference-time evidence only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate import RELATION_TYPE, true_positives
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    NUMERIC_RELATIONS,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.heterogeneous_memory_selector import (
    LogisticCalibrator,
    _key,
    _load_graph,
    _numeric_value,
    _weighted_median,
)


RELATIONS = tuple(sorted(NUMERIC_RELATIONS))
OPTION_SCHEMA = "relation-specific-numeric-option-v1"
FEATURE_SCHEMA = "relation-specific-numeric-features-v1"
DEFAULT_MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _within_tolerance(left: float, right: float, tolerance: float = 0.05) -> bool:
    """Symmetric neighborhood used only to construct inference-time clusters."""
    if left <= 0 or right <= 0:
        return False
    return abs(left - right) / max(abs(right), 1e-12) <= tolerance


def _same_option(left: float, right: float) -> bool:
    return abs(math.log(left / right)) <= 1e-10


def _source_support(node: Mapping[str, Any], agent: str) -> float:
    source = node.get("sources", {}).get(agent)
    return float(source.get("support", 0.0)) if source else 0.0


def _source_samples(graph: Mapping[str, Any], agent: str) -> float:
    value = float(graph["agents"][agent].get("n_samples", 0.0))
    return max(1.0, value)


def _numeric_mad(graph: Mapping[str, Any], agent: str) -> float:
    value = graph["agents"][agent].get("numeric_log_mad")
    if value is None:
        return 0.0
    value = float(value)
    return min(1.0, max(0.0, value) / 2.0)


def _add_option(options: list[dict], value: float | None, kind: str) -> None:
    if value is None or not math.isfinite(value) or value <= 0:
        return
    for option in options:
        if _same_option(float(option["value"]), value):
            option["kinds"].add(kind)
            return
    options.append({
        "schema": OPTION_SCHEMA,
        "value": float(value),
        "kinds": {kind},
    })


def numeric_options(graph: Mapping[str, Any]) -> list[dict]:
    """Enumerate legal outputs without labels.

    Weighted medians preserve an observed value.  Weighted geometric means add
    a smooth representative on the positive numeric scale and are restricted
    to a local 5%-neighborhood so they cannot average unrelated magnitudes.
    """
    if graph["Relation"] not in NUMERIC_RELATIONS:
        raise ContractError(f"numeric option request for nonnumeric row: {_key(graph)}")
    options: list[dict] = []
    for item in graph.get("baseline_objects", []):
        _add_option(options, _numeric_value(item), "baseline")
    nodes: list[tuple[float, Mapping[str, Any]]] = []
    for node in graph.get("candidates", []):
        value = _numeric_value(node.get("item"))
        if value is None:
            continue
        nodes.append((value, node))
        _add_option(options, value, "node")
    for anchor, _ in nodes:
        members = [
            (value, node) for value, node in nodes
            if _within_tolerance(value, anchor)
        ]
        weights = [
            sum(_source_support(node, agent) for agent in (QWEN, GEMMA))
            for _, node in members
        ]
        if not members or sum(weights) <= 0:
            continue
        _add_option(
            options,
            _weighted_median([
                (value, weight)
                for (value, _), weight in zip(members, weights)
            ]),
            "cluster_median",
        )
        log_mean = sum(
            math.log(value) * weight
            for (value, _), weight in zip(members, weights)
        ) / sum(weights)
        _add_option(options, math.exp(log_mean), "cluster_geomean")
    if not options:
        raise ContractError(f"numeric graph has no legal output option: {_key(graph)}")
    if not any("baseline" in option["kinds"] for option in options):
        raise ContractError(f"numeric graph has no legal baseline: {_key(graph)}")
    return options


def feature_names() -> list[str]:
    return [
        "intercept",
        "is_baseline",
        "is_raw_node",
        "is_cluster_median",
        "is_cluster_geomean",
        "qwen_support_mass",
        "gemma_support_mass",
        "cross_model_neighborhood",
        "qwen_selected_neighborhood",
        "gemma_selected_neighborhood",
        "neighborhood_node_count",
        "neighborhood_total_mass",
        "qwen_numeric_log_mad",
        "gemma_numeric_log_mad",
        "log_distance_from_baseline",
        "candidate_count",
    ]


def option_features(graph: Mapping[str, Any], option: Mapping[str, Any]) -> list[float]:
    value = float(option["value"])
    near = []
    for node in graph.get("candidates", []):
        candidate = _numeric_value(node.get("item"))
        if candidate is not None and _within_tolerance(candidate, value):
            near.append(node)
    qmass = sum(_source_support(node, QWEN) for node in near)
    gmass = sum(_source_support(node, GEMMA) for node in near)
    qrate = min(1.0, qmass / _source_samples(graph, QWEN))
    grate = min(1.0, gmass / _source_samples(graph, GEMMA))
    baseline_values = [
        _numeric_value(item) for item in graph.get("baseline_objects", [])]
    baseline_values = [item for item in baseline_values if item is not None]
    if not baseline_values:
        raise ContractError(f"missing numeric baseline: {_key(graph)}")
    baseline = baseline_values[0]
    kinds = set(option["kinds"])
    values = [
        1.0,
        float("baseline" in kinds),
        float("node" in kinds),
        float("cluster_median" in kinds),
        float("cluster_geomean" in kinds),
        qrate,
        grate,
        float(qmass > 0 and gmass > 0),
        float(any(node.get("selected_by", {}).get(QWEN, False) for node in near)),
        float(any(node.get("selected_by", {}).get(GEMMA, False) for node in near)),
        min(1.0, len(near) / 5.0),
        min(1.0, (qrate + grate) / 2.0),
        _numeric_mad(graph, QWEN),
        _numeric_mad(graph, GEMMA),
        min(1.0, abs(math.log(value / baseline)) / 5.0),
        min(1.0, len(graph.get("candidates", [])) / 11.0),
    ]
    if len(values) != len(feature_names()):
        raise AssertionError("numeric option feature schema drift")
    if not all(math.isfinite(item) for item in values):
        raise ContractError(f"non-finite numeric option features: {_key(graph)}")
    return values


def _gold_aliases(gold: Mapping[str, Any]) -> list[list[str]]:
    values = gold.get("ObjectEntities", [])
    return [[str(item)] for item in values] if values and isinstance(values[0], str) else values


def option_label(
        graph: Mapping[str, Any], option: Mapping[str, Any],
        gold: Mapping[str, Any],
) -> float:
    prediction = format(float(option["value"]), ".12g")
    return float(true_positives(
        [prediction], _gold_aliases(gold),
        RELATION_TYPE[str(graph["Relation"])], 0.05,
    ) > 0)


class RelationSpecificNumericModel:
    """One option-correctness calibrator per numeric relation."""

    def __init__(self, l2: float = 2.0):
        self.l2 = float(l2)
        self.models: dict[str, LogisticCalibrator] = {}

    def fit(
            self, graphs: Sequence[Mapping[str, Any]],
            gold_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> "RelationSpecificNumericModel":
        for relation in RELATIONS:
            subset = [graph for graph in graphs if graph["Relation"] == relation]
            if not subset:
                raise ContractError(f"no numeric training rows for {relation}")
            x, y, weights = [], [], []
            for graph in subset:
                if _key(graph) not in gold_by_key:
                    raise ContractError(f"missing training gold for {_key(graph)}")
                options = numeric_options(graph)
                row_weight = 1.0 / len(options)
                for option in options:
                    x.append(option_features(graph, option))
                    y.append(option_label(graph, option, gold_by_key[_key(graph)]))
                    weights.append(row_weight)
            self.models[relation] = LogisticCalibrator(
                feature_names(), l2=self.l2).fit(x, y, weights)
        return self

    def score_options(
            self, graph: Mapping[str, Any],
    ) -> tuple[list[dict], np.ndarray]:
        relation = str(graph["Relation"])
        if relation not in self.models:
            raise ContractError(f"numeric model missing relation {relation}")
        options = numeric_options(graph)
        probabilities = self.models[relation].predict([
            option_features(graph, option) for option in options])
        return options, probabilities

    def decode(
            self, graph: Mapping[str, Any], margin: float,
    ) -> tuple[list[str], dict[str, Any]]:
        options, probabilities = self.score_options(graph)
        baseline_index = next(
            index for index, option in enumerate(options)
            if "baseline" in option["kinds"])
        best_index = max(
            range(len(options)),
            key=lambda index: (float(probabilities[index]), -index),
        )
        improvement = (
            float(probabilities[best_index])
            - float(probabilities[baseline_index]))
        selected_index = best_index if improvement > margin else baseline_index
        selected = options[selected_index]
        return [format(float(selected["value"]), ".12g")], {
            "relation": graph["Relation"],
            "selected_value": float(selected["value"]),
            "selected_kinds": sorted(selected["kinds"]),
            "selected_probability": float(probabilities[selected_index]),
            "best_probability": float(probabilities[best_index]),
            "baseline_value": float(options[baseline_index]["value"]),
            "baseline_probability": float(probabilities[baseline_index]),
            "estimated_improvement": improvement,
            "guard_margin": float(margin),
            "used_baseline": selected_index == baseline_index,
            "options": [
                {
                    "value": float(option["value"]),
                    "kinds": sorted(option["kinds"]),
                    "probability": float(probability),
                }
                for option, probability in zip(options, probabilities)
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "relation-specific-numeric-model-v1",
            "feature_schema": FEATURE_SCHEMA,
            "feature_names": feature_names(),
            "l2": self.l2,
            "models": {
                relation: model.to_dict()
                for relation, model in sorted(self.models.items())
            },
        }


def _numeric_correct(
        graph: Mapping[str, Any], objects: Sequence[str],
        gold: Mapping[str, Any],
) -> float:
    return float(true_positives(
        list(objects), _gold_aliases(gold),
        RELATION_TYPE[str(graph["Relation"])], 0.05,
    ) > 0)


def _fold_policy_diagnostics(
        fold_scores: Mapping[float, Mapping[int, float]],
        baseline_scores: Mapping[int, float],
) -> tuple[float, float, dict[str, Any]]:
    fold_ids = sorted(baseline_scores)
    diagnostics: dict[str, Any] = {}
    for margin, values in sorted(fold_scores.items()):
        if sorted(values) != fold_ids:
            raise ContractError(f"margin {margin} does not cover every fold")
        deltas = [
            float(values[fold] - baseline_scores[fold]) for fold in fold_ids]
        mean_delta = statistics.mean(deltas)
        standard_error = statistics.stdev(deltas) / math.sqrt(len(deltas))
        diagnostics[str(margin)] = {
            "fold_accuracy": {
                str(fold): float(values[fold]) for fold in fold_ids},
            "baseline_fold_accuracy": {
                str(fold): float(baseline_scores[fold]) for fold in fold_ids},
            "paired_fold_delta": {
                str(fold): delta for fold, delta in zip(fold_ids, deltas)},
            "mean_paired_delta": mean_delta,
            "standard_error_paired_delta": standard_error,
        }
    best = max(
        fold_scores,
        key=lambda margin: (
            diagnostics[str(margin)]["mean_paired_delta"], margin),
    )
    threshold = (
        diagnostics[str(best)]["mean_paired_delta"]
        - diagnostics[str(best)]["standard_error_paired_delta"])
    eligible = [
        margin for margin in fold_scores
        if diagnostics[str(margin)]["mean_paired_delta"] >= threshold - 1e-12
    ]
    one_se = max(eligible)
    for margin in fold_scores:
        diagnostics[str(margin)]["within_one_se_of_best"] = margin in eligible
    return float(best), float(one_se), {
        "best_mean_margin": float(best),
        "one_standard_error_margin": float(one_se),
        "one_standard_error_threshold": float(threshold),
        "margins": diagnostics,
    }


def _stable_relation(
        selection: Mapping[str, Any], best_margin: float,
        tolerance: float = 1e-12,
) -> bool:
    """Require positive mean gain and no harmful outer fold.

    This is stricter than selecting the largest mean alone and is evaluated
    entirely on training OOF predictions.  It prevents a noisy relation from
    rewriting validation/test rows merely because gains on a few folds cancel
    a large loss on another fold.
    """
    record = selection["margins"][str(best_margin)]
    deltas = [float(value) for value in record["paired_fold_delta"].values()]
    return (
        float(record["mean_paired_delta"]) > tolerance
        and min(deltas) >= -tolerance
    )


def _merge_numeric(
        control_rows: Sequence[Mapping[str, Any]],
        numeric_rows: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict]:
    merged = []
    control_keys = {_key(row) for row in control_rows}
    if not set(numeric_rows) <= control_keys:
        raise ContractError("numeric predictions are not covered by control")
    for row in control_rows:
        key = _key(row)
        merged.append({
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": list(numeric_rows.get(
                key, row.get("ObjectEntities", []))),
        })
    return merged


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    manifest = {
        "schema": "numeric-decoder-prediction-manifest-v1",
        "contains_labels": False,
        "gold_aware": False,
        "rows": len(read_jsonl(path)),
        "output_sha256": sha256(path),
        **dict(payload),
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = _json(source / "plan/PLAN.json")
    train_graph_path = source / "graphs/train_graph.jsonl"
    validation_graph_path = source / "graphs/validation_graph.jsonl"
    train_graphs = _load_graph(train_graph_path, expected_split="train")
    validation_graphs = _load_graph(
        validation_graph_path, expected_split="validation")
    train_numeric = [
        graph for graph in train_graphs
        if graph["Relation"] in NUMERIC_RELATIONS]
    validation_numeric = [
        graph for graph in validation_graphs
        if graph["Relation"] in NUMERIC_RELATIONS]
    train_gold_rows = read_jsonl(Path(plan["train_gold"]))
    train_gold = {_key(row): row for row in train_gold_rows}
    fold_path = Path(plan["folds"])
    if plan.get("folds_sha256") != sha256(fold_path):
        raise ContractError("fold manifest hash mismatch")
    folds = {_key(row): int(row["fold"]) for row in read_jsonl(fold_path)}
    if set(folds) != {_key(graph) for graph in train_graphs}:
        raise ContractError("fold manifest does not exactly cover train graph")
    margins = tuple(float(value) for value in args.guard_margins.split(","))
    if not margins or any(value < 0 or not math.isfinite(value) for value in margins):
        raise ContractError("guard margins must be finite and nonnegative")

    fold_scores = {
        relation: {margin: {} for margin in margins}
        for relation in RELATIONS}
    baseline_scores = {relation: {} for relation in RELATIONS}
    oof_rows: list[dict] = []
    fold_ids = sorted(set(folds.values()))
    for fold in fold_ids:
        fit_rows = [
            graph for graph in train_numeric
            if folds[_key(graph)] != fold]
        holdout = [
            graph for graph in train_numeric
            if folds[_key(graph)] == fold]
        model = RelationSpecificNumericModel(args.l2).fit(
            fit_rows, train_gold)
        for relation in RELATIONS:
            relation_holdout = [
                graph for graph in holdout if graph["Relation"] == relation]
            baseline_values = []
            margin_values = {margin: [] for margin in margins}
            for graph in relation_holdout:
                gold = train_gold[_key(graph)]
                baseline_values.append(_numeric_correct(
                    graph, graph["baseline_objects"], gold))
                for margin in margins:
                    objects, detail = model.decode(graph, margin)
                    correct = _numeric_correct(graph, objects, gold)
                    margin_values[margin].append(correct)
                    oof_rows.append({
                        "SubjectEntity": graph["SubjectEntity"],
                        "Relation": relation,
                        "fold": fold,
                        "margin": margin,
                        "correct": correct,
                        **detail,
                    })
            baseline_scores[relation][fold] = statistics.mean(baseline_values)
            for margin in margins:
                fold_scores[relation][margin][fold] = statistics.mean(
                    margin_values[margin])

    selected_best: dict[str, float] = {}
    selected_one_se: dict[str, float] = {}
    stable_relations: dict[str, bool] = {}
    selection: dict[str, Any] = {}
    for relation in RELATIONS:
        best, one_se, diagnostics = _fold_policy_diagnostics(
            fold_scores[relation], baseline_scores[relation])
        selected_best[relation] = best
        selected_one_se[relation] = one_se
        selection[relation] = diagnostics
        stable_relations[relation] = _stable_relation(diagnostics, best)

    final_model = RelationSpecificNumericModel(args.l2).fit(
        train_numeric, train_gold)
    model_path = output / "numeric_model.json"
    model_payload = {
        **final_model.to_dict(),
        "training_graph": str(train_graph_path),
        "training_graph_sha256": sha256(train_graph_path),
        "folds": str(fold_path),
        "folds_sha256": sha256(fold_path),
        "best_mean_margins": selected_best,
        "one_standard_error_margins": selected_one_se,
        "stable_relations": stable_relations,
        "stable_relation_rule": (
            "enable only when the best-mean train-OOF margin has positive "
            "mean paired gain and no negative outer-fold delta"),
        "validation_labels_used_for_selection": False,
    }
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")

    control_path = Path(args.control_predictions).resolve()
    control_rows = read_jsonl(control_path)
    if {_key(row) for row in control_rows} != {
            _key(graph) for graph in validation_graphs}:
        raise ContractError("control predictions do not exactly cover validation graph")

    decoded_by_policy: dict[str, list[dict]] = {}
    diagnostics_by_policy: dict[str, list[dict]] = {}
    for policy, relation_margins in (
            ("best_oof", selected_best),
            ("one_se", selected_one_se),
            ("stable_oof", selected_best)):
        replacements = {}
        diagnostics = []
        for graph in validation_numeric:
            relation = str(graph["Relation"])
            if policy == "stable_oof" and not stable_relations[relation]:
                replacements[_key(graph)] = list(
                    next(row for row in control_rows
                         if _key(row) == _key(graph))["ObjectEntities"])
                diagnostics.append({
                    "SubjectEntity": graph["SubjectEntity"],
                    "Relation": relation,
                    "enabled": False,
                    "reason": "negative or nonpositive train-OOF fold evidence",
                    "used_control": True,
                })
                continue
            objects, detail = final_model.decode(
                graph, relation_margins[relation])
            replacements[_key(graph)] = objects
            diagnostics.append({
                "SubjectEntity": graph["SubjectEntity"],
                "Relation": graph["Relation"],
                "enabled": True,
                **detail,
            })
        decoded_by_policy[policy] = _merge_numeric(
            control_rows, replacements)
        diagnostics_by_policy[policy] = diagnostics
        prediction_path = output / f"validation_{policy}.jsonl"
        write_jsonl_atomic(prediction_path, decoded_by_policy[policy])
        write_jsonl_atomic(
            output / f"validation_{policy}.diagnostics.jsonl", diagnostics)
        _write_manifest(prediction_path, {
            "source_graph": str(validation_graph_path),
            "source_graph_sha256": sha256(validation_graph_path),
            "control_predictions": str(control_path),
            "control_predictions_sha256": sha256(control_path),
            "numeric_model": str(model_path),
            "numeric_model_sha256": sha256(model_path),
            "policy": policy,
            "relation_margins": relation_margins,
            "validation_labels_used_for_decoding": False,
        })
    write_jsonl_atomic(output / "train_oof_diagnostics.jsonl", oof_rows)

    # The validation label file is intentionally opened only after every
    # prediction artifact has been decoded and written.
    validation_gold = read_jsonl(Path(plan["validation_gold"]))
    scores = {
        "control": score(control_rows, validation_gold),
        **{
            policy: score(rows, validation_gold)
            for policy, rows in decoded_by_policy.items()
        },
    }
    option_oracle = {}
    for relation in RELATIONS:
        rows = [
            graph for graph in validation_numeric
            if graph["Relation"] == relation]
        correct = 0
        for graph in rows:
            gold = next(row for row in validation_gold if _key(row) == _key(graph))
            correct += max(
                option_label(graph, option, gold)
                for option in numeric_options(graph))
        option_oracle[relation] = correct / len(rows)

    result = {
        "schema": "relation-specific-numeric-decoder-ablation-v1",
        "development_only": True,
        "validation_labels_used_for_selection": False,
        "validation_labels_used_for_posthoc_evaluation": True,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "train_graph": str(train_graph_path),
        "train_graph_sha256": sha256(train_graph_path),
        "validation_graph": str(validation_graph_path),
        "validation_graph_sha256": sha256(validation_graph_path),
        "control_predictions": str(control_path),
        "control_predictions_sha256": sha256(control_path),
        "selection": selection,
        "best_mean_margins": selected_best,
        "one_standard_error_margins": selected_one_se,
        "stable_relations": stable_relations,
        "scores": scores,
        "validation_option_oracle_nondeployable": option_oracle,
        "oracle_warning": (
            "The option oracle opens validation labels and is diagnostic only. "
            "It did not select a margin, model, feature, or prediction."),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Relation-specific numeric risk decoding",
        "",
        "This is a CPU-only development ablation. Validation labels were opened "
        "only after predictions were frozen.",
        "",
        "| policy | pooled | hasArea | hasCapacity | delta vs control |",
        "|---|---:|---:|---:|---:|",
    ]
    control_pooled = scores["control"]["*** All Relations ***"]
    for policy in ("control", "best_oof", "one_se", "stable_oof"):
        values = scores[policy]
        lines.append(
            f"| {policy} | {values['*** All Relations ***']:.9f} | "
            f"{values['hasArea']:.6f} | {values['hasCapacity']:.6f} | "
            f"{values['*** All Relations ***'] - control_pooled:+.9f} |")
    lines += [
        "",
        "## Train-OOF selection",
        "",
        "| relation | best-mean margin | one-SE margin | stable enabled | "
        "baseline OOF | best-mean OOF | option oracle (validation, "
        "nondeployable) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        best = selected_best[relation]
        baseline_mean = statistics.mean(baseline_scores[relation].values())
        best_mean = statistics.mean(fold_scores[relation][best].values())
        lines.append(
            f"| {relation} | {best:.3f} | {selected_one_se[relation]:.3f} | "
            f"{stable_relations[relation]} | "
            f"{baseline_mean:.6f} | {best_mean:.6f} | "
            f"{option_oracle[relation]:.6f} |")
    lines += [
        "",
        "The **best_oof** policy is selected solely by mean paired training-fold "
        "accuracy. The **one_se** policy is the more conservative sensitivity "
        "analysis. The **stable_oof** candidate additionally requires positive "
        "mean gain with no negative outer fold for each enabled relation. The "
        "oracle is gold-aware and must never be deployed.",
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(
        f"complete: stable_oof={scores['stable_oof']['*** All Relations ***']:.9f}; "
        f"best_oof={scores['best_oof']['*** All Relations ***']:.9f}; "
        f"one_se={scores['one_se']['*** All Relations ***']:.9f}; "
        f"report={output / 'RESULT.md'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-predictions", required=True)
    parser.add_argument("--l2", type=float, default=2.0)
    parser.add_argument(
        "--guard-margins",
        default=",".join(str(value) for value in DEFAULT_MARGINS))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
