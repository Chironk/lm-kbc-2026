#!/usr/bin/env python3
"""Train-only explicit-cardinality ablation for heterogeneous memory routing.

This experiment estimates P(K=0), P(K=1), and P(K=many) independently of
candidate identity.  It consumes the already-built label-free graphs and opens
labels only inside five-fold fitting/evaluation, matching the selector's
existing leakage boundary.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    canonical_key,
    ContractError,
    NULLABLE_RELATIONS,
    NUMERIC_RELATIONS,
    SINGLE_RELATIONS,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    LogisticCalibrator,
    _decode_numeric,
    _fit_models,
    _key,
    _load_graph,
    _numeric_value,
    _select_guard_margin_one_se,
    candidate_features,
    decode as decode_standard_selector,
)


CLASSES = ("ZERO", "ONE", "MANY")
DYNAMIC_RELATIONS = {
    "companyTradesAtStockExchange",
    "countryLandBordersCountry",
    "personHasCityOfDeath",
}


def _commitment_probability(
        graph: Mapping[str, Any], agent: str, phase: str, label: str) -> float:
    value = graph["agents"][agent][phase]
    if not value.get("available"):
        return 0.0
    return float(value.get("probabilities", {}).get(label, 0.0))


def cardinality_feature_names() -> list[str]:
    return [
        "intercept",
        "qwen_card_zero", "qwen_card_one", "qwen_card_many",
        "gemma_card_zero", "gemma_card_one", "gemma_card_many",
        "qwen_exist_yes", "qwen_exist_no",
        "gemma_exist_yes", "gemma_exist_no",
        "qwen_none_rate", "gemma_none_rate",
        "candidate_count", "qwen_candidate_count", "gemma_candidate_count",
        "max_qwen_support", "max_gemma_support",
        "shared_candidate_rate", "commitment_agreement",
    ]


def cardinality_features(graph: Mapping[str, Any]) -> list[float]:
    nodes = list(graph.get("candidates", []))
    qnodes = [node for node in nodes if QWEN in node.get("sources", {})]
    gnodes = [node for node in nodes if GEMMA in node.get("sources", {})]
    shared = [node for node in nodes if {QWEN, GEMMA} <= set(node.get("sources", {}))]
    qmax = max((float(node["sources"][QWEN]["support_rate"])
                for node in qnodes), default=0.0)
    gmax = max((float(node["sources"][GEMMA]["support_rate"])
                for node in gnodes), default=0.0)
    qselected = graph["agents"][QWEN]["cardinality"].get("selected")
    gselected = graph["agents"][GEMMA]["cardinality"].get("selected")
    values = [
        1.0,
        *[_commitment_probability(graph, QWEN, "cardinality", label)
          for label in CLASSES],
        *[_commitment_probability(graph, GEMMA, "cardinality", label)
          for label in CLASSES],
        _commitment_probability(graph, QWEN, "existence", "YES"),
        _commitment_probability(graph, QWEN, "existence", "NO"),
        _commitment_probability(graph, GEMMA, "existence", "YES"),
        _commitment_probability(graph, GEMMA, "existence", "NO"),
        float(graph["agents"][QWEN]["none_rate"]),
        float(graph["agents"][GEMMA]["none_rate"]),
        min(1.0, len(nodes) / 20.0),
        min(1.0, len(qnodes) / 20.0),
        min(1.0, len(gnodes) / 20.0),
        qmax, gmax,
        len(shared) / max(1, len(nodes)),
        float(qselected is not None and qselected == gselected),
    ]
    if len(values) != len(cardinality_feature_names()):
        raise AssertionError("cardinality feature schema drift")
    if not all(math.isfinite(value) for value in values):
        raise ContractError(f"non-finite cardinality features for {_key(graph)}")
    return values


def cardinality_label(gold: Mapping[str, Any]) -> str:
    count = len(gold.get("ObjectEntities", []))
    return "ZERO" if count == 0 else "ONE" if count == 1 else "MANY"


class ExplicitCardinalityModel:
    """Relation-specific normalized one-vs-rest cardinality calibrator."""

    def __init__(self, l2: float = 2.0):
        self.l2 = float(l2)
        self.models: dict[str, dict[str, LogisticCalibrator]] = {}
        self.many_mean: dict[str, float] = {}

    def fit(
            self, graphs: Sequence[Mapping[str, Any]],
            gold_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> "ExplicitCardinalityModel":
        names = cardinality_feature_names()
        for relation in sorted(DYNAMIC_RELATIONS):
            subset = [graph for graph in graphs if graph["Relation"] == relation]
            if not subset:
                raise ContractError(f"no cardinality rows for {relation}")
            x = [cardinality_features(graph) for graph in subset]
            labels = [cardinality_label(gold_by_key[_key(graph)]) for graph in subset]
            self.models[relation] = {}
            for label in CLASSES:
                target = [float(value == label) for value in labels]
                self.models[relation][label] = LogisticCalibrator(
                    names, l2=self.l2).fit(x, target)
            many_counts = [
                len(gold_by_key[_key(graph)].get("ObjectEntities", []))
                for graph, label in zip(subset, labels) if label == "MANY"
            ]
            self.many_mean[relation] = (
                statistics.mean(many_counts) if many_counts else 2.0)
        # Deterministic task-schema relations do not consume fitted parameters.
        self.many_mean["awardWonBy"] = statistics.mean([
            len(gold_by_key[_key(graph)].get("ObjectEntities", []))
            for graph in graphs if graph["Relation"] == "awardWonBy"
        ])
        return self

    def predict_one(self, graph: Mapping[str, Any]) -> dict[str, float]:
        relation = str(graph["Relation"])
        if relation in NUMERIC_RELATIONS:
            return {"ZERO": 0.0, "ONE": 1.0, "MANY": 0.0}
        if relation == "awardWonBy":
            return {"ZERO": 0.0, "ONE": 0.0, "MANY": 1.0}
        raw = {
            label: float(model.predict([cardinality_features(graph)])[0])
            for label, model in self.models[relation].items()
        }
        total = sum(raw.values())
        if total <= 0 or not math.isfinite(total):
            raise ContractError(f"invalid cardinality probabilities for {_key(graph)}")
        return {label: value / total for label, value in raw.items()}

    def expected_size(
            self, graph: Mapping[str, Any], probabilities: Mapping[str, float],
            candidate_sum: float) -> float:
        relation = str(graph["Relation"])
        many = max(2.0, float(self.many_mean.get(relation, 2.0)))
        estimate = float(probabilities["ONE"]) + float(probabilities["MANY"]) * many
        # Candidate marginals are an independent lower-bound signal. This
        # prevents a noisy K model from forcing a set smaller than its own
        # accumulated expected true-positive mass.
        return max(estimate, candidate_sum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "explicit-cardinality-ovr-v1",
            "feature_names": cardinality_feature_names(),
            "l2": self.l2,
            "many_mean": self.many_mean,
            "models": {
                relation: {
                    label: model.to_dict() for label, model in models.items()}
                for relation, models in self.models.items()
            },
        }


def _set_utility(
        selected: Sequence[tuple[dict, float]], expected_size: float) -> float:
    if not selected:
        return 0.0
    expected_tp = sum(probability for _, probability in selected)
    denominator = len(selected) + expected_size
    return 2.0 * expected_tp / denominator if denominator > 0 else 0.0


def _objects_utility(
        graph: Mapping[str, Any], objects: Sequence[str],
        scored: Sequence[tuple[dict, float]], cardinality: Mapping[str, float],
        expected_size: float,
) -> float:
    if not objects:
        return float(cardinality["ZERO"]) if graph["Relation"] in NULLABLE_RELATIONS else 0.0
    relation = str(graph["Relation"])
    if relation in NUMERIC_RELATIONS:
        references = [_numeric_value(item) for item in objects]
        references = [value for value in references if value is not None]
        return max((
            probability for node, probability in scored
            if any(
                _numeric_value(node["item"]) is not None
                and abs(_numeric_value(node["item"]) - reference)
                / max(abs(reference), 1e-12) <= 0.05
                for reference in references)
        ), default=0.0)
    keys = {canonical_key(str(item), relation) for item in objects}
    chosen = [
        (node, probability) for node, probability in scored
        if node["key"] in keys
    ]
    if relation in SINGLE_RELATIONS:
        return max((probability for _, probability in chosen), default=0.0)
    return _set_utility(chosen, expected_size)


def decode_graph(
        graph: Mapping[str, Any], candidate_model: LogisticCalibrator,
        cardinality_model: ExplicitCardinalityModel, *, guard_margin: float,
        candidate_feature_fn=candidate_features,
) -> tuple[list[str], dict[str, Any]]:
    nodes = list(graph["candidates"])
    probabilities = (
        candidate_model.predict([candidate_feature_fn(graph, node) for node in nodes])
        if nodes else np.asarray([], dtype=np.float64))
    scored = list(zip(nodes, [float(value) for value in probabilities]))
    cardinality = cardinality_model.predict_one(graph)
    expected_size = cardinality_model.expected_size(
        graph, cardinality, sum(probability for _, probability in scored))
    relation = str(graph["Relation"])
    if relation in NUMERIC_RELATIONS:
        proposed, _ = _decode_numeric(graph, scored)
    elif relation in SINGLE_RELATIONS:
        best = max(scored, key=lambda pair: pair[1], default=None)
        proposed = (
            [] if best is None or cardinality["ZERO"] >= best[1]
            else [best[0]["item"]])
    else:
        ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0]["key"]))
        choices = [
            (_set_utility(ranked[:count], expected_size), ranked[:count])
            for count in range(1, len(ranked) + 1)
        ]
        if choices:
            best_utility, chosen = max(
                choices, key=lambda value: (value[0], -len(value[1])))
            proposed = [node["item"] for node, _ in chosen]
        else:
            best_utility, proposed = 0.0, []
        if relation in NULLABLE_RELATIONS and cardinality["ZERO"] >= best_utility:
            proposed = []
    proposed_utility = _objects_utility(
        graph, proposed, scored, cardinality, expected_size)
    baseline = list(graph.get("baseline_objects", []))
    baseline_utility = _objects_utility(
        graph, baseline, scored, cardinality, expected_size)
    use_baseline = baseline_utility + guard_margin >= proposed_utility
    return (baseline if use_baseline else proposed), {
        "cardinality_probabilities": cardinality,
        "expected_cardinality": expected_size,
        "proposed_objects": proposed,
        "proposed_utility": proposed_utility,
        "baseline_objects": baseline,
        "baseline_utility": baseline_utility,
        "guard_margin": guard_margin,
        "used_baseline": use_baseline,
    }


def _prediction_rows(
        graphs: Sequence[Mapping[str, Any]], candidate: LogisticCalibrator,
        cardinality: ExplicitCardinalityModel, margin: float,
        candidate_feature_fn=candidate_features,
) -> tuple[list[dict], list[dict]]:
    rows, diagnostics = [], []
    for graph in graphs:
        objects, detail = decode_graph(
            graph, candidate, cardinality, guard_margin=margin,
            candidate_feature_fn=candidate_feature_fn)
        rows.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": objects,
        })
        diagnostics.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            **detail,
        })
    return rows, diagnostics


def _cardinality_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["relation"])].append(row)

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict:
        if not rows:
            return {"rows": 0}
        accuracy = sum(
            max(row["probabilities"], key=row["probabilities"].get) == row["label"]
            for row in rows) / len(rows)
        brier = sum(
            sum((float(row["probabilities"][label]) - float(row["label"] == label)) ** 2
                for label in CLASSES)
            for row in rows) / len(rows)
        return {"rows": len(rows), "accuracy": accuracy, "multiclass_brier": brier}

    return {
        "overall": summarize(records),
        "by_relation": {
            relation: summarize(rows) for relation, rows in sorted(grouped.items())},
    }


def run(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    train_graph_path = source / "graphs/train_graph.jsonl"
    validation_graph_path = source / "graphs/validation_graph.jsonl"
    train_graphs = _load_graph(train_graph_path, expected_split="train")
    validation_graphs = _load_graph(validation_graph_path, expected_split="validation")
    plan = json.loads((source / "plan/PLAN.json").read_text())
    gold_rows = read_jsonl(Path(plan["train_gold"]))
    gold_by_key = {_key(row): row for row in gold_rows}
    folds = {
        _key(row): int(row["fold"]) for row in read_jsonl(Path(plan["folds"]))}
    margins = [float(value) for value in args.guard_margins.split(",")]
    fold_scores = {margin: {} for margin in margins}
    baseline_scores: dict[int, float] = {}
    oof_rows = {margin: [] for margin in margins}
    cardinality_records = []
    for fold in sorted(set(folds.values())):
        train = [graph for graph in train_graphs if folds[_key(graph)] != fold]
        holdout = [graph for graph in train_graphs if folds[_key(graph)] == fold]
        candidate, _ = _fit_models(
            train, gold_by_key, args.l2, "row-agent-balanced")
        cardinality = ExplicitCardinalityModel(args.l2).fit(train, gold_by_key)
        holdout_gold = [gold_by_key[_key(graph)] for graph in holdout]
        baseline = [{
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": list(graph["baseline_objects"]),
        } for graph in holdout]
        baseline_scores[fold] = score(
            baseline, holdout_gold)["*** All Relations ***"]
        for graph in holdout:
            cardinality_records.append({
                "relation": graph["Relation"],
                "label": cardinality_label(gold_by_key[_key(graph)]),
                "probabilities": cardinality.predict_one(graph),
            })
        for margin in margins:
            rows, _ = _prediction_rows(holdout, candidate, cardinality, margin)
            oof_rows[margin].extend(rows)
            fold_scores[margin][fold] = score(
                rows, holdout_gold)["*** All Relations ***"]
    selected_margin, guard_diagnostics = _select_guard_margin_one_se(
        fold_scores, baseline_scores)
    candidate, _ = _fit_models(
        train_graphs, gold_by_key, args.l2, "row-agent-balanced")
    cardinality = ExplicitCardinalityModel(args.l2).fit(
        train_graphs, gold_by_key)
    predictions, diagnostics = _prediction_rows(
        validation_graphs, candidate, cardinality, selected_margin)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "validation_predictions.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(
        output / "validation_predictions.diagnostics.jsonl", diagnostics)
    validation_scores = score(
        predictions, read_jsonl(Path(plan["validation_gold"])))

    # Evaluate the pre-existing decoder at the exact same selected margin.
    # This separates the cardinality/decoder effect from any gain caused by
    # choosing a different guard margin.
    matched_control_path = output / "original_decoder_matched_margin.jsonl"
    decode_standard_selector(Namespace(
        graph=str(validation_graph_path),
        selector=str(source / "selector/selector.json"),
        output=str(matched_control_path),
        guard_margin=selected_margin,
        allow_missing_commitments=False,
    ))
    validation_gold = read_jsonl(Path(plan["validation_gold"]))
    matched_control_scores = score(
        read_jsonl(matched_control_path), validation_gold)
    direct_cardinality_delta = (
        validation_scores["*** All Relations ***"]
        - matched_control_scores["*** All Relations ***"])

    source_result = json.loads((source / "selector/RESULT.json").read_text())
    source_scores = source_result["scores"]
    end_to_end_delta = (
        validation_scores["*** All Relations ***"]
        - source_scores["*** All Relations ***"])
    result = {
        "schema": "explicit-cardinality-ablation-v1",
        "development_only": True,
        "validation_labels_used_for_selection": False,
        "train_rows": len(train_graphs),
        "validation_rows": len(validation_graphs),
        "source_train_graph": str(train_graph_path),
        "source_train_graph_sha256": sha256(train_graph_path),
        "guard_margin": selected_margin,
        "guard_selection": guard_diagnostics,
        "cardinality_model": cardinality.to_dict(),
        "oof_cardinality": _cardinality_diagnostics(cardinality_records),
        "scores": validation_scores,
        "source_selected_system_scores": source_scores,
        "end_to_end_delta_vs_source_selected_system": end_to_end_delta,
        "matched_margin_original_decoder_scores": matched_control_scores,
        "direct_cardinality_delta_at_matched_margin": direct_cardinality_delta,
        "attribution_note": (
            "The matched-margin delta isolates the decoder/cardinality change. "
            "The larger end-to-end delta also includes the effect of selecting "
            "a different guard margin from training OOF predictions."
        ),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Explicit cardinality CPU ablation",
        "",
        f"- Pooled validation F1: **{validation_scores['*** All Relations ***']:.9f}**",
        f"- Training-OOF-selected guard: **{selected_margin:.3f}**",
        f"- End-to-end delta vs source selected system: "
        f"**{end_to_end_delta:+.9f}**",
        f"- Original decoder at the same margin: "
        f"**{matched_control_scores['*** All Relations ***']:.9f}**",
        f"- Direct cardinality/decoder delta at the same margin: "
        f"**{direct_cardinality_delta:+.9f}**",
        f"- OOF cardinality accuracy: "
        f"**{result['oof_cardinality']['overall']['accuracy']:.6f}**",
        f"- OOF multiclass Brier: "
        f"**{result['oof_cardinality']['overall']['multiclass_brier']:.6f}**",
        "",
        "The matched-margin delta is the controlled cardinality comparison. "
        "The end-to-end delta also includes the independently selected "
        "guard-margin change.",
        "",
        "| relation | F1 | cardinality accuracy |",
        "|---|---:|---:|",
    ]
    by_relation = result["oof_cardinality"]["by_relation"]
    for relation in sorted(
            key for key in validation_scores if key != "*** All Relations ***"):
        lines.append(
            f"| {relation} | {validation_scores[relation]:.6f} | "
            f"{by_relation[relation]['accuracy']:.6f} |")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(
        f"explicit cardinality pooled="
        f"{validation_scores['*** All Relations ***']:.9f}; "
        f"margin={selected_margin}")
    print(f"report={output / 'RESULT.md'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--l2", type=float, default=2.0)
    parser.add_argument(
        "--guard-margins", default="0,0.02,0.05,0.1,0.15,0.2,0.25,0.3")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
