#!/usr/bin/env python3
"""Nested expected-F1 list decoder over heterogeneous component evidence.

This experiment fixes the translation layer between a useful component graph
and the official set-valued metric.  It consumes two already-frozen,
label-free evidence sources:

* candidate-conditioned TRUE/FALSE judgments from Qwen and Gemma;
* connected same-context pairwise preference-graph scores.

A row-balanced component calibrator is fitted on outer-training folds only.
The explicit cardinality model supplies an expected gold-set size.  The
decoder constructs complete nested answer sets (pure prefixes, incumbent
expansions, incumbent pruning, and prune-then-expand combinations) and scores
each set with plug-in expected F1:

    2 * sum(P(component is correct)) / (predicted size + expected gold size)

An inner cross-fit on the outer-training partition learns only whether each
list relation has positive net utility.  It therefore cannot enable a
relation because of the outer holdout.  The final arm places this set decoder
after the established incumbent-edge/connected-graph precision-recall
cascade.  Validation is structurally absent.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.candidate_truth_evidence import (
    _truth_scores,
    _validated_responses as _validated_truth_responses,
)
from experiments.heterogeneous_agents.comparative_edge_action_selector import (
    EPSILON,
    _decode_sign_gate as _decode_incumbent_gate,
    _fit_relation_reliability as _fit_incumbent_reliability,
    _fit_sign_gate as _fit_incumbent_gate,
    _validated_tournament as _validated_incumbent_edges,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.explicit_cardinality_ablation import (
    ExplicitCardinalityModel,
)
from experiments.heterogeneous_agents.heterogeneous_memory_selector import (
    LogisticCalibrator,
    _load_graph,
)
from experiments.heterogeneous_agents.pairwise_preference_graph_selector import (
    CHANNELS,
    GRAPH_RIDGE,
    _cascade_outputs,
    _decode as _decode_pairwise_graph,
    _fit_relation_reliability as _fit_pairwise_reliability,
    _validated_pairwise_observations,
    build_preference_graphs,
)
from experiments.heterogeneous_agents.relation_specific_structured_decoder import (
    _row_f1,
)
from experiments.heterogeneous_agents.row_grouped_action_ranker import (
    FIXED_L2,
    _load,
    _relation_deltas,
    _utility,
)
from experiments.heterogeneous_agents.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
)
from experiments.heterogeneous_agents.unified_memory_action_graph import _key


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_REVIEW_RUN = (
    RUNS / "baseline_conditioned_action_review_20260727_v2")
DEFAULT_TOURNAMENT_RUN = RUNS / "full_candidate_tournament_20260727_v1"
DEFAULT_TRUTH_RUN = RUNS / "candidate_truth_evidence_20260727_v1"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_OUTPUT = RUNS / "component_expected_f1_set_decoder_20260727_v3"

LIST_RELATIONS = (
    "awardWonBy",
    "companyTradesAtStockExchange",
    "countryLandBordersCountry",
)
NULLABLE_LIST_RELATIONS = {"companyTradesAtStockExchange"}
FEATURE_ARMS = ("truth_only", "truth_graph")
ARMS = tuple(
    f"{feature}_{decoder}"
    for feature in FEATURE_ARMS
    for decoder in ("raw", "nested_reliability", "full_cascade")
)
PRIMARY = "truth_graph_full_cascade"

MIN_INCREMENTAL_DELTA = 0.015
MIN_POSITIVE_FOLDS = 4
MIN_RELATION_DELTA = -0.010
MAX_EXPANSION_PREFIX = 8

MEMORY_FEATURES = (
    "cross_memory",
    "gemma_independent_rank",
    "gemma_independent_share",
    "gemma_selected",
    "gemma_support",
    "memory_count",
    "qwen_selected",
    "qwen_self_consistency_rank",
    "qwen_self_consistency_share",
    "qwen_self_consistency_support",
    "qwen_support",
    "route_count",
    "within_qwen_route_count",
)
TRUTH_FEATURE_NAMES = (
    "intercept",
    "truth_qwen",
    "truth_gemma",
    "truth_mean",
    "truth_minimum",
    "truth_agreement",
)
GRAPH_COMPONENT_FEATURE_NAMES = tuple(
    f"{prompt}_{'qwen' if agent == QWEN else 'gemma'}_component_score"
    for prompt, agent in CHANNELS
) + (
    "preference_mean",
    "preference_minimum",
    "preference_maximum",
    "preference_positive_fraction",
    "preference_qwen_mean",
    "preference_gemma_mean",
    "preference_model_disagreement",
    "is_incumbent_component",
    "component_count",
) + MEMORY_FEATURES
RELATION_FEATURE_NAMES = tuple(
    f"relation_{relation}" for relation in LIST_RELATIONS)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _components(graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        node for node in graph["nodes"]
        if node.get("node_type") == "candidate_component"]


def _validated_truth_evidence(
    truth_run: Path,
    expected_source_graph_sha256: str,
) -> tuple[
    dict[tuple[str, str, str], Mapping[str, float]],
    dict[str, Any],
]:
    """Reconstruct inference-legal scores without reading scored gold output."""
    plan_path = truth_run / "plan/PLAN.json"
    plan = _json(plan_path)
    inventory_path = Path(plan["inventory"])
    if (
        plan.get("schema") != "candidate-truth-evidence-plan-v1"
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or sha256(inventory_path) != plan["inventory_sha256"]
        or plan.get("train_graph_sha256") != expected_source_graph_sha256
    ):
        raise ContractError("candidate-truth plan contract failed")
    responses, tasks = _validated_truth_responses(plan)
    truth_scores = _truth_scores(responses, tasks)
    inventory = read_jsonl(inventory_path)
    output = {}
    for component in inventory:
        component_key = str(component["component_key"])
        if component_key not in truth_scores:
            raise ContractError("candidate-truth inventory coverage failure")
        identity = (
            str(component["SubjectEntity"]),
            str(component["Relation"]),
            str(component["component_id"]),
        )
        if identity in output:
            raise ContractError("duplicate candidate-truth component")
        output[identity] = truth_scores[component_key]
    return output, plan


def _feature_names(arm: str) -> tuple[str, ...]:
    if arm == "truth_only":
        return TRUTH_FEATURE_NAMES + RELATION_FEATURE_NAMES
    if arm == "truth_graph":
        return (
            TRUTH_FEATURE_NAMES
            + GRAPH_COMPONENT_FEATURE_NAMES
            + RELATION_FEATURE_NAMES
        )
    raise ContractError(f"unknown component feature arm: {arm}")


def component_features(
    graph: Mapping[str, Any],
    component: Mapping[str, Any],
    truth_evidence: Mapping[tuple[str, str, str], Mapping[str, float]],
    preference_scores: Mapping[
        tuple[str, str], Mapping[str, float]] | None,
    arm: str,
) -> list[float]:
    identity = (
        str(graph["SubjectEntity"]),
        str(graph["Relation"]),
        str(component["id"]),
    )
    if identity not in truth_evidence:
        raise ContractError(f"missing candidate-truth evidence: {identity}")
    truth = truth_evidence[identity]
    values = [
        1.0,
        float(truth[QWEN]),
        float(truth[GEMMA]),
        float(truth["mean"]),
        float(truth["minimum"]),
        float(truth["agreement"]),
    ]
    if arm == "truth_graph":
        scores = preference_scores or {}
        channel_values = [
            max(-12.0, min(
                12.0,
                float(scores.get(channel, {}).get(component["id"], 0.0)),
            )) / 12.0
            for channel in CHANNELS
        ]
        qwen = [
            channel_values[index]
            for index, (_, agent) in enumerate(CHANNELS)
            if agent == QWEN]
        gemma = [
            channel_values[index]
            for index, (_, agent) in enumerate(CHANNELS)
            if agent == GEMMA]
        relation = str(graph["Relation"])
        incumbent_keys = {
            canonical_key(str(item), relation)
            for item in graph["incumbent_objects"]}
        component_keys = {
            canonical_key(str(item), relation)
            for item in component.get(
                "member_items", [component["representative"]])}
        memory = component.get("memory_evidence", {})
        values.extend([
            *channel_values,
            float(np.mean(channel_values)),
            min(channel_values),
            max(channel_values),
            float(np.mean([value > 0.0 for value in channel_values])),
            float(np.mean(qwen)),
            float(np.mean(gemma)),
            abs(float(np.mean(qwen)) - float(np.mean(gemma))),
            float(bool(incumbent_keys & component_keys)),
            min(1.0, len(_components(graph)) / 20.0),
            *(float(memory.get(name, 0.0)) for name in MEMORY_FEATURES),
        ])
    values.extend(
        float(graph["Relation"] == relation)
        for relation in LIST_RELATIONS)
    if len(values) != len(_feature_names(arm)):
        raise AssertionError("component expected-F1 feature schema drift")
    if not all(math.isfinite(value) for value in values):
        raise ContractError(f"non-finite component features for {identity}")
    return values


def _component_label(
    graph: Mapping[str, Any],
    component: Mapping[str, Any],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    # A component is an equivalence class.  Its representative is only a
    # deterministic rendering and must not determine the training label.
    members = list(dict.fromkeys(map(str, [
        *component.get("member_items", []),
        component["representative"],
    ])))
    return float(any(
        _row_f1(
            [member],
            gold_by[_key(graph)],
            str(graph["Relation"]),
        ) > 0.0
        for member in members
    ))


def fit_component_model(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    truth_evidence: Mapping[tuple[str, str, str], Mapping[str, float]],
    preference_graphs: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    arm: str,
) -> LogisticCalibrator:
    x, y, weights = [], [], []
    for graph in graphs:
        if graph["Relation"] not in LIST_RELATIONS:
            continue
        components = _components(graph)
        row_weight = 1.0 / max(1, len(components))
        for component in components:
            x.append(component_features(
                graph, component, truth_evidence,
                preference_graphs.get(_key(graph)), arm))
            y.append(_component_label(graph, component, gold_by))
            weights.append(row_weight)
    if not x or set(y) != {0.0, 1.0}:
        raise ContractError("component model lacks binary training coverage")
    return LogisticCalibrator(
        _feature_names(arm), l2=FIXED_L2,
    ).fit(x, y, weights)


def _surface_probability_lookup(
    graph: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
) -> dict[str, float]:
    relation = str(graph["Relation"])
    output: dict[str, float] = {}
    for component, probability in zip(
        components, probabilities, strict=True,
    ):
        for item in [
            *component.get("member_items", []),
            component["representative"],
        ]:
            output[canonical_key(str(item), relation)] = float(probability)
    return output


def expected_f1_utility(
    objects: Sequence[str],
    probability_by_surface: Mapping[str, float],
    relation: str,
    expected_cardinality: float,
    zero_probability: float,
) -> float:
    keys = tuple(dict.fromkeys(
        canonical_key(str(item), relation) for item in objects))
    if not keys:
        return (
            float(zero_probability)
            if relation in NULLABLE_LIST_RELATIONS else 0.0)
    expected_true_positives = sum(
        float(probability_by_surface.get(key, 0.0)) for key in keys)
    denominator = len(keys) + float(expected_cardinality)
    return (
        2.0 * expected_true_positives / denominator
        if denominator > 0.0 else 1.0)


def positive_pooled_utility(values: Sequence[float]) -> bool:
    """Use the pooled row objective; fold counts are diagnostics only."""
    return bool(values and sum(map(float, values)) > EPSILON)


def _component_identity_by_surface(
    relation: str,
    components: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for component in components:
        component_id = str(component["id"])
        for item in [
            *component.get("member_items", []),
            component["representative"],
        ]:
            output[canonical_key(str(item), relation)] = component_id
    return output


def _deduplicate_component_objects(
    objects: Sequence[str],
    relation: str,
    component_by_surface: Mapping[str, str],
) -> list[str]:
    """Keep one rendering for each equivalence component or free surface."""
    output, seen = [], set()
    for item in map(str, objects):
        surface = canonical_key(item, relation)
        identity = (
            "component", component_by_surface[surface],
        ) if surface in component_by_surface else ("surface", surface)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(item)
    return output


def nested_set_actions(
    graph: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    """Construct complete prefix/prune/expand sets without consulting labels."""
    relation = str(graph["Relation"])
    ranked = sorted(
        zip(components, map(float, probabilities)),
        key=lambda pair: (-pair[1], str(pair[0]["id"])),
    )
    probability_by_surface = _surface_probability_lookup(
        graph, components, probabilities)
    component_by_surface = _component_identity_by_surface(
        relation, components)
    incumbent = list(map(str, graph["incumbent_objects"]))
    incumbent_ranked = sorted(
        incumbent,
        key=lambda item: (
            -probability_by_surface.get(
                canonical_key(item, relation), 0.0),
            canonical_key(item, relation),
        ),
    )
    top_limit = min(MAX_EXPANSION_PREFIX, len(ranked))
    candidates: list[tuple[str, list[str]]] = [
        ("KEEP", incumbent),
        ("EMPTY", []),
    ]
    for count in range(1, len(ranked) + 1):
        candidates.append((
            "PURE_PREFIX",
            [str(component["representative"])
             for component, _ in ranked[:count]],
        ))
    for keep_count in range(len(incumbent_ranked) + 1):
        retained = incumbent_ranked[:keep_count]
        candidates.append(("PRUNE", retained))
        for add_count in range(1, top_limit + 1):
            additions = [
                str(component["representative"])
                for component, _ in ranked[:add_count]]
            candidates.append((
                "PRUNE_EXPAND",
                list(dict.fromkeys([*retained, *additions])),
            ))
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    incumbent_rendered = _deduplicate_component_objects(
        incumbent, relation, component_by_surface)
    incumbent_key = tuple(sorted(
        (
            "component", component_by_surface[surface],
        ) if surface in component_by_surface else ("surface", surface)
        for surface in (
            canonical_key(item, relation) for item in incumbent_rendered)
    ))
    for family, objects in candidates:
        rendered = _deduplicate_component_objects(
            objects, relation, component_by_surface)
        key = tuple(sorted(
            (
                "component", component_by_surface[surface],
            ) if surface in component_by_surface else ("surface", surface)
            for surface in (
                canonical_key(item, relation) for item in rendered)
        ))
        if key == incumbent_key:
            family = "KEEP"
            rendered = incumbent_rendered
        if key not in unique or family == "KEEP":
            unique[key] = {
                "family": family,
                "objects": rendered,
            }
    return list(unique.values())


def decode_expected_f1(
    graph: Mapping[str, Any],
    source_graph: Mapping[str, Any],
    component_model: LogisticCalibrator,
    cardinality_model: ExplicitCardinalityModel,
    truth_evidence: Mapping[tuple[str, str, str], Mapping[str, float]],
    preference_graphs: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    arm: str,
) -> tuple[list[str], dict[str, Any]]:
    components = _components(graph)
    features = [
        component_features(
            graph, component, truth_evidence,
            preference_graphs.get(_key(graph)), arm)
        for component in components]
    probabilities = (
        list(map(float, component_model.predict(features)))
        if features else [])
    cardinality = cardinality_model.predict_one(source_graph)
    expected_cardinality = cardinality_model.expected_size(
        source_graph, cardinality, sum(probabilities))
    probability_by_surface = _surface_probability_lookup(
        graph, components, probabilities)
    actions = nested_set_actions(graph, components, probabilities)
    utilities = [
        expected_f1_utility(
            action["objects"], probability_by_surface,
            str(graph["Relation"]), expected_cardinality,
            float(cardinality["ZERO"]),
        )
        for action in actions]
    keep = next(
        index for index, action in enumerate(actions)
        if action["family"] == "KEEP")
    selected = max(
        range(len(actions)),
        key=lambda index: (
            utilities[index],
            index == keep,
            -len(actions[index]["objects"]),
            tuple(actions[index]["objects"]),
        ),
    )
    # The comparison uses the same expected-F1 scale on both sides.
    if utilities[selected] <= utilities[keep] + EPSILON:
        selected = keep
    return list(actions[selected]["objects"]), {
        "selected_family": actions[selected]["family"],
        "selected_objects": list(actions[selected]["objects"]),
        "selected_expected_f1": float(utilities[selected]),
        "incumbent_objects": list(actions[keep]["objects"]),
        "incumbent_expected_f1": float(utilities[keep]),
        "estimated_improvement": float(
            utilities[selected] - utilities[keep]),
        "changed": selected != keep,
        "action_count": len(actions),
        "cardinality_probabilities": cardinality,
        "expected_cardinality": float(expected_cardinality),
        "component_probabilities": [
            {
                "component_id": str(component["id"]),
                "representative": str(component["representative"]),
                "probability": probability,
            }
            for component, probability in zip(
                components, probabilities, strict=True)
        ],
    }


def _fit_models(
    graphs: Sequence[Mapping[str, Any]],
    source_by: Mapping[tuple[str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    truth_evidence: Mapping[tuple[str, str, str], Mapping[str, float]],
    preference_graphs: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    arm: str,
) -> tuple[LogisticCalibrator, ExplicitCardinalityModel]:
    component = fit_component_model(
        graphs, gold_by, truth_evidence, preference_graphs, arm)
    cardinality = ExplicitCardinalityModel(FIXED_L2).fit(
        [source_by[_key(graph)] for graph in graphs], gold_by)
    return component, cardinality


def _decode_list_rows(
    graphs: Sequence[Mapping[str, Any]],
    source_by: Mapping[tuple[str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    component_model: LogisticCalibrator,
    cardinality_model: ExplicitCardinalityModel,
    truth_evidence: Mapping[tuple[str, str, str], Mapping[str, float]],
    preference_graphs: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    arm: str,
    relation_reliability: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    predictions, diagnostics = {}, []
    for graph in graphs:
        if graph["Relation"] not in LIST_RELATIONS:
            continue
        key = _key(graph)
        proposed, detail = decode_expected_f1(
            graph, source_by[key], component_model, cardinality_model,
            truth_evidence, preference_graphs, arm)
        relation_allowed = (
            True if relation_reliability is None
            else bool(relation_reliability[
                str(graph["Relation"])]["allowed"]))
        selected = (
            proposed if relation_allowed
            else list(graph["incumbent_objects"]))
        before = (
            _row_f1(
                list(graph["incumbent_objects"]),
                gold_by[key], str(graph["Relation"]))
            if gold_by is not None else None)
        after = (
            _row_f1(selected, gold_by[key], str(graph["Relation"]))
            if gold_by is not None else None)
        predictions[key] = {
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": selected,
        }
        diagnostics.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            **detail,
            "relation_allowed": relation_allowed,
            "changed": (
                sorted(map(str, selected))
                != sorted(map(str, graph["incumbent_objects"]))),
            "utility_delta": (
                after - before
                if after is not None and before is not None else None),
        })
    return predictions, diagnostics


def _inner_relation_reliability(
    fit_graphs: Sequence[Mapping[str, Any]],
    folds: Mapping[tuple[str, str], int],
    source_by: Mapping[tuple[str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    truth_evidence: Mapping[tuple[str, str, str], Mapping[str, float]],
    preference_graphs: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    arm: str,
) -> dict[str, dict[str, Any]]:
    """Fit the relation veto from inner held-out predictions only."""
    available_folds = sorted({folds[_key(graph)] for graph in fit_graphs})
    deltas: dict[str, list[float]] = defaultdict(list)
    fold_deltas: dict[str, dict[int, float]] = defaultdict(
        lambda: defaultdict(float))
    for inner in available_folds:
        inner_train = [
            graph for graph in fit_graphs if folds[_key(graph)] != inner]
        inner_hold = [
            graph for graph in fit_graphs if folds[_key(graph)] == inner]
        component, cardinality = _fit_models(
            inner_train, source_by, gold_by, truth_evidence,
            preference_graphs, arm)
        _, diagnostics = _decode_list_rows(
            inner_hold, source_by, gold_by, component, cardinality,
            truth_evidence, preference_graphs, arm, None)
        for item in diagnostics:
            delta = float(item["utility_delta"])
            deltas[str(item["Relation"])].append(delta)
            fold_deltas[str(item["Relation"])][inner] += delta
    return {
        relation: {
            "allowed": positive_pooled_utility(deltas.get(relation, [])),
            "inner_rows": len(deltas.get(relation, [])),
            "helped": sum(
                value > EPSILON for value in deltas.get(relation, [])),
            "harmed": sum(
                value < -EPSILON for value in deltas.get(relation, [])),
            "net_utility": sum(deltas.get(relation, [])),
            "positive_inner_folds": sum(
                value > EPSILON
                for value in fold_deltas[relation].values()),
            "selection_rule": (
                "positive summed inner-OOF row-F1 utility; fold counts are "
                "diagnostic because the target metric pools rows, not folds"
            ),
            "inner_fold_net_utility": {
                str(fold): value
                for fold, value in sorted(fold_deltas[relation].items())
            },
        }
        for relation in LIST_RELATIONS
    }


def _pairwise_cascade_holdout(
    fit: Sequence[Mapping[str, Any]],
    hold: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    incumbent_registry: Mapping[tuple[str, str], Mapping[str, Any]],
    incumbent_edges: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    preference_graphs: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_gate = _fit_incumbent_gate(
        fit, gold_by, review_evidence,
        incumbent_registry, incumbent_edges,
        include_review=True)
    old_reliability = _fit_incumbent_reliability(
        fit, gold_by, incumbent_registry, incumbent_edges)
    old_predictions, old_detail = _decode_incumbent_gate(
        old_gate, hold, gold_by, review_evidence,
        incumbent_registry, incumbent_edges,
        include_review=True,
        relation_reliability=old_reliability)
    new_reliability = _fit_pairwise_reliability(
        fit, gold_by, preference_graphs, "qwen_submission")
    new_predictions, new_detail = _decode_pairwise_graph(
        hold, gold_by, review_evidence, preference_graphs,
        "qwen_submission", None, new_reliability)
    return _cascade_outputs(
        old_predictions, old_detail, new_predictions, new_detail)


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    review_run = Path(args.review_run).resolve()
    tournament_run = Path(args.tournament_run).resolve()
    truth_run = Path(args.truth_run).resolve()
    gold_path = Path(args.gold).resolve()
    review_plan, graphs, control_rows, gold_by, folds, review_evidence = _load(
        review_run, gold_path)
    control_by = {_key(row): row for row in control_rows}
    source_path = Path(review_plan["source_graph"])
    if sha256(source_path) != review_plan["source_graph_sha256"]:
        raise ContractError("stale source graph")
    source_by = {
        _key(row): row
        for row in _load_graph(source_path, expected_split="train")}
    if set(source_by) != {_key(graph) for graph in graphs}:
        raise ContractError("source/action graph row mismatch")
    truth_evidence, _ = _validated_truth_evidence(
        truth_run, review_plan["source_graph_sha256"])
    tournament_registry, observations, _ = (
        _validated_pairwise_observations(tournament_run))
    incumbent_registry, incumbent_edges, _ = _validated_incumbent_edges(
        tournament_run)
    preference_graphs = build_preference_graphs(
        tournament_registry, observations)
    if set(incumbent_registry) != set(tournament_registry):
        raise ContractError("tournament registry mismatch")

    results: dict[str, Any] = {}
    prediction_artifacts: dict[str, list[dict[str, Any]]] = {}
    diagnostic_artifacts: dict[str, list[dict[str, Any]]] = {}
    reliability_artifacts: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        feature_arm = next(
            feature for feature in FEATURE_ARMS if arm.startswith(feature))
        decoder = arm[len(feature_arm) + 1:]
        oof: dict[tuple[str, str], dict[str, Any]] = {}
        diagnostics: list[dict[str, Any]] = []
        reliability_rows: list[dict[str, Any]] = []
        fold_results = []
        for outer in sorted(set(folds.values())):
            fit = [graph for graph in graphs if folds[_key(graph)] != outer]
            hold = [graph for graph in graphs if folds[_key(graph)] == outer]
            component, cardinality = _fit_models(
                fit, source_by, gold_by, truth_evidence,
                preference_graphs, feature_arm)
            reliability = (
                None if decoder == "raw"
                else _inner_relation_reliability(
                    fit, folds, source_by, gold_by, truth_evidence,
                    preference_graphs, feature_arm))
            list_predictions, list_detail = _decode_list_rows(
                hold, source_by, gold_by, component, cardinality,
                truth_evidence, preference_graphs, feature_arm,
                reliability)
            if decoder == "full_cascade":
                base_predictions, base_detail = _pairwise_cascade_holdout(
                    fit, hold, gold_by, review_evidence,
                    incumbent_registry, incumbent_edges,
                    preference_graphs)
            else:
                base_predictions = [control_by[_key(graph)] for graph in hold]
                base_detail = [{
                    "SubjectEntity": graph["SubjectEntity"],
                    "Relation": graph["Relation"],
                    "changed": False,
                    "utility_delta": 0.0,
                    "cascade_stage": "not_run",
                } for graph in hold]
            base_by = {_key(row): dict(row) for row in base_predictions}
            list_detail_by = {_key(row): row for row in list_detail}
            detail_by = {_key(row): row for row in base_detail}
            predictions = []
            fold_detail = []
            for graph in hold:
                key = _key(graph)
                row = list_predictions.get(key, base_by[key])
                before = _row_f1(
                    control_by[key]["ObjectEntities"],
                    gold_by[key], str(graph["Relation"]))
                after = _row_f1(
                    row["ObjectEntities"],
                    gold_by[key], str(graph["Relation"]))
                merged = {
                    **dict(detail_by[key]),
                    **dict(list_detail_by.get(key, {})),
                    "SubjectEntity": key[0],
                    "Relation": key[1],
                    "outer_fold": outer,
                    "arm": arm,
                    "changed": (
                        sorted(map(str, row["ObjectEntities"]))
                        != sorted(map(
                            str, control_by[key]["ObjectEntities"]))),
                    "utility_delta": after - before,
                    "decoder_source": (
                        "expected_f1_set"
                        if key in list_predictions else "pairwise_cascade"),
                }
                predictions.append(dict(row))
                fold_detail.append(merged)
                oof[key] = dict(row)
            diagnostics.extend(fold_detail)
            if reliability is not None:
                reliability_rows.extend({
                    "outer_fold": outer,
                    "feature_arm": feature_arm,
                    "Relation": relation,
                    **value,
                } for relation, value in reliability.items())
            hold_gold = [gold_by[_key(graph)] for graph in hold]
            hold_control = [control_by[_key(graph)] for graph in hold]
            control_value = score(
                hold_control, hold_gold)["*** All Relations ***"]
            selected_value = score(
                predictions, hold_gold)["*** All Relations ***"]
            fold_results.append({
                "fold": outer,
                "control": control_value,
                "selected": selected_value,
                "delta": selected_value - control_value,
                "changed": sum(item["changed"] for item in fold_detail),
                "helped": sum(
                    item["utility_delta"] > EPSILON for item in fold_detail),
                "harmed": sum(
                    item["utility_delta"] < -EPSILON for item in fold_detail),
            })
        if set(oof) != {_key(graph) for graph in graphs}:
            raise ContractError("expected-F1 OOF coverage failure")
        ordered = [oof[_key(graph)] for graph in graphs]
        ordered_gold = [gold_by[_key(graph)] for graph in graphs]
        ordered_control = [control_by[_key(graph)] for graph in graphs]
        control_scores = score(ordered_control, ordered_gold)
        selected_scores = score(ordered, ordered_gold)
        changed = [item for item in diagnostics if item["changed"]]
        results[arm] = {
            "control_scores": control_scores,
            "selected_scores": selected_scores,
            "incremental_delta": (
                selected_scores["*** All Relations ***"]
                - control_scores["*** All Relations ***"]),
            "relation_deltas": _relation_deltas(
                selected_scores, control_scores),
            "changed_rows": len(changed),
            "helped_rows": sum(
                item["utility_delta"] > EPSILON for item in changed),
            "harmed_rows": sum(
                item["utility_delta"] < -EPSILON for item in changed),
            "neutral_changed_rows": sum(
                abs(item["utility_delta"]) <= EPSILON for item in changed),
            "positive_folds": sum(
                item["delta"] > EPSILON for item in fold_results),
            "folds": fold_results,
        }
        prediction_artifacts[arm] = ordered
        diagnostic_artifacts[arm] = diagnostics
        reliability_artifacts[arm] = reliability_rows

    primary = results[PRIMARY]
    passed = (
        primary["incremental_delta"] >= MIN_INCREMENTAL_DELTA
        and primary["positive_folds"] >= MIN_POSITIVE_FOLDS
        and min(primary["relation_deltas"].values()) >= MIN_RELATION_DELTA
        and primary["helped_rows"] > primary["harmed_rows"])

    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for arm in ARMS:
        prediction_path = output / f"analysis/{arm}_OOF_PREDICTIONS.jsonl"
        diagnostic_path = output / f"analysis/{arm}_OOF_DIAGNOSTICS.jsonl"
        reliability_path = output / f"analysis/{arm}_RELIABILITY.jsonl"
        write_jsonl_atomic(prediction_path, prediction_artifacts[arm])
        write_jsonl_atomic(diagnostic_path, diagnostic_artifacts[arm])
        write_jsonl_atomic(reliability_path, reliability_artifacts[arm])
        artifacts[arm] = {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
            "reliability": str(reliability_path),
            "reliability_sha256": sha256(reliability_path),
        }

    final_component, final_cardinality = _fit_models(
        graphs, source_by, gold_by, truth_evidence,
        preference_graphs, "truth_graph")
    final_reliability = _inner_relation_reliability(
        graphs, folds, source_by, gold_by, truth_evidence,
        preference_graphs, "truth_graph")
    model_path = output / "analysis/TRAIN_FIT_MODEL.json"
    _write_json(model_path, {
        "schema": "component-expected-f1-set-decoder-model-v1",
        "feature_arm": "truth_graph",
        "component_model": final_component.to_dict(),
        "cardinality_model": final_cardinality.to_dict(),
        "relation_reliability": final_reliability,
        "component_l2": FIXED_L2,
        "cardinality_l2": FIXED_L2,
        "max_expansion_prefix": MAX_EXPANSION_PREFIX,
        "expected_f1_rule": (
            "2*sum(component_probability)/(set_size+expected_cardinality)"),
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "promotion_gate_passed": passed,
    })

    result = {
        "schema": "component-expected-f1-set-decoder-result-v1",
        "development_only": True,
        "deployable": False,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "primary_arm": PRIMARY,
        "rows": len(graphs),
        "list_relations": list(LIST_RELATIONS),
        "component_l2": FIXED_L2,
        "cardinality_l2": FIXED_L2,
        "graph_ridge": GRAPH_RIDGE,
        "arms": results,
        "artifacts": artifacts,
        "source_hashes": {
            "review_plan": sha256(review_run / "plan/PLAN.json"),
            "tournament_plan": sha256(tournament_run / "plan/PLAN.json"),
            "truth_plan": sha256(truth_run / "plan/PLAN.json"),
            "source_graph": sha256(source_path),
            "gold": sha256(gold_path),
        },
        "deployment_gate": {
            "passed": passed,
            "minimum_incremental_delta": MIN_INCREMENTAL_DELTA,
            "minimum_positive_folds": MIN_POSITIVE_FOLDS,
            "minimum_relation_delta": MIN_RELATION_DELTA,
        },
    }
    _write_json(output / "RESULT.json", result)
    lines = [
        "# Component-aware expected-F1 set decoder", "",
        "All scores are subject-grouped outer OOF. Relation reliability is "
        "fitted by an additional inner cross-fit.", "",
        "| arm | OOF score | delta | changed | helped | harmed | folds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in results.items():
        lines.append(
            f"| {arm} | "
            f"{item['selected_scores']['*** All Relations ***']:.6f} | "
            f"{item['incremental_delta']:+.6f} | "
            f"{item['changed_rows']} | {item['helped_rows']} | "
            f"{item['harmed_rows']} | {item['positive_folds']}/5 |")
    lines += [
        "",
        f"Predeclared promotion gate: **{passed}**",
        "",
    ]
    (output / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps({
        "arms": {
            arm: {
                "score": item["selected_scores"]["*** All Relations ***"],
                "delta": item["incremental_delta"],
                "changed": item["changed_rows"],
                "helped": item["helped_rows"],
                "harmed": item["harmed_rows"],
                "positive_folds": item["positive_folds"],
                "relation_deltas": item["relation_deltas"],
            }
            for arm, item in results.items()
        },
        "gate_passed": passed,
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--review-run", default=str(DEFAULT_REVIEW_RUN))
    value.add_argument("--tournament-run", default=str(DEFAULT_TOURNAMENT_RUN))
    value.add_argument("--truth-run", default=str(DEFAULT_TRUTH_RUN))
    value.add_argument("--gold", default=str(DEFAULT_GOLD))
    value.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
