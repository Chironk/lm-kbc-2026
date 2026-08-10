#!/usr/bin/env python3
"""Unified heterogeneous-memory graph and counterfactual action selector.

This module is deliberately broader than the earlier relation-specific
decoders.  It represents every subject-relation row with the same hierarchy:

    query -> parametric memory -> evidence route -> candidate component
          -> current answer state -> legal counterfactual action

Qwen routes remain children of one Qwen memory node.  Consequently, multiple
prompts or self-consistency samples from Qwen cannot be counted as independent
memories.  A single pooled ridge model estimates the row-F1 utility of every
legal graph action.  Relation identity affects features, but there are no
per-relation models or tuned switching thresholds.

The train audit is nested, subject-grouped, out-of-fold evaluation.  Gold is
used only to label counterfactual train actions and score held-out train rows;
it is never written into the graph artifact or inference features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    NULLABLE_RELATIONS,
    NUMERIC_RELATIONS,
    SINGLE_RELATIONS,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    _component_for_prediction,
    augment_relational_graph,
    collapse_prediction,
)
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    _row_f1,
)


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/graphs/train_graph.jsonl")
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_AGENTS = (
    ROOT / "experiments/heterogeneous_agents/agents_qwen_gemma_n1_frozen.json")
DEFAULT_OUTPUT = RUNS / "unified_memory_action_graph_20260727_v1"

RELATIONS = (
    "awardWonBy",
    "companyTradesAtStockExchange",
    "countryLandBordersCountry",
    "hasArea",
    "hasCapacity",
    "personHasCityOfDeath",
)
ACTION_TYPES = ("KEEP", "COLLAPSE", "EMPTY", "REPLACE", "ADD", "DROP")
L2_GRID = (0.1, 1.0, 10.0, 100.0)
OUTER_FOLDS = 5
INNER_FOLDS = 3
PARAMETER_CAP = 32_000_000_000

# Broad deployment gates.  These are architecture-level gates, not
# relation-specific thresholds.
MIN_POOLED_DELTA = 0.010
MIN_WINNING_FOLDS = 3
MIN_FOLD_DELTA = -0.005
MIN_WINNING_RELATIONS = 2
MIN_RELATION_DELTA = -0.020


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def _stable_int(*parts: object) -> int:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def relation_family(relation: str) -> str:
    if relation in NUMERIC_RELATIONS:
        return "numeric"
    if relation in SINGLE_RELATIONS:
        return "single"
    return "list"


def grouped_relation_folds(
    rows: Sequence[Mapping[str, Any]], n_folds: int, *, seed: int,
) -> dict[tuple[str, str], int]:
    """Deterministic subject-grouped folds balanced without looking at gold."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["SubjectEntity"])].append(row)
    relation_totals = Counter(str(row["Relation"]) for row in rows)
    target_relation = {
        relation: total / n_folds for relation, total in relation_totals.items()}
    target_size = len(rows) / n_folds
    fold_relation = [Counter() for _ in range(n_folds)]
    fold_size = [0 for _ in range(n_folds)]
    ordered = sorted(
        by_subject.items(),
        key=lambda item: (
            -len(item[1]),
            _stable_int("subject-fold", seed, item[0]),
            item[0],
        ),
    )
    assignments: dict[str, int] = {}
    for subject, subject_rows in ordered:
        counts = Counter(str(row["Relation"]) for row in subject_rows)

        def cost(fold: int) -> tuple[float, int, int]:
            relation_cost = sum(
                ((fold_relation[fold][relation] + count)
                 - target_relation[relation]) ** 2
                - (fold_relation[fold][relation]
                   - target_relation[relation]) ** 2
                for relation, count in counts.items())
            size_cost = (
                (fold_size[fold] + len(subject_rows) - target_size) ** 2
                - (fold_size[fold] - target_size) ** 2)
            return (
                relation_cost + 0.05 * size_cost,
                fold_size[fold],
                fold,
            )

        chosen = min(range(n_folds), key=cost)
        assignments[subject] = chosen
        fold_relation[chosen].update(counts)
        fold_size[chosen] += len(subject_rows)
    result = {
        _key(row): assignments[str(row["SubjectEntity"])] for row in rows}
    for subject, subject_rows in by_subject.items():
        if len({result[_key(row)] for row in subject_rows}) != 1:
            raise AssertionError(f"subject grouping failed for {subject}")
    return result


def _prob(
    row: Mapping[str, Any], agent: str, phase: str, label: str,
) -> float:
    value = row.get("agents", {}).get(agent, {}).get(phase, {})
    if not value.get("available"):
        return 0.0
    return float(value.get("probabilities", {}).get(label, 0.0))


def _memory_route_nodes(row: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    nodes, edges = [], []
    for agent in (QWEN, GEMMA):
        nodes.append({
            "id": f"memory:{agent}",
            "node_type": "parametric_memory",
            "agent_id": agent,
        })
    route_names = sorted({
        str(route)
        for candidate in row.get("candidates", [])
        for route in candidate.get("routes", {})
    } | set(str(route) for route in row.get("proposal_routes", {})))
    for route in route_names:
        metadata = row.get("proposal_routes", {}).get(route, {})
        family = str(metadata.get(
            "model_family",
            GEMMA if route.startswith("gemma:") else QWEN))
        if family not in {QWEN, GEMMA}:
            raise ContractError(f"unknown route memory family {family!r}")
        nodes.append({
            "id": f"route:{route}",
            "node_type": "evidence_route",
            "route": route,
            "memory_id": f"memory:{family}",
        })
        edges.append({
            "source": f"memory:{family}",
            "target": f"route:{route}",
            "edge_type": "contains_route",
            "directed": True,
        })
    return nodes, edges


def _component_evidence(
    row: Mapping[str, Any], component: Mapping[str, Any],
) -> dict[str, float]:
    qwen, gemma = [], []
    qwen_self_consistency, qwen_system2 = [], []
    qwen_selected = gemma_selected = False
    route_count = 0
    for route, values in component.get("routes", {}).items():
        route_count += 1
        metadata = row.get("proposal_routes", {}).get(route, {})
        family = metadata.get(
            "model_family",
            GEMMA if str(route).startswith("gemma:") else QWEN)
        rate = float(values.get("max_support_rate", 0.0))
        selected = bool(values.get("selected", False))
        if family == QWEN:
            qwen.append(rate)
            if str(route) == "qwen:self_consistency":
                qwen_self_consistency.append(rate)
            elif str(route) == "qwen:system2":
                qwen_system2.append(rate)
            qwen_selected = qwen_selected or selected
        elif family == GEMMA:
            gemma.append(rate)
            gemma_selected = gemma_selected or selected
    # Routes within a model are capped by max; model independence is expressed
    # only by the two separate memory-level values.
    q = max(qwen, default=0.0)
    g = max(gemma, default=0.0)
    route_metrics = {}
    for route in (
        "qwen:self_consistency", "qwen:system2", "gemma:independent"):
        metadata = row.get("proposal_routes", {}).get(route, {})
        available = bool(metadata.get("available", False))
        rate = float(component.get("routes", {}).get(
            route, {}).get("max_support_rate", 0.0))
        competitors = [
            float(candidate.get("routes", {}).get(
                route, {}).get("max_support_rate", 0.0))
            for candidate in row["relational_graph"]["components"]
        ]
        mass = sum(competitors)
        strictly_better = sum(value > rate + 1e-12 for value in competitors)
        route_metrics[route] = {
            "available": float(available),
            "silent": float(available and rate <= 0.0),
            "share": rate / mass if mass > 0.0 else 0.0,
            "rank": (
                1.0 - strictly_better / max(len(competitors) - 1, 1)
                if rate > 0.0 else 0.0),
        }
    return {
        "qwen_support": q,
        "gemma_support": g,
        "qwen_self_consistency_support": max(
            qwen_self_consistency, default=0.0),
        "qwen_system2_support": max(qwen_system2, default=0.0),
        "qwen_selected": float(qwen_selected),
        "gemma_selected": float(gemma_selected),
        "cross_memory": float(q > 0 and g > 0),
        "memory_count": float((q > 0) + (g > 0)),
        "route_count": float(route_count),
        "within_qwen_route_count": float(len(qwen)),
        "qwen_self_consistency_share": route_metrics[
            "qwen:self_consistency"]["share"],
        "qwen_system2_share": route_metrics["qwen:system2"]["share"],
        "gemma_independent_share": route_metrics[
            "gemma:independent"]["share"],
        "qwen_self_consistency_rank": route_metrics[
            "qwen:self_consistency"]["rank"],
        "qwen_system2_rank": route_metrics["qwen:system2"]["rank"],
        "gemma_independent_rank": route_metrics[
            "gemma:independent"]["rank"],
        "qwen_self_consistency_silent": route_metrics[
            "qwen:self_consistency"]["silent"],
        "qwen_system2_silent": route_metrics["qwen:system2"]["silent"],
        "gemma_independent_silent": route_metrics[
            "gemma:independent"]["silent"],
    }


def _canonical_objects(values: Sequence[str], relation: str) -> tuple[str, ...]:
    return tuple(sorted({
        canonical_key(str(value), relation) for value in values}))


def _legal_actions(
    row: Mapping[str, Any], incumbent: Sequence[str],
) -> list[dict[str, Any]]:
    relation = str(row["Relation"])
    collapsed = collapse_prediction(row, incumbent)
    components = row["relational_graph"]["components"]
    representatives = [str(item["representative"]) for item in components]
    raw: list[tuple[str, list[str]]] = [("KEEP", list(incumbent))]
    if _canonical_objects(collapsed, relation) != _canonical_objects(
            incumbent, relation):
        raw.append(("COLLAPSE", collapsed))
    if relation in NULLABLE_RELATIONS:
        raw.append(("EMPTY", []))
    if relation in SINGLE_RELATIONS:
        raw.extend(("REPLACE", [item]) for item in representatives)
    else:
        current = collapse_prediction(row, incumbent)
        current_keys = set(_canonical_objects(current, relation))
        for item in representatives:
            if canonical_key(item, relation) not in current_keys:
                raw.append(("ADD", [*current, item]))
        for index in range(len(current)):
            raw.append(("DROP", current[:index] + current[index + 1:]))
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for action_type, objects in raw:
        objects = list(dict.fromkeys(str(item) for item in objects))
        key = _canonical_objects(objects, relation)
        existing = unique.get(key)
        # KEEP is the canonical no-op if another action produces the same set.
        if existing is None or action_type == "KEEP":
            unique[key] = {
                "id": f"action:{len(unique)}",
                "node_type": "counterfactual_action",
                "action_type": action_type,
                "objects": objects,
            }
    actions = list(unique.values())
    if sum(action["action_type"] == "KEEP" for action in actions) != 1:
        raise AssertionError("every row must have exactly one KEEP action")
    return actions


def build_hierarchical_row(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Build a label-free hierarchical graph row with legal action nodes."""
    row = (
        dict(graph)
        if graph.get("relational_graph_schema")
        == "typed-relational-candidate-graph-v1"
        else augment_relational_graph(graph))
    relation = str(row["Relation"])
    incumbent = [str(item) for item in row.get("baseline_objects", [])]
    memory_nodes, memory_edges = _memory_route_nodes(row)
    nodes = [{
        "id": "query",
        "node_type": "subject_relation_query",
        "subject": str(row["SubjectEntity"]),
        "relation": relation,
        "relation_family": relation_family(relation),
    }, {
        "id": "state:incumbent",
        "node_type": "answer_state",
        "objects": incumbent,
    }, {
        "id": "state:null",
        "node_type": "null_state",
        "legal": relation in NULLABLE_RELATIONS,
    }]
    nodes.extend(memory_nodes)
    edges = list(memory_edges)
    components = row["relational_graph"]["components"]
    candidate_by_id = {
        node["id"]: node for node in row["relational_graph"]["nodes"]
        if node["node_type"] == "candidate_surface"}
    for component in components:
        evidence = _component_evidence(row, component)
        nodes.append({**component, "memory_evidence": evidence})
        for route in component.get("routes", {}):
            support_rate = float(
                component["routes"][route].get(
                    "max_support_rate", 0.0))
            if support_rate <= 0.0:
                continue
            route_prefix = {
                "qwen:self_consistency": "qwen_self_consistency",
                "qwen:system2": "qwen_system2",
                "gemma:independent": "gemma_independent",
            }.get(str(route))
            edges.append({
                "source": f"route:{route}",
                "target": str(component["id"]),
                "edge_type": "supports_component",
                "directed": True,
                "support_rate": support_rate,
                "support_share": (
                    evidence[f"{route_prefix}_share"]
                    if route_prefix is not None else 0.0),
                "within_route_rank": (
                    evidence[f"{route_prefix}_rank"]
                    if route_prefix is not None else 0.0),
            })
        for route, metadata in row.get("proposal_routes", {}).items():
            support_rate = float(component.get("routes", {}).get(
                route, {}).get("max_support_rate", 0.0))
            if (
                metadata.get("available")
                and support_rate <= 0.0
            ):
                edges.append({
                    "source": f"route:{route}",
                    "target": str(component["id"]),
                    "edge_type": "silent_on_component",
                    "directed": True,
                    "opportunity_samples": int(
                        metadata.get("n_samples", 0)),
                    "support_rate": 0.0,
                })
        for candidate_id in component["member_candidate_ids"]:
            candidate = candidate_by_id.get(candidate_id)
            if candidate is not None:
                nodes.append(candidate)
                edges.append({
                    "source": str(component["id"]),
                    "target": str(candidate_id),
                    "edge_type": "has_surface",
                    "directed": True,
                })
    for agent in (QWEN, GEMMA):
        for phase, labels in (
            ("existence", ("NO", "YES")),
            ("cardinality", ("ZERO", "ONE", "MANY")),
        ):
            for label in labels:
                probability = _prob(row, agent, phase, label)
                state_id = f"commitment:{agent}:{phase}:{label}"
                nodes.append({
                    "id": state_id,
                    "node_type": f"{phase}_commitment",
                    "label": label,
                    "probability": probability,
                })
                edges.append({
                    "source": f"memory:{agent}",
                    "target": state_id,
                    "edge_type": "commits",
                    "directed": True,
                    "weight": probability,
                })
    actions = _legal_actions(row, incumbent)
    nodes.extend(actions)
    for action in actions:
        edges.append({
            "source": "state:incumbent",
            "target": action["id"],
            "edge_type": "permits_transition",
            "directed": True,
            "action_type": action["action_type"],
        })
        for item in action["objects"]:
            component = _component_for_prediction(row, str(item))
            if component is not None:
                edges.append({
                    "source": str(component["id"]),
                    "target": action["id"],
                    "edge_type": "selected_by_action",
                    "directed": True,
                })
    return {
        "schema": "unified-hierarchical-memory-action-graph-v1",
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": relation,
        "relation_family": relation_family(relation),
        "incumbent_objects": incumbent,
        "source_graph_schema": row.get("schema"),
        "source_graph_sha256": None,
        "nodes": nodes,
        "edges": edges,
        "actions": actions,
        "component_count": len(components),
        "candidate_count": len(row.get("candidates", [])),
        # Retained only in memory for feature extraction; stripped when writing.
        "_source": row,
    }


FEATURE_NAMES = (
    # Shared relation semantics.  Relation identity is intentionally absent:
    # sparse relations must borrow strength from the same output family rather
    # than becoming fragile miniature relation-specific heads.
    "family_numeric", "family_single", "family_list",
    # Legal transition type.
    "action_keep", "action_collapse", "action_empty",
    "action_replace", "action_add", "action_drop",
    # Row/state geometry.
    "incumbent_empty", "incumbent_size", "output_size", "size_delta",
    "candidate_count", "component_count", "changed_component_count",
    # Memory commitments.
    "qwen_none_rate", "gemma_none_rate",
    "qwen_exist_no", "gemma_exist_no",
    "qwen_exist_yes", "gemma_exist_yes",
    "qwen_card_zero", "gemma_card_zero",
    "qwen_card_one", "gemma_card_one",
    "qwen_card_many", "gemma_card_many",
    # Proposed action evidence aggregated at memory level.
    "action_qwen_support_max", "action_gemma_support_max",
    "action_qwen_support_mean", "action_gemma_support_mean",
    "action_cross_memory", "action_memory_balance",
    "action_qwen_selected", "action_gemma_selected",
    "action_route_count", "action_within_qwen_route_count",
    # Route factors remain children of one Qwen memory, but preserve their
    # distinct empirical reliability instead of collapsing both to one max.
    "action_qwen_self_consistency", "action_qwen_system2",
    # Incumbent evidence and contrast.
    "incumbent_qwen_support", "incumbent_gemma_support",
    "incumbent_qwen_self_consistency", "incumbent_qwen_system2",
    "qwen_support_delta", "gemma_support_delta",
    "qwen_self_consistency_delta", "qwen_system2_delta",
    "joint_support_delta", "support_competition_margin",
    # Component structure and numeric geometry.
    "component_member_count", "component_alias_collapsed",
    "numeric_log_distance", "numeric_row_dispersion",
    # Directed graph messages.  These distinguish evidence entering the
    # proposed state from evidence removed from the incumbent state.
    "added_qwen_support", "added_gemma_support", "added_cross_memory",
    "added_qwen_self_consistency", "added_qwen_system2",
    "removed_qwen_support", "removed_gemma_support",
    "removed_cross_memory",
    "removed_qwen_self_consistency", "removed_qwen_system2",
    # Action-conditioned commitment messages.  Raw row commitments above are
    # useful for calibration, while these products can actually change the
    # ranking between actions in a linear model.
    "empty_null_support", "empty_nonnull_conflict",
    "add_many_support", "add_zero_conflict",
    "drop_zero_support", "drop_many_conflict",
    "replace_one_support",
    # Relation-family message channels: one shared model, typed graph edges.
    "numeric_added_joint_support", "numeric_removed_joint_support",
    "single_added_joint_support", "single_removed_joint_support",
    "list_added_joint_support", "list_removed_joint_support",
)


def _component_values(
    source: Mapping[str, Any], objects: Sequence[str],
) -> list[Mapping[str, Any]]:
    output, seen = [], set()
    for item in objects:
        component = _component_for_prediction(source, str(item))
        if component is not None and component["id"] not in seen:
            seen.add(component["id"])
            output.append(component)
    return output


def _evidence_summary(
    source: Mapping[str, Any], components: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    evidence = [_component_evidence(source, component)
                for component in components]
    q = [item["qwen_support"] for item in evidence]
    g = [item["gemma_support"] for item in evidence]
    qsc = [item["qwen_self_consistency_support"] for item in evidence]
    qs2 = [item["qwen_system2_support"] for item in evidence]
    return {
        "qmax": max(q, default=0.0),
        "gmax": max(g, default=0.0),
        "qmean": statistics.mean(q) if q else 0.0,
        "gmean": statistics.mean(g) if g else 0.0,
        "qscmax": max(qsc, default=0.0),
        "qs2max": max(qs2, default=0.0),
        "cross": float(any(item["cross_memory"] for item in evidence)),
        "balance": (
            min(max(q, default=0.0), max(g, default=0.0))
            if evidence else 0.0),
        "qselected": float(any(item["qwen_selected"] for item in evidence)),
        "gselected": float(any(item["gemma_selected"] for item in evidence)),
        "routes": max(
            (item["route_count"] for item in evidence), default=0.0),
        "qroutes": max(
            (item["within_qwen_route_count"] for item in evidence),
            default=0.0),
    }


def action_features(
    graph: Mapping[str, Any], action: Mapping[str, Any],
) -> list[float]:
    cached = action.get("_inference_features")
    if cached is not None:
        if len(cached) != len(FEATURE_NAMES):
            raise ContractError("cached unified action feature schema drift")
        return list(cached)
    source = graph["_source"]
    relation = str(graph["Relation"])
    family = relation_family(relation)
    incumbent = graph["incumbent_objects"]
    output = action["objects"]
    inc_components = _component_values(source, incumbent)
    out_components = _component_values(source, output)
    inc_ids = {item["id"] for item in inc_components}
    out_ids = {item["id"] for item in out_components}
    added_components = [
        item for item in out_components if item["id"] not in inc_ids]
    removed_components = [
        item for item in inc_components if item["id"] not in out_ids]
    changed = [
        item for item in [*out_components, *inc_components]
        if item["id"] in (out_ids ^ inc_ids)]
    # For KEEP/COLLAPSE, evidence on the output state is still informative.
    target = changed if changed else out_components
    action_evidence = _evidence_summary(source, target)
    incumbent_evidence = _evidence_summary(source, inc_components)
    added_evidence = _evidence_summary(source, added_components)
    removed_evidence = _evidence_summary(source, removed_components)
    all_support = sorted(
        (
            max(values["qwen_support"], values["gemma_support"])
            for values in (
                _component_evidence(source, item)
                for item in source["relational_graph"]["components"])
        ),
        reverse=True,
    )
    competition = (
        all_support[0] - all_support[1] if len(all_support) > 1
        else (all_support[0] if all_support else 0.0))
    numeric_distance = 0.0
    if relation in NUMERIC_RELATIONS and incumbent and output:
        try:
            old = float(str(incumbent[0]).replace(",", ""))
            new = float(str(output[0]).replace(",", ""))
            numeric_distance = abs(
                math.log10(max(abs(new), 1e-12))
                - math.log10(max(abs(old), 1e-12)))
        except ValueError:
            numeric_distance = 3.0
    action_type = str(action["action_type"])
    q_exist_no = _prob(source, QWEN, "existence", "NO")
    g_exist_no = _prob(source, GEMMA, "existence", "NO")
    q_exist_yes = _prob(source, QWEN, "existence", "YES")
    g_exist_yes = _prob(source, GEMMA, "existence", "YES")
    q_card_zero = _prob(source, QWEN, "cardinality", "ZERO")
    g_card_zero = _prob(source, GEMMA, "cardinality", "ZERO")
    q_card_one = _prob(source, QWEN, "cardinality", "ONE")
    g_card_one = _prob(source, GEMMA, "cardinality", "ONE")
    q_card_many = _prob(source, QWEN, "cardinality", "MANY")
    g_card_many = _prob(source, GEMMA, "cardinality", "MANY")
    none_support = (
        float(source["agents"][QWEN].get("none_rate", 0.0))
        + float(source["agents"][GEMMA].get("none_rate", 0.0)))
    added_joint = added_evidence["qmax"] + added_evidence["gmax"]
    removed_joint = removed_evidence["qmax"] + removed_evidence["gmax"]
    values = [
        float(family == "numeric"), float(family == "single"),
        float(family == "list"),
        *(float(action_type == item) for item in ACTION_TYPES),
        float(not incumbent), min(len(incumbent), 10) / 10.0,
        min(len(output), 10) / 10.0,
        max(-1.0, min(1.0, (len(output) - len(incumbent)) / 10.0)),
        min(graph["candidate_count"], 20) / 20.0,
        min(graph["component_count"], 20) / 20.0,
        min(len(out_ids ^ inc_ids), 10) / 10.0,
        float(source["agents"][QWEN].get("none_rate", 0.0)),
        float(source["agents"][GEMMA].get("none_rate", 0.0)),
        q_exist_no, g_exist_no,
        q_exist_yes, g_exist_yes,
        q_card_zero, g_card_zero,
        q_card_one, g_card_one,
        q_card_many, g_card_many,
        action_evidence["qmax"], action_evidence["gmax"],
        action_evidence["qmean"], action_evidence["gmean"],
        action_evidence["cross"], action_evidence["balance"],
        action_evidence["qselected"], action_evidence["gselected"],
        min(action_evidence["routes"], 5.0) / 5.0,
        min(action_evidence["qroutes"], 5.0) / 5.0,
        action_evidence["qscmax"], action_evidence["qs2max"],
        incumbent_evidence["qmax"], incumbent_evidence["gmax"],
        incumbent_evidence["qscmax"], incumbent_evidence["qs2max"],
        action_evidence["qmax"] - incumbent_evidence["qmax"],
        action_evidence["gmax"] - incumbent_evidence["gmax"],
        action_evidence["qscmax"] - incumbent_evidence["qscmax"],
        action_evidence["qs2max"] - incumbent_evidence["qs2max"],
        (action_evidence["qmax"] + action_evidence["gmax"])
        - (incumbent_evidence["qmax"] + incumbent_evidence["gmax"]),
        competition,
        min(sum(len(item.get("member_items", [])) for item in target), 10)
        / 10.0,
        float(any(item.get("alias_collapsed") for item in target)),
        min(numeric_distance, 3.0) / 3.0,
        min(float(source["agents"][QWEN].get(
            "numeric_log_mad") or 0.0), 5.0) / 5.0,
        added_evidence["qmax"], added_evidence["gmax"],
        added_evidence["cross"],
        added_evidence["qscmax"], added_evidence["qs2max"],
        removed_evidence["qmax"], removed_evidence["gmax"],
        removed_evidence["cross"],
        removed_evidence["qscmax"], removed_evidence["qs2max"],
        float(action_type == "EMPTY") * (
            q_exist_no + g_exist_no + q_card_zero + g_card_zero
            + none_support) / 5.0,
        float(action_type == "EMPTY") * (
            q_exist_yes + g_exist_yes) / 2.0,
        float(action_type == "ADD") * (
            q_card_many + g_card_many) / 2.0,
        float(action_type == "ADD") * (
            q_card_zero + g_card_zero) / 2.0,
        float(action_type == "DROP") * (
            q_card_zero + g_card_zero) / 2.0,
        float(action_type == "DROP") * (
            q_card_many + g_card_many) / 2.0,
        float(action_type == "REPLACE") * (
            q_card_one + g_card_one) / 2.0,
        float(family == "numeric") * added_joint,
        float(family == "numeric") * removed_joint,
        float(family == "single") * added_joint,
        float(family == "single") * removed_joint,
        float(family == "list") * added_joint,
        float(family == "list") * removed_joint,
    ]
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(
            f"feature schema drift: {len(values)} != {len(FEATURE_NAMES)}")
    if not all(math.isfinite(value) for value in values):
        raise ContractError("non-finite unified graph action feature")
    # This cache contains inference-legal graph features only.  Keeping it in
    # the written action graph makes the exact selector input reproducible.
    action["_inference_features"] = list(values)
    return values


class WeightedRidge:
    """Deterministic weighted ridge with an unpenalized intercept."""

    def __init__(self, l2: float):
        self.l2 = float(l2)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    @property
    def parameter_count(self) -> int:
        return len(FEATURE_NAMES) + 1

    def fit(
        self, x: Sequence[Sequence[float]], y: Sequence[float],
        weights: Sequence[float],
    ) -> "WeightedRidge":
        matrix = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape != (len(target), len(FEATURE_NAMES))
            or weight.shape != target.shape
            or len(target) < 2
            or np.any(weight <= 0)
        ):
            raise ValueError("invalid weighted ridge training arrays")
        weight = weight * (len(weight) / weight.sum())
        self.mean = np.average(matrix, axis=0, weights=weight)
        variance = np.average(
            (matrix - self.mean) ** 2, axis=0, weights=weight)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        standardized = (matrix - self.mean) / self.scale
        design = np.column_stack([np.ones(len(target)), standardized])
        root = np.sqrt(weight)[:, None]
        penalty = np.eye(design.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(
            (design * root).T @ (design * root) + penalty,
            (design * root).T @ (target * root[:, 0]),
        )
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("ridge is not fitted")
        matrix = np.asarray(x, dtype=np.float64)
        design = np.column_stack(
            [np.ones(matrix.shape[0]), (matrix - self.mean) / self.scale])
        return np.clip(design @ self.coef, -1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("ridge is not fitted")
        return {
            "schema": "unified-memory-action-weighted-ridge-v1",
            "l2": self.l2,
            "feature_names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coef.tolist(),
            "parameter_count": self.parameter_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WeightedRidge":
        if (
            value.get("schema") != "unified-memory-action-weighted-ridge-v1"
            or value.get("feature_names") != list(FEATURE_NAMES)
        ):
            raise ContractError("invalid unified action ridge artifact")
        model = cls(float(value["l2"]))
        model.mean = np.asarray(value["mean"], dtype=np.float64)
        model.scale = np.asarray(value["scale"], dtype=np.float64)
        model.coef = np.asarray(value["coefficients"], dtype=np.float64)
        if (
            model.mean.shape != (len(FEATURE_NAMES),)
            or model.scale.shape != model.mean.shape
            or model.coef.shape != (len(FEATURE_NAMES) + 1,)
        ):
            raise ContractError("unified action ridge shape mismatch")
        return model


ROW_GATE_FEATURE_NAMES = (
    "family_numeric", "family_single", "family_list",
    "incumbent_empty", "incumbent_size",
    "candidate_count", "component_count", "alternative_action_count",
    "qwen_none_rate", "gemma_none_rate",
    "qwen_exist_no", "gemma_exist_no",
    "qwen_exist_yes", "gemma_exist_yes",
    "qwen_card_zero", "gemma_card_zero",
    "qwen_card_one", "gemma_card_one",
    "qwen_card_many", "gemma_card_many",
    "incumbent_qwen_support", "incumbent_gemma_support",
    "max_added_qwen_support", "max_added_gemma_support",
    "any_added_cross_memory", "numeric_row_dispersion",
    "incumbent_qwen_self_consistency", "incumbent_qwen_system2",
    "max_added_qwen_self_consistency", "max_added_qwen_system2",
)


def row_gate_features(graph: Mapping[str, Any]) -> list[float]:
    source = graph["_source"]
    family = relation_family(str(graph["Relation"]))
    incumbent = graph["incumbent_objects"]
    inc = _evidence_summary(source, _component_values(source, incumbent))
    action_vectors = [
        action_features(graph, action) for action in graph["actions"]
        if action["action_type"] != "KEEP"]
    index = {name: offset for offset, name in enumerate(FEATURE_NAMES)}

    def maximum(name: str) -> float:
        return max(
            (values[index[name]] for values in action_vectors), default=0.0)

    values = [
        float(family == "numeric"), float(family == "single"),
        float(family == "list"),
        float(not incumbent), min(len(incumbent), 10) / 10.0,
        min(graph["candidate_count"], 20) / 20.0,
        min(graph["component_count"], 20) / 20.0,
        min(max(0, len(graph["actions"]) - 1), 20) / 20.0,
        float(source["agents"][QWEN].get("none_rate", 0.0)),
        float(source["agents"][GEMMA].get("none_rate", 0.0)),
        _prob(source, QWEN, "existence", "NO"),
        _prob(source, GEMMA, "existence", "NO"),
        _prob(source, QWEN, "existence", "YES"),
        _prob(source, GEMMA, "existence", "YES"),
        _prob(source, QWEN, "cardinality", "ZERO"),
        _prob(source, GEMMA, "cardinality", "ZERO"),
        _prob(source, QWEN, "cardinality", "ONE"),
        _prob(source, GEMMA, "cardinality", "ONE"),
        _prob(source, QWEN, "cardinality", "MANY"),
        _prob(source, GEMMA, "cardinality", "MANY"),
        inc["qmax"], inc["gmax"],
        maximum("added_qwen_support"),
        maximum("added_gemma_support"),
        maximum("added_cross_memory"),
        min(float(source["agents"][QWEN].get(
            "numeric_log_mad") or 0.0), 5.0) / 5.0,
        inc["qscmax"], inc["qs2max"],
        maximum("added_qwen_self_consistency"),
        maximum("added_qwen_system2"),
    ]
    if len(values) != len(ROW_GATE_FEATURE_NAMES):
        raise AssertionError("row gate feature schema drift")
    if not all(math.isfinite(value) for value in values):
        raise ContractError("non-finite row gate feature")
    return values


class WeightedLogistic:
    """Small deterministic weighted logistic regression."""

    def __init__(self, l2: float):
        self.l2 = float(l2)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    @property
    def parameter_count(self) -> int:
        return len(ROW_GATE_FEATURE_NAMES) + 1

    def fit(
        self, x: Sequence[Sequence[float]], y: Sequence[float],
        weights: Sequence[float],
    ) -> "WeightedLogistic":
        matrix = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        if (
            matrix.shape != (len(target), len(ROW_GATE_FEATURE_NAMES))
            or weight.shape != target.shape
            or not set(np.unique(target)) <= {0.0, 1.0}
            or len(np.unique(target)) != 2
        ):
            raise ValueError("invalid weighted logistic training arrays")
        weight = weight * (len(weight) / weight.sum())
        self.mean = np.average(matrix, axis=0, weights=weight)
        variance = np.average(
            (matrix - self.mean) ** 2, axis=0, weights=weight)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = np.column_stack(
            [np.ones(len(target)), (matrix - self.mean) / self.scale])
        beta = np.zeros(design.shape[1], dtype=np.float64)
        penalty = np.eye(design.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        for _ in range(100):
            logits = np.clip(design @ beta, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            curvature = np.maximum(
                probability * (1.0 - probability), 1e-8)
            gradient = design.T @ (
                weight * (probability - target)) + penalty @ beta
            hessian = (
                design.T @ (design * (weight * curvature)[:, None])
                + penalty)
            step = np.linalg.solve(hessian, gradient)
            beta -= step
            if float(np.max(np.abs(step))) < 1e-9:
                break
        self.coef = beta
        return self

    def predict_probability(
        self, x: Sequence[Sequence[float]],
    ) -> np.ndarray:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("logistic model is not fitted")
        matrix = np.asarray(x, dtype=np.float64)
        design = np.column_stack(
            [np.ones(matrix.shape[0]), (matrix - self.mean) / self.scale])
        logits = np.clip(design @ self.coef, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("logistic model is not fitted")
        return {
            "schema": "unified-memory-row-edit-logistic-v1",
            "l2": self.l2,
            "feature_names": list(ROW_GATE_FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coef.tolist(),
            "parameter_count": self.parameter_count,
            "decision_boundary": 0.5,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WeightedLogistic":
        if (
            value.get("schema") != "unified-memory-row-edit-logistic-v1"
            or value.get("feature_names") != list(ROW_GATE_FEATURE_NAMES)
            or float(value.get("decision_boundary", -1.0)) != 0.5
        ):
            raise ContractError("invalid unified row gate artifact")
        model = cls(float(value["l2"]))
        model.mean = np.asarray(value["mean"], dtype=np.float64)
        model.scale = np.asarray(value["scale"], dtype=np.float64)
        model.coef = np.asarray(value["coefficients"], dtype=np.float64)
        if (
            model.mean.shape != (len(ROW_GATE_FEATURE_NAMES),)
            or model.scale.shape != model.mean.shape
            or model.coef.shape != (len(ROW_GATE_FEATURE_NAMES) + 1,)
        ):
            raise ContractError("unified row gate shape mismatch")
        return model


class UnifiedSelector:
    """Shared edit-existence gate followed by shared action ranking."""

    def __init__(self, action_model: WeightedRidge, gate: WeightedLogistic):
        self.action_model = action_model
        self.gate = gate

    @property
    def parameter_count(self) -> int:
        return self.action_model.parameter_count + self.gate.parameter_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "unified-two-stage-memory-action-selector-v1",
            "parameter_count": self.parameter_count,
            "row_edit_gate": self.gate.to_dict(),
            "action_ranker": self.action_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UnifiedSelector":
        if value.get("schema") != "unified-two-stage-memory-action-selector-v1":
            raise ContractError("invalid unified selector artifact")
        model = cls(
            WeightedRidge.from_dict(value["action_ranker"]),
            WeightedLogistic.from_dict(value["row_edit_gate"]),
        )
        if int(value.get("parameter_count", -1)) != model.parameter_count:
            raise ContractError("unified selector parameter-count mismatch")
        return model


def _training_arrays(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[list[float]], list[float], list[float]]:
    x, y, weights = [], [], []
    for graph in graphs:
        gold = gold_by[_key(graph)]
        relation = str(graph["Relation"])
        base = _row_f1(graph["incumbent_objects"], gold, relation)
        row_weight = 1.0 / len(graph["actions"])
        for action in graph["actions"]:
            x.append(action_features(graph, action))
            y.append(_row_f1(action["objects"], gold, relation) - base)
            weights.append(row_weight)
    return x, y, weights


def _fit(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]], l2: float,
) -> UnifiedSelector:
    x, y, weights = _training_arrays(graphs, gold_by)
    action_model = WeightedRidge(l2).fit(x, y, weights)
    gate_x, gate_y = [], []
    for graph in graphs:
        gold = gold_by[_key(graph)]
        relation = str(graph["Relation"])
        baseline = _row_f1(
            graph["incumbent_objects"], gold, relation)
        reachable = max(
            _row_f1(action["objects"], gold, relation)
            for action in graph["actions"]) > baseline + 1e-12
        gate_x.append(row_gate_features(graph))
        gate_y.append(float(reachable))
    gate = WeightedLogistic(l2).fit(
        gate_x, gate_y, [1.0] * len(gate_y))
    return UnifiedSelector(action_model, gate)


def decode_one(
    model: WeightedRidge | UnifiedSelector, graph: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    action_model = (
        model.action_model if isinstance(model, UnifiedSelector) else model)
    edit_probability = (
        float(model.gate.predict_probability(
            [row_gate_features(graph)])[0])
        if isinstance(model, UnifiedSelector) else 1.0)
    predictions = action_model.predict([
        action_features(graph, action) for action in graph["actions"]])
    keep_index = next(
        index for index, action in enumerate(graph["actions"])
        if action["action_type"] == "KEEP")
    # Direct action comparison: no relation-specific or tuned threshold.
    best = max(
        range(len(graph["actions"])),
        key=lambda index: (
            float(predictions[index]),
            graph["actions"][index]["action_type"] == "KEEP",
            -len(graph["actions"][index]["objects"]),
            -index,
        ),
    )
    # A conventional posterior decision boundary, fixed globally.  This is
    # not selected per relation or against validation.
    gate_open = edit_probability > 0.5
    if not gate_open:
        best = keep_index
    action = graph["actions"][best]
    return list(action["objects"]), {
        "SubjectEntity": graph["SubjectEntity"],
        "Relation": graph["Relation"],
        "selected_action": action["action_type"],
        "predicted_utility": float(predictions[best]),
        "predicted_keep_utility": float(predictions[keep_index]),
        "predicted_advantage": float(
            predictions[best] - predictions[keep_index]),
        "edit_probability": edit_probability,
        "edit_gate_open": gate_open,
        "incumbent_objects": graph["incumbent_objects"],
        "selected_objects": action["objects"],
    }


def _prediction_rows(
    model: WeightedRidge | UnifiedSelector,
    graphs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, diagnostics = [], []
    for graph in graphs:
        objects, detail = decode_one(model, graph)
        rows.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": objects,
        })
        diagnostics.append(detail)
    return rows, diagnostics


def _subset_gold(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [gold_by[_key(graph)] for graph in graphs]


def _choose_l2_nested(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]], *, seed: int,
) -> tuple[float, dict[str, Any]]:
    folds = grouped_relation_folds(graphs, INNER_FOLDS, seed=seed)
    summaries = {}
    for l2 in L2_GRID:
        deltas = []
        for fold in range(INNER_FOLDS):
            fit_rows = [row for row in graphs if folds[_key(row)] != fold]
            hold_rows = [row for row in graphs if folds[_key(row)] == fold]
            model = _fit(fit_rows, gold_by, l2)
            predictions, _ = _prediction_rows(model, hold_rows)
            control = [{
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "ObjectEntities": row["incumbent_objects"],
            } for row in hold_rows]
            gold = _subset_gold(hold_rows, gold_by)
            deltas.append(
                score(predictions, gold)["*** All Relations ***"]
                - score(control, gold)["*** All Relations ***"])
        summaries[str(l2)] = {
            "fold_deltas": deltas,
            "mean_delta": statistics.mean(deltas),
        }
    best = max(
        L2_GRID,
        key=lambda value: (summaries[str(value)]["mean_delta"], value))
    return float(best), summaries


def _oracle_predictions(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for graph in graphs:
        relation = str(graph["Relation"])
        gold = gold_by[_key(graph)]
        best = max(
            graph["actions"],
            key=lambda action: (
                _row_f1(action["objects"], gold, relation),
                action["action_type"] == "KEEP",
                -len(action["objects"]),
            ),
        )
        rows.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": relation,
            "ObjectEntities": best["objects"],
        })
    return rows


def _relation_deltas(
    selected: Mapping[str, float], control: Mapping[str, float],
) -> dict[str, float]:
    return {
        relation: selected[relation] - control[relation]
        for relation in RELATIONS}


def _deployment_gate(
    pooled_delta: float, fold_deltas: Sequence[float],
    relation_deltas: Mapping[str, float],
) -> dict[str, Any]:
    checks = {
        "pooled_delta": pooled_delta >= MIN_POOLED_DELTA,
        "winning_folds": sum(delta > 0 for delta in fold_deltas)
        >= MIN_WINNING_FOLDS,
        "fold_floor": min(fold_deltas) >= MIN_FOLD_DELTA,
        "winning_relations": sum(delta > 0 for delta in relation_deltas.values())
        >= MIN_WINNING_RELATIONS,
        "relation_floor": min(relation_deltas.values())
        >= MIN_RELATION_DELTA,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_pooled_delta": MIN_POOLED_DELTA,
            "min_winning_folds": MIN_WINNING_FOLDS,
            "min_fold_delta": MIN_FOLD_DELTA,
            "min_winning_relations": MIN_WINNING_RELATIONS,
            "min_relation_delta": MIN_RELATION_DELTA,
        },
    }


def _agent_parameter_total(path: Path) -> int:
    config = json.loads(path.read_text())
    return sum(int(agent["parameter_upper_bound"])
               for agent in config["agents"])


def run_train_audit(args: argparse.Namespace) -> int:
    graph_path = Path(args.train_graph).resolve()
    gold_path = Path(args.train_gold).resolve()
    agents_path = Path(args.agents).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_graphs = read_jsonl(graph_path)
    gold_rows = read_jsonl(gold_path)
    if len(raw_graphs) != len(gold_rows):
        raise ContractError("train graph/gold row count mismatch")
    gold_by = {_key(row): row for row in gold_rows}
    if len(gold_by) != len(gold_rows):
        raise ContractError("duplicate train gold key")
    if {_key(row) for row in raw_graphs} != set(gold_by):
        raise ContractError("train graph/gold keys mismatch")
    graphs = [build_hierarchical_row(row) for row in raw_graphs]
    folds = grouped_relation_folds(graphs, OUTER_FOLDS, seed=args.seed)
    oof_by: dict[tuple[str, str], dict[str, Any]] = {}
    diagnostics, fold_records = [], []
    selected_l2 = []
    for fold in range(OUTER_FOLDS):
        fit_rows = [row for row in graphs if folds[_key(row)] != fold]
        hold_rows = [row for row in graphs if folds[_key(row)] == fold]
        l2, inner = _choose_l2_nested(
            fit_rows, gold_by, seed=args.seed + 1009 * (fold + 1))
        selected_l2.append(l2)
        model = _fit(fit_rows, gold_by, l2)
        predictions, detail = _prediction_rows(model, hold_rows)
        for row in predictions:
            oof_by[_key(row)] = row
        diagnostics.extend([{**item, "outer_fold": fold, "l2": l2}
                            for item in detail])
        control = [{
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": row["incumbent_objects"],
        } for row in hold_rows]
        hold_gold = _subset_gold(hold_rows, gold_by)
        selected_score = score(predictions, hold_gold)
        control_score = score(control, hold_gold)
        fold_records.append({
            "fold": fold,
            "rows": len(hold_rows),
            "selected_l2": l2,
            "inner_cv": inner,
            "control_score": control_score["*** All Relations ***"],
            "selected_score": selected_score["*** All Relations ***"],
            "delta": selected_score["*** All Relations ***"]
            - control_score["*** All Relations ***"],
        })
    if set(oof_by) != {_key(row) for row in graphs}:
        raise ContractError("nested OOF predictions do not cover train graph")
    ordered_predictions = [oof_by[_key(row)] for row in graphs]
    control_predictions = [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": row["incumbent_objects"],
    } for row in graphs]
    oracle_predictions = _oracle_predictions(graphs, gold_by)
    selected_scores = score(ordered_predictions, gold_rows)
    control_scores = score(control_predictions, gold_rows)
    oracle_scores = score(oracle_predictions, gold_rows)
    pooled_delta = (
        selected_scores["*** All Relations ***"]
        - control_scores["*** All Relations ***"])
    relation_deltas = _relation_deltas(selected_scores, control_scores)
    gate = _deployment_gate(
        pooled_delta, [row["delta"] for row in fold_records],
        relation_deltas)
    final_l2 = max(
        sorted(set(selected_l2)),
        key=lambda value: (selected_l2.count(value), value))
    final_model = _fit(graphs, gold_by, final_l2)
    agent_parameters = _agent_parameter_total(agents_path)
    selector_parameters = final_model.parameter_count
    total_parameters = agent_parameters + selector_parameters
    if total_parameters > PARAMETER_CAP:
        raise ContractError("portfolio plus selector exceeds parameter cap")

    graph_artifact = output / "TRAIN_ACTION_GRAPHS.jsonl"
    serializable_graphs = []
    for graph in graphs:
        item = {key: value for key, value in graph.items() if key != "_source"}
        serializable_graphs.append(item)
    write_jsonl_atomic(graph_artifact, serializable_graphs)
    graph_artifact.with_suffix(
        graph_artifact.suffix + ".manifest.json").write_text(json.dumps({
            "schema": "unified-hierarchical-memory-action-graph-manifest-v1",
            "split": "train",
            "rows": len(serializable_graphs),
            "contains_labels": False,
            "gold_aware": False,
            "inference_legal_features_only": True,
            "source_graph": str(graph_path),
            "source_graph_sha256": sha256(graph_path),
            "output_sha256": sha256(graph_artifact),
        }, indent=2, sort_keys=True) + "\n")
    write_jsonl_atomic(output / "TRAIN_OOF_PREDICTIONS.jsonl",
                       ordered_predictions)
    (output / "TRAIN_OOF_PREDICTIONS.jsonl.manifest.json").write_text(
        json.dumps({
            "schema": "unified-memory-action-oof-predictions-manifest-v1",
            "split": "train",
            "rows": len(ordered_predictions),
            "contains_labels": False,
            "gold_aware": True,
            "deployable": False,
            "oof_model_excludes_row": True,
            "selection_uses_train_labels": True,
            "validation_labels_used": False,
            "output_sha256": sha256(
                output / "TRAIN_OOF_PREDICTIONS.jsonl"),
        }, indent=2, sort_keys=True) + "\n")
    write_jsonl_atomic(output / "TRAIN_OOF_DIAGNOSTICS.jsonl", diagnostics)
    write_jsonl_atomic(output / "FOLDS.jsonl", [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "fold": folds[_key(row)],
    } for row in graphs])
    (output / "MODEL.json").write_text(json.dumps({
        "schema": "unified-memory-action-selector-model-v1",
        "development_only": True,
        "train_labels_used": True,
        "validation_labels_used": False,
        "selection_threshold": None,
        "decision_rule": (
            "global P(any beneficial edit)>0.5, then argmax predicted "
            "action utility including KEEP"),
        "nested_oof_selected_l2_values": selected_l2,
        "final_l2": final_l2,
        "model": final_model.to_dict(),
        "agent_parameter_upper_bound": agent_parameters,
        "selector_parameter_count": selector_parameters,
        "combined_parameter_upper_bound": total_parameters,
        "parameter_cap": PARAMETER_CAP,
    }, indent=2, sort_keys=True) + "\n")
    result = {
        "schema": "unified-memory-action-train-audit-v1",
        "development_only": True,
        "contains_labels": True,
        "validation_labels_used": False,
        "rows": len(graphs),
        "subjects": len({row["SubjectEntity"] for row in graphs}),
        "control_scores": control_scores,
        "selected_scores": selected_scores,
        "oracle_action_scores": oracle_scores,
        "pooled_delta": pooled_delta,
        "relation_deltas": relation_deltas,
        "folds": fold_records,
        "deployment_gate": gate,
        "action_counts": dict(Counter(
            item["selected_action"] for item in diagnostics)),
        "changed_rows": sum(
            item["selected_action"] != "KEEP" for item in diagnostics),
        "helped_rows": sum(
            _row_f1(
                oof_by[_key(graph)]["ObjectEntities"],
                gold_by[_key(graph)], graph["Relation"])
            > _row_f1(
                graph["incumbent_objects"], gold_by[_key(graph)],
                graph["Relation"]) + 1e-12
            for graph in graphs),
        "harmed_rows": sum(
            _row_f1(
                oof_by[_key(graph)]["ObjectEntities"],
                gold_by[_key(graph)], graph["Relation"])
            + 1e-12 < _row_f1(
                graph["incumbent_objects"], gold_by[_key(graph)],
                graph["Relation"])
            for graph in graphs),
        "artifacts": {
            "train_graph": str(graph_artifact),
            "train_graph_sha256": sha256(graph_artifact),
            "oof_predictions": str(output / "TRAIN_OOF_PREDICTIONS.jsonl"),
            "oof_predictions_sha256": sha256(
                output / "TRAIN_OOF_PREDICTIONS.jsonl"),
            "model": str(output / "MODEL.json"),
            "model_sha256": sha256(output / "MODEL.json"),
        },
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Unified heterogeneous-memory action graph", "",
        "Train-only nested subject-grouped OOF audit. Validation was not read.",
        "",
        f"- Rows: **{len(graphs)}**; unique subjects: "
        f"**{result['subjects']}**",
        f"- Control pooled F1: "
        f"**{control_scores['*** All Relations ***']:.9f}**",
        f"- Unified selector pooled F1: "
        f"**{selected_scores['*** All Relations ***']:.9f}**",
        f"- Pooled delta: **{pooled_delta:+.9f}**",
        f"- Reachable one-action oracle: "
        f"**{oracle_scores['*** All Relations ***']:.9f}**",
        f"- Changed/helped/harmed: **{result['changed_rows']} / "
        f"{result['helped_rows']} / {result['harmed_rows']}**",
        f"- Broad deployment gate: **{gate['passed']}**",
        f"- Learned selector parameters: **{selector_parameters}**",
        f"- Counted portfolio total: **{total_parameters:,} / "
        f"{PARAMETER_CAP:,}**", "",
        "## Relation deltas", "",
        "| relation | control | selector | delta |",
        "|---|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        lines.append(
            f"| {relation} | {control_scores[relation]:.6f} | "
            f"{selected_scores[relation]:.6f} | "
            f"{relation_deltas[relation]:+.6f} |")
    lines.extend(["", "## Outer folds", "",
                  "| fold | rows | L2 | delta |",
                  "|---:|---:|---:|---:|"])
    for item in fold_records:
        lines.append(
            f"| {item['fold']} | {item['rows']} | "
            f"{item['selected_l2']:.1f} | {item['delta']:+.6f} |")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "pooled_delta": pooled_delta,
        "gate_passed": gate["passed"],
        "helped": result["helped_rows"],
        "harmed": result["harmed_rows"],
        "output": str(output),
    }, indent=2))
    return 0


def run_decode(args: argparse.Namespace) -> int:
    """Decode a label-free split only after the broad train gate passes."""
    model_dir = Path(args.model_dir).resolve()
    result_path = model_dir / "RESULT.json"
    model_path = model_dir / "MODEL.json"
    if not result_path.is_file() or not model_path.is_file():
        raise ContractError("missing unified selector train audit")
    result = json.loads(result_path.read_text())
    if not result.get("deployment_gate", {}).get("passed"):
        raise ContractError(
            "unified selector failed the broad train-only deployment gate")
    model_artifact = json.loads(model_path.read_text())
    if (
        model_artifact.get("validation_labels_used") is not False
        or model_artifact.get("train_labels_used") is not True
    ):
        raise ContractError("invalid unified selector label provenance")
    selector = UnifiedSelector.from_dict(model_artifact["model"])
    graph_path = Path(args.graph).resolve()
    output_path = Path(args.output).resolve()
    raw = read_jsonl(graph_path)
    graphs = [build_hierarchical_row(row) for row in raw]
    predictions, diagnostics = _prediction_rows(selector, graphs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_path, predictions)
    diagnostics_path = output_path.with_name(
        output_path.stem + ".diagnostics.jsonl")
    write_jsonl_atomic(diagnostics_path, diagnostics)
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps({
            "schema": "unified-memory-action-predictions-manifest-v1",
            "rows": len(predictions),
            "contains_labels": False,
            "gold_aware": False,
            "validation_labels_used": False,
            "train_gate_passed": True,
            "source_graph": str(graph_path),
            "source_graph_sha256": sha256(graph_path),
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "output_sha256": sha256(output_path),
        }, indent=2, sort_keys=True) + "\n")
    print(f"predictions frozen: {output_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("train-audit")
    audit.add_argument("--train-graph", default=str(DEFAULT_GRAPH))
    audit.add_argument("--train-gold", default=str(DEFAULT_GOLD))
    audit.add_argument("--agents", default=str(DEFAULT_AGENTS))
    audit.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    audit.add_argument("--seed", type=int, default=20260727)
    audit.set_defaults(func=run_train_audit)
    decode = sub.add_parser("decode")
    decode.add_argument("--model-dir", default=str(DEFAULT_OUTPUT))
    decode.add_argument(
        "--graph",
        default=str(
            RUNS / "targeted_company_gemma_n3_20260724_v1/"
            "graphs/validation_graph.jsonl"))
    decode.add_argument(
        "--output", default=str(DEFAULT_OUTPUT / "PREDICTIONS.jsonl"))
    decode.set_defaults(func=run_decode)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
