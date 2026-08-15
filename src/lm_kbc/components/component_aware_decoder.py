#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence
from evaluate import try_parse_number
from lm_kbc.components.baseline_relative_route_decoder import ResidualRidge
from lm_kbc.core import ContractError, canonical_key
from lm_kbc.components.dual_model_validation import GEMMA, QWEN
from lm_kbc.components.relation_specific_structured_decoder import _prob
from lm_kbc.components.relational_candidate_graph import LIST_RELATIONS, NUMERIC_RELATIONS, SINGLE_RELATIONS, _component_for_prediction, collapse_prediction
from lm_kbc.components.route_aware_candidate_graph import ROUTE_GEMMA, ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2

def _object_key(item: str, relation: str) -> str:
    return canonical_key(str(item), relation)

def _surface_action_key(
        objects: Sequence[str], relation: str,
) -> tuple[str, ...]:
    return tuple(sorted({
        _object_key(str(item), relation) for item in objects
        if _object_key(str(item), relation)
    }))

def _component_token(
        graph: Mapping[str, Any], item: str,
) -> str:
    component = _component_for_prediction(graph, str(item))
    if component is not None:
        return str(component["id"])
    return f"surface:{_object_key(str(item), str(graph['Relation']))}"

def _action_tokens(
        graph: Mapping[str, Any], objects: Sequence[str], arm: str,
) -> tuple[str, ...]:
    if arm == "surface":
        return _surface_action_key(objects, str(graph["Relation"]))
    if arm != "component":
        raise ContractError(f"unknown decoder arm {arm}")
    return tuple(sorted({
        _component_token(graph, str(item)) for item in objects
    }))

def _dedupe_actions(
        graph: Mapping[str, Any], actions: Sequence[Sequence[str]], arm: str,
) -> list[list[str]]:
    unique: dict[tuple[str, ...], list[str]] = {}
    for action in actions:
        values = list(dict.fromkeys(str(item) for item in action))
        unique.setdefault(_action_tokens(graph, values, arm), values)
    return list(unique.values())

def _system2_only_candidate(node: Mapping[str, Any]) -> bool:
    summary = node.get("route_summary", {})
    if isinstance(summary, Mapping):
        return bool(summary.get("system2_only", False))
    routes = set(node.get("routes", {}))
    return routes == {ROUTE_QWEN_SYSTEM2}

def _eligible_candidates(
        graph: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    return [
        node for node in graph.get("candidates", [])
        if not _system2_only_candidate(node)
    ]

def _candidate_by_id(
        graph: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        f"candidate:{index}": node
        for index, node in enumerate(graph.get("candidates", []))
    }

def _eligible_components(
        graph: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    candidate_by_id = _candidate_by_id(graph)
    output = []
    for component in graph["relational_graph"]["components"]:
        members = [
            candidate_by_id[candidate_id]
            for candidate_id in component["member_candidate_ids"]]
        if any(not _system2_only_candidate(node) for node in members):
            output.append(component)
    return output

def actions_for(
        graph: Mapping[str, Any], control: Sequence[str], arm: str,
) -> list[list[str]]:
    """Enumerate bounded, inference-legal edits around the incumbent."""
    relation = str(graph["Relation"])
    if arm == "surface":
        representatives = [
            str(node["item"]) for node in _eligible_candidates(graph)]
        normalized_control = list(control)
    elif arm == "component":
        representatives = [
            str(component["representative"])
            for component in _eligible_components(graph)]
        normalized_control = collapse_prediction(graph, control)
    else:
        raise ContractError(f"unknown decoder arm {arm}")

    actions: list[list[str]] = [list(control), [], normalized_control]
    if relation in SINGLE_RELATIONS | NUMERIC_RELATIONS:
        actions.extend([[item] for item in representatives])
    elif relation in LIST_RELATIONS:
        current_tokens = set(_action_tokens(graph, normalized_control, arm))
        for item in representatives:
            if _action_tokens(graph, [item], arm)[0] not in current_tokens:
                actions.append([*normalized_control, item])
        for index in range(len(normalized_control)):
            actions.append(
                normalized_control[:index] + normalized_control[index + 1:])
    else:
        raise ContractError(f"unsupported relation {relation}")
    return _dedupe_actions(graph, actions, arm)

def _members(
        graph: Mapping[str, Any], component: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    candidate_by_id = _candidate_by_id(graph)
    return [
        candidate_by_id[candidate_id]
        for candidate_id in component["member_candidate_ids"]
        if candidate_id in candidate_by_id
        and not _system2_only_candidate(candidate_by_id[candidate_id])
    ]

def _route_support(
        members: Sequence[Mapping[str, Any]], route: str,
) -> float:
    return max([
        float(node.get("routes", {}).get(
            route, {}).get("support_rate", 0.0))
        for node in members
    ] or [0.0])

def _route_selected(
        members: Sequence[Mapping[str, Any]], route: str,
) -> float:
    return float(any(
        bool(node.get("routes", {}).get(route, {}).get("selected", False))
        for node in members))

def _component_summary(
        graph: Mapping[str, Any], component: Mapping[str, Any] | None,
        surface_node: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    if component is not None:
        members = _members(graph, component)
        alias_collapsed = float(len(members) > 1)
    elif surface_node is not None:
        members = [surface_node]
        alias_collapsed = 0.0
    else:
        members, alias_collapsed = [], 0.0
    qwen = _route_support(members, ROUTE_QWEN_SC)
    system2 = _route_support(members, ROUTE_QWEN_SYSTEM2)
    gemma = _route_support(members, ROUTE_GEMMA)
    routes_present = sum(value > 0 for value in (qwen, system2, gemma))
    independent_families = int(qwen > 0 or system2 > 0) + int(gemma > 0)
    numeric_values = [
        float(value) for node in members
        if (value := try_parse_number(str(node["item"]))) is not None
        and float(value) > 0]
    numeric_spread = 0.0
    if len(numeric_values) > 1:
        numeric_spread = min(
            1.0, math.log(max(numeric_values) / min(numeric_values))
            / math.log(1.05))
    return {
        "member_count": min(1.0, len(members) / 5.0),
        "alias_collapsed": alias_collapsed,
        "qwen_support": qwen,
        "system2_support": system2,
        "gemma_support": gemma,
        "qwen_selected": _route_selected(members, ROUTE_QWEN_SC),
        "system2_selected": _route_selected(
            members, ROUTE_QWEN_SYSTEM2),
        "gemma_selected": _route_selected(members, ROUTE_GEMMA),
        "route_count": routes_present / 3.0,
        "independent_family_count": independent_families / 2.0,
        "cross_model": float(independent_families == 2),
        "within_qwen": float(qwen > 0 and system2 > 0),
        "numeric_spread": numeric_spread,
    }

def _component_by_id(
        graph: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(component["id"]): component
        for component in graph["relational_graph"]["components"]
    }

def _surface_by_key(
        graph: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        _object_key(str(node["item"]), str(graph["Relation"])): node
        for node in _eligible_candidates(graph)
    }

SUMMARY_NAMES = (
    "member_count", "alias_collapsed", "qwen_support",
    "system2_support", "gemma_support", "qwen_selected",
    "system2_selected", "gemma_selected", "route_count",
    "independent_family_count", "cross_model", "within_qwen",
    "numeric_spread",
)

def feature_names() -> list[str]:
    return [
        "control_empty", "action_empty", "control_size", "action_size",
        "size_delta", "noop", "add", "drop", "replace", "multi_edit",
        "overlap", "component_arm", "collapse_action",
        *[f"added_{name}" for name in SUMMARY_NAMES],
        *[f"dropped_{name}" for name in SUMMARY_NAMES],
        "added_component_count", "dropped_component_count",
        "candidate_count", "component_count",
        "collapsed_surface_rate", "co_support_density",
        "qwen_none_rate", "gemma_none_rate",
        "qwen_exist_yes", "gemma_exist_yes",
        "qwen_exist_no", "gemma_exist_no",
        "qwen_card_zero", "gemma_card_zero",
        "qwen_card_one", "gemma_card_one",
        "qwen_card_many", "gemma_card_many",
        "action_cardinality_gap", "numeric_log_distance_from_control",
    ]

def _mean_summary(
        summaries: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    return {
        name: statistics.mean(summary[name] for summary in summaries)
        if summaries else 0.0
        for name in SUMMARY_NAMES
    }

def _token_summaries(
        graph: Mapping[str, Any], tokens: Sequence[str], arm: str,
) -> list[dict[str, float]]:
    if arm == "component":
        by_id = _component_by_id(graph)
        return [
            _component_summary(graph, by_id.get(token))
            for token in tokens]
    by_key = _surface_by_key(graph)
    return [
        _component_summary(
            graph, None,
            by_key.get(token.removeprefix("surface:")))
        for token in tokens]

def action_features(
        graph: Mapping[str, Any], control: Sequence[str],
        action: Sequence[str], arm: str,
) -> list[float]:
    relation = str(graph["Relation"])
    control_tokens = set(_action_tokens(graph, control, arm))
    action_tokens = set(_action_tokens(graph, action, arm))
    added, dropped = action_tokens - control_tokens, control_tokens - action_tokens
    edit_count = len(added) + len(dropped)
    added_summaries = _token_summaries(
        graph, sorted(added), arm)
    dropped_summaries = _token_summaries(
        graph, sorted(dropped), arm)
    added_summary = _mean_summary(added_summaries)
    dropped_summary = _mean_summary(dropped_summaries)
    expected_cardinality = statistics.mean([
        _prob(graph, agent, "cardinality", "ONE")
        + 2.0 * _prob(graph, agent, "cardinality", "MANY")
        for agent in (QWEN, GEMMA)
    ])
    relational_stats = graph["relational_graph"]["statistics"]
    surface_count = int(relational_stats["surface_candidate_count"])
    component_count = int(relational_stats["component_count"])
    co_support = int(relational_stats["co_support_edge_count"])
    possible_pairs = component_count * (component_count - 1) / 2
    numeric_distance = 0.0
    if relation in NUMERIC_RELATIONS and control and action:
        before = try_parse_number(str(control[0]))
        after = try_parse_number(str(action[0]))
        if before is not None and after is not None and before > 0 and after > 0:
            numeric_distance = min(
                1.0, abs(math.log(float(after) / float(before))) / 3.0)
    collapsed_control = collapse_prediction(graph, control)
    values = [
        float(not control_tokens), float(not action_tokens),
        min(1.0, len(control_tokens) / 5.0),
        min(1.0, len(action_tokens) / 5.0),
        max(-1.0, min(
            1.0, (len(action_tokens) - len(control_tokens)) / 3.0)),
        float(edit_count == 0),
        float(bool(added) and not dropped),
        float(bool(dropped) and not added),
        float(bool(added) and bool(dropped) and edit_count == 2),
        float(edit_count > 2),
        len(control_tokens & action_tokens)
        / max(1, len(control_tokens | action_tokens)),
        float(arm == "component"),
        float(
            arm == "component"
            and _surface_action_key(action, relation)
            == _surface_action_key(collapsed_control, relation)
            and _surface_action_key(control, relation)
            != _surface_action_key(collapsed_control, relation)),
        *[added_summary[name] for name in SUMMARY_NAMES],
        *[dropped_summary[name] for name in SUMMARY_NAMES],
        min(1.0, len(added_summaries) / 3.0),
        min(1.0, len(dropped_summaries) / 3.0),
        min(1.0, surface_count / 15.0),
        min(1.0, component_count / 15.0),
        (surface_count - component_count) / max(1, surface_count),
        co_support / max(1.0, possible_pairs),
        float(graph["agents"][QWEN]["none_rate"]),
        float(graph["agents"][GEMMA]["none_rate"]),
        _prob(graph, QWEN, "existence", "YES"),
        _prob(graph, GEMMA, "existence", "YES"),
        _prob(graph, QWEN, "existence", "NO"),
        _prob(graph, GEMMA, "existence", "NO"),
        _prob(graph, QWEN, "cardinality", "ZERO"),
        _prob(graph, GEMMA, "cardinality", "ZERO"),
        _prob(graph, QWEN, "cardinality", "ONE"),
        _prob(graph, GEMMA, "cardinality", "ONE"),
        _prob(graph, QWEN, "cardinality", "MANY"),
        _prob(graph, GEMMA, "cardinality", "MANY"),
        min(1.0, abs(len(action_tokens) - expected_cardinality) / 4.0),
        numeric_distance,
    ]
    if len(values) != len(feature_names()):
        raise AssertionError("component action feature schema drift")
    return values

def decode(
        model: ResidualRidge, graph: Mapping[str, Any],
        control: Sequence[str], arm: str, margin: float,
) -> tuple[list[str], dict[str, Any]]:
    actions = actions_for(graph, control, arm)
    estimates = model.predict([
        action_features(graph, control, action, arm) for action in actions])
    best = max(range(len(actions)), key=lambda index: (
        float(estimates[index]), -len(actions[index]),
        _action_tokens(graph, actions[index], arm)))
    proposed = actions[best]
    estimated_delta = float(estimates[best])
    use_control = estimated_delta <= margin
    return (list(control) if use_control else list(proposed)), {
        "arm": arm,
        "control_objects": list(control),
        "proposed_objects": list(proposed),
        "estimated_f1_delta": estimated_delta,
        "guard_margin": float(margin),
        "used_control": use_control,
        "action_count": len(actions),
    }
