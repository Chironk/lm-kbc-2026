#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence
from evaluate import try_parse_number
from lm_kbc.core import ContractError
from lm_kbc.components.heterogeneous_memory_selector import _key

RELATION = "hasCapacity"

FAMILIES = (
    "qwen_recall",
    "gemma_independent",
    "ministral_independent",
)

TOLERANCE = 0.05

def _near(left: float, right: float) -> bool:
    if left <= 0 or right <= 0:
        return False
    return abs(left - right) / max(abs(right), 1e-12) <= TOLERANCE

def _format(value: float) -> str:
    return format(float(value), ".12g")

def _component_values(graph: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for component in graph["relational_graph"].get("components", []):
        value = try_parse_number(str(component.get("representative", "")))
        if value is None or not math.isfinite(value) or value <= 0:
            continue
        values[str(component["id"])] = float(value)
    return values

def _event_support(
    graph: Mapping[str, Any], component_values: Mapping[str, float],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    relational = graph["relational_graph"]
    nodes = {
        str(node["id"]): node
        for node in relational.get("nodes", [])
        if node.get("node_type") == "evidence_event"
    }
    targets: dict[str, set[str]] = defaultdict(set)
    for edge in relational.get("edges", []):
        if str(edge.get("edge_type")) != "supports":
            continue
        source, target = str(edge["source"]), str(edge["target"])
        if source in nodes and target in component_values:
            targets[source].add(target)
    events: list[dict[str, Any]] = []
    denominators: Counter[str] = Counter()
    for event_id, node in sorted(nodes.items()):
        family = str(node.get("model_family"))
        if family not in FAMILIES:
            continue
        # Every exact generation belongs in the denominator.  Counting only
        # parsed non-empty scalars would erase abstentions and parse failures,
        # artificially inflating a route's apparent confidence.
        denominators[family] += 1
        supported = targets.get(event_id, set())
        if str(node.get("status")) != "candidate_set" or not supported:
            continue
        # Capacity is single-valued.  A malformed multi-object event is not
        # silently converted into several independent votes.
        if len(supported) != 1:
            continue
        component_id = next(iter(supported))
        events.append({
            "id": event_id,
            "family": family,
            "component": component_id,
            "value": float(component_values[component_id]),
        })
    if set(denominators) != set(FAMILIES):
        raise ContractError(f"{_key(graph)}: incomplete family evidence")
    return events, denominators

def _complete_link_groups(
    component_values: Mapping[str, float],
) -> list[list[str]]:
    """Deterministic, non-transitive 5% groups over observed components."""
    groups: list[list[str]] = []
    ordered = sorted(component_values, key=lambda key: (
        component_values[key], key))
    for component_id in ordered:
        compatible = [
            index for index, group in enumerate(groups)
            if all(_near(component_values[component_id],
                         component_values[member]) for member in group)
        ]
        if not compatible:
            groups.append([component_id])
            continue
        # Prefer the group whose log-median is closest.  Complete-link avoids
        # the A~B~C transitivity error that can merge values >5% apart.
        index = min(compatible, key=lambda candidate: abs(
            math.log(component_values[component_id])
            - statistics.median(
                math.log(component_values[member])
                for member in groups[candidate])
        ))
        groups[index].append(component_id)
    return groups

def capacity_options(
    graph: Mapping[str, Any], incumbent_objects: Sequence[str],
) -> list[dict[str, Any]]:
    if str(graph["Relation"]) != RELATION:
        raise ContractError(f"capacity options requested for {_key(graph)}")
    if len(incumbent_objects) != 1:
        raise ContractError(f"{_key(graph)}: capacity incumbent must be scalar")
    incumbent = try_parse_number(str(incumbent_objects[0]))
    if incumbent is None or not math.isfinite(incumbent) or incumbent <= 0:
        raise ContractError(f"{_key(graph)}: invalid capacity incumbent")
    components = _component_values(graph)
    events, denominators = _event_support(graph, components)
    groups = _complete_link_groups(components)
    compatible_incumbent_groups = [
        index for index, group in enumerate(groups)
        if all(_near(incumbent, components[member]) for member in group)
    ]
    if not compatible_incumbent_groups:
        pseudo = "incumbent:pseudo"
        components[pseudo] = float(incumbent)
        groups = _complete_link_groups(components)
        compatible_incumbent_groups = [
            index for index, group in enumerate(groups)
            if pseudo in group
        ]
    incumbent_group = min(
        compatible_incumbent_groups,
        key=lambda index: (
            abs(
                math.log(incumbent)
                - statistics.median(
                    math.log(components[member]) for member in groups[index])
            ),
            index,
        ),
    )

    total_events = sum(denominators.values())
    options: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        group_set = set(group)
        grouped_events = [
            event for event in events if event["component"] in group_set]
        counts = Counter(event["family"] for event in grouped_events)
        rates = {
            family: counts[family] / denominators[family]
            for family in FAMILIES
        }
        supported_values = [float(event["value"]) for event in grouped_events]
        observed_values = [components[member] for member in group]
        representative_pool = supported_values or observed_values
        # Preserve an observed surface: select the support-weighted medoid.
        representative = min(
            sorted(set(representative_pool)),
            key=lambda value: (
                sum(abs(math.log(value / other)) for other in representative_pool),
                value,
            ),
        )
        is_incumbent = group_index == incumbent_group
        if is_incumbent:
            representative = float(incumbent)
        family_fraction = sum(counts[family] > 0 for family in FAMILIES) / 3.0
        family_rates = [rates[family] for family in FAMILIES]
        options.append({
            "value": float(representative),
            "component_ids": sorted(group),
            "is_incumbent": bool(is_incumbent),
            "family_fraction": family_fraction,
            "rates": rates,
            "mean_family_rate": statistics.mean(family_rates),
            "minimum_family_rate": min(family_rates),
            "maximum_family_rate": max(family_rates),
            "total_event_rate": len(grouped_events) / total_events,
            "component_fraction": len(group) / max(1, len(components)),
        })
    incumbents = [index for index, option in enumerate(options)
                  if option["is_incumbent"]]
    if len(incumbents) != 1:
        raise ContractError(f"{_key(graph)}: expected one incumbent option")
    return options
