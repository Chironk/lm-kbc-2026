#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
from lm_kbc.core import canonical_key
from lm_kbc.components.relational_candidate_graph import NUMERIC_RELATIONS, collapse_prediction

def _component_id_map(graph: Mapping[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for component in graph["relational_graph"]["components"]:
        component_id = str(component["id"])
        for member in component.get("member_items", []):
            mapping[canonical_key(str(member), str(graph["Relation"]))] = (
                component_id
            )
        mapping[canonical_key(
            str(component["representative"]), str(graph["Relation"])
        )] = component_id
    return mapping

def _component_ids(
    graph: Mapping[str, Any], objects: Sequence[str],
) -> set[str]:
    mapping = _component_id_map(graph)
    relation = str(graph["Relation"])
    result: set[str] = set()
    for item in collapse_prediction(graph, objects):
        key = canonical_key(str(item), relation)
        if key in mapping:
            result.add(mapping[key])
    return result

def _objects_for_ids(
    graph: Mapping[str, Any], component_ids: Iterable[str],
) -> list[str]:
    wanted = set(component_ids)
    return [
        str(component["representative"])
        for component in graph["relational_graph"]["components"]
        if str(component["id"]) in wanted
    ]
