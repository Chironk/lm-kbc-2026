#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence
from lm_kbc.core import ContractError, canonical_key
from lm_kbc.components.dual_model_validation import GEMMA, QWEN
from lm_kbc.components.heterogeneous_memory_selector import _key

TARGET_RELATIONS = {
    "companyTradesAtStockExchange",
    "personHasCityOfDeath",
}

ROUTE_QWEN_SC = "qwen:self_consistency"

ROUTE_QWEN_SYSTEM2 = "qwen:system2"

ROUTE_GEMMA = "gemma:independent"

def _route(
        *, model_family: str, support: int, samples: int, selected: bool,
        route_type: str,
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "route_type": route_type,
        "support": int(support),
        "samples": int(samples),
        "support_rate": support / samples if samples else 0.0,
        "selected": bool(selected),
    }

def _summarize_routes(routes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    families = {
        str(route["model_family"]) for route in routes.values()}
    qwen_routes = {
        name for name, route in routes.items()
        if route["model_family"] == QWEN}
    return {
        "route_count": len(routes),
        "model_family_count": len(families),
        "cross_model_agreement": len(families) > 1,
        "within_qwen_route_agreement": len(qwen_routes) > 1,
        "system2_supported": ROUTE_QWEN_SYSTEM2 in routes,
        "system2_only": (
            set(routes) == {ROUTE_QWEN_SYSTEM2}),
        "gemma_only": set(routes) == {ROUTE_GEMMA},
        "qwen_sc_only": set(routes) == {ROUTE_QWEN_SC},
    }

def _selected_keys(
    graph: Mapping[str, Any],
    agent: str,
    relation: str,
    *,
    required: bool,
) -> set[str]:
    """Return canonical selected outputs, failing closed when required.

    Historical numeric train graphs used untyped candidate keys (``"38"``)
    while current canonicalization produces ``"numeric:38"``.  Their
    ``selected_by`` flags were consequently all false even though
    ``agent_outputs`` contained a selected numeric answer.  Recompute route
    selection exclusively from the actual output surfaces.  A graph that
    claims an agent/route but lacks its output is invalid; silently inheriting
    ``selected_by`` would reintroduce split-dependent legacy semantics.
    """
    outputs = graph.get("agent_outputs", {})
    if agent not in outputs:
        if required:
            raise ContractError(
                f"{_key(graph)}: required agent_outputs missing {agent}")
        return set()
    if not isinstance(outputs[agent], list):
        raise ContractError(
            f"{_key(graph)}: agent_outputs[{agent!r}] must be a list")
    return {
        key for item in outputs.get(agent, [])
        if (key := canonical_key(str(item), relation))
    }

def augment_graph(
        graph: Mapping[str, Any], system2_objects: Sequence[str],
) -> dict[str, Any]:
    """Return one graph with explicit route evidence and merged identities."""
    row = copy.deepcopy(graph)
    relation = str(row["Relation"])
    selected_keys = {
        agent: _selected_keys(row, agent, relation, required=True)
        for agent in (QWEN, GEMMA)
    }
    nodes: dict[str, dict] = {}
    for original in row.get("candidates", []):
        node = copy.deepcopy(original)
        item_key = canonical_key(str(node.get("item", "")), relation)
        routes: dict[str, dict] = {}
        qwen = node.get("sources", {}).get(QWEN)
        gemma = node.get("sources", {}).get(GEMMA)
        if qwen is not None:
            qwen_selected = bool(
                item_key and item_key in selected_keys[QWEN])
            routes[ROUTE_QWEN_SC] = _route(
                model_family=QWEN,
                support=int(qwen.get("support", 0)),
                samples=int(qwen.get("samples", 0)),
                selected=qwen_selected,
                route_type="sampled-self-consistency",
            )
            node.setdefault("selected_by", {})[QWEN] = qwen_selected
        if gemma is not None:
            gemma_selected = bool(
                item_key and item_key in selected_keys[GEMMA])
            routes[ROUTE_GEMMA] = _route(
                model_family=GEMMA,
                support=int(gemma.get("support", 0)),
                samples=int(gemma.get("samples", 0)),
                selected=gemma_selected,
                route_type="independent-direct-recall",
            )
            node.setdefault("selected_by", {})[GEMMA] = gemma_selected
        node["routes"] = routes
        node["route_summary"] = _summarize_routes(routes)
        nodes[str(node["key"])] = node

    system2_keys = set()
    for item in system2_objects:
        key = canonical_key(str(item), relation)
        if not key or key in system2_keys:
            continue
        system2_keys.add(key)
        node = nodes.setdefault(key, {
            "key": key,
            "item": str(item),
            "type": (
                "numeric" if relation in {"hasArea", "hasCapacity"}
                else "string"),
            # ``sources`` retains its original model-proposal semantics for
            # backward compatibility. New consumers must use ``routes``.
            "sources": {},
            "selected_by": {QWEN: False, GEMMA: False},
            "routes": {},
            "route_summary": {},
        })
        node["routes"][ROUTE_QWEN_SYSTEM2] = _route(
            model_family=QWEN,
            support=1,
            samples=1,
            selected=True,
            route_type="alternate-prompt-judge",
        )
        node["route_summary"] = _summarize_routes(node["routes"])
    for node in nodes.values():
        # Recompute summaries for pre-existing nodes after System-2 merging.
        node["route_summary"] = _summarize_routes(node["routes"])

    row["candidates"] = sorted(nodes.values(), key=lambda node: (
        -int(node["route_summary"]["model_family_count"]),
        -int(node["route_summary"]["route_count"]),
        -sum(float(route["support_rate"])
             for route in node["routes"].values()),
        str(node["key"]),
    ))
    row["proposal_routes"] = {
        ROUTE_QWEN_SC: {
            "model_family": QWEN,
            "available": True,
            "n_samples": int(row["agents"][QWEN]["n_samples"]),
        },
        ROUTE_QWEN_SYSTEM2: {
            "model_family": QWEN,
            "available": relation in TARGET_RELATIONS,
            "n_samples": 1 if relation in TARGET_RELATIONS else 0,
            "objects": list(system2_objects),
        },
        ROUTE_GEMMA: {
            "model_family": GEMMA,
            "available": True,
            "n_samples": int(row["agents"][GEMMA]["n_samples"]),
        },
    }
    row["route_graph_schema"] = "explicit-proposal-routes-v1"
    row["route_selection_normalization"] = {
        "schema": "canonical-agent-output-selection-v2",
        "qwen_outputs_required": True,
        "gemma_outputs_required": True,
        "qwen_outputs_available": True,
        "gemma_outputs_available": True,
        "legacy_selected_by_fallback_allowed": False,
    }
    return row
