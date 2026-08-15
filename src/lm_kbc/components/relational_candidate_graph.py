#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence
from evaluate import normalize_string, try_parse_number
from lm_kbc.core import canonical_key
from lm_kbc.components.route_aware_candidate_graph import ROUTE_GEMMA, ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2

LIST_RELATIONS = {
    "awardWonBy",
    "companyTradesAtStockExchange",
    "countryLandBordersCountry",
}

SINGLE_RELATIONS = {"personHasCityOfDeath"}

NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}

STOPWORDS = {"a", "an", "and", "at", "for", "in", "of", "on", "the", "to"}

class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left

def _parenthetical_base(value: str) -> str:
    return normalize_string(re.sub(r"\s*\([^()]{1,40}\)\s*$", "", value))

def _explicit_acronym(value: str) -> str | None:
    letters = re.sub(r"[^A-Za-z]", "", value)
    if not 2 <= len(letters) <= 6:
        return None
    alpha = "".join(character for character in value if character.isalpha())
    if not alpha or not alpha.isupper():
        return None
    return alpha.lower()

def _initials(value: str) -> str | None:
    tokens = [
        token for token in re.findall(r"[A-Za-z]+", value)
        if token.lower() not in STOPWORDS]
    if len(tokens) < 2:
        return None
    result = "".join(token[0].lower() for token in tokens)
    return result if 2 <= len(result) <= 8 else None

def equivalence_rule(
        left: str, right: str, relation: str,
) -> tuple[str, float] | None:
    """Return only high-precision, inference-legal equivalence evidence."""
    if normalize_string(left) == normalize_string(right):
        return "normalized_exact", 1.0
    if relation in NUMERIC_RELATIONS:
        a, b = try_parse_number(left), try_parse_number(right)
        if a is None or b is None:
            return None
        scale = max(abs(a), abs(b))
        if scale == 0:
            return ("numeric_exact_zero", 1.0) if a == b else None
        distance = abs(a - b) / scale
        if distance <= 0.05 + 1e-12:
            return "numeric_within_official_tolerance", 1.0 - distance
        return None
    left_base, right_base = _parenthetical_base(left), _parenthetical_base(right)
    if left_base and left_base == right_base:
        return "parenthetical_alias", 0.98
    left_acronym, right_acronym = _explicit_acronym(left), _explicit_acronym(right)
    if left_acronym and left_acronym == _initials(right):
        return "explicit_acronym", 0.97
    if right_acronym and right_acronym == _initials(left):
        return "explicit_acronym", 0.97
    return None

def _acronym_expansions(
        candidates: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Return the distinct long-form expansions observed for each acronym.

    Acronym equivalence is only safe in row context.  For example, ``TSE``
    matches both ``Tokyo Stock Exchange`` and ``Toronto Stock Exchange``.
    Pairwise matches are true as strings but cannot be transitively collapsed
    into one entity component.
    """
    expansions: dict[str, set[str]] = {}
    surfaces = [str(node["item"]) for node in candidates]
    for surface in surfaces:
        acronym = _explicit_acronym(surface)
        if acronym is None:
            continue
        for possible_expansion in surfaces:
            if _explicit_acronym(possible_expansion) is not None:
                continue
            if acronym == _initials(possible_expansion):
                expansions.setdefault(acronym, set()).add(
                    normalize_string(possible_expansion))
    return expansions

def _row_safe_equivalence_rule(
        left: str, right: str, relation: str,
        acronym_expansions: Mapping[str, set[str]],
) -> tuple[str, float] | None:
    rule = equivalence_rule(left, right, relation)
    if rule is None or rule[0] != "explicit_acronym":
        return rule
    acronym = _explicit_acronym(left) or _explicit_acronym(right)
    return (
        rule if acronym is not None
        and len(acronym_expansions.get(acronym, set())) == 1
        else None)

def _route_names(node: Mapping[str, Any]) -> set[str]:
    routes = node.get("routes")
    if isinstance(routes, Mapping):
        return set(str(name) for name in routes)
    names = set()
    sources = set(node.get("sources", {}))
    if "qwen_recall" in sources:
        names.add(ROUTE_QWEN_SC)
    if "gemma_independent" in sources:
        names.add(ROUTE_GEMMA)
    return names

def _node_strength(node: Mapping[str, Any]) -> float:
    routes = node.get("routes", {})
    if isinstance(routes, Mapping):
        return sum(float(value.get("support_rate", 0.0))
                   for value in routes.values())
    return sum(float(value.get("support_rate", 0.0))
               for value in node.get("sources", {}).values())

def _representative(
        members: Sequence[Mapping[str, Any]], relation: str,
) -> str:
    if relation in NUMERIC_RELATIONS:
        parsed = [
            (try_parse_number(str(node["item"])),
             max(1e-6, _node_strength(node)), str(node["item"]))
            for node in members]
        parsed = [item for item in parsed if item[0] is not None]
        if not parsed:
            return str(members[0]["item"])
        parsed.sort(key=lambda item: item[0])
        midpoint = sum(item[1] for item in parsed) / 2.0
        cumulative = 0.0
        for _, weight, original in parsed:
            cumulative += weight
            if cumulative >= midpoint:
                return original
        return parsed[-1][2]
    return str(max(members, key=lambda node: (
        int(_explicit_acronym(str(node["item"])) is None),
        sum(bool(value) for value in node.get("selected_by", {}).values()),
        _node_strength(node),
        min(len(str(node["item"])), 80),
        str(node["item"]),
    ))["item"])

def _component_routes(
        members: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    # New proposal routes must survive graph construction.  The original
    # implementation iterated a frozen three-route tuple, which silently
    # discarded component-level evidence for any later route even when the
    # candidate node itself retained it.
    route_names = sorted({
        str(route)
        for node in members
        for route in node.get("routes", {})
    })
    for route in route_names:
        route_members = [
            node for node in members if route in _route_names(node)]
        if not route_members:
            continue
        route_values = [
            node.get("routes", {}).get(route, {})
            for node in route_members
        ]
        generation_indices = sorted({
            int(generation)
            for value in route_values
            for generation in value.get("generation_indices", [])
        })
        sample_counts = [
            int(value.get("samples", 0))
            for value in route_values
            if int(value.get("samples", 0)) > 0
        ]
        samples = max(sample_counts, default=0)
        # Exact surfaces and aliases from the same generation are one piece
        # of evidence.  When generation provenance is available, component
        # support is therefore the union of generation IDs, not a sum over
        # surfaces and not merely the maximum support of one surface.
        distinct_generation_support = len(generation_indices)
        max_support_rate = max(
            float(value.get("support_rate", 0.0))
            for value in route_values)
        component_support_rate = (
            distinct_generation_support / samples
            if generation_indices and samples > 0
            else max_support_rate
        )
        output[route] = {
            "member_count": len(route_members),
            "max_support_rate": max_support_rate,
            "component_support_rate": component_support_rate,
            "distinct_generation_support": distinct_generation_support,
            "generation_indices": generation_indices,
            "samples": samples,
            "selected": any(
                bool(value.get("selected", False))
                for value in route_values),
        }
    return output

def augment_relational_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(graph)
    relation = str(row["Relation"])
    candidates = list(row.get("candidates", []))
    equivalent_edges: list[dict[str, Any]] = []
    blocked_equivalences: list[dict[str, Any]] = []
    if relation in NUMERIC_RELATIONS:
        # Complete-link clustering prevents tolerance chaining: 100~104 and
        # 104~108 must not silently make 100 equivalent to 108.
        parsed = sorted([
            (float(value), index)
            for index, node in enumerate(candidates)
            if (value := try_parse_number(str(node["item"]))) is not None
        ])
        unparsed = [
            index for index, node in enumerate(candidates)
            if try_parse_number(str(node["item"])) is None]
        ordered_groups: list[list[int]] = []
        for value, index in parsed:
            if not ordered_groups:
                ordered_groups.append([index])
                continue
            existing_values = [
                try_parse_number(str(candidates[item]["item"]))
                for item in ordered_groups[-1]]
            minimum = min(float(item) for item in existing_values)
            maximum = max([float(item) for item in existing_values] + [value])
            scale = max(abs(minimum), abs(maximum))
            within = (
                maximum == minimum if scale == 0
                else abs(maximum - minimum) / scale <= 0.05 + 1e-12)
            if within:
                ordered_groups[-1].append(index)
            else:
                ordered_groups.append([index])
        ordered_groups.extend([[index] for index in unparsed])
        ordered_groups.sort(key=lambda values: min(values))
        for indices in ordered_groups:
            for offset, left in enumerate(indices):
                for right in indices[offset + 1:]:
                    rule = equivalence_rule(
                        str(candidates[left]["item"]),
                        str(candidates[right]["item"]), relation)
                    if rule is None:
                        raise AssertionError(
                            "numeric complete-link component exceeds tolerance")
                    equivalent_edges.append({
                        "source": f"candidate:{left}",
                        "target": f"candidate:{right}",
                        "edge_type": "equivalent_to",
                        "directed": False,
                        "rule": rule[0],
                        "confidence": rule[1],
                    })
    else:
        union = _UnionFind(len(candidates))
        acronym_expansions = _acronym_expansions(candidates)
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                rule = equivalence_rule(
                    str(candidates[left]["item"]),
                    str(candidates[right]["item"]), relation)
                if rule is None:
                    continue
                if rule[0] == "explicit_acronym":
                    acronym = (
                        _explicit_acronym(str(candidates[left]["item"]))
                        or _explicit_acronym(
                            str(candidates[right]["item"])))
                    expansions = acronym_expansions.get(
                        str(acronym), set())
                    if len(expansions) != 1:
                        blocked_equivalences.append({
                            "source": f"candidate:{left}",
                            "target": f"candidate:{right}",
                            "rule": "explicit_acronym",
                            "reason": "ambiguous_row_level_expansion",
                            "acronym": acronym,
                            "distinct_expansions": sorted(expansions),
                        })
                        continue
                union.union(left, right)
                equivalent_edges.append({
                    "source": f"candidate:{left}",
                    "target": f"candidate:{right}",
                    "edge_type": "equivalent_to",
                    "directed": False,
                    "rule": rule[0],
                    "confidence": rule[1],
                })
        groups: dict[int, list[int]] = {}
        for index in range(len(candidates)):
            groups.setdefault(union.find(index), []).append(index)
        ordered_groups = sorted(
            groups.values(), key=lambda values: min(values))
    component_for: dict[int, str] = {}
    components: list[dict[str, Any]] = []
    for component_index, indices in enumerate(ordered_groups):
        component_id = f"component:{component_index}"
        for index in indices:
            component_for[index] = component_id
        members = [candidates[index] for index in indices]
        components.append({
            "id": component_id,
            "node_type": "candidate_component",
            "member_candidate_ids": [
                f"candidate:{index}" for index in indices],
            "member_keys": [str(node["key"]) for node in members],
            "member_items": [str(node["item"]) for node in members],
            "representative": _representative(members, relation),
            "routes": _component_routes(members),
            "alias_collapsed": len(members) > 1,
        })

    nodes: list[dict[str, Any]] = [{
        "id": "hypothesis:null",
        "node_type": "null_hypothesis",
        "meaning": "the relation has no answer for this subject",
    }]
    available_routes = row.get("proposal_routes", {})
    route_names = sorted(
        set(str(route) for route in available_routes)
        | {
            str(route)
            for candidate in candidates
            for route in candidate.get("routes", {})
        })
    for route in route_names:
        metadata = available_routes.get(route, {})
        if metadata.get("available", route != ROUTE_QWEN_SYSTEM2):
            route_members = [
                candidate.get("routes", {}).get(route, {})
                for candidate in candidates
                if route in candidate.get("routes", {})
            ]
            nodes.append({
                "id": f"route:{route}",
                "node_type": "evidence_route",
                "route": route,
                "model_family": metadata.get(
                    "model_family",
                    next((
                        value.get("model_family")
                        for value in route_members
                        if value.get("model_family")
                    ), "gemma_independent" if route == ROUTE_GEMMA
                       else "qwen_recall")),
                "n_samples": int(metadata.get(
                    "n_samples",
                    max((int(value.get("samples", 0))
                         for value in route_members), default=0))),
            })
    for index, candidate in enumerate(candidates):
        nodes.append({
            "id": f"candidate:{index}",
            "node_type": "candidate_surface",
            "key": str(candidate["key"]),
            "item": str(candidate["item"]),
        })
    nodes.extend(copy.deepcopy(components))

    edges = list(equivalent_edges)
    for index, candidate in enumerate(candidates):
        edges.append({
            "source": f"candidate:{index}",
            "target": component_for[index],
            "edge_type": "member_of",
            "directed": True,
            "weight": 1.0,
        })
        for route in sorted(_route_names(candidate)):
            route_data = candidate.get("routes", {}).get(route, {})
            edges.append({
                "source": f"candidate:{index}",
                "target": f"route:{route}",
                "edge_type": "proposed_by",
                "directed": True,
                "support_rate": float(route_data.get(
                    "support_rate", 0.0)),
                "selected": bool(route_data.get("selected", False)),
                "generation_indices": sorted({
                    int(value)
                    for value in route_data.get(
                        "generation_indices", [])
                }),
            })
    route_node_ids = {node["id"] for node in nodes
                      if node["node_type"] == "evidence_route"}
    if ({f"route:{ROUTE_QWEN_SC}", f"route:{ROUTE_QWEN_SYSTEM2}"}
            <= route_node_ids):
        edges.append({
            "source": f"route:{ROUTE_QWEN_SC}",
            "target": f"route:{ROUTE_QWEN_SYSTEM2}",
            "edge_type": "dependent_with",
            "directed": False,
            "reason": "shared_qwen_checkpoint",
            "independence_weight": 0.0,
        })
    for component in components:
        edges.append({
            "source": "hypothesis:null",
            "target": component["id"],
            "edge_type": "contradicts",
            "directed": False,
            "reason": "null_vs_nonnull",
            "weight": 1.0,
        })
    if relation in SINGLE_RELATIONS | NUMERIC_RELATIONS:
        for left in range(len(components)):
            for right in range(left + 1, len(components)):
                edges.append({
                    "source": components[left]["id"],
                    "target": components[right]["id"],
                    "edge_type": "contradicts",
                    "directed": False,
                    "reason": (
                        "single_valued_relation" if relation in SINGLE_RELATIONS
                        else "distinct_numeric_clusters"),
                    "weight": 1.0,
                })
    if relation in LIST_RELATIONS:
        for left in range(len(components)):
            for right in range(left + 1, len(components)):
                common = (
                    set(components[left]["routes"])
                    & set(components[right]["routes"]))
                if common:
                    cooccurrence_by_route: dict[str, dict[str, Any]] = {}
                    for route in sorted(common):
                        left_route = components[left]["routes"][route]
                        right_route = components[right]["routes"][route]
                        left_generations = set(
                            left_route.get("generation_indices", []))
                        right_generations = set(
                            right_route.get("generation_indices", []))
                        intersection = sorted(
                            left_generations & right_generations)
                        samples = max(
                            int(left_route.get("samples", 0)),
                            int(right_route.get("samples", 0)),
                        )
                        cooccurrence_by_route[route] = {
                            "generation_indices": intersection,
                            "count": len(intersection),
                            "rate": (
                                len(intersection) / samples
                                if samples > 0 else 0.0
                            ),
                            "generation_provenance_available": bool(
                                left_generations or right_generations),
                        }
                    cooccurring_routes = sorted(
                        route for route, value
                        in cooccurrence_by_route.items()
                        if value["count"] > 0
                    )
                    selected_together_routes = sorted(
                        route for route in common
                        if (
                            components[left]["routes"][route].get(
                                "selected", False)
                            and components[right]["routes"][route].get(
                                "selected", False)
                        )
                    )
                    # Sharing a sampled route does not mean two facts were
                    # proposed together.  Keep route overlap as provenance,
                    # and reserve co_supported_with for observed
                    # same-generation evidence.  Otherwise every Qwen
                    # self-consistency row becomes a false candidate clique.
                    edge_type = (
                        "co_supported_with"
                        if cooccurring_routes
                        else "co_selected_with"
                        if selected_together_routes
                        else "shares_route_with"
                    )
                    edges.append({
                        "source": components[left]["id"],
                        "target": components[right]["id"],
                        "edge_type": edge_type,
                        "directed": False,
                        "routes": sorted(common),
                        "cooccurring_routes": cooccurring_routes,
                        "selected_together_routes": selected_together_routes,
                        "weight": len(common) / max(1, len(route_names)),
                        "cooccurrence_by_route": cooccurrence_by_route,
                        "cooccurrence_count": sum(
                            value["count"]
                            for value in cooccurrence_by_route.values()
                        ),
                        "cooccurrence_rate": max(
                            (
                                value["rate"]
                                for value in cooccurrence_by_route.values()
                            ),
                            default=0.0,
                        ),
                    })
    row["relational_graph"] = {
        "schema": "typed-relational-candidate-graph-v1",
        "relation": relation,
        "nodes": nodes,
        "edges": edges,
        "components": components,
        "blocked_equivalences": blocked_equivalences,
        "statistics": {
            "surface_candidate_count": len(candidates),
            "component_count": len(components),
            "collapsed_surface_count": len(candidates) - len(components),
            "equivalence_edge_count": len(equivalent_edges),
            "blocked_equivalence_count": len(blocked_equivalences),
            "contradiction_edge_count": sum(
                edge["edge_type"] == "contradicts" for edge in edges),
            "co_support_edge_count": sum(
                edge["edge_type"] == "co_supported_with" for edge in edges),
            "co_selected_edge_count": sum(
                edge["edge_type"] == "co_selected_with" for edge in edges),
            "shared_route_edge_count": sum(
                edge["edge_type"] == "shares_route_with" for edge in edges),
        },
    }
    row["relational_graph_schema"] = "typed-relational-candidate-graph-v1"
    return row

def _component_for_prediction(
        graph: Mapping[str, Any], item: str,
) -> Mapping[str, Any] | None:
    relation = str(graph["Relation"])
    key = canonical_key(str(item), relation)
    relational = graph["relational_graph"]
    acronym_expansions = _acronym_expansions(
        graph.get("candidates", []))
    # Exact membership must win globally before any approximate fallback is
    # considered.  Otherwise an exact member of a later numeric complete-link
    # component can be captured by an earlier component whose boundary member
    # happens to be within 5% (the same tolerance-chaining ambiguity that the
    # builder deliberately avoids).
    for component in relational["components"]:
        if key in set(component["member_keys"]):
            return component
    approximate_matches = []
    for component in relational["components"]:
        for member in component["member_items"]:
            if _row_safe_equivalence_rule(
                    str(item), str(member), relation,
                    acronym_expansions) is not None:
                approximate_matches.append(component)
                break
    # An external surface that approximately matches multiple components is
    # ambiguous.  Failing closed preserves component separation.
    return approximate_matches[0] if len(approximate_matches) == 1 else None

def collapse_prediction(
        graph: Mapping[str, Any], objects: Sequence[str],
) -> list[str]:
    relation = str(graph["Relation"])
    output: list[str] = []
    seen_components: set[str] = set()
    seen_keys: set[str] = set()
    for item in objects:
        component = _component_for_prediction(graph, str(item))
        if component is not None:
            if component["id"] in seen_components:
                continue
            seen_components.add(component["id"])
            value = str(component["representative"])
        else:
            value = str(item)
        key = canonical_key(value, relation)
        if key not in seen_keys:
            seen_keys.add(key)
            output.append(value)
    return output
