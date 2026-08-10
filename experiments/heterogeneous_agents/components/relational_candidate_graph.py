#!/usr/bin/env python3
"""Build and audit a genuinely relational heterogeneous candidate graph.

``build`` is label-free.  It augments each existing candidate row with typed
candidate, component, route, and null-hypothesis nodes plus explicit edges:

* candidate -> component membership;
* candidate -> proposal-route provenance;
* high-precision alias/numeric equivalence;
* null/candidate and single-valued candidate contradictions;
* list-candidate co-support;
* same-model dependence between the two Qwen routes.

``analyze`` is deliberately gold-aware and writes only artifacts whose
manifests say ``contains_labels=true`` and ``gold_aware=true``.  It produces an
error ledger that separates candidate-supply failures from reachable selection
failures, null/cardinality mistakes, alias fragmentation, and numeric-cluster
mistakes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluate import RELATION_TYPE, normalize_string, true_positives, try_parse_number
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    _key,
    _load_graph,
)
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    _row_f1,
)
from experiments.heterogeneous_agents.components.route_aware_candidate_graph import (
    ROUTE_GEMMA,
    ROUTE_QWEN_SC,
    ROUTE_QWEN_SYSTEM2,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "route_aware_graph_20260723_v1")
DEFAULT_VALIDATION_PREDICTIONS = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "route_residual_decoder_20260723_v1/"
    "validation_train_selected.jsonl")
LIST_RELATIONS = {
    "awardWonBy",
    "companyTradesAtStockExchange",
    "countryLandBordersCountry",
}
SINGLE_RELATIONS = {"personHasCityOfDeath"}
NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}
ROUTES = (ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2, ROUTE_GEMMA)
STOPWORDS = {"a", "an", "and", "at", "for", "in", "of", "on", "the", "to"}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


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


def _write_graph(
        path: Path, rows: Sequence[Mapping[str, Any]], *, split: str,
        source_path: Path,
) -> None:
    write_jsonl_atomic(path, rows)
    manifest = {
        "schema": "heterogeneous-memory-graph-manifest-v1",
        "split": split,
        "rows": len(rows),
        "contains_labels": False,
        "gold_aware": False,
        "output_sha256": sha256(path),
        "source_graph": str(source_path),
        "source_graph_sha256": sha256(source_path),
        "relational_graph_schema": "typed-relational-candidate-graph-v1",
        "equivalence_rules": [
            "normalized_exact", "parenthetical_alias", "explicit_acronym",
            "numeric_within_official_tolerance"],
        "parameter_count_delta": 0,
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    graph_dir, plan_dir = output / "graphs", output / "plan"
    graph_dir.mkdir(parents=True, exist_ok=True)
    plan_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    graph_statistics: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        source_path = source / f"graphs/{split}_graph.jsonl"
        rows = _load_graph(source_path, expected_split=split)
        augmented = [augment_relational_graph(row) for row in rows]
        target = graph_dir / f"{split}_graph.jsonl"
        _write_graph(
            target, augmented, split=split, source_path=source_path)
        counts[split] = len(augmented)
        relations = sorted({str(row["Relation"]) for row in augmented})
        graph_statistics[split] = {}
        for relation in relations:
            subset = [
                row["relational_graph"] for row in augmented
                if row["Relation"] == relation]
            edge_counts: dict[str, int] = {}
            for relational in subset:
                for edge in relational["edges"]:
                    edge_type = str(edge["edge_type"])
                    edge_counts[edge_type] = edge_counts.get(
                        edge_type, 0) + 1
            graph_statistics[split][relation] = {
                "rows": len(subset),
                "surface_candidates": sum(
                    int(item["statistics"]["surface_candidate_count"])
                    for item in subset),
                "components": sum(
                    int(item["statistics"]["component_count"])
                    for item in subset),
                "collapsed_surfaces": sum(
                    int(item["statistics"]["collapsed_surface_count"])
                    for item in subset),
                "blocked_equivalences": sum(
                    int(item["statistics"]["blocked_equivalence_count"])
                    for item in subset),
                "edge_counts": edge_counts,
            }
    source_plan = source / "plan/PLAN.json"
    if not source_plan.is_file():
        raise ContractError(f"missing source plan {source_plan}")
    shutil.copy2(source_plan, plan_dir / "PLAN.json")
    record = {
        "schema": "typed-relational-candidate-graph-build-v1",
        "labels_opened": False,
        "validation_labels_opened": False,
        "source": str(source),
        "source_train_sha256": sha256(
            source / "graphs/train_graph.jsonl"),
        "source_validation_sha256": sha256(
            source / "graphs/validation_graph.jsonl"),
        "train_rows": counts["train"],
        "validation_rows": counts["validation"],
        "parameter_count_delta": 0,
        "graph_statistics": graph_statistics,
    }
    (output / "BUILD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"relational graph ready: {counts['train']} train, "
        f"{counts['validation']} validation")
    return 0


def _aliases(gold: Mapping[str, Any]) -> list[list[str]]:
    values = gold.get("ObjectEntities", [])
    return [[str(item)] for item in values] if values and isinstance(
        values[0], str) else values


def _candidate_hit(
        graph: Mapping[str, Any], gold: Mapping[str, Any],
) -> bool:
    relation = str(graph["Relation"])
    return true_positives(
        [str(node["item"]) for node in graph.get("candidates", [])],
        _aliases(gold), RELATION_TYPE[relation], 0.05) > 0


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


def component_actions(
        graph: Mapping[str, Any], current: Sequence[str],
) -> list[list[str]]:
    relation = str(graph["Relation"])
    representatives = [
        str(component["representative"])
        for component in graph["relational_graph"]["components"]]
    actions: list[list[str]] = [list(current), [], collapse_prediction(
        graph, current)]
    if relation in SINGLE_RELATIONS | NUMERIC_RELATIONS:
        actions.extend([[item] for item in representatives])
    else:
        collapsed = collapse_prediction(graph, current)
        current_components = {
            component["id"] for item in current
            if (component := _component_for_prediction(graph, str(item)))
            is not None}
        for component in graph["relational_graph"]["components"]:
            if component["id"] not in current_components:
                actions.append([*collapsed, str(component["representative"])])
        for item in current:
            item_key = canonical_key(str(item), relation)
            actions.append([
                value for value in current
                if canonical_key(str(value), relation) != item_key])
    unique: dict[tuple[str, ...], list[str]] = {}
    for action in actions:
        values = list(dict.fromkeys(str(item) for item in action))
        key = tuple(sorted(canonical_key(item, relation) for item in values))
        unique.setdefault(key, values)
    return list(unique.values())


def classify_error(
        graph: Mapping[str, Any], prediction: Sequence[str],
        gold: Mapping[str, Any],
) -> dict[str, Any]:
    relation = str(graph["Relation"])
    current = list(prediction)
    current_f1 = _row_f1(current, gold, relation)
    actions = component_actions(graph, current)
    action_scores = [_row_f1(action, gold, relation) for action in actions]
    best_index = max(
        range(len(actions)),
        key=lambda index: (action_scores[index], -len(actions[index])))
    oracle_f1 = action_scores[best_index]
    collapsed = collapse_prediction(graph, current)
    collapsed_f1 = _row_f1(collapsed, gold, relation)
    gold_nonnull = bool(gold.get("ObjectEntities"))
    prediction_nonnull = bool(current)
    hit = _candidate_hit(graph, gold)
    alias_fragmentation = (
        len(collapsed) < len(current)
        and collapsed_f1 > current_f1 + 1e-12)
    flags = {
        "false_positive_null": not gold_nonnull and prediction_nonnull,
        "false_negative_null": gold_nonnull and not prediction_nonnull,
        "candidate_hit": hit,
        "reachable_improvement": oracle_f1 > current_f1 + 1e-12,
        "alias_fragmentation": alias_fragmentation,
        "component_representative_improvement": (
            collapsed != current
            and collapsed_f1 > current_f1 + 1e-12),
        "cardinality_under": (
            gold_nonnull and len(current) < len(_aliases(gold))),
        "cardinality_over": (
            len(current) > len(_aliases(gold))),
        "numeric_cluster_failure": (
            relation in NUMERIC_RELATIONS and hit
            and oracle_f1 > current_f1 + 1e-12),
    }
    if current_f1 >= 1.0 - 1e-12:
        primary = "correct"
    elif flags["false_positive_null"]:
        primary = "false_positive_on_null"
    elif flags["false_negative_null"]:
        primary = (
            "false_negative_candidate_available" if hit
            else "false_negative_candidate_missing")
    elif flags["alias_fragmentation"]:
        primary = "alias_fragmentation"
    elif flags["numeric_cluster_failure"]:
        primary = "numeric_cluster_selection"
    elif hit and flags["reachable_improvement"]:
        primary = "candidate_present_selection_failure"
    elif not hit and gold_nonnull:
        primary = "candidate_supply_missing"
    elif flags["cardinality_under"] or flags["cardinality_over"]:
        primary = "cardinality_error"
    elif flags["reachable_improvement"]:
        primary = "reachable_structured_edit"
    else:
        primary = "unresolved_or_unreachable"
    return {
        "SubjectEntity": graph["SubjectEntity"],
        "Relation": relation,
        "prediction_objects": current,
        "gold_objects": gold.get("ObjectEntities", []),
        "current_f1": current_f1,
        "component_action_oracle_f1": oracle_f1,
        "available_gain": oracle_f1 - current_f1,
        "best_component_action": actions[best_index],
        "collapsed_prediction": collapsed,
        "collapsed_f1": collapsed_f1,
        "primary_failure": primary,
        "flags": flags,
        "surface_candidate_count": len(graph.get("candidates", [])),
        "component_count": len(
            graph["relational_graph"]["components"]),
    }


def _summarize(
        rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    relations = sorted({str(row["Relation"]) for row in rows})
    for relation in relations:
        subset = [row for row in rows if row["Relation"] == relation]
        failures: dict[str, int] = {}
        for row in subset:
            failures[row["primary_failure"]] = (
                failures.get(row["primary_failure"], 0) + 1)
        output[relation] = {
            "rows": len(subset),
            "current_mean_f1": statistics.mean(
                float(row["current_f1"]) for row in subset),
            "component_action_oracle_mean_f1": statistics.mean(
                float(row["component_action_oracle_f1"]) for row in subset),
            "mean_available_gain": statistics.mean(
                float(row["available_gain"]) for row in subset),
            "candidate_supply_missing_rows": failures.get(
                "candidate_supply_missing", 0)
                + failures.get("false_negative_candidate_missing", 0),
            "candidate_present_selection_failure_rows": failures.get(
                "candidate_present_selection_failure", 0)
                + failures.get("false_negative_candidate_available", 0)
                + failures.get("numeric_cluster_selection", 0),
            "reachable_improvement_rows": sum(
                bool(row["flags"]["reachable_improvement"]) for row in subset),
            "alias_fragmentation_rows": sum(
                bool(row["flags"]["alias_fragmentation"]) for row in subset),
            "false_positive_null_rows": sum(
                bool(row["flags"]["false_positive_null"]) for row in subset),
            "false_negative_null_rows": sum(
                bool(row["flags"]["false_negative_null"]) for row in subset),
            "primary_failure_counts": failures,
            "surface_candidates": sum(
                int(row["surface_candidate_count"]) for row in subset),
            "components": sum(
                int(row["component_count"]) for row in subset),
        }
    return output


def _overall(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failure_gain: dict[str, dict[str, float | int]] = {}
    for row in rows:
        name = str(row["primary_failure"])
        bucket = failure_gain.setdefault(name, {
            "rows": 0, "total_available_gain": 0.0})
        bucket["rows"] = int(bucket["rows"]) + 1
        bucket["total_available_gain"] = (
            float(bucket["total_available_gain"])
            + float(row["available_gain"]))
    count = len(rows)
    for bucket in failure_gain.values():
        bucket["pooled_gain_contribution"] = (
            float(bucket["total_available_gain"]) / count)
    return {
        "rows": count,
        "current_pooled_macro_f1": statistics.mean(
            float(row["current_f1"]) for row in rows),
        "component_action_oracle_pooled_macro_f1": statistics.mean(
            float(row["component_action_oracle_f1"]) for row in rows),
        "pooled_available_gain": statistics.mean(
            float(row["available_gain"]) for row in rows),
        "gain_attribution_by_primary_failure": failure_gain,
    }


def _prediction_map(
        rows: Sequence[Mapping[str, Any]], label: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ContractError(f"{label}: duplicate row {key}")
        result[key] = row
    return result


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _json(output / "plan/PLAN.json")
    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    ledger_paths: dict[str, str] = {}
    for split in ("train", "validation"):
        graph_path = output / f"graphs/{split}_graph.jsonl"
        graphs = _load_graph(graph_path, expected_split=split)
        graph_by = {_key(graph): graph for graph in graphs}
        gold_path = Path(
            plan["train_gold"] if split == "train"
            else plan["validation_gold"])
        gold = _prediction_map(read_jsonl(gold_path), f"{split} gold")
        if split == "train":
            predictions = {
                key: {
                    "ObjectEntities": list(
                        graph.get("baseline_objects", []))}
                for key, graph in graph_by.items()
                if graph.get("calibration_eligible", True) is not False}
            graph_by = {
                key: graph for key, graph in graph_by.items()
                if graph.get("calibration_eligible", True) is not False}
            prediction_source = "production-matched train OOF baseline_objects"
        else:
            prediction_path = Path(
                args.validation_predictions).resolve()
            predictions = _prediction_map(
                read_jsonl(prediction_path), "validation predictions")
            prediction_source = str(prediction_path)
        if set(predictions) != set(graph_by):
            missing = sorted(set(graph_by) - set(predictions))[:3]
            extra = sorted(set(predictions) - set(graph_by))[:3]
            raise ContractError(
                f"{split} prediction/graph mismatch; missing={missing}, "
                f"extra={extra}")
        ledger = [
            classify_error(
                graph_by[key],
                predictions[key].get("ObjectEntities", []),
                gold[key])
            for key in sorted(graph_by)]
        ledger_path = analysis_dir / f"{split}_error_ledger.jsonl"
        write_jsonl_atomic(ledger_path, ledger)
        ledger_path.with_suffix(
            ledger_path.suffix + ".manifest.json").write_text(json.dumps({
                "schema": "relational-error-ledger-manifest-v1",
                "split": split,
                "rows": len(ledger),
                "contains_labels": True,
                "gold_aware": True,
                "deployable": False,
                "output_sha256": sha256(ledger_path),
                "source_graph": str(graph_path),
                "source_graph_sha256": sha256(graph_path),
                "gold": str(gold_path),
                "gold_sha256": sha256(gold_path),
                "prediction_source": prediction_source,
            }, indent=2, sort_keys=True) + "\n")
        reports[split] = {
            "overall": _overall(ledger),
            "relations": _summarize(ledger),
        }
        ledger_paths[split] = str(ledger_path)
    result = {
        "schema": "relational-candidate-graph-error-audit-v1",
        "gold_aware": True,
        "deployable": False,
        "validation_labels_used_for_model_selection": False,
        "purpose": (
            "posthoc failure attribution and component-action ceiling; "
            "not a decoder or threshold selection"),
        "reports": reports,
        "ledgers": ledger_paths,
    }
    result_path = analysis_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Relational candidate graph and error ledger",
        "",
        "Gold-aware diagnostic only. No decoder or threshold is selected here.",
        "",
    ]
    for split in ("train", "validation"):
        overall = reports[split]["overall"]
        lines += [
            f"## {split}",
            "",
            f"Overall current `{overall['current_pooled_macro_f1']:.6f}`; "
            f"component-action oracle "
            f"`{overall['component_action_oracle_pooled_macro_f1']:.6f}`; "
            f"available pooled gain "
            f"`{overall['pooled_available_gain']:+.6f}`.",
            "",
            "| relation | current | component-action oracle | available gain | "
            "missing supply | present but misselected | reachable rows | "
            "alias rows | FP null | FN null |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for relation, values in reports[split]["relations"].items():
            lines.append(
                f"| {relation} | {values['current_mean_f1']:.6f} | "
                f"{values['component_action_oracle_mean_f1']:.6f} | "
                f"{values['mean_available_gain']:+.6f} | "
                f"{values['candidate_supply_missing_rows']} | "
                f"{values['candidate_present_selection_failure_rows']} | "
                f"{values['reachable_improvement_rows']} | "
                f"{values['alias_fragmentation_rows']} | "
                f"{values['false_positive_null_rows']} | "
                f"{values['false_negative_null_rows']} |")
        lines += [
            "",
            "| primary failure | rows | pooled available-gain contribution |",
            "|---|---:|---:|",
        ]
        for failure, values in sorted(
                overall["gain_attribution_by_primary_failure"].items(),
                key=lambda pair: -float(
                    pair[1]["pooled_gain_contribution"])):
            lines.append(
                f"| {failure} | {values['rows']} | "
                f"{float(values['pooled_gain_contribution']):+.6f} |")
        lines.append("")
    lines += [
        "The component-action oracle is gold-aware and nondeployable. It "
        "measures whether the new relational action space contains a better "
        "bounded edit; it does not claim that the decoder can identify it.",
    ]
    (analysis_dir / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(f"analysis complete: {analysis_dir / 'RESULT.md'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("build", "analyze", "all"), nargs="?",
        default="all")
    parser.add_argument(
        "--source-output-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--validation-predictions",
        default=str(DEFAULT_VALIDATION_PREDICTIONS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"build", "all"}:
        build(args)
    if args.stage in {"analyze", "all"}:
        analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
