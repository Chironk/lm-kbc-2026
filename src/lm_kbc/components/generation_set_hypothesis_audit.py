#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence
from evaluate import try_parse_number
from lm_kbc.core import ContractError, canonical_key
from lm_kbc.components.cot40_evidence_edge_ablation import FAMILIES
from lm_kbc.components.cot40_graph_native_decoder import NUMERIC_RELATIONS, _component_ids, _objects_for_ids
from lm_kbc.components.heterogeneous_memory_selector import _key

def set_f1(left: Iterable[str], right: Iterable[str]) -> float:
    """Symmetric set F1 used only between graph component identities."""
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))

def _incumbent_tokens(
    graph: Mapping[str, Any], objects: Sequence[str],
) -> frozenset[str]:
    """Map KEEP to components, with a guarded numeric-tolerance fallback."""
    relation = str(graph["Relation"])
    tokens: list[str] = []
    for index, item in enumerate(objects):
        exact = _component_ids(graph, [str(item)])
        token: str | None = None
        if len(exact) == 1:
            token = next(iter(exact))
        elif relation in NUMERIC_RELATIONS:
            target = try_parse_number(str(item))
            candidates: list[tuple[float, str]] = []
            if target is not None and target > 0:
                for component in graph["relational_graph"]["components"]:
                    numbers = [
                        try_parse_number(str(value))
                        for value in (
                            *component.get("member_items", []),
                            component.get("representative", ""),
                        )
                    ]
                    distances = [
                        abs(value - target) / abs(target)
                        for value in numbers
                        if value is not None and value > 0
                    ]
                    if distances and min(distances) <= 0.05 + 1e-12:
                        candidates.append((
                            min(distances), str(component["id"])))
            candidates.sort()
            if candidates and (
                len(candidates) == 1
                or candidates[0][0] < candidates[1][0] - 1e-12
            ):
                token = candidates[0][1]
        if token is None:
            token = (
                f"incumbent:{index}:"
                f"{canonical_key(str(item), relation)}"
            )
        if token not in tokens:
            tokens.append(token)
    return frozenset(tokens)

def build_hypotheses(
    graph: Mapping[str, Any], incumbent: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, list[frozenset[str]]], dict[str, int]]:
    """Traverse exact event->component edges into unique complete sets."""
    relational = graph["relational_graph"]
    events = {
        str(node["id"]): node
        for node in relational["nodes"]
        if node.get("node_type") == "evidence_event"
    }
    support: dict[str, set[str]] = {event: set() for event in events}
    for edge in relational["edges"]:
        if edge.get("edge_type") != "supports":
            continue
        source, target = str(edge["source"]), str(edge["target"])
        if source not in support:
            raise ContractError(f"{_key(graph)}: orphan supports edge")
        support[source].add(target)

    incumbent_tokens = _incumbent_tokens(graph, incumbent)
    by_tokens: dict[frozenset[str], dict[str, Any]] = {
        incumbent_tokens: {
            "tokens": incumbent_tokens,
            "objects": list(incumbent),
            "is_incumbent": True,
            "proposer_families": set(),
            "event_ids": [],
        }
    }
    family_events: dict[str, list[frozenset[str]]] = {
        family: [] for family in FAMILIES
    }
    status_counts: Counter[str] = Counter()
    for event_id, node in sorted(events.items()):
        status = str(node.get("status"))
        status_counts[status] += 1
        if status not in ("candidate_set", "explicit_none"):
            continue
        family = str(node.get("model_family"))
        if family not in family_events:
            raise ContractError(f"{_key(graph)}: unknown family {family}")
        tokens = frozenset(support[event_id])
        if status == "candidate_set" and not tokens:
            raise ContractError(f"{_key(graph)}: empty candidate-set event")
        if status == "explicit_none" and tokens:
            raise ContractError(f"{_key(graph)}: supported explicit-none event")
        family_events[family].append(tokens)
        hypothesis = by_tokens.setdefault(tokens, {
            "tokens": tokens,
            "objects": _objects_for_ids(graph, tokens),
            "is_incumbent": False,
            "proposer_families": set(),
            "event_ids": [],
        })
        hypothesis["proposer_families"].add(family)
        hypothesis["event_ids"].append(event_id)

    if any(not values for values in family_events.values()):
        raise ContractError(f"{_key(graph)}: family has no parseable event")
    hypotheses = sorted(by_tokens.values(), key=lambda value: (
        not bool(value["is_incumbent"]),
        tuple(sorted(value["tokens"])),
    ))
    return hypotheses, family_events, dict(sorted(status_counts.items()))

def hypothesis_stats(
    hypothesis: Mapping[str, Any],
    family_events: Mapping[str, Sequence[frozenset[str]]],
) -> dict[str, Any]:
    tokens = frozenset(map(str, hypothesis["tokens"]))
    similarities = {
        family: max(
            (set_f1(tokens, event) for event in family_events[family]),
            default=0.0,
        )
        for family in FAMILIES
    }
    exact_rates = {
        family: (
            sum(event == tokens for event in family_events[family])
            / len(family_events[family])
        )
        for family in FAMILIES
    }
    proposers = set(map(str, hypothesis["proposer_families"]))
    reviewers = [family for family in FAMILIES if family not in proposers]
    independent = (
        1.0 if len(proposers) == len(FAMILIES)
        else statistics.mean(similarities[family] for family in reviewers)
        if reviewers else 0.5
    )
    return {
        "mean_similarity": statistics.mean(similarities.values()),
        "minimum_similarity": min(similarities.values()),
        "exact_family_fraction": len(proposers) / len(FAMILIES),
        "within_family_exact_rate_mean": statistics.mean(exact_rates.values()),
        "independent_similarity": independent,
        "family_similarities": similarities,
        "family_exact_rates": exact_rates,
    }
