#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence
from sample_evidence import EXPLICIT_ABSTENTION, classify_samples
from evaluate import normalize_string
from lm_kbc.core import ContractError, canonical_key, proposal_parse_status
from lm_kbc.components.graph_event_contract import repair_unsupported_candidate_set_events
from lm_kbc.components.heterogeneous_memory_selector import _key

FAMILIES = (
    "qwen_recall",
    "gemma_independent",
    "ministral_independent",
)

RISK_CLAIMS = (
    "area_non_km_unit",
    "award_partial_scope",
    "border_maritime_only",
    "capacity_variant",
    "city_noncity_scope",
    "listing_historical",
    "listing_inactive",
)

def _reasoning(text: str) -> str:
    value = str(text)
    match = re.search(r"<think>(.*?)(?:</think>|$)", value, re.I | re.S)
    if match:
        return match.group(1).strip()
    before = re.split(r"\bANSWER\s*:", value, maxsplit=1, flags=re.I)[0]
    before = re.sub(r"^\s*REASONING\s*:\s*", "", before, flags=re.I)
    return before.strip()

def reasoning_claims(text: str, relation: str) -> list[str]:
    """Extract only explicit, relation-scoped claims from model reasoning.

    These are lexical observations, not truth labels or confidence scores.
    Risk/positive interpretation is applied only by the train-side selector.
    """
    value = re.sub(r"\s+", " ", _reasoning(text).casefold())
    claims: set[str] = set()
    if relation == "hasArea":
        has_km = bool(re.search(
            r"\b(square\s+kilomet(?:er|re)s?|sq\.?\s*km|km\s*[²2])\b",
            value,
        ))
        has_other = bool(re.search(
            r"\b(square\s+miles?|sq\.?\s*mi|acres?|hectares?)\b", value))
        if has_km:
            claims.add("area_km2_unit")
        if has_other and not has_km and "convert" not in value:
            claims.add("area_non_km_unit")
    elif relation == "hasCapacity":
        if re.search(
            r"\b(seated|seating|standing|concert|temporary|configuration)\b",
            value,
        ):
            claims.add("capacity_variant")
        if re.search(
            r"\b(total|official|spectator)\s+(?:seating\s+)?capacity\b",
            value,
        ):
            claims.add("capacity_total")
    elif relation == "companyTradesAtStockExchange":
        if re.search(
            r"\b(formerly|previously|historically|used to|at one time)\b",
            value,
        ):
            claims.add("listing_historical")
        if re.search(
            r"\b(delisted|no longer (?:listed|traded)|taken private|"
            r"privately held|not publicly traded|acquired)\b",
            value,
        ):
            claims.add("listing_inactive")
        if (
            re.search(
                r"\b(currently|shares (?:are )?traded|shares trade|"
                r"listed on|publicly traded)\b",
                value,
            )
            and not ({"listing_historical", "listing_inactive"} & claims)
        ):
            claims.add("listing_current")
    elif relation == "awardWonBy":
        if (
            re.search(r"\b(?:19|20)\d{2}\b", value)
            or re.search(
                r"\b(recent|latest|for example|focus on|specific year)\b",
                value,
            )
        ):
            claims.add("award_partial_scope")
    elif relation == "countryLandBordersCountry":
        has_land = bool(re.search(r"\bland border", value))
        has_maritime = bool(re.search(r"\bmaritime (?:border|boundary)", value))
        if has_land:
            claims.add("border_land")
        if has_maritime and not has_land:
            claims.add("border_maritime_only")
    elif relation == "personHasCityOfDeath":
        if re.search(r"\b(city of|died in|death (?:in|at))\b", value):
            claims.add("city_explicit")
        if (
            re.search(r"\b(province|state|region|country)\b", value)
            and not re.search(r"\b(city|town|municipality)\b", value)
        ):
            claims.add("city_noncity_scope")
    return sorted(claims)

def _component_lookup(
    graph: Mapping[str, Any],
) -> tuple[dict[str, str], set[str]]:
    relation = str(graph["Relation"])
    lookup: dict[str, str] = {}
    invalid = set()
    for component in graph["relational_graph"]["components"]:
        component_id = str(component["id"])
        keys = {
            str(value) for value in component.get("member_keys", [])
        } | {
            canonical_key(str(value), relation)
            for value in component.get("member_items", [])
        } | {
            canonical_key(str(component.get("representative", "")), relation)
        }
        if relation in {"hasArea", "hasCapacity"}:
            keys |= {
                value.removeprefix("numeric:")
                for value in list(keys)
                if value.startswith("numeric:")
            }
        for key in keys:
            if not key:
                continue
            if key in lookup and lookup[key] != component_id:
                invalid.add(key)
            lookup[key] = component_id
    if invalid:
        raise ContractError(
            f"{_key(graph)}: ambiguous component keys {sorted(invalid)}")
    return lookup, invalid

def _map_items(
    graph: Mapping[str, Any],
    items: Iterable[str],
) -> tuple[list[str], list[str]]:
    relation = str(graph["Relation"])
    lookup, _ = _component_lookup(graph)
    components: list[str] = []
    unknown: list[str] = []
    seen = set()
    for item in items:
        key = canonical_key(str(item), relation)
        # Historical numeric candidate tables retained a small number of
        # invalid/zero surfaces under their literal legacy key.  They remain
        # legal components in the frozen action inventory, so exact event
        # recovery must preserve their observed membership even though the
        # current positive-number canonicalizer intentionally rejects them.
        candidates = [key, str(item).strip(), normalize_string(str(item))]
        if key.startswith("numeric:"):
            candidates.append(key.removeprefix("numeric:"))
        component = next(
            (lookup[value] for value in candidates if value in lookup), None)
        if component is None:
            if key:
                unknown.append(str(item))
            continue
        if component not in seen:
            components.append(component)
            seen.add(component)
    return components, unknown

def _generic_record(
    graph: Mapping[str, Any], text: str,
) -> tuple[str, list[str], list[str]]:
    status, items = proposal_parse_status(str(text), str(graph["Relation"]))
    components, unknown = _map_items(graph, items)
    if unknown:
        raise ContractError(
            f"{_key(graph)}: generation has unmapped candidates {unknown}")
    mapped_status = (
        "candidate_set" if components else
        "explicit_none" if status == "explicit_none" else
        "unparsed_or_no_candidate"
    )
    return mapped_status, components, reasoning_claims(
        str(text), str(graph["Relation"]))

def _qwen_records(
    graph: Mapping[str, Any], samples: Sequence[str],
) -> list[tuple[str, list[str], list[str]]]:
    relation = str(graph["Relation"])
    evidence = classify_samples(samples, relation, "legacy-cot")
    records = []
    for raw, parsed in zip(samples, evidence):
        components, unknown = _map_items(graph, parsed.items)
        if unknown:
            raise ContractError(
                f"{_key(graph)}: Qwen generation has unmapped {unknown}")
        status = (
            "candidate_set" if components else
            "explicit_none"
            if parsed.kind == EXPLICIT_ABSTENTION else
            "unparsed_or_no_candidate"
        )
        records.append((
            status,
            components,
            reasoning_claims(raw, relation),
        ))
    return records

def _event_node(
    *, route: str, family: str, generation: int, samples: int,
    status: str, claims: Sequence[str], provenance: str, text: str,
) -> dict[str, Any]:
    return {
        "id": f"evidence:{route}:generation:{generation}",
        "node_type": "evidence_event",
        "evidence_kind": "exact_generation",
        "route": route,
        "model_family": family,
        "generation_index": generation,
        "samples": samples,
        "status": status,
        "claims": list(claims),
        "provenance_mode": provenance,
        "reasoning_sha256": hashlib.sha256(
            _reasoning(text).encode()).hexdigest(),
    }

def _replace_route_events(
    graph: dict[str, Any],
    *, route: str, family: str,
    records: Sequence[tuple[str, list[str], list[str]]],
    raw_texts: Sequence[str],
    provenance: str,
) -> dict[str, Any]:
    if len(records) != len(raw_texts):
        raise ContractError("event record/raw count mismatch")
    relational = graph["relational_graph"]
    old_nodes = list(relational["nodes"])
    removed_ids = {
        str(node["id"])
        for node in old_nodes
        if (
            node.get("node_type") == "evidence_event"
            and node.get("route") == route
        )
    }
    relational["nodes"] = [
        node for node in old_nodes if str(node.get("id")) not in removed_ids]
    relational["edges"] = [
        edge for edge in relational["edges"]
        if (
            str(edge.get("source")) not in removed_ids
            and str(edge.get("target")) not in removed_ids
        )
    ]
    samples = len(records)
    recovered: Counter[str] = Counter()
    for generation, ((status, components, claims), raw) in enumerate(
        zip(records, raw_texts)
    ):
        node = _event_node(
            route=route,
            family=family,
            generation=generation,
            samples=samples,
            status=status,
            claims=claims,
            provenance=provenance,
            text=raw,
        )
        relational["nodes"].append(node)
        for component in components:
            recovered[component] += 1
            metadata = next(
                value["routes"][route]
                for value in relational["components"]
                if (
                    str(value["id"]) == component
                    and route in value.get("routes", {})
                )
            )
            relational["edges"].append({
                "source": node["id"],
                "target": component,
                "edge_type": "supports",
                "evidence_kind": "exact_generation",
                "route": route,
                "model_family": family,
                "generation_index": generation,
                "weight": 1.0 / samples,
                "selected": bool(metadata.get("selected", False)),
            })

    # Exact component-union support may be greater than the historical maximum
    # surface support, but can never be less.  This catches wrong raw lineage.
    for component in relational["components"]:
        metadata = component.get("routes", {}).get(route)
        if metadata is None:
            continue
        lower_bound = int(math.ceil(
            float(metadata.get(
                "max_support_rate",
                metadata.get("component_support_rate", 0.0),
            )) * samples - 1e-9
        ))
        actual = recovered[str(component["id"])]
        if actual < lower_bound or actual > samples:
            raise ContractError(
                f"{_key(graph)}: {route} exact support violates "
                f"historical lower bound for {component['id']}: "
                f"{actual} not in [{lower_bound}, {samples}]")
    return {
        "events_replaced": len(removed_ids),
        "exact_events": samples,
        "support_edges": sum(recovered.values()),
        "support_union_increase": sum(
            max(
                0,
                recovered[str(component["id"])]
                - int(round(float(
                    component.get("routes", {}).get(route, {}).get(
                        "component_support_rate", 0.0)
                ) * samples)),
            )
            for component in relational["components"]
        ),
    }

def _state_and_relation_edges(graph: dict[str, Any]) -> dict[str, int]:
    # A candidate-set status without a mapped support edge is a parse/mapping
    # failure, not evidence for an empty answer.  Repair it before deriving
    # cardinality and existence assertions.
    _, repair = repair_unsupported_candidate_set_events(graph, in_place=True)
    relational = graph["relational_graph"]
    events = {
        str(node["id"]): node
        for node in relational["nodes"]
        if node.get("node_type") == "evidence_event"
    }
    supports: dict[str, set[str]] = {event: set() for event in events}
    for edge in relational["edges"]:
        if edge.get("edge_type") == "supports":
            supports[str(edge["source"])].add(str(edge["target"]))

    state_nodes = [
        {
            "id": f"cardinality:{value}",
            "node_type": "cardinality_state",
            "value": value,
        }
        for value in ("ZERO", "ONE", "MANY")
    ] + [
        {
            "id": f"existence:{value}",
            "node_type": "existence_state",
            "value": value,
        }
        for value in ("EMPTY", "NONEMPTY")
    ]
    claims = sorted({
        str(claim)
        for event in events.values()
        for claim in event.get("claims", [])
    })
    claim_nodes = [{
        "id": f"claim:{claim}",
        "node_type": "claim_state",
        "claim": claim,
        "risk": claim in RISK_CLAIMS,
    } for claim in claims]
    relational["nodes"].extend([*state_nodes, *claim_nodes])

    extra_edges: list[dict[str, Any]] = []
    pair_events: dict[tuple[str, str], list[str]] = defaultdict(list)
    parseable_events = 0
    for event_id, event in sorted(events.items()):
        status = str(event.get("status"))
        members = sorted(supports[event_id])
        if status not in ("candidate_set", "explicit_none"):
            continue
        parseable_events += 1
        cardinality = (
            "ZERO" if not members else "ONE" if len(members) == 1 else "MANY")
        existence = "EMPTY" if not members else "NONEMPTY"
        extra_edges.extend([
            {
                "source": event_id,
                "target": f"cardinality:{cardinality}",
                "edge_type": "asserts_cardinality",
            },
            {
                "source": event_id,
                "target": f"existence:{existence}",
                "edge_type": "asserts_existence",
            },
        ])
        for claim in event.get("claims", []):
            extra_edges.append({
                "source": event_id,
                "target": f"claim:{claim}",
                "edge_type": "asserts_claim",
                "risk": claim in RISK_CLAIMS,
            })
        for left, right in itertools.combinations(members, 2):
            pair_events[(left, right)].append(event_id)
    for (left, right), event_ids in sorted(pair_events.items()):
        family_counts = Counter(
            str(events[event]["model_family"]) for event in event_ids)
        route_counts = Counter(str(events[event]["route"]) for event in event_ids)
        extra_edges.append({
            "source": left,
            "target": right,
            "edge_type": "co_occurs_with",
            "directed": False,
            "count": len(event_ids),
            "rate": len(event_ids) / max(1, parseable_events),
            "event_ids": event_ids,
            "family_counts": dict(sorted(family_counts.items())),
            "route_counts": dict(sorted(route_counts.items())),
        })
    relational["edges"].extend(extra_edges)
    return {
        "unsupported_candidate_events_repaired": repair["repaired_events"],
        "parseable_events": parseable_events,
        "co_occurrence_edges": sum(
            edge["edge_type"] == "co_occurs_with"
            for edge in extra_edges
        ),
        "cardinality_edges": sum(
            edge["edge_type"] == "asserts_cardinality"
            for edge in extra_edges
        ),
        "existence_edges": sum(
            edge["edge_type"] == "asserts_existence"
            for edge in extra_edges
        ),
        "claim_nodes": len(claim_nodes),
        "claim_edges": sum(
            edge["edge_type"] == "asserts_claim"
            for edge in extra_edges
        ),
    }
