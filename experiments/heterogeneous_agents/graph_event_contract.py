"""Inference-safe contracts for evidence-event support edges.

An exact-generation event marked ``candidate_set`` must support at least one
candidate component.  Historical graphs contained a small number of events
whose parser reported a candidate but whose surface could not be mapped into
the graph.  Treating those events as empty candidate sets fabricates ZERO
cardinality evidence; treating them as valid candidate sets makes downstream
graph traversal fail.  The only label-free repair is to preserve the event but
mark it as unparsed/no-candidate evidence.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from experiments.heterogeneous_agents.core import ContractError


PARSEABLE_EVENT_STATUSES = frozenset({"candidate_set", "explicit_none"})
UNMAPPED_STATUS = "unparsed_or_no_candidate"


def _key(graph: Mapping[str, Any]) -> tuple[str, str]:
    return str(graph["SubjectEntity"]), str(graph["Relation"])


def _event_support_index(
    graph: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, set[str]], set[str]]:
    relational = graph.get("relational_graph")
    if not isinstance(relational, Mapping):
        raise ContractError(f"{_key(graph)}: missing relational graph")
    nodes = relational.get("nodes", [])
    edges = relational.get("edges", [])
    components = {
        str(node["id"])
        for node in nodes
        if node.get("node_type") == "candidate_component"
    }
    events = {
        str(node["id"]): node
        for node in nodes
        if node.get("node_type") == "evidence_event"
    }
    support = {event_id: set() for event_id in events}
    for edge in edges:
        if edge.get("edge_type") != "supports":
            continue
        source, target = str(edge["source"]), str(edge["target"])
        if source not in events:
            raise ContractError(f"{_key(graph)}: orphan support source {source}")
        if target not in components:
            raise ContractError(f"{_key(graph)}: unknown support target {target}")
        support[source].add(target)
    return events, support, components


def assert_event_support_invariants(graph: Mapping[str, Any]) -> None:
    """Reject contradictory event status/support combinations."""
    events, support, _ = _event_support_index(graph)
    for event_id, event in events.items():
        status = str(event.get("status"))
        members = support[event_id]
        if status == "candidate_set" and not members:
            raise ContractError(
                f"{_key(graph)}: candidate_set event has no support: {event_id}")
        if status == "explicit_none" and members:
            raise ContractError(
                f"{_key(graph)}: explicit_none event has support: {event_id}")


def repair_unsupported_candidate_set_events(
    graph: Mapping[str, Any], *, in_place: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Repair unmapped candidate events without inventing a component edge.

    A missing edge cannot safely be reconstructed from the graph alone because
    the original surface is not available.  Such events are therefore marked
    unparsed.  Typed assertions emitted from the invalid status are removed;
    all provenance and hashes on the event itself are retained.
    """
    value = graph if in_place else copy.deepcopy(graph)
    if not isinstance(value, dict):
        value = dict(value)
    events, support, _ = _event_support_index(value)
    repaired_ids = {
        event_id
        for event_id, event in events.items()
        if str(event.get("status")) == "candidate_set" and not support[event_id]
    }
    for event_id in sorted(repaired_ids):
        event = events[event_id]
        event["status"] = UNMAPPED_STATUS
        event["status_repair"] = "candidate_set_without_support"

    removed_assertions = 0
    if repaired_ids:
        retained = []
        for edge in value["relational_graph"].get("edges", []):
            remove = (
                str(edge.get("source")) in repaired_ids
                and str(edge.get("edge_type", "")).startswith("asserts_")
            )
            removed_assertions += int(remove)
            if not remove:
                retained.append(edge)
        value["relational_graph"]["edges"] = retained

    assert_event_support_invariants(value)
    return value, {
        "repaired_events": len(repaired_ids),
        "repaired_event_ids": sorted(repaired_ids),
        "removed_assertion_edges": removed_assertions,
        "repair_policy": "mark_unmapped_candidate_event_unparsed",
    }
