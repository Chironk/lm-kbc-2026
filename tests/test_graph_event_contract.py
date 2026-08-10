from __future__ import annotations

import pytest

from experiments.heterogeneous_agents.core import ContractError
from experiments.heterogeneous_agents.graph_event_contract import (
    assert_event_support_invariants,
    repair_unsupported_candidate_set_events,
)


def _graph(status: str, *, support: bool) -> dict:
    edges = []
    if support:
        edges.append({
            "source": "event:0", "target": "component:0",
            "edge_type": "supports",
        })
    edges.extend([{
        "source": "event:0", "target": "cardinality:ZERO",
        "edge_type": "asserts_cardinality",
    }, {
        "source": "event:0", "target": "existence:EMPTY",
        "edge_type": "asserts_existence",
    }])
    return {
        "SubjectEntity": "X", "Relation": "hasCapacity",
        "relational_graph": {
            "nodes": [{
                "id": "component:0", "node_type": "candidate_component",
                "representative": "10000",
            }, {
                "id": "event:0", "node_type": "evidence_event",
                "status": status,
            }],
            "edges": edges,
        },
    }


def test_candidate_set_without_support_is_repaired_to_unparsed():
    graph = _graph("candidate_set", support=False)
    with pytest.raises(ContractError, match="candidate_set event has no support"):
        assert_event_support_invariants(graph)
    repaired, audit = repair_unsupported_candidate_set_events(graph)
    event = next(node for node in repaired["relational_graph"]["nodes"]
                 if node["node_type"] == "evidence_event")
    assert event["status"] == "unparsed_or_no_candidate"
    assert audit["repaired_events"] == 1
    assert audit["removed_assertion_edges"] == 2
    assert repaired["relational_graph"]["edges"] == []
    assert_event_support_invariants(repaired)


def test_valid_candidate_set_is_unchanged():
    graph = _graph("candidate_set", support=True)
    repaired, audit = repair_unsupported_candidate_set_events(graph)
    assert audit["repaired_events"] == 0
    assert any(edge["edge_type"] == "supports"
               for edge in repaired["relational_graph"]["edges"])


def test_explicit_none_with_support_is_rejected():
    with pytest.raises(ContractError, match="explicit_none event has support"):
        repair_unsupported_candidate_set_events(
            _graph("explicit_none", support=True))
