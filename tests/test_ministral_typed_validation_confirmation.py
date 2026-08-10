from experiments.heterogeneous_agents.ministral_consistency_admission import ROUTE
from experiments.heterogeneous_agents.ministral_typed_validation_confirmation import (
    _unanimous_new_area,
    apply_frozen_policy,
)


def _graph(relation="hasArea", support=3, reason="numeric_complete_link_self_consistent_new"):
    return {
        "SubjectEntity": "X",
        "Relation": relation,
        "candidates": [{
            "item": "100",
            "routes": {ROUTE: {
                "admission_reason": reason,
                "support": support,
                "samples": 3,
            }},
        }],
    }


def test_only_single_unanimous_new_area_is_legal():
    assert _unanimous_new_area(_graph()) == ["100"]
    assert _unanimous_new_area(_graph(support=2)) == []
    assert _unanimous_new_area(_graph(relation="hasCapacity")) == []
    assert _unanimous_new_area(_graph(
        reason="numeric_component_corroborates_source")) == []


def test_ambiguous_unanimous_components_fail_closed():
    graph = _graph()
    graph["candidates"].append({
        "item": "200",
        "routes": {ROUTE: {
            "admission_reason": "numeric_complete_link_self_consistent_new",
            "support": 3,
            "samples": 3,
        }},
    })
    assert _unanimous_new_area(graph) == []


def test_frozen_policy_replaces_only_legal_area():
    baseline = [{
        "SubjectEntity": "X",
        "Relation": "hasArea",
        "ObjectEntities": ["90"],
    }]
    predictions, decisions = apply_frozen_policy(baseline, [_graph()])
    assert predictions[0]["ObjectEntities"] == ["100"]
    assert decisions[0]["changed"] is True
