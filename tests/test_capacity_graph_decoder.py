from __future__ import annotations

from experiments.heterogeneous_agents.capacity_graph_decoder import (
    _complete_link_groups,
    _event_support,
    _merge,
    capacity_options,
    option_features,
)


def _graph(values, events):
    components = [{
        "id": f"component:{index}",
        "node_type": "candidate_component",
        "representative": str(value),
    } for index, value in enumerate(values)]
    nodes = list(components)
    edges = []
    for index, (family, component) in enumerate(events):
        event_id = f"event:{index}"
        nodes.append({
            "id": event_id,
            "node_type": "evidence_event",
            "model_family": family,
            "status": "candidate_set",
        })
        edges.append({
            "source": event_id,
            "target": f"component:{component}",
            "edge_type": "supports",
        })
    return {
        "SubjectEntity": "Example Stadium",
        "Relation": "hasCapacity",
        "contains_labels": False,
        "gold_aware": False,
        "relational_graph": {
            "components": components,
            "nodes": nodes,
            "edges": edges,
        },
    }


def test_complete_link_does_not_chain_numeric_equivalence():
    values = {"a": 100.0, "b": 104.0, "c": 108.0}
    groups = _complete_link_groups(values)
    assert groups == [["a", "b"], ["c"]]


def test_three_family_support_is_exposed_to_decoder():
    graph = _graph(
        [10_000, 20_000],
        [
            ("qwen_recall", 0),
            ("qwen_recall", 1),
            ("gemma_independent", 0),
            ("ministral_independent", 0),
            ("ministral_independent", 0),
        ],
    )
    options = capacity_options(graph, ["10000"])
    incumbent = next(option for option in options if option["is_incumbent"])
    assert incumbent["family_fraction"] == 1.0
    assert incumbent["rates"] == {
        "qwen_recall": 0.5,
        "gemma_independent": 1.0,
        "ministral_independent": 1.0,
    }
    assert len(option_features(incumbent, incumbent)) == 16


def test_incumbent_absent_from_graph_is_preserved_as_option():
    graph = _graph(
        [10_000, 20_000],
        [
            ("qwen_recall", 0),
            ("gemma_independent", 0),
            ("ministral_independent", 1),
        ],
    )
    options = capacity_options(graph, ["30000"])
    incumbent = [option for option in options if option["is_incumbent"]]
    assert len(incumbent) == 1
    assert incumbent[0]["value"] == 30_000.0
    assert incumbent[0]["family_fraction"] == 0.0


def test_incumbent_is_assigned_to_one_of_overlapping_tolerance_groups():
    graph = _graph(
        [9_600, 10_400],
        [
            ("qwen_recall", 0),
            ("gemma_independent", 1),
            ("ministral_independent", 1),
        ],
    )
    options = capacity_options(graph, ["10000"])
    assert sum(option["is_incumbent"] for option in options) == 1


def test_empty_replacement_map_is_identity_fallback():
    rows = [{
        "SubjectEntity": "Example Stadium",
        "Relation": "hasCapacity",
        "ObjectEntities": ["10000"],
    }]
    assert _merge(rows, {}) == rows


def test_abstentions_remain_in_family_support_denominator():
    graph = _graph(
        [10_000],
        [
            ("qwen_recall", 0),
            ("gemma_independent", 0),
            ("ministral_independent", 0),
        ],
    )
    graph["relational_graph"]["nodes"].append({
        "id": "event:qwen:none",
        "node_type": "evidence_event",
        "model_family": "qwen_recall",
        "status": "explicit_none",
    })
    events, denominators = _event_support(
        graph, {"component:0": 10_000.0})
    assert len(events) == 3
    assert denominators["qwen_recall"] == 2
    assert denominators["gemma_independent"] == 1
