from experiments.heterogeneous_agents.components.comparative_edge_action_selector import (
    EDGE_FEATURE_NAMES,
    _identity_action_indices,
    edge_action_features,
)


def test_keep_has_zero_comparative_edge_features():
    graph = {
        "Relation": "hasArea",
        "nodes": [{
            "node_type": "candidate_component",
            "id": "component:0",
            "member_items": ["10"],
            "representative": "10",
        }],
    }
    action = {"objects": ["10"]}
    registry = {"nodes": [{
        "node_id": "component:0", "is_incumbent": True,
    }]}
    values = edge_action_features(graph, action, registry, {})
    assert values == [0.0] * len(EDGE_FEATURE_NAMES)


def test_challenger_edges_are_context_anchored():
    graph = {
        "Relation": "hasArea",
        "nodes": [
            {
                "node_type": "candidate_component",
                "id": "component:0",
                "member_items": ["10"],
                "representative": "10",
            },
            {
                "node_type": "candidate_component",
                "id": "component:1",
                "member_items": ["20"],
                "representative": "20",
            },
        ],
    }
    action = {"objects": ["20"]}
    registry = {"nodes": [
        {"node_id": "component:0", "is_incumbent": True},
        {"node_id": "component:1", "is_incumbent": False},
    ]}
    edges = {
        (prompt, agent): {"component:1": 2.0}
        for prompt in ("recognition", "skeptical", "submission")
        for agent in ("qwen_recall", "gemma_independent")
    }
    values = edge_action_features(graph, action, registry, edges)
    assert len(values) == len(EDGE_FEATURE_NAMES)
    assert values[-3] == 1.0  # every directed edge supports the challenger
    assert values[-1] == 1.0  # complete evidence coverage


def test_rejected_challenger_is_not_forwarded_to_gate():
    graph = {
        "Relation": "hasArea",
        "actions": [
            {"action_type": "KEEP", "objects": ["10"]},
            {"action_type": "REPLACE", "objects": ["20"]},
        ],
        "nodes": [
            {
                "node_type": "candidate_component",
                "id": "component:0",
                "member_items": ["10"],
                "representative": "10",
            },
            {
                "node_type": "candidate_component",
                "id": "component:1",
                "member_items": ["20"],
                "representative": "20",
            },
        ],
    }
    registry = {"nodes": [
        {"node_id": "component:0", "is_incumbent": True},
        {"node_id": "component:1", "is_incumbent": False},
    ]}
    edges = {
        (prompt, agent): {"component:1": -1.0}
        for prompt in ("recognition", "skeptical", "submission")
        for agent in ("qwen_recall", "gemma_independent")
    }
    assert _identity_action_indices(graph, registry, edges) == (0,)
