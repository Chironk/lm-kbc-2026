from __future__ import annotations

from experiments.heterogeneous_agents.core import canonical_key
from experiments.heterogeneous_agents.components.multi_challenger_graph_decoder import (
    _aggregate,
    _task,
    action_shortlist_score,
    shortlist_actions,
)


def _component(identifier: str, value: str) -> dict[str, object]:
    return {
        "id": identifier,
        "node_type": "candidate_component",
        "representative": value,
        "member_items": [value],
    }


def _action(identifier: str, action_type: str, value: str) -> dict[str, object]:
    return {
        "id": identifier,
        "action_type": action_type,
        "objects": [value],
    }


def test_shortlist_preserves_multiple_challengers() -> None:
    graph = {
        "SubjectEntity": "Example",
        "Relation": "hasArea",
        "nodes": [
            _component("component:0", "10"),
            _component("component:1", "20"),
            _component("component:2", "30"),
        ],
        "actions": [
            _action("action:0", "KEEP", "10"),
            _action("action:1", "REPLACE", "20"),
            _action("action:2", "REPLACE", "30"),
        ],
    }
    truth = {
        ("Example", "hasArea", "component:0"): {
            "qwen_recall": 0.4, "gemma_independent": 0.4},
        ("Example", "hasArea", "component:1"): {
            "qwen_recall": 0.7, "gemma_independent": 0.7},
        ("Example", "hasArea", "component:2"): {
            "qwen_recall": 0.6, "gemma_independent": 0.6},
    }
    keep, challengers = shortlist_actions(graph, truth, limit=2)
    assert keep["id"] == "action:0"
    assert [action["id"] for action in challengers] == [
        "action:1", "action:2"]


def test_unmapped_incumbent_is_neutral_not_deletion_evidence() -> None:
    graph = {"Relation": "hasCapacity"}
    keep = _action("action:0", "KEEP", "100")
    replacement = _action("action:1", "REPLACE", "200")
    score = action_shortlist_score(
        graph, replacement, keep, {
            canonical_key("200", "hasCapacity"): 0.8})
    assert abs(score - 0.3) < 1e-12


def test_dual_argmax_fails_closed_on_disagreement() -> None:
    selected = _aggregate(
        "dual_argmax",
        {"action:1": 1.0, "action:2": 0.5},
        {"action:1": 0.1, "action:2": 1.0},
        "action:0",
    )
    assert selected == "action:0"


def test_task_never_exceeds_balanced_code_capacity() -> None:
    graph = {"SubjectEntity": "Example", "Relation": "hasArea"}
    keep = _action("action:0", "KEEP", "10")
    challengers = [
        _action("action:1", "REPLACE", "20"),
        _action("action:2", "REPLACE", "30"),
    ]
    task = _task(
        graph, keep, challengers, "qwen_recall", 0, 0)
    assert len(task["choices"]) == 4
    assert len(task["choice_variants"]) == 4
    assert task["contains_labels"] is False
