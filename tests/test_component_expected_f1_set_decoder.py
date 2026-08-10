from __future__ import annotations

import math

from experiments.heterogeneous_agents.component_expected_f1_set_decoder import (
    _component_label,
    expected_f1_utility,
    nested_set_actions,
    positive_pooled_utility,
)


def _component(identifier: str, representative: str) -> dict[str, object]:
    return {
        "id": identifier,
        "node_type": "candidate_component",
        "representative": representative,
        "member_items": [representative],
    }


def test_component_label_checks_non_representative_members() -> None:
    graph = {
        "SubjectEntity": "Example",
        "Relation": "companyTradesAtStockExchange",
    }
    component = {
        "representative": "NYSE",
        "member_items": ["NYSE", "New York Stock Exchange"],
    }
    gold = {
        ("Example", "companyTradesAtStockExchange"): {
            "SubjectEntity": "Example",
            "Relation": "companyTradesAtStockExchange",
            "ObjectEntities": [["New York Stock Exchange"]],
        }
    }
    assert _component_label(graph, component, gold) == 1.0


def test_expected_f1_scores_complete_set_on_one_scale() -> None:
    probabilities = {"alpha": 0.8, "beta": 0.6}
    utility = expected_f1_utility(
        ["Alpha", "Beta"],
        probabilities,
        "awardWonBy",
        expected_cardinality=2.0,
        zero_probability=0.0,
    )
    assert math.isclose(utility, 0.7)


def test_empty_action_uses_null_probability_only_when_nullable() -> None:
    assert math.isclose(expected_f1_utility(
        [], {}, "companyTradesAtStockExchange",
        expected_cardinality=0.2, zero_probability=0.75,
    ), 0.75)
    assert expected_f1_utility(
        [], {}, "awardWonBy",
        expected_cardinality=0.2, zero_probability=0.75,
    ) == 0.0


def test_nested_actions_include_prune_then_expand_sets() -> None:
    graph = {
        "Relation": "awardWonBy",
        "incumbent_objects": ["Old A", "Old B"],
    }
    components = [
        _component("old-a", "Old A"),
        _component("old-b", "Old B"),
        _component("new-c", "New C"),
    ]
    actions = nested_set_actions(graph, components, [0.1, 0.2, 0.9])
    rendered = {
        frozenset(map(str, action["objects"])): str(action["family"])
        for action in actions
    }
    assert frozenset(("Old B", "New C")) in rendered
    assert rendered[frozenset(("Old A", "Old B"))] == "KEEP"


def test_nested_actions_do_not_duplicate_equivalent_component() -> None:
    graph = {
        "Relation": "companyTradesAtStockExchange",
        "incumbent_objects": ["New York Stock Exchange"],
    }
    components = [{
        **_component("nyse", "NYSE"),
        "member_items": ["NYSE", "New York Stock Exchange"],
    }]
    actions = nested_set_actions(graph, components, [0.9])
    assert all(len(action["objects"]) <= 1 for action in actions)
    keep = next(action for action in actions if action["family"] == "KEEP")
    assert keep["objects"] == ["New York Stock Exchange"]


def test_relation_gate_uses_pooled_utility_not_fold_majority() -> None:
    # Two positive and three negative folds can still improve the pooled
    # row-level target when the positive folds contain more utility.
    assert positive_pooled_utility([1.0, 1.0, -0.2, -0.2, -0.2])
    assert not positive_pooled_utility([0.1, 0.1, -0.2, -0.2, -0.2])
