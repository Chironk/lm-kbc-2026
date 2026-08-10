from __future__ import annotations

import math

from experiments.heterogeneous_agents.components.pairwise_preference_graph_selector import (
    _cascade_outputs,
    solve_preference_scores,
)


def test_preference_solver_recovers_connected_log_odds() -> None:
    scores = solve_preference_scores(
        ("incumbent", "a", "b"),
        (
            ("a", "incumbent", 2.0),
            ("b", "incumbent", -1.0),
            ("a", "b", 3.0),
        ),
        ("incumbent",),
        ridge=1e-9,
    )
    assert math.isclose(scores["incumbent"], 0.0, abs_tol=1e-8)
    assert math.isclose(scores["a"], 2.0, abs_tol=1e-7)
    assert math.isclose(scores["b"], -1.0, abs_tol=1e-7)


def test_global_edge_propagates_through_incumbent_star() -> None:
    without_global = solve_preference_scores(
        ("incumbent", "a", "b"),
        (
            ("a", "incumbent", 0.2),
            ("b", "incumbent", 0.2),
        ),
        ("incumbent",),
        ridge=0.05,
    )
    with_global = solve_preference_scores(
        ("incumbent", "a", "b"),
        (
            ("a", "incumbent", 0.2),
            ("b", "incumbent", 0.2),
            ("a", "b", 2.0),
        ),
        ("incumbent",),
        ridge=0.05,
    )
    assert math.isclose(
        without_global["a"], without_global["b"], abs_tol=1e-12)
    assert with_global["a"] > with_global["b"]


def test_cascade_prefers_precision_stage_then_recall_stage() -> None:
    old_predictions = [
        {
            "SubjectEntity": "one",
            "Relation": "hasArea",
            "ObjectEntities": ["old"],
        },
        {
            "SubjectEntity": "two",
            "Relation": "hasArea",
            "ObjectEntities": ["keep"],
        },
    ]
    new_predictions = [
        {
            "SubjectEntity": "one",
            "Relation": "hasArea",
            "ObjectEntities": ["new"],
        },
        {
            "SubjectEntity": "two",
            "Relation": "hasArea",
            "ObjectEntities": ["new"],
        },
    ]
    old_diagnostics = [
        {"changed": True, "utility_delta": 1.0},
        {"changed": False, "utility_delta": 0.0},
    ]
    new_diagnostics = [
        {"changed": True, "utility_delta": -1.0},
        {"changed": True, "utility_delta": 1.0},
    ]
    predictions, diagnostics = _cascade_outputs(
        old_predictions, old_diagnostics,
        new_predictions, new_diagnostics,
    )
    assert predictions[0]["ObjectEntities"] == ["old"]
    assert predictions[1]["ObjectEntities"] == ["new"]
    assert diagnostics[0]["cascade_stage"] == "incumbent_edge_precision"
    assert diagnostics[1]["cascade_stage"] == "pairwise_graph_recall"
