from __future__ import annotations

from experiments.heterogeneous_agents.components.proof_carrying_graph_decoder import (
    _candidate_is_eligible,
    _decode,
    _event_records,
    _proof_metrics,
    _select,
)


def _metric(**values):
    result = {
        "exact_family_fraction": 1 / 3,
        "minimum_similarity": 0.4,
        "mean_similarity": 0.5,
        "within_family_exact_rate_mean": 0.1,
        "independent_similarity": 0.5,
        "exact_cardinality_match_rate": 0.3,
        "existence_match_rate": 1.0,
        "set_size": 2.0,
    }
    result.update(values)
    return result


def test_proof_requires_cardinality_advantage_and_no_expansion():
    keep = _metric()
    good = _metric(
        exact_family_fraction=2 / 3,
        minimum_similarity=0.5,
        exact_cardinality_match_rate=0.4,
        set_size=1.0,
    )
    assert _candidate_is_eligible("loose_proof_graph", good, keep)
    assert not _candidate_is_eligible(
        "loose_proof_graph",
        {**good, "exact_cardinality_match_rate": 0.3}, keep)
    assert not _candidate_is_eligible(
        "loose_proof_graph", {**good, "set_size": 3.0}, keep)


def test_strict_proof_requires_two_family_advantage_over_keep():
    keep = _metric(exact_family_fraction=1 / 3)
    two_family = _metric(
        exact_family_fraction=2 / 3,
        minimum_similarity=0.5,
        exact_cardinality_match_rate=0.4,
        set_size=1.0,
    )
    assert _candidate_is_eligible("loose_proof_graph", two_family, keep)
    assert not _candidate_is_eligible("strict_proof_graph", two_family, keep)
    all_family = {**two_family, "exact_family_fraction": 1.0}
    assert _candidate_is_eligible("strict_proof_graph", all_family, keep)


def test_support_arm_does_not_consume_cardinality_or_expansion_guard():
    keep = _metric()
    challenger = _metric(
        exact_family_fraction=2 / 3,
        minimum_similarity=0.5,
        exact_cardinality_match_rate=0.0,
        existence_match_rate=0.0,
        set_size=20.0,
    )
    assert _candidate_is_eligible("support_consensus", challenger, keep)


def test_event_records_validate_typed_state_edges():
    graph = {
        "SubjectEntity": "X", "Relation": "hasCapacity",
        "relational_graph": {
            "nodes": [{
                "id": "e:q:0", "node_type": "evidence_event",
                "status": "candidate_set", "model_family": "qwen_recall",
            }, {
                "id": "e:g:0", "node_type": "evidence_event",
                "status": "candidate_set", "model_family": "gemma_independent",
            }, {
                "id": "e:m:0", "node_type": "evidence_event",
                "status": "explicit_none", "model_family": "ministral_independent",
            }],
            "edges": [
                {"source": "e:q:0", "target": "candidate:1", "edge_type": "supports"},
                {"source": "e:g:0", "target": "candidate:1", "edge_type": "supports"},
                {"source": "e:q:0", "target": "cardinality:ONE", "edge_type": "asserts_cardinality"},
                {"source": "e:g:0", "target": "cardinality:ONE", "edge_type": "asserts_cardinality"},
                {"source": "e:m:0", "target": "cardinality:ZERO", "edge_type": "asserts_cardinality"},
            ],
        },
    }
    records = _event_records(graph)
    assert [row["exact_cardinality"] for row in records] == [1, 0, 1]
    assert {row["family"] for row in records} == {
        "qwen_recall", "gemma_independent", "ministral_independent"}


def test_proof_metrics_use_exact_event_degree():
    records = [
        {"exact_cardinality": 2, "exists": True},
        {"exact_cardinality": 2, "exists": True},
        {"exact_cardinality": 1, "exists": True},
    ]
    stats = {
        "exact_family_fraction": 2 / 3,
        "minimum_similarity": 0.5,
        "mean_similarity": 0.6,
        "within_family_exact_rate_mean": 0.2,
        "independent_similarity": 0.7,
    }
    metrics = _proof_metrics({"tokens": {"a", "b"}}, stats, records)
    assert metrics["exact_cardinality_match_rate"] == 2 / 3
    assert metrics["existence_match_rate"] == 1.0


def test_select_keeps_when_proof_is_incomplete():
    context = {
        "hypotheses": [
            {"tokens": {"a"}, "objects": ["A"], "is_incumbent": True},
            {"tokens": {"b"}, "objects": ["B"], "is_incumbent": False},
        ],
        "metrics": [
            _metric(set_size=1.0),
            _metric(exact_family_fraction=2 / 3, minimum_similarity=0.5,
                    exact_cardinality_match_rate=0.3, set_size=1.0),
        ],
        "keep_index": 0,
    }
    selected, decision = _select(context, "loose_proof_graph")
    assert selected == 0
    assert decision["changed"] is False


def test_validation_decode_fails_closed_on_unmapped_candidate_event():
    graph = {
        "SubjectEntity": "X", "Relation": "hasArea",
        "relational_graph": {
            "nodes": [
                {"id": "q", "node_type": "evidence_event",
                 "status": "candidate_set", "model_family": "qwen_recall"},
                {"id": "g", "node_type": "evidence_event",
                 "status": "explicit_none", "model_family": "gemma_independent"},
                {"id": "m", "node_type": "evidence_event",
                 "status": "explicit_none", "model_family": "ministral_independent"},
            ],
            "edges": [
                {"source": "q", "target": "cardinality:ZERO",
                 "edge_type": "asserts_cardinality"},
                {"source": "g", "target": "cardinality:ZERO",
                 "edge_type": "asserts_cardinality"},
                {"source": "m", "target": "cardinality:ZERO",
                 "edge_type": "asserts_cardinality"},
            ],
            "components": [],
        },
    }
    predictions, decisions = _decode(
        [graph], {("X", "hasArea"): ["12"]}, "strict_proof_graph",
        fail_closed_invalid_evidence=True,
    )
    assert predictions[0]["ObjectEntities"] == ["12"]
    assert decisions[0]["evidence_invalid_fallback"] is True
