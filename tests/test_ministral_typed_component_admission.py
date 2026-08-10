from experiments.heterogeneous_agents.components.ministral_typed_component_admission import (
    _complete_link_numeric_clusters,
    _distance,
    _merge_typed_row,
)


def test_official_relative_distance() -> None:
    assert _distance(590.0, 603.67) < 0.05
    assert _distance(100.0, 106.0) > 0.05


def test_numeric_support_counts_distinct_generations() -> None:
    occurrences = [
        {"generation": 0, "item": "590", "key": "numeric:590", "value": 590.0},
        {
            "generation": 0,
            "item": "592.55",
            "key": "numeric:592.55",
            "value": 592.55,
        },
        {
            "generation": 1,
            "item": "603.67",
            "key": "numeric:603.67",
            "value": 603.67,
        },
        {
            "generation": 2,
            "item": "1000",
            "key": "numeric:1000",
            "value": 1000.0,
        },
    ]
    clusters = _complete_link_numeric_clusters(occurrences)
    strong = next(
        cluster for cluster in clusters if "590" in cluster["member_items"])
    assert strong["generation_support"] == 2
    assert set(strong["member_items"]) == {"590", "592.55", "603.67"}


def test_complete_link_does_not_bridge_distant_endpoints() -> None:
    occurrences = [
        {"generation": 0, "item": "100", "key": "numeric:100", "value": 100.0},
        {"generation": 1, "item": "104", "key": "numeric:104", "value": 104.0},
        {"generation": 2, "item": "108", "key": "numeric:108", "value": 108.0},
    ]
    clusters = _complete_link_numeric_clusters(occurrences)
    assert len(clusters) == 2
    assert not any(
        {"100", "108"}.issubset(set(cluster["member_items"]))
        for cluster in clusters)


def test_typed_area_cluster_is_admitted_and_string_is_queued() -> None:
    base = {
        "SubjectEntity": "Island",
        "Relation": "hasArea",
        "candidates": [],
        "agents": {},
        "proposal_routes": {},
        "production_match": {},
    }
    third = {
        "SubjectEntity": "Island",
        "Relation": "hasArea",
        "candidates": [],
        "commitments": {
            "ministral_independent": {
                "existence": "YES",
                "existence_probabilities": {"YES": 0.9, "NO": 0.1},
                "cardinality": "ONE",
                "cardinality_probabilities": {"ONE": 0.8, "MANY": 0.2},
            },
        },
    }
    proposal = {
        "generations": [
            "ANSWER: 590",
            "ANSWER: 592.55",
            "ANSWER: 603.67",
        ],
    }
    merged, queue, audit = _merge_typed_row(base, third, proposal)
    assert audit["admitted_new_candidates"] == 1
    assert queue == []
    assert len(merged["candidates"]) == 1
    assert len(merged["relational_graph"]["components"]) == 1
    agent = merged["agents"]["ministral_independent"]
    assert agent["decoder_commitments_enabled"] is True
    assert agent["existence"]["probabilities"]["YES"] == 0.9


def test_queued_candidate_remains_dormant_and_not_output_eligible() -> None:
    base = {
        "SubjectEntity": "Person",
        "Relation": "personHasCityOfDeath",
        "candidates": [],
        "agents": {},
        "proposal_routes": {},
        "production_match": {},
    }
    third = {
        "SubjectEntity": "Person",
        "Relation": "personHasCityOfDeath",
        "candidates": [{
            "item": "Paris",
            "proposal_support": {"ministral_independent": 1},
        }],
        "commitments": {
            "ministral_independent": {
                "existence": "YES",
                "existence_probabilities": {"YES": 1.0},
                "cardinality": "ONE",
                "cardinality_probabilities": {"ONE": 1.0},
            },
        },
    }
    proposal = {"generations": [
        "ANSWER: Paris", "ANSWER: Lyon", "ANSWER: Rome"]}
    merged, queue, _ = _merge_typed_row(base, third, proposal)
    assert len(queue) == 1
    assert merged["candidates"] == []
    assert len(merged["dormant_candidates"]) == 1
    assert merged["dormant_candidates"][0]["output_eligible"] is False
