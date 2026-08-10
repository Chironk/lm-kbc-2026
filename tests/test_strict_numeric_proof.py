from __future__ import annotations

from experiments.heterogeneous_agents.strict_numeric_proof import (
    strict_numeric_eligible,
)


def _graph(relation: str = "hasCapacity") -> dict:
    return {
        "SubjectEntity": "X", "Relation": relation,
        "relational_graph": {
            "components": [{
                "id": "component:0", "node_type": "candidate_component",
                "representative": "10000",
            }, {
                "id": "component:1", "node_type": "candidate_component",
                "representative": "10400",
            }],
            "nodes": [], "edges": [],
        },
    }


def _option(*, fraction: float, value: float = 10000.0,
            rates: dict[str, float] | None = None) -> dict:
    return {
        "value": value,
        "component_ids": ["component:0", "component:1"],
        "family_fraction": fraction,
        "rates": rates or {
            "qwen_recall": 0.2,
            "gemma_independent": 1.0,
            "ministral_independent": 0.1,
        },
    }


def test_numeric_proof_accepts_equal_cardinality_with_strict_family_advantage():
    incumbent = _option(fraction=1 / 3, value=9000.0)
    challenger = _option(fraction=1.0)
    assert strict_numeric_eligible(_graph(), challenger, incumbent)


def test_numeric_proof_rejects_two_versus_one_family_margin():
    incumbent = _option(fraction=1 / 3, value=9000.0)
    challenger = _option(fraction=2 / 3, rates={
        "qwen_recall": 0.2,
        "gemma_independent": 1.0,
        "ministral_independent": 0.0,
    })
    assert not strict_numeric_eligible(_graph(), challenger, incumbent)


def test_numeric_proof_rejects_incoherent_five_percent_component():
    graph = _graph()
    graph["relational_graph"]["components"][1]["representative"] = "12000"
    assert not strict_numeric_eligible(
        graph, _option(fraction=1.0), _option(fraction=1 / 3, value=9000.0))
