from experiments.heterogeneous_agents import standalone_cot_ablation as ablation
from experiments.heterogeneous_agents.standalone_cot_ablation import (
    _generic_items,
    _route_native_base_predictions,
    _supported_objects,
    _thresholds,
)


def test_thresholds_are_predeclared() -> None:
    assert _thresholds(10) == {
        "any": 1,
        "majority": 6,
        "two_thirds": 7,
        "unanimous": 10,
    }
    assert _thresholds(3) == {
        "any": 1,
        "majority": 2,
        "two_thirds": 2,
        "unanimous": 3,
    }


def test_string_support_counts_distinct_generations() -> None:
    response = {
        "relation": "companyTradesAtStockExchange",
        "generations": [
            "ANSWER: Nasdaq; Nasdaq",
            "ANSWER: Nasdaq",
            "ANSWER: None",
        ],
    }
    assert _supported_objects(
        response, parser=_generic_items, threshold=2) == ["Nasdaq"]


def test_numeric_tie_abstains() -> None:
    response = {
        "relation": "hasArea",
        "generations": ["ANSWER: 100", "ANSWER: 200"],
    }
    assert _supported_objects(
        response, parser=_generic_items, threshold=1) == []


def test_route_native_uses_repository_aggregator(monkeypatch) -> None:
    plan = {
        "cot_agents": "agents.json",
        "jobs": {
            "qwen:self_consistency": {"agent_id": "qwen_recall"},
            "gemma:independent": {"agent_id": "gemma_independent"},
        },
    }
    source = [{"SubjectEntity": "S", "Relation": "hasArea"}]
    monkeypatch.setattr(ablation, "load_agent_config", lambda path: {})
    monkeypatch.setattr(
        ablation.e2e, "_agent_of", lambda config, agent_id: {"id": agent_id})

    def responses(job):
        return [
            {
                "subject": "S",
                "relation": "hasArea",
                "phase": phase,
            }
            for phase in ("commit_existence", "commit_cardinality", "propose")
        ]

    monkeypatch.setattr(ablation.e2e, "_validated_responses", responses)
    monkeypatch.setattr(
        ablation, "assemble_graphs",
        lambda rows, agents, response_rows: [{
            "SubjectEntity": "S", "Relation": "hasArea"}])
    monkeypatch.setattr(
        ablation, "prediction_for_agent",
        lambda graph, agent_id: [agent_id])

    result = _route_native_base_predictions(plan, source)
    assert result["qwen_recall"][0]["ObjectEntities"] == ["qwen_recall"]
    assert result["gemma_independent"][0]["ObjectEntities"] == [
        "gemma_independent"]
