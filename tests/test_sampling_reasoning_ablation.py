from experiments.heterogeneous_agents.sampling_reasoning_ablation import (
    CAP_N,
    GEMMA,
    MINISTRAL,
    N_VALUES,
    REASONING_CAPS,
    _prefix_evidence,
    _reasoning_words,
    _residual_prediction,
    _thresholds,
    arms,
)


def _response(relation: str, *answers: str) -> dict:
    return {
        "relation": relation,
        "generations": [
            f"REASONING: short factual recall\nANSWER: {answer}"
            for answer in answers
        ],
    }


def test_design_is_nested_and_orthogonal() -> None:
    values = arms()
    assert N_VALUES == (1, 3, 5, 10)
    assert REASONING_CAPS == (20, 40, 80)
    assert CAP_N == 3
    assert {arm.agent_id for arm in values} == {GEMMA, MINISTRAL}
    assert len(values) == 6
    assert sum(arm.axis == "sample_count" for arm in values) == 2
    assert sum(arm.axis == "reasoning_cap" for arm in values) == 4


def test_nested_prefix_does_not_read_later_generations() -> None:
    responses = {
        ("Venue", "hasCapacity"): _response(
            "hasCapacity", "100", "100", "200", "300", "400"),
    }
    n3, telemetry = _prefix_evidence(responses, 3, 20)
    assert n3[("Venue", "hasCapacity")]["counts"]["numeric:100"] == 2
    assert "numeric:300" not in n3[("Venue", "hasCapacity")]["counts"]
    assert telemetry["generations"] == 3
    assert telemetry["reasoning_cap_violation_rate"] == 0.0


def test_two_thirds_threshold_is_predeclared() -> None:
    assert _thresholds(1)["two_thirds"] == 1
    assert _thresholds(3)["two_thirds"] == 2
    assert _thresholds(5)["two_thirds"] == 4
    assert _thresholds(10)["two_thirds"] == 7


def test_numeric_residual_requires_unique_top_vote() -> None:
    incumbent = {
        "SubjectEntity": "Venue",
        "Relation": "hasCapacity",
        "ObjectEntities": ["100"],
    }
    tied = {
        "counts": {"numeric:200": 2, "numeric:300": 2},
        "displays": {"numeric:200": "200", "numeric:300": "300"},
    }
    assert _residual_prediction(
        incumbent, tied, 2)["ObjectEntities"] == ["100"]
    winner = {
        "counts": {"numeric:200": 3, "numeric:300": 1},
        "displays": {"numeric:200": "200", "numeric:300": "300"},
    }
    assert _residual_prediction(
        incumbent, winner, 2)["ObjectEntities"] == ["200"]


def test_string_residual_is_add_only_and_reasoning_words_are_audited() -> None:
    incumbent = {
        "SubjectEntity": "Company",
        "Relation": "companyTradesAtStockExchange",
        "ObjectEntities": ["NYSE"],
    }
    evidence = {
        "counts": {"nasdaq": 2},
        "displays": {"nasdaq": "NASDAQ"},
    }
    assert _residual_prediction(
        incumbent, evidence, 2)["ObjectEntities"] == ["NYSE", "NASDAQ"]
    assert _reasoning_words(
        "REASONING: one two three\nANSWER: value") == 3
