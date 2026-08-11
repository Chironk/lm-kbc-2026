import pytest

from experiments.heterogeneous_agents import graphless_cot_ensemble_ablation as ablation
from experiments.heterogeneous_agents.graphless_cot_ensemble_ablation import (
    ARMS,
    ROUTES,
    _route_predictions,
    fuse_source_outputs,
)


def test_predeclared_arms_cover_requested_combinations() -> None:
    assert ARMS == {
        "qwen_gemma_union": (("qwen", "gemma"), 1),
        "qwen_gemma_agreement": (("qwen", "gemma"), 2),
        "qwen_gemma_ministral_union": (
            ("qwen", "gemma", "ministral"), 1),
        "qwen_gemma_ministral_majority": (
            ("qwen", "gemma", "ministral"), 2),
        "qwen_gemma_ministral_unanimous": (
            ("qwen", "gemma", "ministral"), 3),
    }


def test_string_votes_count_distinct_models_after_canonicalization() -> None:
    outputs = [
        ["NYSE", "NYSE"],
        ["New York Stock Exchange", "NASDAQ"],
        ["NASDAQ"],
    ]
    relation = "companyTradesAtStockExchange"
    assert fuse_source_outputs(
        outputs, relation, minimum_sources=1) == ["NYSE", "NASDAQ"]
    assert fuse_source_outputs(
        outputs, relation, minimum_sources=2) == ["NYSE", "NASDAQ"]
    assert fuse_source_outputs(
        outputs, relation, minimum_sources=3) == []


def test_numeric_fusion_is_plain_median_with_availability_gate() -> None:
    outputs = [["100"], ["120"], ["1000"]]
    assert fuse_source_outputs(
        outputs, "hasArea", minimum_sources=2) == ["120"]
    assert fuse_source_outputs(
        [["100"], [], []], "hasArea", minimum_sources=2) == []


def test_invalid_source_threshold_fails() -> None:
    with pytest.raises(ValueError):
        fuse_source_outputs([[]], "awardWonBy", minimum_sources=2)


def test_route_predictions_ignore_commitment_phases(monkeypatch, tmp_path) -> None:
    source = [{"SubjectEntity": "S", "Relation": "personHasCityOfDeath"}]
    plan = {"jobs": {}}
    response_rows = {}
    reasoning_words = {"qwen": None, "gemma": 20, "ministral": 40}
    for source_name, (route, samples, shots) in ROUTES.items():
        response_path = tmp_path / f"{source_name}.jsonl"
        response_path.write_text("validated-cache\n")
        job = {
            "route": route,
            "model": source_name,
            "revision": f"{source_name}-revision",
            "n_proposals": samples,
            "synthetic_shots": shots,
            "reasoning_words": reasoning_words[source_name],
            "response_path": str(response_path),
        }
        plan["jobs"][route] = job
        response_rows[route] = [
            {
                "subject": "S",
                "relation": "personHasCityOfDeath",
                "phase": "commit_existence",
                "selected_choice": "NO",
            },
            {
                "subject": "S",
                "relation": "personHasCityOfDeath",
                "phase": "propose",
                "generations": ["ANSWER: Paris"] * samples,
            },
        ]

    monkeypatch.setattr(
        ablation.e2e,
        "_validated_responses",
        lambda job: response_rows[job["route"]],
    )
    decoded_phases = []

    def decode(response):
        decoded_phases.append(response["phase"])
        return ["Paris"]

    monkeypatch.setattr(ablation, "proposal_only_prediction", decode)
    predictions, provenance = _route_predictions(plan, source)

    assert decoded_phases == ["propose", "propose", "propose"]
    assert set(predictions) == {"qwen", "gemma", "ministral"}
    assert all(rows[0]["ObjectEntities"] == ["Paris"]
               for rows in predictions.values())
    assert all(record["synthetic_shots"] == 5
               for record in provenance.values())
