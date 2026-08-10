from __future__ import annotations

from experiments.heterogeneous_agents.components.generation_set_hypothesis_audit import (
    artifact_contract_matches,
    hypothesis_stats,
    select_hypothesis,
    set_f1,
    shuffled_values,
)
from experiments.heterogeneous_agents.core import sha256


def test_set_f1_is_symmetric_and_handles_empty() -> None:
    assert set_f1([], []) == 1.0
    assert set_f1([], ["a"]) == 0.0
    assert set_f1(["a", "b"], ["b", "c"]) == 0.5
    assert set_f1(["b", "c"], ["a", "b"]) == 0.5


def test_family_balance_caps_repeated_samples() -> None:
    hypothesis = {
        "tokens": frozenset({"a"}),
        "proposer_families": {"qwen_recall"},
    }
    base = {
        "qwen_recall": [frozenset({"a"})],
        "gemma_independent": [frozenset({"b"})],
        "ministral_independent": [frozenset({"c"})],
    }
    repeated = dict(base)
    repeated["qwen_recall"] = [frozenset({"a"})] * 20
    left = hypothesis_stats(hypothesis, base)
    right = hypothesis_stats(hypothesis, repeated)
    assert left["mean_similarity"] == right["mean_similarity"] == 1 / 3
    assert left["exact_family_fraction"] == right["exact_family_fraction"]


def test_selector_prefers_incumbent_on_complete_tie() -> None:
    hypotheses = [
        {
            "tokens": frozenset({"a"}),
            "is_incumbent": True,
            "proposer_families": set(),
        },
        {
            "tokens": frozenset({"b"}),
            "is_incumbent": False,
            "proposer_families": set(),
        },
    ]
    stats = [{
        "mean_similarity": 0.5,
        "minimum_similarity": 0.4,
        "exact_family_fraction": 0.0,
        "independent_similarity": 0.5,
    }] * 2
    assert select_hypothesis(
        hypotheses, stats, "family_mean_medoid") == 0


def test_shuffle_is_deterministic_and_preserves_values() -> None:
    values = [0.1, 0.2, 0.3, 0.4]
    left = shuffled_values(values, subject="s", relation="hasArea")
    right = shuffled_values(values, subject="s", relation="hasArea")
    assert left == right
    assert left != values
    assert sorted(left) == values


def test_artifact_contract_hashes_path_values(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    plan = {
        "artifact_path": str(artifact),
        "artifact_sha256": sha256(artifact),
    }
    required = (("artifact_path", "artifact_sha256"),)
    assert artifact_contract_matches(plan, required)
    plan["artifact_sha256"] = "0" * 64
    assert not artifact_contract_matches(plan, required)
