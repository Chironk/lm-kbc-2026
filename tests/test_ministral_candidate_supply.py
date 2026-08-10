from pathlib import Path

from experiments.heterogeneous_agents.core import load_agent_config
from experiments.heterogeneous_agents.components.ministral_candidate_supply import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    GEMMA,
    MINISTRAL,
    QWEN,
    _combined_graph,
    _correlation,
    _has_correct,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final/portfolio_supply.json"


def test_legal_portfolio_uses_exact_pinned_ministral() -> None:
    config = load_agent_config(CONFIG)
    agents = {agent["id"]: agent for agent in config["agents"]}
    assert set(agents) == {QWEN, GEMMA, MINISTRAL}
    assert agents[MINISTRAL]["model"] == EXPECTED_MODEL
    assert agents[MINISTRAL]["revision"] == EXPECTED_REVISION
    assert agents[MINISTRAL]["fix_mistral_regex"] is True
    assert agents[MINISTRAL]["synthetic_shots"] == 0
    assert agents[MINISTRAL]["proposal_output"] == "bounded_reasoning"
    assert config["verified_parameter_total"] == 30_515_165_024
    assert config["declared_parameter_headroom"] == 1_484_834_976


def test_combined_graph_deduplicates_canonical_candidates() -> None:
    base = {
        "SubjectEntity": "Example",
        "Relation": "companyTradesAtStockExchange",
        "candidates": [{"item": "New York Stock Exchange"}],
    }
    third = {
        "SubjectEntity": "Example",
        "Relation": "companyTradesAtStockExchange",
        "candidates": [
            {
                "item": "New York Stock Exchange",
                "proposer_agents": [MINISTRAL],
            },
            {"item": "NASDAQ", "proposer_agents": [MINISTRAL]},
        ],
    }
    combined = _combined_graph(base, third)
    assert [row["item"] for row in combined["candidates"]] == [
        "New York Stock Exchange", "NASDAQ"]


def test_correct_candidate_uses_official_string_matching() -> None:
    gold = {
        "SubjectEntity": "Example",
        "Relation": "companyTradesAtStockExchange",
        "ObjectEntities": [["New York Stock Exchange", "NYSE"]],
    }
    assert _has_correct(
        ["NYSE"], gold, "companyTradesAtStockExchange")
    assert not _has_correct(
        ["NASDAQ"], gold, "companyTradesAtStockExchange")


def test_numeric_correct_candidate_uses_five_percent_tolerance() -> None:
    gold = {
        "SubjectEntity": "Arena",
        "Relation": "hasCapacity",
        "ObjectEntities": [["10000"]],
    }
    assert _has_correct(["10400"], gold, "hasCapacity")
    assert not _has_correct(["10600"], gold, "hasCapacity")


def test_correlation_is_well_defined_and_symmetric() -> None:
    left = [0.0, 0.0, 1.0, 1.0]
    right = [0.0, 1.0, 0.0, 1.0]
    assert _correlation(left, right) == _correlation(right, left)
    assert _correlation(left, right) == 0.0
