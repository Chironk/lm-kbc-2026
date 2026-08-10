import copy

from experiments.heterogeneous_agents.components.ministral_consistency_admission import (
    ADMISSION_SUPPORT,
    MINISTRAL,
    N_PROPOSALS,
    ROUTE,
    _merge_row,
)


def _base(relation: str = "companyTradesAtStockExchange") -> dict:
    return {
        "SubjectEntity": "Example",
        "Relation": relation,
        "candidates": [{
            "key": "newyorkstockexchange",
            "item": "New York Stock Exchange",
            "sources": {
                "qwen_recall": {
                    "samples": 10,
                    "support": 8,
                    "support_rate": 0.8,
                },
            },
            "routes": {
                "qwen:self_consistency": {
                    "model_family": "qwen_recall",
                    "support_rate": 0.8,
                    "selected": True,
                },
            },
            "selected_by": {"qwen_recall": True},
            "route_summary": {},
        }],
        "agents": {},
        "agent_outputs": {},
        "proposal_routes": {},
        "production_match": {},
    }


def _third(candidates: list[tuple[str, int]]) -> dict:
    return {
        "SubjectEntity": "Example",
        "Relation": "companyTradesAtStockExchange",
        "commitments": {
            MINISTRAL: {
                "existence": "YES",
                "existence_probabilities": {"YES": 0.8},
                "cardinality": "ONE",
                "cardinality_probabilities": {"ONE": 0.7},
            },
        },
        "candidates": [
            {
                "key": item.lower().replace(" ", ""),
                "item": item,
                "proposer_agents": [MINISTRAL],
                "proposal_support": {MINISTRAL: support},
                "reviews": {},
            }
            for item, support in candidates
        ],
    }


def test_admission_preserves_source_and_rejects_single_sample_new() -> None:
    source = _base()
    original = copy.deepcopy(source)
    merged, audit = _merge_row(source, _third([
        ("New York Stock Exchange", 1),
        ("NASDAQ", 1),
    ]))
    assert source == original
    assert [node["item"] for node in merged["candidates"]] == [
        "New York Stock Exchange"]
    assert audit["corroborated_source_candidates"] == 1
    assert audit["rejected_new_candidates"] == 1
    node = merged["candidates"][0]
    assert node["sources"][MINISTRAL]["support"] == 1
    assert node["routes"][ROUTE]["admission_reason"] == "cross_model_corroborated"


def test_two_of_three_new_candidate_is_admitted_and_componentized() -> None:
    merged, audit = _merge_row(_base(), _third([("NASDAQ", 2)]))
    assert N_PROPOSALS == 3
    assert ADMISSION_SUPPORT == 2
    assert audit["admitted_new_candidates"] == 1
    nasdaq = next(
        node for node in merged["candidates"] if node["item"] == "NASDAQ")
    assert nasdaq["sources"][MINISTRAL]["support_rate"] == 2 / 3
    assert nasdaq["selected_by"][MINISTRAL] is True
    components = merged["relational_graph"]["components"]
    assert any("NASDAQ" in component["member_items"] for component in components)


def test_ministral_commitments_are_recorded_but_decoder_disabled() -> None:
    merged, _ = _merge_row(_base(), _third([("NASDAQ", 3)]))
    agent = merged["agents"][MINISTRAL]
    assert agent["existence"]["selected"] == "YES"
    assert agent["cardinality"]["selected"] == "ONE"
    assert agent["decoder_commitments_enabled"] is False


def test_legacy_duplicate_numeric_nodes_are_coalesced_without_double_count() -> None:
    source = _base("hasArea")
    source["candidates"] = [
        {
            "key": "numeric:923768",
            "item": "923768",
            "sources": {
                "gemma_independent": {
                    "samples": 1, "support": 1, "support_rate": 1.0}},
            "routes": {
                "gemma:independent": {
                    "model_family": "gemma_independent",
                    "support": 1,
                    "support_rate": 1.0,
                    "selected": True,
                },
            },
            "selected_by": {"gemma_independent": True},
            "route_summary": {},
        },
        {
            "key": "923768",
            "item": "923768",
            "sources": {
                "qwen_recall": {
                    "samples": 10, "support": 7, "support_rate": 0.7}},
            "routes": {
                "qwen:self_consistency": {
                    "model_family": "qwen_recall",
                    "support": 7,
                    "support_rate": 0.7,
                    "selected": False,
                },
            },
            "selected_by": {"qwen_recall": False},
            "route_summary": {},
        },
    ]
    third = {
        "SubjectEntity": "Example",
        "Relation": "hasArea",
        "commitments": {},
        "candidates": [],
    }
    merged, _ = _merge_row(source, third)
    assert len(merged["candidates"]) == 1
    node = merged["candidates"][0]
    assert node["key"] == "numeric:923768"
    assert set(node["sources"]) == {"gemma_independent", "qwen_recall"}
    assert node["sources"]["qwen_recall"]["support"] == 7
    assert node["route_summary"]["cross_model_agreement"] is True
