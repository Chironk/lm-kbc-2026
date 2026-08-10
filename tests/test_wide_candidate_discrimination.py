from types import SimpleNamespace

from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.wide_candidate_discrimination import (
    WIDE_FEATURE_NAMES,
    WideEvidence,
    _binary_auroc,
    aggregate_anchor_margins,
    wide_anchor_groups,
)


def test_wide_groups_share_anchor_and_cover_every_challenger_once() -> None:
    nodes = [
        {
            "node_id": f"n{index}",
            "representative": str(index),
            "is_incumbent": index == 2,
        }
        for index in range(16)
    ]
    anchor, groups = wide_anchor_groups(nodes)
    assert anchor == "n2"
    assert all(group[0]["node_id"] == anchor for group in groups)
    assert all(1 <= len(group) <= 7 for group in groups)
    challengers = [
        node["node_id"] for group in groups for node in group[1:]
    ]
    assert sorted(challengers) == sorted(
        node["node_id"] for node in nodes if node["node_id"] != anchor)


def test_empty_incumbent_uses_unknown_anchor_and_seven_fact_groups() -> None:
    nodes = [
        {
            "node_id": f"n{index}",
            "representative": str(index),
            "is_incumbent": False,
        }
        for index in range(15)
    ]
    anchor, groups = wide_anchor_groups(nodes)
    assert anchor == "UNKNOWN"
    assert [len(group) for group in groups] == [7, 7, 1]
    assert sorted(
        node["node_id"] for group in groups for node in group
    ) == sorted(f"n{index}" for index in range(15))


def test_anchor_log_odds_cancel_group_normalizer() -> None:
    task = {
        "task_id": "t",
        "subject": "s",
        "relation": "hasArea",
        "agent_id": QWEN,
        "anchor_node_id": "old",
        "group_node_ids": ["old", "new"],
    }
    response = {
        "task_id": "t",
        "choice_probabilities": {
            "old": 0.2,
            "new": 0.6,
            "UNKNOWN": 0.2,
        },
    }
    evidence = aggregate_anchor_margins([task], {"t": response})
    assert abs(evidence[("s", "hasArea")][QWEN]["nodes"]["old"]) < 1e-12
    assert evidence[("s", "hasArea")][QWEN]["nodes"]["new"] > 1.09


def test_action_features_express_candidate_advantage() -> None:
    evidence = object.__new__(WideEvidence)
    evidence.nodes = {
        ("s", "hasArea"): {
            "raw": {"10": "old", "12": "new"},
            "canonical": {"numeric:10": "old", "numeric:12": "new"},
        },
    }
    evidence.scores = {
        ("s", "hasArea"): {
            QWEN: {
                "nodes": {"old": 0.0, "new": 2.0},
                "unknown_margin": -1.0,
            },
            GEMMA: {
                "nodes": {"old": 0.0, "new": 1.0},
                "unknown_margin": -0.5,
            },
        },
    }
    row = SimpleNamespace(
        key=("s", "hasArea"),
        relation="hasArea",
        source=SimpleNamespace(keep={"objects": ["10"]}),
    )
    action = SimpleNamespace(
        source=SimpleNamespace(action={"objects": ["12"]}))
    features = evidence.features(row, action)
    assert len(features) == len(WIDE_FEATURE_NAMES)
    assert features[0] == 1.0
    assert features[1] == 1.0
    assert features[2] > features[3] > 0.0


def test_auroc_handles_ties_and_missing_class() -> None:
    assert _binary_auroc([True, False], [1.0, 0.0]) == 1.0
    assert _binary_auroc([True, False], [0.0, 1.0]) == 0.0
    assert _binary_auroc([True, False], [0.0, 0.0]) == 0.5
    assert _binary_auroc([True], [1.0]) is None


def test_exact_alias_surfaces_override_ambiguous_canonical_fallback() -> None:
    evidence = object.__new__(WideEvidence)
    evidence.nodes = {
        ("Energen", "companyTradesAtStockExchange"): {
            "raw": {
                "nyse": "incumbent",
                "new york stock exchange": "component",
            },
            # The shared canonical key is intentionally absent because it is
            # ambiguous. Exact raw surfaces remain legal and distinguishable.
            "canonical": {},
        },
    }
    evidence.scores = {
        ("Energen", "companyTradesAtStockExchange"): {
            QWEN: {
                "nodes": {"incumbent": 0.0, "component": 1.0},
                "unknown_margin": -1.0,
            },
            GEMMA: {
                "nodes": {"incumbent": 0.0, "component": 0.5},
                "unknown_margin": -1.0,
            },
        },
    }
    row = SimpleNamespace(
        key=("Energen", "companyTradesAtStockExchange"),
        relation="companyTradesAtStockExchange",
        source=SimpleNamespace(keep={"objects": ["NYSE"]}),
    )
    action = SimpleNamespace(source=SimpleNamespace(
        action={"objects": ["New York Stock Exchange"]}))
    features = evidence.features(row, action)
    assert features[0] == 1.0
    assert features[1] == 1.0
    assert features[2] > 0.0
