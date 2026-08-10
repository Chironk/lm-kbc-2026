import unittest

import numpy as np

from experiments.heterogeneous_agents.explicit_cardinality_ablation import (
    ExplicitCardinalityModel,
    cardinality_feature_names,
    cardinality_features,
    cardinality_label,
)
from experiments.heterogeneous_agents.heterogeneous_memory_selector import (
    GEMMA,
    QWEN,
    _make_graph,
)


def _proposal(agent, relation, answers):
    return {
        "agent_id": agent,
        "subject": "Target",
        "relation": relation,
        "phase": "propose",
        "mode": "generate",
        "generations": answers,
    }


def _commitment(selected, probabilities):
    return {
        "available": True,
        "selected": selected,
        "probabilities": probabilities,
    }


def _graph(relation="companyTradesAtStockExchange"):
    commitments = {
        QWEN: {
            "existence": _commitment("YES", {"YES": 0.8, "NO": 0.2}),
            "cardinality": _commitment(
                "ONE", {"ZERO": 0.1, "ONE": 0.8, "MANY": 0.1}),
        },
        GEMMA: {
            "existence": _commitment("YES", {"YES": 0.7, "NO": 0.3}),
            "cardinality": _commitment(
                "MANY", {"ZERO": 0.1, "ONE": 0.2, "MANY": 0.7}),
        },
    }
    return _make_graph(
        0,
        "Target",
        relation,
        {
            QWEN: _proposal(QWEN, relation, ["ANSWER: NASDAQ"]),
            GEMMA: _proposal(
                GEMMA, relation, ["ANSWER: NASDAQ, New York Stock Exchange"]),
        },
        {QWEN: ["NASDAQ"], GEMMA: ["NASDAQ", "New York Stock Exchange"]},
        commitments,
        "direct",
    )


class ExplicitCardinalityTests(unittest.TestCase):
    def test_feature_schema_is_fixed_and_finite(self):
        values = cardinality_features(_graph())
        self.assertEqual(len(values), len(cardinality_feature_names()))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_cardinality_labels_follow_gold_object_count(self):
        self.assertEqual(cardinality_label({"ObjectEntities": []}), "ZERO")
        self.assertEqual(cardinality_label({"ObjectEntities": [["A"]]}), "ONE")
        self.assertEqual(
            cardinality_label({"ObjectEntities": [["A"], ["B"]]}), "MANY")

    def test_numeric_and_award_cardinality_are_schema_deterministic(self):
        model = ExplicitCardinalityModel()
        self.assertEqual(
            model.predict_one(_graph("hasArea")),
            {"ZERO": 0.0, "ONE": 1.0, "MANY": 0.0},
        )
        self.assertEqual(
            model.predict_one(_graph("awardWonBy")),
            {"ZERO": 0.0, "ONE": 0.0, "MANY": 1.0},
        )

    def test_expected_size_never_undercuts_candidate_mass(self):
        model = ExplicitCardinalityModel()
        model.many_mean["companyTradesAtStockExchange"] = 3.0
        expected = model.expected_size(
            _graph(),
            {"ZERO": 0.2, "ONE": 0.5, "MANY": 0.3},
            candidate_sum=2.0,
        )
        self.assertEqual(expected, 2.0)


if __name__ == "__main__":
    unittest.main()
