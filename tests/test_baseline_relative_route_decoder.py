import unittest

import numpy as np

from experiments.heterogeneous_agents.components.baseline_relative_route_decoder import (
    ResidualRidge,
    _eligible_nodes,
    _route_values,
    actions_for,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.route_aware_candidate_graph import (
    ROUTE_GEMMA,
    ROUTE_QWEN_SC,
    ROUTE_QWEN_SYSTEM2,
)


def graph(relation="companyTradesAtStockExchange"):
    return {
        "SubjectEntity": "S",
        "Relation": relation,
        "agents": {
            QWEN: {
                "n_samples": 10, "none_rate": 0.1,
                "existence": {"available": True,
                              "probabilities": {"YES": 0.9, "NO": 0.1}},
                "cardinality": {
                    "available": True,
                    "probabilities": {"ZERO": 0.1, "ONE": 0.3, "MANY": 0.6}},
            },
            GEMMA: {
                "n_samples": 1, "none_rate": 0.0,
                "existence": {"available": True,
                              "probabilities": {"YES": 0.8, "NO": 0.2}},
                "cardinality": {
                    "available": True,
                    "probabilities": {"ZERO": 0.2, "ONE": 0.5, "MANY": 0.3}},
            },
        },
        "candidates": [
            {
                "key": "x", "item": "X",
                "sources": {QWEN: {"support_rate": 0.6}},
                "selected_by": {QWEN: True, GEMMA: False},
                "routes": {
                    ROUTE_QWEN_SC: {"support_rate": 0.6},
                    ROUTE_QWEN_SYSTEM2: {"support_rate": 1.0},
                    ROUTE_GEMMA: {"support_rate": 1.0},
                },
                "route_summary": {
                    "model_family_count": 2,
                    "cross_model_agreement": True,
                    "within_qwen_route_agreement": True,
                    "system2_only": False,
                    "qwen_sc_only": False,
                    "gemma_only": False,
                },
            },
            {
                "key": "y", "item": "Y", "sources": {},
                "selected_by": {QWEN: False, GEMMA: False},
                "routes": {
                    ROUTE_QWEN_SYSTEM2: {"support_rate": 1.0}},
                "route_summary": {
                    "model_family_count": 1,
                    "cross_model_agreement": False,
                    "within_qwen_route_agreement": False,
                    "system2_only": True,
                    "qwen_sc_only": False,
                    "gemma_only": False,
                },
            },
        ],
    }


class BaselineRelativeRouteDecoderTest(unittest.TestCase):
    def test_agreement_arm_excludes_only_system2_only_nodes(self):
        self.assertEqual(
            [node["item"] for node in _eligible_nodes(
                graph(), "route_agreement")],
            ["X"])
        self.assertEqual(
            [node["item"] for node in _eligible_nodes(graph(), "route_full")],
            ["X", "Y"])

    def test_company_actions_are_bounded_incumbent_edits(self):
        actions = actions_for(graph(), ["A", "B"], "route_agreement")
        keys = {tuple(sorted(action)) for action in actions}
        self.assertIn(("A", "B"), keys)
        self.assertIn((), keys)
        self.assertIn(("A", "B", "X"), keys)
        self.assertIn(("A",), keys)
        self.assertIn(("B",), keys)
        self.assertNotIn(("A", "B", "Y"), keys)

    def test_city_actions_include_null_control_and_each_candidate(self):
        row = graph("personHasCityOfDeath")
        actions = actions_for(row, ["A"], "route_full")
        self.assertEqual(
            {tuple(action) for action in actions},
            {("A",), (), ("X",), ("Y",)})

    def test_all_three_route_is_explicit(self):
        values = _route_values(graph()["candidates"][0])
        self.assertEqual(values["all_three"], 1.0)
        self.assertEqual(values["cross_model"], 1.0)
        self.assertEqual(values["within_qwen"], 1.0)

    def test_residual_ridge_preserves_signed_outputs(self):
        model = ResidualRidge(["x"], l2=0.01).fit(
            [[-2.0], [-1.0], [1.0], [2.0]],
            [-1.0, -0.5, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0])
        predictions = model.predict([[-1.5], [1.5]])
        self.assertLess(float(predictions[0]), 0.0)
        self.assertGreater(float(predictions[1]), 0.0)
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()
