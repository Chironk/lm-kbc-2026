import unittest

import numpy as np

from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.relation_specific_structured_decoder import (
    CITY,
    COMPANY,
    CityConditionalDecoder,
    _selection,
    company_actions,
    company_set_feature_names,
    company_set_features,
)


def graph(relation):
    return {
        "SubjectEntity": "Example",
        "Relation": relation,
        "baseline_objects": [],
        "candidates": [
            {
                "item": "Alpha",
                "key": "alpha",
                "sources": {
                    QWEN: {"support": 6, "samples": 10, "support_rate": 0.6},
                    GEMMA: {"support": 1, "samples": 1, "support_rate": 1.0},
                },
                "selected_by": {QWEN: True, GEMMA: True},
            },
            {
                "item": "Beta",
                "key": "beta",
                "sources": {
                    QWEN: {"support": 2, "samples": 10, "support_rate": 0.2}},
                "selected_by": {QWEN: False, GEMMA: False},
            },
        ],
        "agents": {
            QWEN: {
                "n_samples": 10, "none_rate": 0.2, "parse_failures": 1,
                "existence": {"available": True, "probabilities": {
                    "YES": 0.8, "NO": 0.2}},
                "cardinality": {"available": True, "probabilities": {
                    "ZERO": 0.1, "ONE": 0.7, "MANY": 0.2}},
            },
            GEMMA: {
                "n_samples": 1, "none_rate": 0.0, "parse_failures": 0,
                "existence": {"available": True, "probabilities": {
                    "YES": 0.7, "NO": 0.3}},
                "cardinality": {"available": True, "probabilities": {
                    "ZERO": 0.2, "ONE": 0.6, "MANY": 0.2}},
            },
        },
    }


class StructuredDecoderTests(unittest.TestCase):
    def test_company_actions_include_control_empty_and_additions(self):
        actions = company_actions(
            graph(COMPANY), np.asarray([0.8, 0.3]), ["Alpha"])
        keys = {tuple(sorted(action)) for action in actions}
        self.assertIn((), keys)
        self.assertIn(("Alpha",), keys)
        self.assertIn(("Alpha", "Beta"), keys)

    def test_company_actions_support_empty_candidate_graph(self):
        empty = graph(COMPANY)
        empty["candidates"] = []
        self.assertEqual(company_actions(
            empty, np.asarray([], dtype=np.float64), []), [[]])

    def test_company_set_features_have_fixed_schema(self):
        values = company_set_features(
            graph(COMPANY), ["Alpha"], np.asarray([0.8, 0.3]), [],
            {"ZERO": 0.1, "ONE": 0.7, "MANY": 0.2})
        self.assertEqual(len(values), len(company_set_feature_names()))
        self.assertTrue(all(np.isfinite(value) for value in values))

    def test_city_factorization_can_choose_null(self):
        decoder = CityConditionalDecoder()

        class Null:
            def predict(self, rows):
                return np.asarray([0.9])

        class Candidate:
            def predict(self, rows):
                return np.asarray([0.8] * len(rows))

        decoder.null_model = Null()
        decoder.candidate_model = Candidate()
        objects, detail = decoder.decode(graph(CITY), [], 0.0)
        self.assertEqual(objects, [])
        self.assertGreater(detail["p_null"], detail["proposed_utility"] - 1e-12)

    def test_city_control_matching_is_canonical(self):
        decoder = CityConditionalDecoder()

        class Null:
            def predict(self, rows):
                return np.asarray([0.1])

        class Candidate:
            def predict(self, rows):
                return np.asarray([0.9, 0.1])

        decoder.null_model = Null()
        decoder.candidate_model = Candidate()
        objects, detail = decoder.decode(graph(CITY), ["ALPHA"], 0.5)
        self.assertEqual(objects, ["ALPHA"])
        self.assertGreater(detail["control_utility"], 0.0)

    def test_deployment_gate_rejects_one_large_bad_fold(self):
        baseline = {fold: 0.4 for fold in range(5)}
        scores = {
            0.0: {0: 0.3, 1: 0.5, 2: 0.45, 3: 0.45, 4: 0.45},
            0.1: {fold: 0.4 for fold in range(5)},
        }
        margin, enabled, detail = _selection(scores, baseline)
        self.assertEqual(margin, 0.1)
        self.assertFalse(enabled)
        self.assertFalse(detail["enabled"])

    def test_one_se_prefers_conservative_eligible_margin(self):
        baseline = {fold: 0.4 for fold in range(5)}
        scores = {
            0.0: {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.4, 4: 0.4},
            0.2: {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.4, 4: 0.4},
        }
        margin, enabled, detail = _selection(scores, baseline)
        self.assertEqual(margin, 0.2)
        self.assertTrue(enabled)
        self.assertEqual(detail["best_mean_margin"], 0.2)


if __name__ == "__main__":
    unittest.main()
