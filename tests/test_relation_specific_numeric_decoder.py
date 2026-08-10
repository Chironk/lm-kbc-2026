import unittest

from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.relation_specific_numeric_decoder import (
    RelationSpecificNumericModel,
    _fold_policy_diagnostics,
    _stable_relation,
    feature_names,
    numeric_options,
    option_features,
)


def graph(relation="hasArea"):
    return {
        "SubjectEntity": "Example",
        "Relation": relation,
        "baseline_objects": ["100"],
        "candidates": [
            {
                "item": "100",
                "key": "numeric:100",
                "sources": {
                    QWEN: {"support": 5, "samples": 10, "support_rate": 0.5}},
                "selected_by": {QWEN: True, GEMMA: False},
                "type": "numeric",
            },
            {
                "item": "104",
                "key": "numeric:104",
                "sources": {
                    GEMMA: {"support": 1, "samples": 1, "support_rate": 1.0}},
                "selected_by": {QWEN: False, GEMMA: True},
                "type": "numeric",
            },
            {
                "item": "300",
                "key": "numeric:300",
                "sources": {
                    QWEN: {"support": 1, "samples": 10, "support_rate": 0.1}},
                "selected_by": {QWEN: False, GEMMA: False},
                "type": "numeric",
            },
        ],
        "agents": {
            QWEN: {"n_samples": 10, "numeric_log_mad": 0.2},
            GEMMA: {"n_samples": 1, "numeric_log_mad": 0.0},
        },
    }


class RelationSpecificNumericDecoderTests(unittest.TestCase):
    def test_options_include_baseline_nodes_and_local_representative(self):
        options = numeric_options(graph())
        kinds = [option["kinds"] for option in options]
        self.assertTrue(any("baseline" in value for value in kinds))
        self.assertTrue(any("node" in value for value in kinds))
        self.assertTrue(any("cluster_geomean" in value for value in kinds))
        self.assertFalse(any(
            104 < option["value"] < 300 for option in options))

    def test_gemma_n1_is_not_treated_as_ten_votes(self):
        options = numeric_options(graph())
        gemma = next(option for option in options if option["value"] == 104)
        features = dict(zip(feature_names(), option_features(graph(), gemma)))
        self.assertEqual(features["gemma_support_mass"], 1.0)
        self.assertEqual(features["qwen_support_mass"], 0.5)

    def test_one_se_selects_largest_eligible_margin(self):
        folds = {
            0.0: {0: 0.4, 1: 0.6, 2: 0.5, 3: 0.5, 4: 0.5},
            0.1: {0: 0.4, 1: 0.6, 2: 0.5, 3: 0.5, 4: 0.5},
            0.2: {0: 0.4, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
        }
        baseline = {fold: 0.4 for fold in range(5)}
        best, one_se, details = _fold_policy_diagnostics(folds, baseline)
        self.assertEqual(best, 0.1)
        self.assertEqual(one_se, 0.2)
        self.assertTrue(details["margins"]["0.2"]["within_one_se_of_best"])

    def test_stable_relation_requires_no_negative_fold(self):
        positive = {
            "margins": {"0.0": {
                "mean_paired_delta": 0.02,
                "paired_fold_delta": {"0": 0.0, "1": 0.04},
            }}}
        unstable = {
            "margins": {"0.0": {
                "mean_paired_delta": 0.02,
                "paired_fold_delta": {"0": -0.1, "1": 0.14},
            }}}
        self.assertTrue(_stable_relation(positive, 0.0))
        self.assertFalse(_stable_relation(unstable, 0.0))

    def test_decode_guard_can_preserve_baseline(self):
        model = RelationSpecificNumericModel()
        # A deterministic fake model is enough to test the output guard.
        class Fake:
            def predict(self, rows):
                return __import__("numpy").asarray(
                    [0.8 if row[1] else 0.81 for row in rows])
        model.models["hasArea"] = Fake()
        objects, details = model.decode(graph(), margin=0.02)
        self.assertEqual(objects, ["100"])
        self.assertTrue(details["used_baseline"])


if __name__ == "__main__":
    unittest.main()
