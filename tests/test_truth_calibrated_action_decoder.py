import unittest

from experiments.heterogeneous_agents.components.truth_calibrated_action_decoder import (
    ACTION_FEATURE_NAMES,
    CALIBRATOR_FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    PAIRWISE_FEATURE_NAMES,
    RANK_GATE_FEATURE_NAMES,
    StandardizedLinear,
    _within_row_rank_z,
    apply_starting_predictions,
    augmented_action_features,
    augmented_gate_features,
    calibrated_rank_predictions,
    calibrator_features,
    counterfactual_incumbents,
    direct_utility_predictions,
    expand_fixed_cardinality_states,
    fit_direct_utility_selector,
    SetAwareRanker,
    fixed_cardinality_actions,
    pairwise_action_features,
    rank_gate_features,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import (
    build_hierarchical_row,
)
from tests.test_unified_memory_action_graph import graph


def evidence(row):
    records, calibrated = {}, {}
    for component in row["_source"]["relational_graph"]["components"]:
        key = (
            row["SubjectEntity"], row["Relation"], component["id"])
        is_london = component["representative"] == "London"
        records[key] = {
            "component_key": component["id"],
            "proposer_agents": (
                ["gemma_independent"] if is_london else ["qwen_recall"]),
            "raw": {
                "qwen_recall": 0.8 if not is_london else 0.6,
                "gemma_independent": 0.2 if not is_london else 0.95,
            },
        }
        calibrated[key] = 0.9 if is_london else 0.1
    return records, calibrated


class TruthCalibratedActionDecoderTest(unittest.TestCase):
    def test_calibrator_feature_schema_is_finite(self):
        record = {
            "proposer_agents": ["qwen_recall"],
            "raw": {
                "qwen_recall": 0.8,
                "gemma_independent": 0.2,
            },
        }
        values = calibrator_features(
            "personHasCityOfDeath", record)
        self.assertEqual(len(values), len(CALIBRATOR_FEATURE_NAMES))
        self.assertTrue(all(value == value for value in values))

    def test_likelihood_normalization_is_tie_stable(self):
        normalized = _within_row_rank_z({
            "a": -2.0, "b": -1.0, "c": -1.0,
        })
        self.assertEqual(normalized["b"][0], normalized["c"][0])
        self.assertGreater(normalized["b"][0], normalized["a"][0])

    def test_calibrator_accepts_within_question_likelihood(self):
        record = {
            "proposer_agents": ["qwen_recall"],
            "raw": {
                "qwen_recall": 0.8,
                "gemma_independent": 0.2,
            },
            "likelihood": {
                "available": 1.0,
                "qwen_recall": {
                    signal: {"rank": 0.75, "z": 0.25}
                    for signal in ("subject", "masked", "pmi")
                },
                "gemma_independent": {
                    signal: {"rank": 0.75, "z": 0.25}
                    for signal in ("subject", "masked", "pmi")
                },
            },
        }
        values = calibrator_features("hasArea", record)
        self.assertEqual(len(values), len(CALIBRATOR_FEATURE_NAMES))
        self.assertIn(1.0, values)

    def test_augmented_features_express_challenger_advantage(self):
        row = build_hierarchical_row(graph())
        records, calibrated = evidence(row)
        london = next(
            action for action in row["actions"]
            if action["objects"] == ["London"])
        action_values = augmented_action_features(
            row, london, calibrated, records)
        gate_values = augmented_gate_features(
            row, calibrated, records)
        self.assertEqual(len(action_values), len(ACTION_FEATURE_NAMES))
        self.assertEqual(len(gate_values), len(GATE_FEATURE_NAMES))
        self.assertAlmostEqual(
            action_values[ACTION_FEATURE_NAMES.index(
                "truth_challenger_advantage")],
            0.8,
        )
        self.assertAlmostEqual(
            gate_values[GATE_FEATURE_NAMES.index(
                "truth_challenger_advantage")],
            0.8,
        )

    def test_named_logistic_outputs_probabilities(self):
        model = StandardizedLinear(
            ("signal",), 1.0, logistic=True).fit(
                [[0.0], [0.2], [0.8], [1.0]],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            )
        low, high = model.predict([[0.1], [0.9]])
        self.assertGreater(high, low)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_named_model_round_trip_is_exact(self):
        model = StandardizedLinear(
            ("left", "right"), 1.0, logistic=False).fit(
                [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            )
        restored = StandardizedLinear.from_dict(model.to_dict())
        expected = model.predict([[0.25, 0.75], [0.75, 0.25]])
        actual = restored.predict([[0.25, 0.75], [0.75, 0.25]])
        self.assertEqual(expected.tolist(), actual.tolist())

    def test_pairwise_features_include_signed_keep_contrast(self):
        row = build_hierarchical_row(graph())
        records, calibrated = evidence(row)
        keep = next(
            action for action in row["actions"]
            if action["action_type"] == "KEEP")
        london = next(
            action for action in row["actions"]
            if action["objects"] == ["London"])
        action = augmented_action_features(
            row, london, calibrated, records)
        baseline = augmented_action_features(
            row, keep, calibrated, records)
        values = pairwise_action_features(
            row, london, keep, calibrated, records)
        self.assertEqual(len(values), len(PAIRWISE_FEATURE_NAMES))
        self.assertEqual(values[:len(ACTION_FEATURE_NAMES)], action)
        self.assertEqual(
            values[len(ACTION_FEATURE_NAMES):],
            [left - right for left, right in zip(action, baseline)],
        )

    def test_starting_prediction_becomes_explicit_keep_state(self):
        row = build_hierarchical_row(graph())
        apply_starting_predictions([row], [{
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": ["London"],
        }])
        self.assertEqual(row["incumbent_objects"], ["London"])
        keep = next(
            action for action in row["actions"]
            if action["action_type"] == "KEEP")
        self.assertEqual(keep["objects"], ["London"])

    def test_fixed_cardinality_actions_never_change_answer_size(self):
        row = build_hierarchical_row(graph())
        actions = fixed_cardinality_actions(row, ["Paris"])
        self.assertTrue(any(
            action["objects"] == ["London"] for action in actions))
        self.assertTrue(all(len(action["objects"]) == 1 for action in actions))
        self.assertFalse(any(
            action["action_type"] in {"ADD", "DROP", "EMPTY"}
            for action in actions))

    def test_empty_state_cannot_invent_candidate_truth(self):
        row = build_hierarchical_row(graph())
        actions = fixed_cardinality_actions(row, [])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action_type"], "KEEP")
        self.assertEqual(actions[0]["objects"], [])

    def test_fixed_cardinality_expansion_preserves_root_size(self):
        row = build_hierarchical_row(graph())
        row["incumbent_objects"] = ["Paris"]
        states = expand_fixed_cardinality_states(row)
        self.assertEqual(len(states), 1)
        self.assertGreater(len(states[0]["actions"]), 1)
        self.assertTrue(all(
            len(state["incumbent_objects"]) == 1 for state in states))

    def test_calibrated_rank_selects_best_identity_at_fixed_size(self):
        row = build_hierarchical_row(graph())
        records, calibrated = evidence(row)
        del records
        predictions, diagnostics = calibrated_rank_predictions(
            [row], calibrated)
        self.assertEqual(predictions[0]["ObjectEntities"], ["London"])
        self.assertEqual(
            len(predictions[0]["ObjectEntities"]),
            len(row["incumbent_objects"]),
        )
        self.assertTrue(diagnostics[0]["changed"])

    def test_counterfactual_states_are_label_free_and_deduplicated(self):
        source = graph()
        source["agent_outputs"] = {
            "qwen_recall": ["Paris"],
            "gemma_independent": ["London"],
        }
        source["proposal_routes"]["gemma:independent"]["objects"] = [
            "London"]
        row = build_hierarchical_row(source)
        states = counterfactual_incumbents(row)
        self.assertEqual(
            {tuple(item["objects"]) for item in states},
            {("Paris",), ("London",)},
        )
        london = next(
            item for item in states if item["objects"] == ["London"])
        self.assertGreaterEqual(len(london["origins"]), 2)
        self.assertIn("agent:gemma_independent", london["origins"])
        self.assertIn("route:gemma:independent", london["origins"])

    def test_rank_gate_features_compare_proposal_with_incumbent(self):
        row = build_hierarchical_row(graph())
        records, calibrated = evidence(row)
        values = rank_gate_features(
            row, ["Paris"], ["London"],
            calibrated, calibrated, records)
        self.assertEqual(len(values), len(RANK_GATE_FEATURE_NAMES))
        self.assertTrue(all(value == value for value in values))
        self.assertGreater(
            values[RANK_GATE_FEATURE_NAMES.index(
                "rank_mean_advantage")],
            0.0,
        )

    def test_direct_utility_uses_dense_incumbent_relative_actions(self):
        rows, records, calibrated, gold = [], {}, {}, {}
        for subject, answer in (
            ("Subject A", "London"), ("Subject B", "Paris")
        ):
            source = graph()
            source["SubjectEntity"] = subject
            row = build_hierarchical_row(source)
            row_records, row_calibrated = evidence(row)
            rows.append(row)
            records.update(row_records)
            calibrated.update(row_calibrated)
            gold[(subject, row["Relation"])] = {
                "SubjectEntity": subject,
                "Relation": row["Relation"],
                "ObjectEntities": [answer],
            }
        model, detail = fit_direct_utility_selector(
            rows, gold, calibrated, records)
        predictions, diagnostics = direct_utility_predictions(
            model, rows, calibrated, records)
        self.assertEqual(detail["target"], "row_f1_action_minus_incumbent")
        self.assertFalse(detail["hurdle_model"])
        self.assertEqual(detail["actions"], 2)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(
            len(item["ObjectEntities"]) == 1 for item in predictions))
        self.assertTrue(all(
            "predicted_expected_f1_delta" in item["trace"][0]
            for item in diagnostics))

    def test_set_aware_ranker_is_permutation_equivariant_and_serializable(
        self,
    ):
        model = SetAwareRanker(("a", "b"), hidden_size=4).fit(
            [
                ([[2.0, 0.0], [0.0, 1.0]], [1.0, 0.0]),
                ([[0.0, 2.0], [1.0, 0.0]], [1.0, 0.0]),
            ],
            seed=7,
        )
        original = model.predict_rows(
            [[[2.0, 0.0], [0.0, 1.0]]])[0]
        permuted = model.predict_rows(
            [[[0.0, 1.0], [2.0, 0.0]]])[0]
        restored = SetAwareRanker.from_dict(model.to_dict())
        roundtrip = restored.predict_rows(
            [[[2.0, 0.0], [0.0, 1.0]]])[0]
        empty = restored.predict_rows([[]])[0]
        self.assertAlmostEqual(original[0], permuted[1], places=10)
        self.assertAlmostEqual(original[1], permuted[0], places=10)
        self.assertAlmostEqual(original[0], roundtrip[0], places=10)
        self.assertAlmostEqual(original[1], roundtrip[1], places=10)
        self.assertEqual(len(empty), 0)


if __name__ == "__main__":
    unittest.main()
