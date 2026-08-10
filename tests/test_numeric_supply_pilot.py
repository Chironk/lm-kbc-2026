import unittest

from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.numeric_supply_pilot import (
    ARMS,
    N_SAMPLES,
    _gold_value,
    _sample_values,
    arm_prompt,
    build_pilot_tasks,
    uncertainty_features,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks


def _graph(subject="Stadium X", relation="hasCapacity"):
    return {
        "SubjectEntity": subject,
        "Relation": relation,
        "baseline_objects": ["10000"],
        "candidates": [
            {"key": "numeric:10000", "item": "10000",
             "routes": {"qwen:self_consistency": {"support_rate": 0.6}}},
            {"key": "numeric:25000", "item": "25000",
             "routes": {"gemma:independent": {"support_rate": 1.0}}},
        ],
        "relational_graph": {"components": [{"id": "c0"}, {"id": "c1"}]},
    }


class NumericSupplyPilotTests(unittest.TestCase):
    def test_prompts_are_label_free_compact_and_relation_specific(self):
        for arm in ARMS:
            area = arm_prompt(arm, "Elba", "hasArea")
            capacity = arm_prompt(arm, "Camp Nou", "hasCapacity")
            self.assertIn("square kilometres", area)
            self.assertIn("spectator capacity", capacity)
            self.assertIn("ANSWER:", area)
            self.assertNotIn("ObjectEntities", area)
        self.assertIn("MAGNITUDE:", arm_prompt("gemma_magnitude", "X", "hasArea"))
        self.assertNotIn("MAGNITUDE:", arm_prompt("gemma_direct", "X", "hasArea"))

    def test_tasks_validate_under_run_agent_contract(self):
        tasks = build_pilot_tasks([_graph(), _graph("Venue Y", "hasArea")], seed=7)
        self.assertEqual(len(tasks[GEMMA]), 4)   # two gemma arms x two rows
        self.assertEqual(len(tasks[QWEN]), 2)    # one qwen arm x two rows
        validate_tasks(tasks[GEMMA], GEMMA)
        validate_tasks(tasks[QWEN], QWEN)
        for task in tasks[GEMMA] + tasks[QWEN]:
            self.assertEqual(task["n_samples"], N_SAMPLES)
            self.assertNotIn("gold", str(task).lower())

    def test_uncertainty_features_are_label_free_and_finite(self):
        features = uncertainty_features(_graph())
        self.assertEqual(features["component_count"], 2.0)
        self.assertEqual(features["gemma_distinct_from_incumbent"], 1.0)
        self.assertGreater(features["log_value_spread"], 0.0)

    def test_sample_parsing_tolerates_magnitude_line_and_prose(self):
        values = _sample_values([
            "MAGNITUDE: 10^4\nANSWER: 41000",
            "ANSWER: 39,500\nThat is my estimate.",
            "I do not know.",
        ], "hasCapacity")
        self.assertEqual(values[0], 41000.0)
        self.assertEqual(values[1], 39500.0)
        self.assertIsNone(values[2])

    def test_gold_value_reads_alias_lists(self):
        self.assertEqual(_gold_value({"ObjectEntities": [["40000"]]}), 40000.0)
        self.assertIsNone(_gold_value({"ObjectEntities": []}))


if __name__ == "__main__":
    unittest.main()
