import unittest

from experiments.heterogeneous_agents.components.baseline_conditioned_action_review import (
    ALL_FEATURE_NAMES,
    CHOICES,
    _review_features,
    _task,
    comparison_prompt,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import (
    FEATURE_NAMES,
)


class BaselineConditionedActionReviewTest(unittest.TestCase):
    def test_prompt_contains_complete_outputs_and_balances_order(self):
        codebook = {
            "KEEP_CURRENT": "A",
            "USE_ALTERNATIVE": "B",
            "UNCERTAIN": "C",
        }
        current_first = comparison_prompt(
            "Example Stadium", "hasCapacity", ["10000"], ["12000"],
            codebook, alternative_first=False)
        alternative_first = comparison_prompt(
            "Example Stadium", "hasCapacity", ["10000"], ["12000"],
            codebook, alternative_first=True)
        self.assertIn('CURRENT OUTPUT: ["10000"]', current_first)
        self.assertIn('ALTERNATIVE OUTPUT: ["12000"]', current_first)
        self.assertLess(
            current_first.index("CURRENT OUTPUT"),
            current_first.index("ALTERNATIVE OUTPUT"))
        self.assertLess(
            alternative_first.index("ALTERNATIVE OUTPUT"),
            alternative_first.index("CURRENT OUTPUT"))

    def test_task_has_three_balanced_variants_and_no_labels(self):
        graph = {
            "SubjectEntity": "Example Stadium",
            "Relation": "hasCapacity",
            "incumbent_objects": ["10000"],
            "_source": {"input_index": 7},
        }
        action = {
            "id": "action:1",
            "action_type": "REPLACE",
            "objects": ["12000"],
        }
        task = _task(graph, action, "qwen_recall")
        self.assertEqual(task["choices"], list(CHOICES))
        self.assertEqual(len(task["choice_variants"]), 3)
        self.assertFalse(task["contains_labels"])
        self.assertFalse(task["gold_aware"])
        self.assertNotIn("qwen_recall", task["prompt"].lower())
        orders = {
            variant["prompt"].index("ALTERNATIVE OUTPUT")
            < variant["prompt"].index("CURRENT OUTPUT")
            for variant in task["choice_variants"]
        }
        self.assertEqual(orders, {False, True})

    def test_evidence_arms_keep_feature_schema_fixed(self):
        qwen = {
            "KEEP_CURRENT": 0.2,
            "USE_ALTERNATIVE": 0.7,
            "UNCERTAIN": 0.1,
        }
        gemma = {
            "KEEP_CURRENT": 0.6,
            "USE_ALTERNATIVE": 0.3,
            "UNCERTAIN": 0.1,
        }
        for arm in ("qwen", "gemma", "mean", "dual"):
            values = _review_features(
                "hasCapacity", qwen, gemma, arm)
            self.assertEqual(
                len(values),
                len(ALL_FEATURE_NAMES) - len(FEATURE_NAMES))
        dual = _review_features(
            "hasCapacity", qwen, gemma, "dual")
        self.assertEqual(dual[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
