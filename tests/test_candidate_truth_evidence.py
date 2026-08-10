import unittest

from experiments.heterogeneous_agents.candidate_truth_evidence import (
    CHOICES,
    _raw_action_audit,
    _truth_scores,
    component_inventory,
    truth_prompt,
    truth_task,
)
from experiments.heterogeneous_agents.dual_model_validation import QWEN


def graph():
    return {
        "SubjectEntity": "Example Corp",
        "Relation": "companyTradesAtStockExchange",
        "proposal_routes": {
            "qwen:self_consistency": {
                "model_family": "qwen_recall",
                "available": True,
            },
            "gemma:independent": {
                "model_family": "gemma_independent",
                "available": True,
            },
        },
        "relational_graph": {
            "components": [{
                "id": "component:0",
                "representative": "New York Stock Exchange",
                "member_items": ["NYSE", "New York Stock Exchange"],
                "routes": {
                    "qwen:self_consistency": {
                        "max_support_rate": 0.8,
                    },
                },
            }, {
                "id": "component:1",
                "representative": "Nasdaq",
                "member_items": ["Nasdaq"],
                "routes": {
                    "gemma:independent": {
                        "max_support_rate": 1.0,
                    },
                },
            }],
        },
    }


class CandidateTruthEvidenceTest(unittest.TestCase):
    def test_inventory_is_label_free_and_preserves_provenance(self):
        rows = component_inventory([graph()])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["proposer_agents"], ["qwen_recall"])
        self.assertEqual(rows[1]["proposer_agents"], ["gemma_independent"])
        self.assertTrue(all(
            row["contains_labels"] is False
            and row["gold_aware"] is False for row in rows))
        self.assertEqual(len({
            row["component_key"] for row in rows}), len(rows))

    def test_forced_binary_task_balances_code_tokens(self):
        component = component_inventory([graph()])[0]
        task = truth_task(component, QWEN)
        self.assertEqual(tuple(task["choices"]), CHOICES)
        self.assertEqual(len(task["choice_variants"]), 2)
        self.assertNotIn("UNKNOWN", task["choices"])
        true_codes = {
            variant["choice_codes"]["TRUE"]
            for variant in task["choice_variants"]}
        false_codes = {
            variant["choice_codes"]["FALSE"]
            for variant in task["choice_variants"]}
        self.assertEqual(true_codes, {"A", "B"})
        self.assertEqual(false_codes, {"A", "B"})
        self.assertFalse(task["reviewer_is_independent"])
        self.assertTrue(task["prompt_masks_provenance"])
        self.assertNotIn("qwen", task["prompt"].casefold())

    def test_list_prompt_judges_membership_not_completeness(self):
        prompt = truth_prompt(
            subject="Example Corp",
            relation="companyTradesAtStockExchange",
            candidate="Nasdaq",
            codebook={"TRUE": "A", "FALSE": "B"},
        )
        self.assertIn("judge only whether this candidate is a true member", prompt)
        self.assertIn("A = TRUE", prompt)
        self.assertIn("B = FALSE", prompt)

    def test_truth_scores_require_both_memories(self):
        component = component_inventory([graph()])[0]
        qwen = truth_task(component, "qwen_recall")
        gemma = truth_task(component, "gemma_independent")
        responses = {
            qwen["task_id"]: {
                "choice_probabilities": {"TRUE": 0.8, "FALSE": 0.2}},
            gemma["task_id"]: {
                "choice_probabilities": {"TRUE": 0.4, "FALSE": 0.6}},
        }
        scores = _truth_scores(responses, [qwen, gemma])[
            component["component_key"]]
        self.assertAlmostEqual(scores["mean"], 0.6)
        self.assertAlmostEqual(scores["minimum"], 0.4)
        self.assertAlmostEqual(scores["agreement"], 0.6)

    def test_fixed_boundary_action_audit_can_replace_bad_list_incumbent(self):
        source_graph = graph()
        source_graph["baseline_objects"] = ["New York Stock Exchange"]
        inventory = component_inventory([source_graph])
        scores = {
            inventory[0]["component_key"]: {"mean": 0.1},
            inventory[1]["component_key"]: {"mean": 0.9},
        }
        gold = {
            ("Example Corp", "companyTradesAtStockExchange"): {
                "SubjectEntity": "Example Corp",
                "Relation": "companyTradesAtStockExchange",
                "ObjectEntities": [["Nasdaq"]],
            },
        }
        result = _raw_action_audit(
            inventory, scores, [source_graph], gold, "mean")
        self.assertEqual(result["changed"], 1)
        self.assertEqual(result["helped"], 1)
        self.assertAlmostEqual(result["mean_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
