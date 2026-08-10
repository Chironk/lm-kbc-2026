import unittest

from experiments.heterogeneous_agents.dual_model_validation import (
    PROMPT_POLICY,
    apply_prompt_policy,
    candidate_oracle,
    complementarity_diagnostics,
    complete_answer_oracle,
    fuse_objects,
    proposal_only_prediction,
)
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import build_agent_tasks


QWEN = {"id": "qwen_recall", "role": "synthetic_cot_recall", "synthetic_shots": 2}
GEMMA = {"id": "gemma_independent", "role": "synthetic_cot_recall", "synthetic_shots": 2}


def synthetic_pool(relation, count=8):
    return [
        {"SubjectEntity": f"Example {index}", "Relation": relation,
         "Question": f"Question {index}?", "think": f"Reason {index}",
         "Answer": f"Answer {index}"}
        for index in range(count)
    ]


class PromptPolicyTests(unittest.TestCase):
    def test_area_is_direct_and_company_is_minimum_overlap(self):
        rows = [
            {"SubjectEntity": "Target Area", "Relation": "hasArea"},
            {"SubjectEntity": "Target Company", "Relation": "companyTradesAtStockExchange"},
            {"SubjectEntity": "Target Border", "Relation": "countryLandBordersCountry"},
        ]
        synthetic = {row["Relation"]: synthetic_pool(row["Relation"]) for row in rows}
        tasks = build_agent_tasks(rows, GEMMA, synthetic, seed=9, n_proposals=3)
        diagnostics = apply_prompt_policy(
            tasks, gemma_agent=GEMMA, qwen_agent=QWEN, synthetic=synthetic, seed=9)
        proposals = {task["relation"]: task for task in tasks if task["phase"] == "propose"}
        area = proposals["hasArea"]
        self.assertEqual((area["prompt_policy"], area["shot_subjects"]), ("direct", []))
        self.assertNotIn("PRIVATE RECALL DEMONSTRATIONS", area["prompt"])
        company = proposals["companyTradesAtStockExchange"]
        self.assertEqual(company["prompt_policy"], "disjoint_cot5")
        self.assertFalse(set(company["shot_subjects"]) &
                         set(company["reference_qwen_shot_subjects"]))
        border = proposals["countryLandBordersCountry"]
        self.assertEqual(border["shot_subjects"], border["reference_qwen_shot_subjects"])
        self.assertTrue(diagnostics)

    def test_policy_covers_all_relations(self):
        self.assertEqual(set(PROMPT_POLICY), {
            "countryLandBordersCountry", "companyTradesAtStockExchange",
            "personHasCityOfDeath", "hasArea", "hasCapacity", "awardWonBy"})
        self.assertEqual(PROMPT_POLICY["personHasCityOfDeath"], "shared_cot5")


class SimpleFusionTests(unittest.TestCase):
    def test_string_union_and_intersection_are_canonical(self):
        relation = "companyTradesAtStockExchange"
        self.assertEqual(
            fuse_objects(["NYSE"], ["New York Stock Exchange", "NASDAQ"], relation, "union"),
            ["NYSE", "NASDAQ"])
        self.assertEqual(
            fuse_objects(["NYSE"], ["New York Stock Exchange", "NASDAQ"], relation,
                         "intersection"), ["NYSE"])

    def test_numeric_fusion_is_plain_median(self):
        self.assertEqual(fuse_objects(["100"], ["120"], "hasArea", "union"), ["110"])
        self.assertEqual(fuse_objects(["100"], [], "hasArea", "intersection"), [])

    def test_proposal_only_control_uses_no_blind_commitment(self):
        city = {"relation": "personHasCityOfDeath",
                "generations": ["ANSWER: Paris", "ANSWER: None", "ANSWER: None"]}
        self.assertEqual(proposal_only_prediction(city), [])
        city["generations"] = ["ANSWER: Paris", "ANSWER: Paris", "ANSWER: None"]
        self.assertEqual(proposal_only_prediction(city), ["Paris"])
        area = {"relation": "hasArea",
                "generations": ["ANSWER: 100", "ANSWER: 110", "ANSWER: 1000"]}
        self.assertEqual(proposal_only_prediction(area), ["110"])

    def test_gold_aware_oracles_dominate_each_source(self):
        gold = [
            {"SubjectEntity": "A", "Relation": "hasArea", "ObjectEntities": [["100"]]},
            {"SubjectEntity": "B", "Relation": "personHasCityOfDeath",
             "ObjectEntities": [["Paris"]]},
        ]
        qwen = [
            {"SubjectEntity": "A", "Relation": "hasArea", "ObjectEntities": ["100"]},
            {"SubjectEntity": "B", "Relation": "personHasCityOfDeath", "ObjectEntities": []},
        ]
        gemma = [
            {"SubjectEntity": "A", "Relation": "hasArea", "ObjectEntities": ["1000"]},
            {"SubjectEntity": "B", "Relation": "personHasCityOfDeath",
             "ObjectEntities": ["Paris"]},
        ]
        for oracle in (complete_answer_oracle(qwen, gemma, gold),
                       candidate_oracle(qwen, gemma, gold)):
            self.assertGreaterEqual(score(oracle, gold)["*** All Relations ***"],
                                    score(qwen, gold)["*** All Relations ***"])
            self.assertGreaterEqual(score(oracle, gold)["*** All Relations ***"],
                                    score(gemma, gold)["*** All Relations ***"])

    def test_complementarity_separates_unique_sources(self):
        gold = [
            {"SubjectEntity": "A", "Relation": "personHasCityOfDeath",
             "ObjectEntities": [["Paris"]]},
            {"SubjectEntity": "B", "Relation": "personHasCityOfDeath",
             "ObjectEntities": [["Rome"]]},
            {"SubjectEntity": "C", "Relation": "personHasCityOfDeath",
             "ObjectEntities": []},
        ]
        result = complementarity_diagnostics(
            {("A", "personHasCityOfDeath"): ["Paris"]},
            {("B", "personHasCityOfDeath"): ["Rome"]}, gold)
        self.assertEqual(result["nonempty_gold_rows"], 2)
        self.assertEqual(result["overall"], {"qwen_only": 1, "gemma_only": 1})


if __name__ == "__main__":
    unittest.main()
