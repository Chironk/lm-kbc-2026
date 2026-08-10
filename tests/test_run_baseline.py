import unittest

from run_baseline import order_raw_records
from models.baseline_qwen import BaselineQwenModel


class EchoTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


def prompt_model(policy):
    model = BaselineQwenModel.__new__(BaselineQwenModel)
    model.reasoning_demo_policy = policy
    model.exclude_target_from_shots = False
    model.enable_thinking = False
    model.tokenizer = EchoTokenizer()
    model.prompt_templates = {"personHasCityOfDeath": "Where did {subject_entity} die?"}
    model.few_shot_examples = {"personHasCityOfDeath": [
        {"SubjectEntity": "NoReason", "ObjectEntities": [["Paris"]]},
        {"SubjectEntity": "HasReason", "ObjectEntities": [["Rome"]],
         "Reason": "A curated factual reason."},
    ]}
    return model


class BaselineRawOrderingTests(unittest.TestCase):
    def test_relation_grouped_traces_are_restored_to_input_order(self):
        inputs = [
            {"SubjectEntity": "A", "Relation": "hasArea"},
            {"SubjectEntity": "B", "Relation": "personHasCityOfDeath"},
            {"SubjectEntity": "C", "Relation": "hasArea"},
        ]
        raw = [
            {"SubjectEntity": "A", "Relation": "hasArea"},
            {"SubjectEntity": "C", "Relation": "hasArea"},
            {"SubjectEntity": "B", "Relation": "personHasCityOfDeath"},
        ]
        ordered = order_raw_records(raw, inputs)
        self.assertEqual(
            [(row["SubjectEntity"], row["Relation"]) for row in ordered],
            [(row["SubjectEntity"], row["Relation"]) for row in inputs])

    def test_duplicate_trace_keys_fail_closed(self):
        inputs = [{"SubjectEntity": "A", "Relation": "hasArea"}]
        raw = [
            {"SubjectEntity": "A", "Relation": "hasArea"},
            {"SubjectEntity": "A", "Relation": "hasArea"},
        ]
        with self.assertRaises(RuntimeError):
            order_raw_records(raw, inputs)


class BaselineReasoningDemonstrationTests(unittest.TestCase):
    def test_frozen_policy_reports_only_actually_rendered_reasoning_shots(self):
        model = prompt_model("require-curated-reason")
        prompt = model._build_prompt(
            "Target", "personHasCityOfDeath", with_reasoning=True)
        self.assertNotIn("NoReason", prompt)
        self.assertIn("HasReason", prompt)
        self.assertEqual(
            model._shot_subjects("personHasCityOfDeath", "Target"), ["HasReason"])

    def test_answer_only_ablation_keeps_reasonless_examples_without_faking_reason(self):
        model = prompt_model("answer-only")
        prompt = model._build_prompt(
            "Target", "personHasCityOfDeath", with_reasoning=True)
        self.assertIn("NoReason", prompt)
        self.assertIn("A: Paris", prompt)
        self.assertNotIn("Reason: this is", prompt)
        self.assertEqual(
            model._shot_subjects("personHasCityOfDeath", "Target"),
            ["NoReason", "HasReason"])


if __name__ == "__main__":
    unittest.main()
