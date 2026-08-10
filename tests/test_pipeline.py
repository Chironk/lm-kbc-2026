import random
import unittest

from run_inference import (
    aggregate,
    aggregate_city_support_gate,
    aggregate_numeric_cluster,
    build_prompt,
    extract_answer_with_status,
    parse_answer_items,
    parse_null_shots,
    sample_diverse_shot_sets,
    sample_null_stratified_shots,
    sample_subject_balanced_shots,
    select_curated_shots,
)
from validate_artifact import validate_predictions, validate_raw


class ExtractionTests(unittest.TestCase):
    def test_answer_first_prompt_closes_qwen_thinking(self):
        prompt = build_prompt(
            None, "Where did {subject_entity} die?", "personHasCityOfDeath",
            "Example Person", [], response_protocol="answer-first")
        self.assertTrue(prompt.endswith("<think></think>\nANSWER: "))

    def test_legacy_think_prefill_uses_native_opening(self):
        prompt = build_prompt(
            None, "Where did {subject_entity} die?", "personHasCityOfDeath",
            "Example Person", [], response_protocol="legacy-cot",
            legacy_think_prefill=True)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))

    def test_legacy_unclosed_is_not_semantic_none(self):
        answer, status = extract_answer_with_status(
            "<think>Person died in Paris.", "legacy-cot")
        self.assertEqual(answer, "")
        self.assertEqual(status, "unclosed-think")

    def test_answer_first_prefilled_continuation(self):
        answer, status = extract_answer_with_status(
            "New York City\nEVIDENCE: A short fact.", "answer-first")
        self.assertEqual((answer, status), ("New York City", "valid"))

    def test_answer_first_explicit_none(self):
        answer, status = extract_answer_with_status(
            "ANSWER: None\nEVIDENCE: Still living.", "answer-first")
        self.assertEqual(status, "explicit-none")
        self.assertEqual(answer, "None")

    def test_delimiter_preserves_commas_inside_entity(self):
        items = parse_answer_items(
            "Woman, Life, Freedom movement || Guy L. Steele, Jr.",
            "awardWonBy", "answer-first")
        self.assertEqual(items, ["Woman, Life, Freedom movement", "Guy L. Steele, Jr"])


class AggregationTests(unittest.TestCase):
    def test_city_gate_requires_actual_city_support(self):
        answers = ["Paris"] * 5 + ["None"] * 3 + [""] * 2
        self.assertEqual(aggregate_city_support_gate(answers, 5), ["Paris"])
        self.assertEqual(aggregate_city_support_gate(answers, 6), [])

    def test_numeric_cluster_ignores_distant_mode(self):
        answers = ["10000", "10200", "9800", "50000", "60000"]
        self.assertEqual(aggregate_numeric_cluster(answers, 0.20), ["10000"])

    def test_robust_list_denominator_uses_valid_samples(self):
        raw = ["NYSE", "NYSE", "", ""]
        # 2/2 valid answers clears company theta=.35. Format failures do not
        # silently change the meaning of the threshold.
        self.assertEqual(
            aggregate("companyTradesAtStockExchange", "Example Co", raw,
                      response_protocol="answer-first", aggregation_profile="robust"),
            ["NYSE"],
        )

    def test_relation_v1_uses_valid_denominator_only_for_borders(self):
        valid = "<think>recall</think>\nFrance"
        invalid = "<think>unfinished"
        raw = [valid, valid, invalid, invalid, invalid, invalid, invalid, invalid,
               invalid, invalid]
        self.assertEqual(
            aggregate("countryLandBordersCountry", "Belgium", raw,
                      aggregation_profile="relation-v1"),
            ["France"],
        )
        self.assertEqual(
            aggregate("companyTradesAtStockExchange", "Example Co", raw,
                      aggregation_profile="relation-v1"),
            [],
        )

    def test_relation_v1_city_requires_five_named_votes(self):
        paris = "<think>recall</think>\nParis"
        none = "<think>uncertain</think>\nNone"
        self.assertEqual(
            aggregate("personHasCityOfDeath", "Example", [paris] * 5 + [none] * 5,
                      aggregation_profile="relation-v1"),
            ["Paris"],
        )
        self.assertEqual(
            aggregate("personHasCityOfDeath", "Example", [paris] * 4 + [none] * 6,
                      aggregation_profile="relation-v1"),
            [],
        )


class SamplingTests(unittest.TestCase):
    @staticmethod
    def example(subject, is_null, path):
        return {
            "SubjectEntity": subject,
            "Relation": "companyTradesAtStockExchange",
            "Question": f"Where does {subject} trade?",
            "think": "short evidence.",
            "Answer": "None" if is_null else "NYSE",
            "ObjectEntities": [] if is_null else [["New York Stock Exchange"]],
            "path": path,
        }

    def test_subject_balanced_shots_are_unique_and_null_balanced(self):
        pool = []
        for i in range(8):
            for path in range(3):
                pool.append(self.example(f"S{i}", i < 3, path))
        shots = sample_subject_balanced_shots(
            pool, "companyTradesAtStockExchange", 5, random.Random(7))
        self.assertEqual(len({s["SubjectEntity"] for s in shots}), 5)
        self.assertEqual(sum(not s["ObjectEntities"] for s in shots), 2)

    def test_null_stratified_changes_only_class_composition(self):
        pool = [self.example(f"N{i}", True, 0) for i in range(6)]
        pool += [self.example(f"V{i}", False, 0) for i in range(8)]
        spec = parse_null_shots("companyTradesAtStockExchange=3")
        shots = sample_null_stratified_shots(
            pool, "companyTradesAtStockExchange", 5, random.Random(9), spec)
        self.assertEqual(len(shots), 5)
        self.assertEqual(len({id(x) for x in shots}), 5)
        self.assertEqual(sum(not x["ObjectEntities"] for x in shots), 3)

    def test_null_stratified_omission_preserves_legacy_sampling(self):
        pool = [self.example(f"N{i}", True, 0) for i in range(6)]
        pool += [self.example(f"V{i}", False, 0) for i in range(8)]
        seed = 17
        expected = random.Random(seed).sample(pool, 5)
        actual = sample_null_stratified_shots(
            pool, "personHasCityOfDeath", 5, random.Random(seed),
            {"companyTradesAtStockExchange": 2})
        self.assertEqual(actual, expected)

    def test_null_shot_parser_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            parse_null_shots("companyTradesAtStockExchange=6")
        with self.assertRaises(ValueError):
            parse_null_shots("notARelation=2")

    def test_curated_shots_exclude_evaluated_subject(self):
        curated = {"personHasCityOfDeath": [
            {"SubjectEntity": "Target"},
            {"SubjectEntity": "Other 1"},
            {"SubjectEntity": "Other 2"},
        ]}
        shots = select_curated_shots(
            curated, "personHasCityOfDeath", "Target", 2, True)
        self.assertEqual(
            [shot["SubjectEntity"] for shot in shots], ["Other 1", "Other 2"])

    def test_per_sample_shot_sets_are_deterministic_and_diverse(self):
        pool = []
        for i in range(12):
            for path in range(2):
                pool.append(self.example(f"S{i}", i < 4, path))
        first = sample_diverse_shot_sets(
            pool, "companyTradesAtStockExchange", 5, 10, 42)
        second = sample_diverse_shot_sets(
            pool, "companyTradesAtStockExchange", 5, 10, 42)
        signatures = [tuple((x["SubjectEntity"], x["path"]) for x in shots)
                      for shots in first]
        self.assertEqual(signatures, [
            tuple((x["SubjectEntity"], x["path"]) for x in shots)
            for shots in second])
        self.assertEqual(len(set(signatures)), 10)
        self.assertTrue(all(len({x["SubjectEntity"] for x in shots}) == 5
                            for shots in first))


class ValidatorTests(unittest.TestCase):
    def test_missing_key_is_rejected(self):
        reference = [
            {"SubjectEntity": "A", "Relation": "hasArea", "ObjectEntities": []},
            {"SubjectEntity": "B", "Relation": "hasArea", "ObjectEntities": []},
        ]
        predictions = [
            {"SubjectEntity": "A", "Relation": "hasArea", "ObjectEntities": ["1"]}
        ]
        self.assertTrue(validate_predictions(predictions, reference))

    def test_raw_sample_count_is_rejected(self):
        reference = [
            {"SubjectEntity": "A", "Relation": "hasArea", "ObjectEntities": []}
        ]
        raw = [{"SubjectEntity": "A", "Relation": "hasArea", "raw_samples": ["1"]}]
        self.assertTrue(validate_raw(raw, reference, 10))


if __name__ == "__main__":
    unittest.main()


class ClusterPermutationInvarianceTests(unittest.TestCase):
    """Audit P1-6: the densest-cluster aggregator must be a pure function of
    the sample MULTISET, not the arrival order."""

    def test_permutation_invariant(self):
        import random
        from run_inference import aggregate_numeric_cluster
        answers = ["11000", "15000", "39000", "20000", "3000",
                   "15200", "14800", "39500", "100", "15100"]
        base = aggregate_numeric_cluster(answers, 0.30)
        for seed in range(50):
            shuffled = answers[:]
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(aggregate_numeric_cluster(shuffled, 0.30), base)

    def test_exact_tie_resolves_deterministically(self):
        from run_inference import aggregate_numeric_cluster
        # two disjoint clusters with identical support and zero log-spread:
        # the smaller center must win regardless of order.
        a = aggregate_numeric_cluster(["100", "100", "900", "900"], 0.10)
        b = aggregate_numeric_cluster(["900", "900", "100", "100"], 0.10)
        self.assertEqual(a, b)
