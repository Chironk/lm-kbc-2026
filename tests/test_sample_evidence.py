import unittest

from sample_evidence import (ENTITY, EXPLICIT_ABSTENTION, INVALID,
                             candidate_table, classify_sample,
                             evidence_summary, validate_recorded_statuses)


class SampleEvidenceTests(unittest.TestCase):
    def test_explicit_none_is_not_invalid(self):
        sample = classify_sample("<think>unknown</think>\nNone",
                                 "personHasCityOfDeath")
        self.assertEqual(sample.kind, EXPLICIT_ABSTENTION)
        self.assertEqual(sample.parser_status, "explicit-none")

    def test_unclosed_and_empty_are_invalid(self):
        self.assertEqual(
            classify_sample("<think>still reasoning", "personHasCityOfDeath").kind,
            INVALID)
        self.assertEqual(classify_sample("", "personHasCityOfDeath").kind, INVALID)

    def test_entity_items_and_per_sample_deduplication(self):
        samples = [
            classify_sample("<think>x</think>\nNYSE, NYSE",
                            "companyTradesAtStockExchange"),
            classify_sample("<think>x</think>\nNew York Stock Exchange",
                            "companyTradesAtStockExchange"),
        ]
        self.assertTrue(all(sample.kind == ENTITY for sample in samples))
        table = candidate_table(samples)
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]["votes"], 2)

    def test_summary_keeps_three_outcomes_separate(self):
        summary = evidence_summary(
            ["<think>x</think>\nParis", "<think>x</think>\nNone", "<think>x"],
            "personHasCityOfDeath")
        self.assertEqual(summary["entity_samples"], 1)
        self.assertEqual(summary["explicit_abstentions"], 1)
        self.assertEqual(summary["invalid_samples"], 1)

    def test_recorded_status_mismatch_is_detected(self):
        errors = validate_recorded_statuses(
            ["<think>x</think>\nNone"], ["valid"], "legacy-cot")
        self.assertEqual(len(errors), 1)
        self.assertIn("recomputed='explicit-none'", errors[0])

    def test_recovery_prose_uses_final_bare_answer_line(self):
        sample = classify_sample(
            "<think></think>\nThis company is publicly traded in the United "
            "States and its shares trade on a major technology stock exchange.\nNASDAQ",
            "companyTradesAtStockExchange")
        self.assertEqual(sample.kind, ENTITY)
        self.assertEqual(sample.answer, "NASDAQ")
        self.assertEqual(sample.items, ("NASDAQ",))

    def test_company_final_prose_extracts_explicit_listing_only(self):
        sample = classify_sample(
            "<think></think>\nThe bank is publicly traded and its shares are "
            "listed on the Dubai Financial Market.",
            "companyTradesAtStockExchange")
        self.assertEqual(sample.items, ("Dubai Financial Market",))
        negative = classify_sample(
            "<think></think>\nThe company was acquired and delisted from the stock market.",
            "companyTradesAtStockExchange")
        self.assertNotIn("stock market", [item.lower() for item in negative.items])


if __name__ == "__main__":
    unittest.main()
