import unittest

from architecture_candidate_v3 import (
    city_prediction, company_evidence, company_prediction,
    plausible_exchange_candidate,
)


def sample(answer):
    return f"<think>independent recall</think>\n{answer}"


class ArchitectureCandidateV3Tests(unittest.TestCase):
    def test_company_uses_structured_candidates_and_k4_k3_rules(self):
        samples = ([sample("NASDAQ")] * 3 + [sample("NYSE")] * 4
                   + [sample("None")] * 3)
        evidence = company_evidence(samples, ["NASDAQ"])
        self.assertEqual(evidence["baseline_objects"], ["NYSE", "NASDAQ"])
        self.assertEqual(company_prediction("Acme", samples, ["NASDAQ"]),
                         ["NYSE", "NASDAQ"])

    def test_company_rejects_explanatory_prose_candidates(self):
        self.assertFalse(plausible_exchange_candidate({
            "item": "The company is publicly traded on the NASDAQ exchange"}))
        self.assertTrue(plausible_exchange_candidate({"item": "NASDAQ"}))

    def test_city_system2_can_select_a_co_top_not_first_seen(self):
        samples = ([sample("London")] * 2 + [sample("Paris")] * 2
                   + [sample("None")] * 6)
        self.assertEqual(city_prediction("Person", samples, ["Paris"]),
                         ["Paris"])
        self.assertEqual(city_prediction("Person", samples, ["Rome"]), [])

    def test_city_without_corroboration_keeps_k6(self):
        samples = [sample("Paris")] * 5 + [sample("None")] * 5
        self.assertEqual(city_prediction(
            "Person", samples, [], corroborate=False), [])
        self.assertEqual(city_prediction(
            "Person", samples + [sample("Paris")], [], corroborate=False),
            ["Paris"])


if __name__ == "__main__":
    unittest.main()
