import unittest

from architecture_ensemble import company_prediction, city_prediction


def sample(answer):
    return f"<think>independent recall</think>\n{answer}"


class ArchitectureEnsembleTests(unittest.TestCase):
    def test_company_only_promotes_system2_with_three_system1_votes(self):
        raw = [sample("NYSE")] * 3 + [sample("None")] * 7
        self.assertEqual(company_prediction("Example", raw, ["NYSE"], False), [])
        self.assertEqual(company_prediction("Example", raw, ["NYSE"], True), ["NYSE"])

        weak = [sample("NYSE")] * 2 + [sample("None")] * 8
        self.assertEqual(company_prediction("Example", weak, ["NYSE"], True), [])

    def test_city_only_promotes_system2_when_it_confirms_top_candidate(self):
        raw = ([sample("Paris")] * 3 + [sample("London")] * 2
               + [sample("None")] * 5)
        self.assertEqual(city_prediction("Example", raw, ["Paris"], False), [])
        self.assertEqual(city_prediction("Example", raw, ["Paris"], True), ["Paris"])
        self.assertEqual(city_prediction("Example", raw, ["London"], True), [])

    def test_city_k5_does_not_need_system2(self):
        raw = [sample("Paris")] * 5 + [sample("None")] * 5
        self.assertEqual(city_prediction("Example", raw, [], True), ["Paris"])


if __name__ == "__main__":
    unittest.main()
