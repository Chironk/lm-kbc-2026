import unittest

from architecture_candidate_v2 import build_rows, city_prediction


def sample(answer):
    return f"<think>independent recall</think>\n{answer}"


def raw(subject, relation, answers):
    return {"SubjectEntity": subject, "Relation": relation,
            "raw_samples": [sample(answer) for answer in answers]}


class ArchitectureCandidateV2Tests(unittest.TestCase):
    def test_city_uses_k6_and_conservative_system2_promotion(self):
        five = [sample("Paris")] * 5 + [sample("None")] * 5
        self.assertEqual(city_prediction("Person", five, [], True), [])
        self.assertEqual(city_prediction("Person", five, ["Paris"], True), ["Paris"])
        self.assertEqual(city_prediction("Person", five, ["London"], True), [])

        six = [sample("Paris")] * 6 + [sample("None")] * 4
        self.assertEqual(city_prediction("Person", six, [], True), ["Paris"])

    def test_composer_routes_every_relation_by_frozen_policy(self):
        reference = [
            {"SubjectEntity": "Country", "Relation": "countryLandBordersCountry",
             "ObjectEntities": []},
            {"SubjectEntity": "Company", "Relation": "companyTradesAtStockExchange",
             "ObjectEntities": []},
            {"SubjectEntity": "Person", "Relation": "personHasCityOfDeath",
             "ObjectEntities": []},
            {"SubjectEntity": "Place", "Relation": "hasArea", "ObjectEntities": []},
            {"SubjectEntity": "Venue", "Relation": "hasCapacity", "ObjectEntities": []},
            {"SubjectEntity": "Award", "Relation": "awardWonBy", "ObjectEntities": []},
        ]
        border = [raw("Country", "countryLandBordersCountry", ["Neighbor"] * 3 + ["None"] * 7)]
        fp16 = [
            raw("Company", "companyTradesAtStockExchange", ["NYSE"] * 3 + ["None"] * 7),
            raw("Person", "personHasCityOfDeath", ["Paris"] * 3 + ["None"] * 7),
            raw("Place", "hasArea", ["100", "101", "102", "500", "600"] * 2),
            raw("Venue", "hasCapacity", ["10000", "10100", "10200", "50000", "60000"] * 2),
        ]
        system2 = [
            {"SubjectEntity": "Company", "Relation": "companyTradesAtStockExchange",
             "ObjectEntities": ["NYSE"]},
            {"SubjectEntity": "Person", "Relation": "personHasCityOfDeath",
             "ObjectEntities": ["Paris"]},
        ]
        awards = [{"SubjectEntity": "Award", "Relation": "awardWonBy",
                   "ObjectEntities": ["Winner"]}]
        result = {row["Relation"]: row["ObjectEntities"] for row in
                  build_rows(reference, border, fp16, system2, awards)}
        self.assertEqual(result["countryLandBordersCountry"], ["Neighbor"])
        self.assertEqual(result["companyTradesAtStockExchange"], ["NYSE"])
        self.assertEqual(result["personHasCityOfDeath"], ["Paris"])
        self.assertEqual(result["hasArea"], ["102"])
        self.assertEqual(result["hasCapacity"], ["10100"])
        self.assertEqual(result["awardWonBy"], ["Winner"])

    def test_missing_required_source_fails_closed(self):
        reference = [{"SubjectEntity": "Person", "Relation": "personHasCityOfDeath",
                      "ObjectEntities": []}]
        with self.assertRaisesRegex(ValueError, "Missing 1 required rows"):
            build_rows(reference, [], [], [], [])


if __name__ == "__main__":
    unittest.main()
