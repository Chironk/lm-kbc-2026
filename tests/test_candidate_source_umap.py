import unittest

from experiments.heterogeneous_agents.analysis.candidate_source_umap import (
    build_candidate_records,
    candidate_matches_gold,
    component_families,
    component_selected,
    semantic_text,
    source_category,
)


class CandidateSourceUmapTest(unittest.TestCase):
    def test_old_and_new_route_schemas_resolve_identically(self):
        old = {
            "routes": {
                "qwen:self_consistency": {},
                "ministral:cot5_cap40_n10": {},
            }
        }
        new = {
            "routes": {
                "qwen:self_consistency": {"model_family": "qwen_recall"},
                "ministral:cot5_cap40_n10": {
                    "model_family": "ministral_independent"},
            }
        }
        self.assertEqual(component_families(old), ("ministral", "qwen"))
        self.assertEqual(component_families(old), component_families(new))
        self.assertEqual(source_category(component_families(old)), "cross_model")

    def test_official_style_string_alias_and_numeric_tolerance(self):
        self.assertTrue(candidate_matches_gold(
            "companyTradesAtStockExchange",
            ["New York Stock Exchange"],
            [["NYSE", "New York Stock Exchange"]],
        ))
        self.assertTrue(candidate_matches_gold(
            "hasCapacity", ["10,400"], [["10000"]],
        ))
        self.assertFalse(candidate_matches_gold(
            "hasCapacity", ["10,600"], [["10000"]],
        ))

    def test_decoder_selection_uses_component_aliases(self):
        self.assertTrue(component_selected(
            "companyTradesAtStockExchange",
            ["NYSE", "New York Stock Exchange"],
            ["New York Stock Exchange"],
        ))
        self.assertTrue(component_selected(
            "hasArea", ["100", "104"], ["103"],
        ))

    def test_projection_text_has_no_labels_or_provenance(self):
        value = semantic_text(
            "Example Corp.", "companyTradesAtStockExchange", "NASDAQ")
        self.assertEqual(
            value,
            "subject: Example Corp.. relation: stock exchanges where the company "
            "trades. candidate: NASDAQ.",
        )
        self.assertNotIn("qwen", value.casefold())
        self.assertNotIn("correct", value.casefold())
        self.assertNotIn("selected", value.casefold())

    def test_records_attach_gold_and_selection_after_text(self):
        graph = [{
            "SubjectEntity": "Example Corp.",
            "Relation": "companyTradesAtStockExchange",
            "input_index": 0,
            "relational_graph": {
                "components": [{
                    "id": "component:0",
                    "representative": "NASDAQ",
                    "member_items": ["NASDAQ", "Nasdaq Stock Market"],
                    "routes": {"ministral:cot5_cap40_n10": {}},
                }]
            },
        }]
        gold = [{
            "SubjectEntity": "Example Corp.",
            "Relation": "companyTradesAtStockExchange",
            "ObjectEntities": [["NASDAQ", "Nasdaq Stock Market"]],
        }]
        predictions = [{
            "SubjectEntity": "Example Corp.",
            "Relation": "companyTradesAtStockExchange",
            "ObjectEntities": ["Nasdaq Stock Market"],
        }]
        records = build_candidate_records("dev", graph, gold, predictions)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_category"], "ministral_only")
        self.assertTrue(records[0]["correct"])
        self.assertTrue(records[0]["selected_by_decoder"])


if __name__ == "__main__":
    unittest.main()
