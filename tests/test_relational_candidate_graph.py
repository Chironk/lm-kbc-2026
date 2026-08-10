import unittest

from experiments.heterogeneous_agents.relational_candidate_graph import (
    augment_relational_graph,
    collapse_prediction,
    component_actions,
    equivalence_rule,
)
from experiments.heterogeneous_agents.route_aware_candidate_graph import (
    ROUTE_GEMMA,
    ROUTE_QWEN_SC,
    ROUTE_QWEN_SYSTEM2,
)


def node(key, item, routes):
    return {
        "key": key,
        "item": item,
        "sources": {},
        "selected_by": {
            "qwen_recall": False, "gemma_independent": False},
        "routes": {
            route: {
                "support_rate": support, "selected": selected}
            for route, support, selected in routes},
    }


def graph(relation, candidates):
    return {
        "schema": "heterogeneous-memory-graph-row-v1",
        "SubjectEntity": "S",
        "Relation": relation,
        "candidates": candidates,
        "proposal_routes": {
            ROUTE_QWEN_SC: {
                "available": True, "model_family": "qwen_recall",
                "n_samples": 10},
            ROUTE_QWEN_SYSTEM2: {
                "available": True, "model_family": "qwen_recall",
                "n_samples": 1},
            ROUTE_GEMMA: {
                "available": True, "model_family": "gemma_independent",
                "n_samples": 1},
        },
    }


class RelationalCandidateGraphTest(unittest.TestCase):
    def test_high_precision_acronym_equivalence(self):
        self.assertEqual(
            equivalence_rule(
                "BSE", "Bombay Stock Exchange",
                "companyTradesAtStockExchange")[0],
            "explicit_acronym")
        self.assertIsNone(equivalence_rule(
            "NYSE", "Nasdaq", "companyTradesAtStockExchange"))

    def test_ambiguous_acronym_does_not_merge_distinct_entities(self):
        row = augment_relational_graph(graph(
            "companyTradesAtStockExchange", [
                node("tse", "TSE", [
                    (ROUTE_QWEN_SC, 0.2, False)]),
                node("tokyo stock exchange", "Tokyo Stock Exchange", [
                    (ROUTE_QWEN_SC, 0.6, True)]),
                node("toronto stock exchange", "Toronto Stock Exchange", [
                    (ROUTE_GEMMA, 1.0, True)]),
            ]))
        relational = row["relational_graph"]
        self.assertEqual(len(relational["components"]), 3)
        self.assertEqual(
            relational["statistics"]["blocked_equivalence_count"], 2)
        self.assertEqual(
            {item["reason"] for item in relational["blocked_equivalences"]},
            {"ambiguous_row_level_expansion"})
        self.assertEqual(
            collapse_prediction(
                row, ["Tokyo Stock Exchange", "Toronto Stock Exchange"]),
            ["Tokyo Stock Exchange", "Toronto Stock Exchange"])

    def test_numeric_official_tolerance_cluster(self):
        self.assertIsNotNone(equivalence_rule(
            "100", "104", "hasCapacity"))
        self.assertIsNone(equivalence_rule(
            "100", "110", "hasCapacity"))

    def test_numeric_components_do_not_tolerance_chain(self):
        row = augment_relational_graph(graph(
            "hasCapacity", [
                node("numeric:100", "100", [
                    (ROUTE_QWEN_SC, 0.2, False)]),
                node("numeric:104", "104", [
                    (ROUTE_QWEN_SC, 0.2, False)]),
                node("numeric:108", "108", [
                    (ROUTE_GEMMA, 1.0, True)]),
            ]))
        components = row["relational_graph"]["components"]
        self.assertEqual(len(components), 2)
        self.assertEqual(components[0]["member_items"], ["100", "104"])
        self.assertEqual(components[1]["member_items"], ["108"])

        # Lookup must prefer exact membership in the later component instead
        # of capturing 108 through its pairwise 104~108 tolerance match.
        self.assertEqual(
            collapse_prediction(row, ["108"]),
            [components[1]["representative"]])

    def test_graph_has_real_typed_edges_and_route_dependence(self):
        row = augment_relational_graph(graph(
            "personHasCityOfDeath", [
                node("paris", "Paris", [
                    (ROUTE_QWEN_SC, 0.6, True)]),
                node("london", "London", [
                    (ROUTE_GEMMA, 1.0, True)]),
            ]))
        edge_types = [
            edge["edge_type"]
            for edge in row["relational_graph"]["edges"]]
        self.assertIn("member_of", edge_types)
        self.assertIn("proposed_by", edge_types)
        self.assertIn("dependent_with", edge_types)
        self.assertEqual(edge_types.count("contradicts"), 3)

    def test_alias_component_collapses_duplicate_list_prediction(self):
        row = augment_relational_graph(graph(
            "companyTradesAtStockExchange", [
                node("bse", "BSE", [
                    (ROUTE_QWEN_SC, 0.2, False)]),
                node("bombay stock exchange", "Bombay Stock Exchange", [
                    (ROUTE_GEMMA, 1.0, True)]),
            ]))
        self.assertEqual(
            len(row["relational_graph"]["components"]), 1)
        self.assertEqual(
            collapse_prediction(row, ["BSE", "Bombay Stock Exchange"]),
            ["Bombay Stock Exchange"])

    def test_component_support_unions_generation_provenance(self):
        first = node("bse", "BSE", [
            (ROUTE_QWEN_SC, 0.2, False)])
        second = node("bombay stock exchange", "Bombay Stock Exchange", [
            (ROUTE_QWEN_SC, 0.2, False)])
        first["routes"][ROUTE_QWEN_SC].update({
            "samples": 10, "generation_indices": [0, 2]})
        second["routes"][ROUTE_QWEN_SC].update({
            "samples": 10, "generation_indices": [2, 4]})
        row = augment_relational_graph(graph(
            "companyTradesAtStockExchange", [first, second]))
        route = row["relational_graph"]["components"][0]["routes"][
            ROUTE_QWEN_SC]
        self.assertEqual(route["generation_indices"], [0, 2, 4])
        self.assertEqual(route["distinct_generation_support"], 3)
        self.assertAlmostEqual(route["component_support_rate"], 0.3)

    def test_co_support_edge_tracks_same_generation_overlap(self):
        first = node("alpha", "Alpha", [
            (ROUTE_QWEN_SC, 0.3, False)])
        second = node("beta", "Beta", [
            (ROUTE_QWEN_SC, 0.3, False)])
        first["routes"][ROUTE_QWEN_SC].update({
            "samples": 10, "generation_indices": [0, 2, 4]})
        second["routes"][ROUTE_QWEN_SC].update({
            "samples": 10, "generation_indices": [2, 4, 6]})
        row = augment_relational_graph(graph(
            "awardWonBy", [first, second]))
        edge = next(
            value for value in row["relational_graph"]["edges"]
            if value["edge_type"] == "co_supported_with")
        self.assertEqual(edge["cooccurrence_count"], 2)
        self.assertAlmostEqual(edge["cooccurrence_rate"], 0.2)
        self.assertEqual(
            edge["cooccurrence_by_route"][ROUTE_QWEN_SC][
                "generation_indices"],
            [2, 4],
        )

    def test_component_actions_include_add_drop_and_collapse(self):
        row = augment_relational_graph(graph(
            "companyTradesAtStockExchange", [
                node("bse", "BSE", [
                    (ROUTE_QWEN_SC, 0.2, False)]),
                node("bombay stock exchange", "Bombay Stock Exchange", [
                    (ROUTE_GEMMA, 1.0, True)]),
                node("nyse", "NYSE", [
                    (ROUTE_GEMMA, 1.0, True)]),
            ]))
        actions = component_actions(
            row, ["BSE", "Bombay Stock Exchange"])
        canonical = {tuple(action) for action in actions}
        self.assertIn(("Bombay Stock Exchange",), canonical)
        self.assertIn(("Bombay Stock Exchange", "NYSE"), canonical)
        self.assertIn((), canonical)


if __name__ == "__main__":
    unittest.main()
