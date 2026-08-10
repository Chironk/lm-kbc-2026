import unittest

from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.route_aware_candidate_graph import (
    ROUTE_GEMMA,
    ROUTE_QWEN_SC,
    ROUTE_QWEN_SYSTEM2,
    augment_graph,
    normalize_route_selection,
)


def graph():
    return {
        "schema": "heterogeneous-memory-graph-row-v1",
        "SubjectEntity": "Person",
        "Relation": "personHasCityOfDeath",
        "agents": {
            QWEN: {"n_samples": 10},
            GEMMA: {"n_samples": 1},
        },
        "agent_outputs": {
            QWEN: [],
            GEMMA: ["Lyon"],
        },
        "candidates": [
            {
                "key": "paris",
                "item": "Paris",
                "type": "string",
                "sources": {
                    QWEN: {
                        "support": 4, "samples": 10, "support_rate": 0.4}},
                "selected_by": {QWEN: False, GEMMA: False},
            },
            {
                "key": "lyon",
                "item": "Lyon",
                "type": "string",
                "sources": {
                    GEMMA: {
                        "support": 1, "samples": 1, "support_rate": 1.0}},
                "selected_by": {QWEN: False, GEMMA: True},
            },
        ],
    }


class RouteAwareCandidateGraphTests(unittest.TestCase):
    def test_system2_merges_canonical_identity_and_adds_unique_node(self):
        enriched = augment_graph(
            graph(), ["PARIS", "Marseille", "Marseille"])
        nodes = {node["key"]: node for node in enriched["candidates"]}
        self.assertEqual(set(nodes["paris"]["routes"]), {
            ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2})
        self.assertTrue(
            nodes["paris"]["route_summary"]["within_qwen_route_agreement"])
        self.assertTrue(
            nodes["marseille"]["route_summary"]["system2_only"])
        self.assertEqual(len(nodes), 3)

    def test_cross_model_agreement_is_distinct_from_cross_route(self):
        enriched = augment_graph(graph(), ["Lyon"])
        nodes = {node["key"]: node for node in enriched["candidates"]}
        self.assertEqual(set(nodes["lyon"]["routes"]), {
            ROUTE_GEMMA, ROUTE_QWEN_SYSTEM2})
        self.assertTrue(nodes["lyon"]["route_summary"]["cross_model_agreement"])
        self.assertFalse(
            nodes["lyon"]["route_summary"]["within_qwen_route_agreement"])

    def test_existing_sources_are_not_redefined(self):
        enriched = augment_graph(graph(), ["Marseille"])
        nodes = {node["key"]: node for node in enriched["candidates"]}
        self.assertEqual(nodes["paris"]["sources"][QWEN]["samples"], 10)
        self.assertEqual(nodes["marseille"]["sources"], {})
        self.assertEqual(
            enriched["route_graph_schema"], "explicit-proposal-routes-v1")

    def test_numeric_selection_uses_canonical_agent_output_not_legacy_key(self):
        value = graph()
        value["Relation"] = "hasArea"
        value["agent_outputs"][QWEN] = ["38"]
        value["candidates"] = [{
            "key": "38",
            "item": "38",
            "type": "numeric",
            "sources": {
                QWEN: {
                    "support": 3, "samples": 10, "support_rate": 0.3}},
            "selected_by": {QWEN: False, GEMMA: False},
        }]
        enriched = augment_graph(value, [])
        route = enriched["candidates"][0]["routes"][ROUTE_QWEN_SC]
        self.assertTrue(route["selected"])
        self.assertTrue(
            enriched["candidates"][0]["selected_by"][QWEN])

    def test_augment_fails_closed_without_required_agent_output(self):
        value = graph()
        del value["agent_outputs"][QWEN]
        with self.assertRaisesRegex(
            Exception, "required agent_outputs missing qwen_recall"
        ):
            augment_graph(value, [])

    def test_normalization_fails_closed_without_required_agent_output(self):
        value = augment_graph(graph(), [])
        del value["agent_outputs"][GEMMA]
        with self.assertRaisesRegex(
            Exception, "required agent_outputs missing gemma_independent"
        ):
            normalize_route_selection(value)


if __name__ == "__main__":
    unittest.main()
