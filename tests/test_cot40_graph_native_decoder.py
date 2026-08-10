import unittest

from experiments.heterogeneous_agents.cot40_graph_native_decoder import (
    EDGE_NAMES,
    LOCAL_NAMES,
    ROUTE,
    _summary,
    action_features,
    cot40_count_anchor,
    edge_features,
    enrich_row,
    legal_actions,
)


class Cot40GraphNativeDecoderTest(unittest.TestCase):
    def source(self):
        return {
            "schema": "heterogeneous-memory-graph-row-v1",
            "SubjectEntity": "Example Prize",
            "Relation": "awardWonBy",
            "baseline_objects": [],
            "candidates": [],
            "agents": {
                "qwen_recall": {"none_rate": 0.0},
                "gemma_independent": {"none_rate": 0.0},
            },
            "proposal_routes": {},
        }

    def response(self):
        return {
            "agent_id": "ministral_independent",
            "phase": "propose",
            "subject": "Example Prize",
            "relation": "awardWonBy",
            "generations": [
                *["ANSWER: Alpha, Beta"] * 7,
                "ANSWER: ANSWER: Alpha",
                "ANSWER: None",
                "ANSWER: None",
            ],
        }

    def test_enrichment_materializes_generation_edges(self):
        graph = enrich_row(self.source(), self.response())
        components = graph["relational_graph"]["components"]
        self.assertEqual(
            {value["representative"] for value in components},
            {"Alpha", "Beta"},
        )
        alpha = next(
            value for value in components
            if value["representative"] == "Alpha")
        self.assertEqual(
            alpha["routes"][ROUTE]["generation_indices"],
            list(range(8)),
        )
        edge = next(
            value for value in graph["relational_graph"]["edges"]
            if value["edge_type"] == "co_supported_with")
        self.assertEqual(edge["cooccurrence_count"], 7)
        self.assertAlmostEqual(edge["cooccurrence_rate"], 0.7)

    def test_edge_arm_sees_topology_beyond_local_arm(self):
        graph = enrich_row(self.source(), self.response())
        actions = legal_actions(graph, [])
        action = next(
            value for value in actions
            if set(value["objects"]) == {"Alpha", "Beta"})
        local = action_features(graph, [], action, "component_local")
        typed = action_features(graph, [], action, "typed_edges")
        edge = edge_features(graph, action)
        self.assertEqual(len(local), len(LOCAL_NAMES))
        self.assertEqual(len(edge), len(EDGE_NAMES))
        self.assertEqual(typed[:len(local)], local)
        self.assertEqual(typed[len(local):], edge)
        self.assertGreater(edge[2], 0.0)
        self.assertGreater(edge[3], 0.0)

    def test_graph_decoder_starts_after_frozen_count_policy(self):
        graph = enrich_row(self.source(), self.response())
        self.assertEqual(
            set(cot40_count_anchor(graph, [])),
            {"Alpha", "Beta"},
        )

    def test_support_summary_is_order_stable(self):
        values = [1.0, 1e-16, 1e-16, 1e-16]
        self.assertEqual(_summary(values), _summary(list(reversed(values))))


if __name__ == "__main__":
    unittest.main()
