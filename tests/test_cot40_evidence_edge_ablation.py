import math
import unittest

from experiments.heterogeneous_agents.components.cot40_evidence_edge_ablation import (
    ARMS,
    LOCAL_NAMES,
    TYPED_NAMES,
    _arm_names,
    _attach_existing_claims,
    _replace_route_events,
    _state_and_relation_edges,
    reasoning_claims,
    typed_features,
)
from experiments.heterogeneous_agents.components.cot40_graph_native_decoder import (
    enrich_row,
    legal_actions,
)
from experiments.heterogeneous_agents.components.cot40_minimal_evidence_graph import (
    minimalize_row,
)


class Cot40EvidenceEdgeAblationTest(unittest.TestCase):
    def source(self):
        return {
            "schema": "heterogeneous-memory-graph-row-v1",
            "SubjectEntity": "Example Prize",
            "Relation": "awardWonBy",
            "baseline_objects": [],
            "candidates": [{
                "key": "alpha",
                "item": "Alpha",
                "type": "string",
                "routes": {
                    "qwen:self_consistency": {
                        "model_family": "qwen_recall",
                        "support": 2,
                        "samples": 2,
                        "support_rate": 1.0,
                        "selected": True,
                        "generation_indices": [],
                    },
                },
                "sources": {
                    "qwen_recall": {
                        "support": 2,
                        "samples": 2,
                        "support_rate": 1.0,
                    },
                },
                "selected_by": {"qwen_recall": True},
                "output_eligible": True,
            }],
            "agents": {
                "qwen_recall": {
                    "n_samples": 2,
                    "none_rate": 0.0,
                },
            },
            "proposal_routes": {
                "qwen:self_consistency": {
                    "available": True,
                    "model_family": "qwen_recall",
                    "n_samples": 2,
                },
            },
        }

    def response(self):
        return {
            "agent_id": "ministral_independent",
            "phase": "propose",
            "subject": "Example Prize",
            "relation": "awardWonBy",
            "generations": [
                "REASONING: 2024 example only\nANSWER: Alpha, Beta",
                *["ANSWER: Alpha, Beta"] * 6,
                *["ANSWER: Alpha"] * 2,
                "ANSWER: None",
            ],
        }

    def graph(self):
        graph = minimalize_row(enrich_row(self.source(), self.response()))
        component = next(
            value["id"]
            for value in graph["relational_graph"]["components"]
            if value["representative"] == "Alpha"
        )
        records = [
            ("candidate_set", [component], []),
            ("candidate_set", [component], []),
        ]
        _replace_route_events(
            graph,
            route="qwen:self_consistency",
            family="qwen_recall",
            records=records,
            raw_texts=["ANSWER: Alpha", "ANSWER: Alpha"],
            provenance="unit_test_exact_raw",
        )
        _attach_existing_claims(
            graph,
            route="ministral:cot5_cap40_n10",
            raw_texts=self.response()["generations"],
        )
        _state_and_relation_edges(graph)
        return graph

    def test_reasoning_claims_are_relation_scoped_and_deterministic(self):
        self.assertEqual(
            reasoning_claims(
                "REASONING: It was formerly listed but is now delisted.\n"
                "ANSWER: Example Exchange",
                "companyTradesAtStockExchange",
            ),
            ["listing_historical", "listing_inactive"],
        )
        self.assertEqual(
            reasoning_claims(
                "REASONING: The official total capacity is 42,000.\n"
                "ANSWER: 42000",
                "hasCapacity",
            ),
            ["capacity_total"],
        )
        self.assertEqual(
            reasoning_claims(
                "REASONING: The area is 100 square miles.\nANSWER: 100",
                "hasArea",
            ),
            ["area_non_km_unit"],
        )

    def test_graph_contains_only_declared_typed_edge_families(self):
        graph = self.graph()
        edge_types = {
            edge["edge_type"]
            for edge in graph["relational_graph"]["edges"]
        }
        self.assertEqual(
            edge_types,
            {
                "supports",
                "co_occurs_with",
                "asserts_cardinality",
                "asserts_existence",
                "asserts_claim",
            },
        )
        node_types = {
            node["node_type"]
            for node in graph["relational_graph"]["nodes"]
        }
        self.assertEqual(
            node_types,
            {
                "candidate_component",
                "evidence_event",
                "cardinality_state",
                "existence_state",
                "claim_state",
            },
        )

    def test_exact_route_replacement_removes_aggregate_event(self):
        graph = self.graph()
        qwen = [
            node
            for node in graph["relational_graph"]["nodes"]
            if (
                node["node_type"] == "evidence_event"
                and node["route"] == "qwen:self_consistency"
            )
        ]
        self.assertEqual(len(qwen), 2)
        self.assertEqual(
            {node["evidence_kind"] for node in qwen},
            {"exact_generation"},
        )
        self.assertEqual(
            {node["provenance_mode"] for node in qwen},
            {"unit_test_exact_raw"},
        )

    def test_typed_features_are_finite_and_arm_schemas_are_distinct(self):
        graph = self.graph()
        actions = legal_actions(graph, [])
        self.assertTrue(actions)
        for action in actions:
            values = typed_features(graph, action)
            self.assertEqual(len(values), len(TYPED_NAMES))
            self.assertTrue(all(math.isfinite(value) for value in values))
        schemas = {arm: _arm_names(arm) for arm in ARMS}
        self.assertEqual(
            len(schemas["component_table"]), len(LOCAL_NAMES))
        self.assertGreater(
            len(schemas["all_typed_edges"]),
            len(schemas["exact_support"]),
        )
        self.assertEqual(
            schemas["all_typed_edges"],
            schemas["all_typed_edges_shifted"],
        )

    def test_raw_reasoning_is_not_stored_in_graph(self):
        graph = self.graph()
        serialized = repr(graph)
        self.assertNotIn("2024 example only", serialized)
        claim_nodes = [
            node
            for node in graph["relational_graph"]["nodes"]
            if node["node_type"] == "claim_state"
        ]
        self.assertEqual(
            {node["claim"] for node in claim_nodes},
            {"award_partial_scope"},
        )


if __name__ == "__main__":
    unittest.main()
