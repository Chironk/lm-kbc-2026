import unittest

from experiments.heterogeneous_agents.cot40_graph_native_decoder import (
    enrich_row,
    legal_actions,
)
from experiments.heterogeneous_agents.cot40_minimal_evidence_graph import (
    EVENT_NAMES,
    _prepare_event_cache,
    event_features,
    minimalize_row,
    parity_audit,
)


class Cot40MinimalEvidenceGraphTest(unittest.TestCase):
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
                        "support": 6,
                        "samples": 10,
                        "support_rate": 0.6,
                        "selected": True,
                        "generation_indices": [],
                    },
                },
                "sources": {
                    "qwen_recall": {
                        "support": 6,
                        "samples": 10,
                        "support_rate": 0.6,
                    },
                },
                "selected_by": {"qwen_recall": True},
                "output_eligible": True,
            }],
            "agents": {
                "qwen_recall": {
                    "n_samples": 10,
                    "none_rate": 0.0,
                },
            },
            "proposal_routes": {
                "qwen:self_consistency": {
                    "available": True,
                    "model_family": "qwen_recall",
                    "n_samples": 10,
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
                *["ANSWER: Alpha, Beta"] * 7,
                *["ANSWER: Alpha"] * 2,
                "ANSWER: None",
            ],
        }

    def graph(self):
        return enrich_row(self.source(), self.response())

    def test_minimal_schema_contains_only_components_events_and_supports(self):
        minimal = minimalize_row(self.graph())
        relational = minimal["relational_graph"]
        self.assertEqual(
            {node["node_type"] for node in relational["nodes"]},
            {"candidate_component", "evidence_event"},
        )
        self.assertEqual(
            {edge["edge_type"] for edge in relational["edges"]},
            {"supports"},
        )
        self.assertEqual(
            minimal["minimal_evidence_contract"][
                "hard_constraints_outside_graph"
            ],
            [
                "null_legality",
                "relation_cardinality",
                "bounded_action_inventory",
            ],
        )

    def test_exact_and_aggregate_provenance_are_not_conflated(self):
        minimal = minimalize_row(self.graph())
        events = {
            node["id"]: node
            for node in minimal["relational_graph"]["nodes"]
            if node["node_type"] == "evidence_event"
        }
        exact = [
            value for value in events.values()
            if value["evidence_kind"] == "exact_generation"
        ]
        aggregate = [
            value for value in events.values()
            if value["evidence_kind"] == "aggregate_route"
        ]
        self.assertEqual(len(exact), 10)
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["route"], "qwen:self_consistency")
        none = [
            value for value in exact
            if value["status"] == "explicit_none"
        ]
        self.assertEqual(
            [value["generation_index"] for value in none], [9])
        self.assertTrue(
            minimal["minimal_evidence_contract"][
                "no_fabricated_generation_provenance"
            ]
        )

    def test_exact_empty_only_route_is_not_dropped(self):
        response = self.response()
        response["generations"] = ["ANSWER: None"] * 10
        minimal = minimalize_row(enrich_row(self.source(), response))
        exact = [
            value
            for value in minimal["relational_graph"]["nodes"]
            if (
                value["node_type"] == "evidence_event"
                and value["evidence_kind"] == "exact_generation"
            )
        ]
        self.assertEqual(len(exact), 10)
        self.assertEqual(
            {value["status"] for value in exact}, {"explicit_none"})
        self.assertEqual(
            minimal["minimal_evidence_contract"]["exact_event_count"], 10)

    def test_single_sample_route_is_safely_reconstructed_as_exact(self):
        source = self.source()
        route = source["candidates"][0]["routes"][
            "qwen:self_consistency"]
        route.update({
            "support": 1,
            "samples": 1,
            "support_rate": 1.0,
        })
        source["candidates"][0]["sources"]["qwen_recall"].update({
            "support": 1,
            "samples": 1,
            "support_rate": 1.0,
        })
        source["agents"]["qwen_recall"]["n_samples"] = 1
        source["proposal_routes"]["qwen:self_consistency"][
            "n_samples"] = 1
        minimal = minimalize_row(enrich_row(source, self.response()))
        qwen = [
            value
            for value in minimal["relational_graph"]["nodes"]
            if (
                value["node_type"] == "evidence_event"
                and value["route"] == "qwen:self_consistency"
            )
        ]
        self.assertEqual(len(qwen), 1)
        self.assertEqual(qwen[0]["evidence_kind"], "exact_generation")
        self.assertEqual(
            qwen[0]["provenance_mode"], "inferred_single_generation")
        self.assertEqual(qwen[0]["status"], "candidate_set")

    def test_exact_generation_membership_and_aggregate_weight(self):
        minimal = minimalize_row(self.graph())
        components = {
            value["representative"]: value["id"]
            for value in minimal["relational_graph"]["components"]
        }
        exact_edges = [
            edge for edge in minimal["relational_graph"]["edges"]
            if edge["evidence_kind"] == "exact_generation"
        ]
        aggregate_edges = [
            edge for edge in minimal["relational_graph"]["edges"]
            if edge["evidence_kind"] == "aggregate_route"
        ]
        alpha_exact = [
            edge for edge in exact_edges
            if edge["target"] == components["Alpha"]
        ]
        beta_exact = [
            edge for edge in exact_edges
            if edge["target"] == components["Beta"]
        ]
        self.assertEqual(len(alpha_exact), 9)
        self.assertEqual(len(beta_exact), 7)
        qwen_alpha = [
            edge for edge in aggregate_edges
            if (
                edge["route"] == "qwen:self_consistency"
                and edge["target"] == components["Alpha"]
            )
        ]
        self.assertEqual(len(qwen_alpha), 1)
        self.assertAlmostEqual(qwen_alpha[0]["weight"], 0.6)

    def test_minimal_graph_preserves_count_actions_and_component_features(self):
        source = self.graph()
        minimal = minimalize_row(source)
        audit = parity_audit(
            [source],
            [minimal],
            {("Example Prize", "awardWonBy"): []},
        )
        self.assertTrue(audit["parity_passed"])
        self.assertEqual(audit["count_anchor_matches"], 1)
        self.assertEqual(audit["legal_action_inventory_matches"], 1)
        self.assertGreater(audit["component_feature_matches"], 0)

    def test_event_features_are_finite_and_have_fixed_schema(self):
        minimal = minimalize_row(self.graph())
        actions = legal_actions(minimal, [])
        for action in actions:
            values = event_features(minimal, action)
            self.assertEqual(len(values), len(EVENT_NAMES))
            self.assertTrue(all(value == value for value in values))

    def test_subject_shift_control_changes_evidence_without_action_drift(self):
        left = minimalize_row(self.graph())
        right_source = self.source()
        right_source["SubjectEntity"] = "Other Prize"
        response = self.response()
        response["subject"] = "Other Prize"
        right = minimalize_row(enrich_row(right_source, response))
        controls = {
            ("Example Prize", "awardWonBy"): [],
            ("Other Prize", "awardWonBy"): [],
        }
        audit = _prepare_event_cache([left, right], controls)
        self.assertEqual(audit["same_subject_assignments"], 0)
        self.assertGreater(audit["actions"], 0)
        for graph in (left, right):
            for action in legal_actions(graph, []):
                self.assertIn(
                    "_minimal_event_features_shifted", action)


if __name__ == "__main__":
    unittest.main()
