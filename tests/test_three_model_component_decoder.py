import unittest

from experiments.heterogeneous_agents.components.ministral_consistency_admission import ROUTE
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    augment_relational_graph,
)
from experiments.heterogeneous_agents.components.route_aware_candidate_graph import (
    ROUTE_GEMMA,
    ROUTE_QWEN_SC,
)
from experiments.heterogeneous_agents.components.three_model_component_decoder import (
    action_features,
    assert_route_flag_population_parity,
    assert_route_selection_provenance,
    component_summary,
    feature_names,
    legal_actions,
    propose,
    subject_grouped_folds,
)


def _agent():
    return {
        "none_rate": 0.0,
        "existence": {
            "available": True,
            "probabilities": {"YES": 1.0, "NO": 0.0},
        },
        "cardinality": {
            "available": True,
            "probabilities": {"ZERO": 0.0, "ONE": 1.0, "MANY": 0.0},
        },
    }


def _graph():
    return augment_relational_graph({
        "SubjectEntity": "S",
        "Relation": "hasArea",
        "agents": {
            "qwen_recall": _agent(),
            "gemma_independent": _agent(),
            "ministral_independent": {
                **_agent(),
                "candidate_supply_only": False,
                "decoder_commitments_enabled": True,
            },
        },
        "candidates": [{
            "item": "100",
            "key": "numeric:100",
            "sources": {},
            "routes": {
                ROUTE_QWEN_SC: {"support_rate": 0.6, "selected": True},
                ROUTE_GEMMA: {"support_rate": 1.0, "selected": True},
                ROUTE: {
                    "support_rate": 1.0,
                    "support": 3,
                    "samples": 3,
                    "selected": True,
                    "admission_reason":
                        "numeric_component_corroborates_source",
                    "cluster_members": ["98", "100", "102"],
                },
            },
            "route_summary": {"system2_only": False},
            "selected_by": {},
        }, {
            "item": "200",
            "key": "numeric:200",
            "sources": {},
            "routes": {
                ROUTE: {
                    "support_rate": 2 / 3,
                    "support": 2,
                    "samples": 3,
                    "selected": True,
                    "admission_reason":
                        "numeric_complete_link_self_consistent_new",
                },
            },
            "route_summary": {"system2_only": False},
            "selected_by": {},
        }],
        "proposal_routes": {},
    })


class ThreeModelComponentDecoderTests(unittest.TestCase):
    def test_summary_exposes_three_model_evidence(self):
        graph = _graph()
        summary = component_summary(
            graph, graph["relational_graph"]["components"][0])
        self.assertEqual(summary["three_model"], 1.0)
        self.assertEqual(summary["ministral_unanimous"], 1.0)
        self.assertEqual(summary["ministral_corroborates"], 1.0)

    def test_new_ministral_component_is_explicit(self):
        graph = _graph()
        component = graph["relational_graph"]["components"][1]
        summary = component_summary(graph, component)
        self.assertEqual(summary["ministral_new"], 1.0)
        self.assertEqual(summary["ministral_numeric_new"], 1.0)
        self.assertEqual(summary["ministral_two_of_three"], 1.0)

    def test_locked_anchor_has_no_alternative_action(self):
        graph = _graph()
        self.assertEqual(legal_actions(graph, ["100"], True), [["100"]])
        self.assertGreater(len(legal_actions(graph, ["100"], False)), 1)

    def test_schema_names_ministral(self):
        self.assertIn("added_ministral_support", feature_names())
        self.assertIn("action_ministral_unanimous", feature_names())
        self.assertIn("ministral_card_one", feature_names())
        self.assertIn("dormant_candidate_count", feature_names())

    def test_numeric_spread_uses_ministral_cluster_members(self):
        graph = _graph()
        component = graph["relational_graph"]["components"][0]
        summary = component_summary(graph, component)
        self.assertGreater(summary["numeric_spread"], 0.0)
        self.assertEqual(summary["component_numeric_surface_spread"], 0.0)

    def test_ministral_commitment_changes_feature_vector(self):
        graph = _graph()
        before = action_features(graph, ["100"], ["200"])
        graph["agents"]["ministral_independent"]["cardinality"] = {
            "available": True,
            "probabilities": {"ZERO": 0.0, "ONE": 0.0, "MANY": 1.0},
        }
        after = action_features(graph, ["100"], ["200"])
        self.assertNotEqual(before, after)

    def test_dormant_candidate_never_becomes_legal_action(self):
        graph = _graph()
        graph["dormant_candidates"] = [{
            "item": "999",
            "key": "numeric:999",
            "type": "numeric",
            "output_eligible": False,
            "dormant": True,
            "routes": {},
        }]
        surfaces = {
            item for action in legal_actions(graph, ["100"], False)
            for item in action
        }
        self.assertNotIn("999", surfaces)

    def test_proposal_uses_advantage_over_keep(self):
        graph = _graph()

        class FakeModel:
            def predict(self, values):
                # A large shared intercept must cancel. The alternative has
                # only 0.2 advantage over KEEP, not an absolute score of 9.2.
                return [9.0 + 0.2 * index for index in range(len(values))]

        _, advantage, _ = propose(FakeModel(), graph, ["100"], False)
        self.assertGreaterEqual(advantage, 0.0)
        self.assertLess(advantage, 1.0)

    def test_subject_folds_never_split_a_subject(self):
        left = _graph()
        right = _graph()
        right["Relation"] = "hasCapacity"
        other = _graph()
        other["SubjectEntity"] = "T"
        folds = subject_grouped_folds([left, right, other], n_folds=2)
        self.assertEqual(
            folds[("S", "hasArea")],
            folds[("S", "hasCapacity")],
        )

    def test_route_population_rejects_numeric_split_degeneracy(self):
        train = []
        validation = []
        for index in range(5):
            left = _graph()
            right = _graph()
            left["SubjectEntity"] = f"train-{index}"
            right["SubjectEntity"] = f"validation-{index}"
            for node in left["candidates"]:
                route = node.get("routes", {}).get(ROUTE_QWEN_SC)
                if route:
                    route["selected"] = False
            train.append(left)
            validation.append(right)
        with self.assertRaisesRegex(Exception, "route-selection parity"):
            assert_route_flag_population_parity(train, validation)

    def test_route_provenance_rejects_missing_output_even_when_rare(self):
        graph = _graph()
        graph["agent_outputs"] = {
            "qwen_recall": ["100"],
            "gemma_independent": ["100"],
        }
        graph["route_selection_normalization"] = {
            "schema": "canonical-agent-output-selection-v2",
            "qwen_outputs_required": True,
            "gemma_outputs_required": True,
            "qwen_outputs_available": True,
            "gemma_outputs_available": True,
            "legacy_selected_by_fallback_allowed": False,
        }
        assert_route_selection_provenance(graph)
        del graph["agent_outputs"]["qwen_recall"]
        with self.assertRaisesRegex(Exception, "missing canonical output"):
            assert_route_selection_provenance(graph)

    def test_route_provenance_rejects_stale_selected_flag(self):
        graph = _graph()
        graph["agent_outputs"] = {
            "qwen_recall": [],
            "gemma_independent": ["100"],
        }
        graph["route_selection_normalization"] = {
            "schema": "canonical-agent-output-selection-v2",
            "qwen_outputs_required": True,
            "gemma_outputs_required": True,
            "qwen_outputs_available": True,
            "gemma_outputs_available": True,
            "legacy_selected_by_fallback_allowed": False,
        }
        with self.assertRaisesRegex(Exception, "selection disagrees"):
            assert_route_selection_provenance(graph)


if __name__ == "__main__":
    unittest.main()
