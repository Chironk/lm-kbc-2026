from __future__ import annotations

import copy
import unittest

from experiments.heterogeneous_agents.unified_graph_decoder import (
    AREA_ADMISSION_REASON,
    CHAIN_ROUTES,
    COT40_SAMPLES,
    COT40_SUPPORT_REQUIRED,
    GEMMA,
    MINISTRAL_COT40,
    MINISTRAL_N3,
    QWEN_SC,
    apply_area_unanimity,
    apply_capacity_veto,
    apply_cot40_support,
    cot40_route_evidence,
    legacy_proposal_parse_status,
    scope_graph,
)


def _candidate(item, routes, **extra):
    node = {"item": item, "key": item.lower(), "routes": dict(routes)}
    node.update(extra)
    return node


class ScopeGraphTest(unittest.TestCase):
    def test_out_of_scope_routes_are_hidden(self):
        graph = {
            "SubjectEntity": "S", "Relation": "hasArea",
            "candidates": [
                _candidate("100", {QWEN_SC: {"support": 4}}),
                _candidate("200", {MINISTRAL_COT40: {"support": 9}}),
            ],
            "relational_graph": {"components": [], "nodes": [], "edges": []},
        }
        scoped = scope_graph(graph, CHAIN_ROUTES)
        items = [node["item"] for node in scoped["candidates"]]
        self.assertEqual(items, ["100"])

    def test_mixed_candidate_survives_with_only_in_scope_routes(self):
        graph = {
            "SubjectEntity": "S", "Relation": "hasArea",
            "candidates": [
                _candidate("100", {
                    QWEN_SC: {"support": 4},
                    MINISTRAL_COT40: {"support": 9},
                }),
            ],
            "relational_graph": {"components": [], "nodes": [], "edges": []},
        }
        scoped = scope_graph(graph, CHAIN_ROUTES)
        self.assertEqual(
            sorted(scoped["candidates"][0]["routes"]), [QWEN_SC])

    def test_scoping_does_not_mutate_the_unified_graph(self):
        graph = {
            "SubjectEntity": "S", "Relation": "hasArea",
            "candidates": [
                _candidate("200", {MINISTRAL_COT40: {"support": 9}})],
            "relational_graph": {"components": [], "nodes": [], "edges": []},
        }
        before = copy.deepcopy(graph)
        scope_graph(graph, CHAIN_ROUTES)
        self.assertEqual(graph, before)


class Cot40EvidenceTest(unittest.TestCase):
    def test_support_counts_distinct_generations_not_mentions(self):
        generations = [
            "ANSWER: Madrid",
            "ANSWER: Madrid, Madrid",
            "ANSWER: Toledo",
        ]
        occurrences, none_generations = cot40_route_evidence(
            generations, "personHasCityOfDeath")
        self.assertEqual(none_generations, [])
        self.assertEqual(
            occurrences["madrid"]["generation_indices"], [0, 1])
        self.assertEqual(occurrences["toledo"]["generation_indices"], [2])

    def test_explicit_none_generations_are_tracked(self):
        occurrences, none_generations = cot40_route_evidence(
            ["ANSWER: None", "ANSWER: Madrid"], "personHasCityOfDeath")
        self.assertEqual(none_generations, [0])
        self.assertIn("madrid", occurrences)

    def test_legacy_mode_preserves_repeated_answer_prefix(self):
        occurrences, _ = cot40_route_evidence(
            ["ANSWER: ANSWER: Madrid"], "personHasCityOfDeath",
            parser_mode="legacy-20260729")
        self.assertIn("answer madrid", occurrences)
        self.assertNotIn("madrid", occurrences)

    def test_corrected_mode_removes_repeated_answer_prefix(self):
        occurrences, _ = cot40_route_evidence(
            ["ANSWER: ANSWER: Madrid"], "personHasCityOfDeath",
            parser_mode="corrected")
        self.assertIn("madrid", occurrences)
        self.assertNotIn("answer madrid", occurrences)

    def test_legacy_parser_remains_explicit_compatibility_behavior(self):
        status, items = legacy_proposal_parse_status(
            "ANSWER: ANSWER: None", "personHasCityOfDeath")
        self.assertEqual(status, "parsed_nonempty")
        self.assertEqual(items, ["ANSWER: None"])


class PolicyTest(unittest.TestCase):
    def _cot40_graph(self, relation, supports):
        return {
            "SubjectEntity": "S", "Relation": relation,
            "candidates": [
                _candidate(item, {MINISTRAL_COT40: {
                    "support": support,
                    "samples": COT40_SAMPLES,
                    "display_item": item,
                }})
                for item, support in supports.items()
            ],
            "relational_graph": {"components": [], "nodes": [], "edges": []},
        }

    def test_numeric_replacement_requires_a_unique_top_candidate(self):
        graph = self._cot40_graph("hasArea", {"100": 8, "200": 3})
        selected, detail = apply_cot40_support(graph, ["999"])
        self.assertEqual(selected, ["100"])
        self.assertTrue(detail["applied"])

    def test_numeric_tie_at_threshold_keeps_the_incumbent(self):
        graph = self._cot40_graph("hasArea", {"100": 8, "200": 8})
        selected, _ = apply_cot40_support(graph, ["999"])
        self.assertEqual(selected, ["999"])

    def test_numeric_below_threshold_keeps_the_incumbent(self):
        graph = self._cot40_graph(
            "hasArea", {"100": COT40_SUPPORT_REQUIRED - 1})
        selected, _ = apply_cot40_support(graph, ["999"])
        self.assertEqual(selected, ["999"])

    def test_string_relation_adds_supported_candidates_to_incumbent(self):
        graph = self._cot40_graph(
            "companyTradesAtStockExchange", {"NASDAQ": 9, "LSE": 2})
        selected, _ = apply_cot40_support(graph, ["New York Stock Exchange"])
        self.assertEqual(selected, ["New York Stock Exchange", "NASDAQ"])

    def test_area_rule_requires_a_single_unanimous_new_component(self):
        graph = {
            "SubjectEntity": "S", "Relation": "hasArea",
            "candidates": [
                _candidate("500", {MINISTRAL_N3: {
                    "admission_reason": AREA_ADMISSION_REASON,
                    "support": 3, "samples": 3}}),
            ],
            "relational_graph": {"components": [], "nodes": [], "edges": []},
        }
        selected, detail = apply_area_unanimity(graph, ["100"])
        self.assertEqual(selected, ["500"])
        self.assertTrue(detail["applied"])

    def test_area_rule_fails_closed_on_two_unanimous_components(self):
        route = {
            "admission_reason": AREA_ADMISSION_REASON,
            "support": 3, "samples": 3,
        }
        graph = {
            "SubjectEntity": "S", "Relation": "hasArea",
            "candidates": [
                _candidate("500", {MINISTRAL_N3: dict(route)}),
                _candidate("700", {MINISTRAL_N3: dict(route)}),
            ],
            "relational_graph": {"components": [], "nodes": [], "edges": []},
        }
        selected, detail = apply_area_unanimity(graph, ["100"])
        self.assertEqual(selected, ["100"])
        self.assertFalse(detail["applied"])

    def test_area_rule_ignores_non_area_relations(self):
        graph = {
            "SubjectEntity": "S", "Relation": "hasCapacity",
            "candidates": [
                _candidate("500", {MINISTRAL_N3: {
                    "admission_reason": AREA_ADMISSION_REASON,
                    "support": 3, "samples": 3}}),
            ],
            "relational_graph": {"components": [], "nodes": [], "edges": []},
        }
        selected, _ = apply_area_unanimity(graph, ["100"])
        self.assertEqual(selected, ["100"])

    def test_capacity_veto_applies_only_recorded_switches(self):
        graph = {
            "SubjectEntity": "S", "Relation": "hasCapacity",
            "frozen_policy_ledger": {
                "action_id": "abc", "proposal": ["2200"], "switched": True},
        }
        selected, detail = apply_capacity_veto(graph, ["6000"])
        self.assertEqual(selected, ["2200"])
        self.assertTrue(detail["applied"])

    def test_capacity_veto_without_a_switch_keeps_the_incumbent(self):
        graph = {
            "SubjectEntity": "S", "Relation": "hasCapacity",
            "frozen_policy_ledger": {
                "action_id": None, "proposal": [], "switched": False},
        }
        selected, detail = apply_capacity_veto(graph, ["6000"])
        self.assertEqual(selected, ["6000"])
        self.assertFalse(detail["applied"])


class RouteContractTest(unittest.TestCase):
    def test_chain_scope_excludes_both_ministral_routes(self):
        self.assertNotIn(MINISTRAL_N3, CHAIN_ROUTES)
        self.assertNotIn(MINISTRAL_COT40, CHAIN_ROUTES)
        self.assertIn(GEMMA, CHAIN_ROUTES)


if __name__ == "__main__":
    unittest.main()
