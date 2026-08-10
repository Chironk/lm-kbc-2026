import unittest

from experiments.heterogeneous_agents.component_aware_decoder import (
    _action_tokens,
    _component_summary,
    actions_for,
)
from experiments.heterogeneous_agents.relational_candidate_graph import (
    augment_relational_graph,
)
from experiments.heterogeneous_agents.route_aware_candidate_graph import (
    ROUTE_GEMMA,
    ROUTE_QWEN_SC,
)


def candidate(item, key, routes):
    return {
        "item": item,
        "key": key,
        "routes": routes,
        "route_summary": {"system2_only": False},
        "sources": {},
        "selected_by": {},
    }


def graph(relation="companyTradesAtStockExchange"):
    return augment_relational_graph({
        "SubjectEntity": "S",
        "Relation": relation,
        "agents": {
            "qwen_recall": {
                "none_rate": 0.0,
                "existence": {"probabilities": {"YES": 1.0, "NO": 0.0}},
                "cardinality": {
                    "probabilities": {
                        "ZERO": 0.0, "ONE": 1.0, "MANY": 0.0}},
            },
            "gemma_independent": {
                "none_rate": 0.0,
                "existence": {"probabilities": {"YES": 1.0, "NO": 0.0}},
                "cardinality": {
                    "probabilities": {
                        "ZERO": 0.0, "ONE": 1.0, "MANY": 0.0}},
            },
        },
        "candidates": [
            candidate("Bombay Stock Exchange", "bombay stock exchange", {
                ROUTE_QWEN_SC: {
                    "support_rate": 0.6, "selected": True}}),
            candidate("BSE", "bse", {
                ROUTE_GEMMA: {
                    "support_rate": 1.0, "selected": True}}),
            candidate("New York Stock Exchange", "new york stock exchange", {
                ROUTE_QWEN_SC: {
                    "support_rate": 0.2, "selected": False}}),
        ],
        "proposal_routes": {},
    })


class ComponentAwareDecoderTests(unittest.TestCase):
    def test_component_arm_collapses_alias_actions(self):
        row = graph()
        component_actions = actions_for(row, [], "component")
        surface_actions = actions_for(row, [], "surface")
        self.assertEqual(
            {tuple(action) for action in component_actions},
            {(), ("Bombay Stock Exchange",),
             ("New York Stock Exchange",)})
        self.assertIn(("BSE",), {tuple(action) for action in surface_actions})

    def test_component_pools_routes_across_alias_members(self):
        row = graph()
        component = row["relational_graph"]["components"][0]
        summary = _component_summary(row, component)
        self.assertEqual(summary["alias_collapsed"], 1.0)
        self.assertEqual(summary["qwen_support"], 0.6)
        self.assertEqual(summary["gemma_support"], 1.0)
        self.assertEqual(summary["cross_model"], 1.0)

    def test_action_tokens_use_component_identity(self):
        row = graph()
        self.assertEqual(
            _action_tokens(
                row, ["Bombay Stock Exchange"], "component"),
            _action_tokens(row, ["BSE"], "component"))
        self.assertNotEqual(
            _action_tokens(row, ["Bombay Stock Exchange"], "surface"),
            _action_tokens(row, ["BSE"], "surface"))

    def test_numeric_component_produces_one_action(self):
        row = augment_relational_graph({
            "SubjectEntity": "N",
            "Relation": "hasArea",
            "agents": graph()["agents"],
            "candidates": [
                candidate("100", "numeric:100", {
                    ROUTE_QWEN_SC: {
                        "support_rate": 0.5, "selected": True}}),
                candidate("104", "numeric:104", {
                    ROUTE_GEMMA: {
                        "support_rate": 1.0, "selected": True}}),
                candidate("130", "numeric:130", {
                    ROUTE_QWEN_SC: {
                        "support_rate": 0.2, "selected": False}}),
            ],
            "proposal_routes": {},
        })
        actions = actions_for(row, [], "component")
        self.assertEqual(len(actions), 3)
        self.assertIn((), {tuple(action) for action in actions})
        self.assertIn(("130",), {tuple(action) for action in actions})
        clustered = [
            action for action in actions
            if action and action[0] in {"100", "104"}]
        self.assertEqual(len(clustered), 1)


if __name__ == "__main__":
    unittest.main()
