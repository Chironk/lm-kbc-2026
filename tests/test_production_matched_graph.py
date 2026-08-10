import unittest

from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.production_matched_graph import (
    _rebase_qwen,
)


class ProductionMatchedGraphTests(unittest.TestCase):
    def test_rebase_preserves_gemma_and_separates_invalid_from_none(self):
        graph = {
            "schema": "heterogeneous-memory-graph-row-v1",
            "SubjectEntity": "Example Corp",
            "Relation": "companyTradesAtStockExchange",
            "baseline_objects": [],
            "agent_outputs": {QWEN: [], GEMMA: ["Nasdaq"]},
            "agents": {
                QWEN: {
                    "n_samples": 1,
                    "none_count": 0,
                    "none_rate": 0.0,
                    "parse_failures": 0,
                    "numeric_log_mad": None,
                    "existence": {"available": True},
                    "cardinality": {"available": True},
                },
                GEMMA: {
                    "n_samples": 1,
                    "none_count": 0,
                    "none_rate": 0.0,
                    "parse_failures": 0,
                    "numeric_log_mad": None,
                    "existence": {"available": True},
                    "cardinality": {"available": True},
                },
            },
            "candidates": [{
                "key": "nasdaq",
                "item": "Nasdaq",
                "type": "string",
                "sources": {
                    GEMMA: {
                        "support": 1, "samples": 1, "support_rate": 1.0}},
                "selected_by": {QWEN: False, GEMMA: True},
            }],
        }
        raw = {
            "SubjectEntity": "Example Corp",
            "Relation": "companyTradesAtStockExchange",
            "raw_samples": [
                "</think>None",
                "<think>unfinished",
                "</think>The company is listed on the New York Stock Exchange.",
            ],
        }
        rebased = _rebase_qwen(
            graph, raw, ["New York Stock Exchange"])
        self.assertEqual(rebased["agents"][QWEN]["n_samples"], 3)
        self.assertEqual(rebased["agents"][QWEN]["none_count"], 1)
        self.assertEqual(rebased["agents"][QWEN]["parse_failures"], 1)
        nodes = {node["key"]: node for node in rebased["candidates"]}
        self.assertIn("nasdaq", nodes)
        self.assertEqual(nodes["nasdaq"]["sources"][GEMMA]["support"], 1)
        self.assertIn("new york stock exchange", nodes)
        self.assertTrue(
            nodes["new york stock exchange"]["selected_by"][QWEN])
        self.assertEqual(
            rebased["production_match"]["candidate_extraction"],
            "relation-aware-sample-evidence-v1")


if __name__ == "__main__":
    unittest.main()
