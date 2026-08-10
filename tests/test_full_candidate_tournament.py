import unittest

from experiments.heterogeneous_agents.components.full_candidate_tournament import (
    decode_row,
    partition_groups,
    row_nodes,
)


class FullCandidateTournamentTest(unittest.TestCase):
    def test_views_cover_every_node_with_legal_group_size(self):
        nodes = [
            {"node_id": f"c{index}", "representative": str(index),
             "is_incumbent": index == 0}
            for index in range(8)
        ]
        for view in ("global3", "incumbent3"):
            groups = partition_groups(nodes, view)
            covered = {
                item["node_id"] for group in groups for item in group}
            self.assertEqual(covered, {item["node_id"] for item in nodes})
            self.assertTrue(all(1 <= len(group) <= 3 for group in groups))
        self.assertTrue(all(
            group[0]["is_incumbent"]
            for group in partition_groups(nodes, "incumbent3")))

    def test_row_nodes_adds_incumbent_absent_from_graph(self):
        graph = {
            "SubjectEntity": "Venue",
            "relational_graph": {
                "components": [{
                    "id": "component:0",
                    "representative": "100",
                    "member_items": ["100"],
                }],
            },
        }
        nodes = row_nodes(graph, ["200"])
        self.assertEqual(len(nodes), 2)
        self.assertEqual(sum(item["is_incumbent"] for item in nodes), 1)

    def test_dual_advantage_requires_both_memories(self):
        registry = {
            "cardinality": 1,
            "incumbent_objects": ["old"],
            "nodes": [
                {"node_id": "old", "representative": "old",
                 "is_incumbent": True},
                {"node_id": "new", "representative": "new",
                 "is_incumbent": False},
            ],
        }
        evidence = {
            view: {
                prompt: {
                    "qwen_recall": {"old": 0.0, "new": 2.0},
                    "gemma_independent": {"old": 0.0, "new": -1.0},
                }
                for prompt in ("recognition", "skeptical", "submission")
            }
            for view in ("global3", "incumbent3")
        }
        selected, detail = decode_row(
            registry, evidence, evidence_arm="combined:recognition", arm="mean",
            gate="dual_advantage")
        self.assertEqual(selected, ["old"])
        self.assertFalse(detail["changed"])
        selected, detail = decode_row(
            registry, evidence, evidence_arm="ensemble", arm="qwen",
            gate="advantage")
        self.assertEqual(selected, ["new"])
        self.assertTrue(detail["changed"])


if __name__ == "__main__":
    unittest.main()
