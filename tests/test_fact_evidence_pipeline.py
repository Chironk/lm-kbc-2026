import unittest

from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.fact_evidence_pipeline import (
    PROPOSAL_ARMS,
    REVIEW_CHOICES,
    _action_inventory,
    _add_route_candidate,
    _auroc,
    _component_stats,
    _component_surface_rows,
    build_proposal_tasks,
    proposal_prompt,
    verification_task,
)
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    augment_relational_graph,
)


def _candidate(item, route, agent):
    return {
        "key": f"numeric:{item}",
        "item": item,
        "type": "numeric",
        "sources": {},
        "selected_by": {},
        "routes": {
            route: {
                "model_family": agent,
                "samples": 1,
                "support": 1,
                "support_rate": 1.0,
                "selected": True,
            },
        },
        "route_summary": {},
    }


def _graph():
    return augment_relational_graph({
        "SubjectEntity": "Example Stadium",
        "Relation": "hasCapacity",
        "baseline_objects": ["10000"],
        "candidates": [
            _candidate("10000", "qwen:self_consistency", QWEN),
            _candidate("20000", "gemma:independent", GEMMA),
            _candidate("30000", "qwen:capacity_direct", QWEN),
        ],
        "proposal_routes": {},
    })


class FactEvidencePipelineTests(unittest.TestCase):
    def test_proposal_arms_are_distinct_compact_routes(self):
        prompts = {
            arm: proposal_prompt(arm, "Example Stadium")
            for arm in PROPOSAL_ARMS}
        self.assertEqual(len(set(prompts.values())), 3)
        for prompt in prompts.values():
            self.assertIn("ANSWER: <single number>", prompt)
            self.assertNotIn("ObjectEntities", prompt)
        tasks = build_proposal_tasks([_graph()], seed=7)
        self.assertEqual(sum(map(len, tasks.values())), len(PROPOSAL_ARMS))
        self.assertTrue(all(task["n_samples"] == 1
                            for rows in tasks.values() for task in rows))

    def test_new_route_becomes_selectable_component(self):
        row = _add_route_candidate(
            _graph(), 40000.0, "gemma:capacity_identity", GEMMA)
        component = next(
            value for value in row["relational_graph"]["components"]
            if "40000" in value["member_items"])
        self.assertIn("gemma:capacity_identity", component["routes"])
        support, routes, agents, new = _component_stats(row, "40000")
        self.assertEqual(support, 1.0)
        self.assertEqual(routes, 1)
        self.assertEqual(agents, {GEMMA})
        self.assertTrue(new)

    def test_inventory_preserves_surfaces_and_proposer(self):
        rows = _action_inventory(
            [_graph()],
            {("Example Stadium", "hasCapacity"): ["10000"]})
        by_value = {row["alternative"][0]: row for row in rows}
        self.assertEqual(by_value["20000"]["proposer_agents"], [GEMMA])
        self.assertEqual(by_value["30000"]["proposer_agents"], [QWEN])
        self.assertTrue(all(
            row["schema"] == "fact-evidence-surface-action-v2"
            for row in rows))

    def test_component_surface_rows_do_not_force_representative(self):
        graph = _graph()
        graph["candidates"].append(
            _candidate("20400", "qwen:capacity_direct", QWEN))
        graph = augment_relational_graph(graph)
        component = next(
            value for value in graph["relational_graph"]["components"]
            if {"20000", "20400"} <= set(value["member_items"]))
        surfaces = _component_surface_rows(graph, component)
        self.assertEqual(
            {row["surface"] for row in surfaces}, {"20000", "20400"})
        self.assertEqual(
            sum(row["surface_is_component_representative"]
                for row in surfaces), 1)

    def test_single_origin_candidate_uses_other_reviewer(self):
        action = {
            "SubjectEntity": "Example Stadium",
            "Relation": "hasCapacity",
            "baseline": ["10000"],
            "alternative": ["20000"],
            "action_id": "a",
            "proposer_agents": [GEMMA],
        }
        task = verification_task(action, QWEN, "exact_memory")
        self.assertTrue(task["reviewer_is_independent"])
        self.assertEqual(len(task["choice_variants"]), 6)
        for choice in REVIEW_CHOICES:
            codes = [
                variant["choice_codes"][choice]
                for variant in task["choice_variants"]]
            self.assertEqual(sorted(codes), ["A", "A", "B", "B", "C", "C"])
        text = "\n".join(
            variant["prompt"] for variant in task["choice_variants"])
        self.assertNotIn("gemma_independent", text)
        self.assertNotIn("qwen_recall", text)

    def test_auroc_gate_signal(self):
        self.assertEqual(_auroc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]), 1.0)
        self.assertEqual(_auroc([1, 0], [0.5, 0.5]), 0.5)
        self.assertIsNone(_auroc([1, 1], [0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
