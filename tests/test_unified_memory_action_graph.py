import unittest

from experiments.heterogeneous_agents.components.unified_memory_action_graph import (
    ACTION_TYPES,
    FEATURE_NAMES,
    WeightedRidge,
    WeightedLogistic,
    UnifiedSelector,
    action_features,
    build_hierarchical_row,
    decode_one,
    grouped_relation_folds,
    row_gate_features,
)


def candidate(item, qwen=0.0, gemma=0.0):
    routes = {}
    if qwen:
        routes["qwen:self_consistency"] = {
            "model_family": "qwen_recall",
            "support_rate": qwen,
            "selected": qwen >= 0.5,
        }
    if gemma:
        routes["gemma:independent"] = {
            "model_family": "gemma_independent",
            "support_rate": gemma,
            "selected": gemma >= 0.5,
        }
    return {
        "item": item,
        "key": item.casefold(),
        "routes": routes,
        "sources": {},
        "selected_by": {
            "qwen_recall": bool(qwen), "gemma_independent": bool(gemma)},
    }


def graph(relation="personHasCityOfDeath"):
    return {
        "schema": "heterogeneous-memory-graph-row-v1",
        "SubjectEntity": "Repeated Subject",
        "Relation": relation,
        "baseline_objects": ["Paris"],
        "agents": {
            agent: {
                "n_samples": 1,
                "none_rate": 0.0,
                "numeric_log_mad": 0.0,
                "existence": {
                    "available": True,
                    "probabilities": {"YES": 0.8, "NO": 0.2}},
                "cardinality": {
                    "available": True,
                    "probabilities": {
                        "ZERO": 0.1, "ONE": 0.8, "MANY": 0.1}},
            }
            for agent in ("qwen_recall", "gemma_independent")
        },
        "proposal_routes": {
            "qwen:self_consistency": {
                "model_family": "qwen_recall", "available": True},
            "gemma:independent": {
                "model_family": "gemma_independent", "available": True},
        },
        "candidates": [
            candidate("Paris", qwen=0.8),
            candidate("London", gemma=1.0),
        ],
    }


class UnifiedMemoryActionGraphTest(unittest.TestCase):
    def test_hierarchy_nests_routes_under_memories(self):
        row = build_hierarchical_row(graph())
        edges = row["edges"]
        self.assertIn({
            "source": "memory:qwen_recall",
            "target": "route:qwen:self_consistency",
            "edge_type": "contains_route",
            "directed": True,
        }, edges)
        self.assertIn({
            "source": "memory:gemma_independent",
            "target": "route:gemma:independent",
            "edge_type": "contains_route",
            "directed": True,
        }, edges)
        self.assertEqual(
            {item["action_type"] for item in row["actions"]},
            {"KEEP", "EMPTY", "REPLACE"})

    def test_nonnullable_numeric_has_no_empty_action(self):
        row = build_hierarchical_row(graph("hasArea"))
        self.assertNotIn(
            "EMPTY", {item["action_type"] for item in row["actions"]})

    def test_feature_schema_is_shared_and_finite(self):
        row = build_hierarchical_row(graph())
        for action in row["actions"]:
            values = action_features(row, action)
            self.assertEqual(len(values), len(FEATURE_NAMES))
            self.assertTrue(all(value == value for value in values))
        self.assertEqual(set(ACTION_TYPES), {
            "KEEP", "COLLAPSE", "EMPTY", "REPLACE", "ADD", "DROP"})

    def test_qwen_routes_remain_dependent_but_distinguishable(self):
        source = graph()
        source["proposal_routes"]["qwen:system2"] = {
            "model_family": "qwen_recall", "available": True}
        source["candidates"].append({
            **candidate("Berlin"),
            "routes": {
                "qwen:system2": {
                    "model_family": "qwen_recall",
                    "support_rate": 1.0,
                    "selected": True,
                }
            },
        })
        row = build_hierarchical_row(source)
        berlin = next(
            action for action in row["actions"]
            if action["objects"] == ["Berlin"])
        values = action_features(row, berlin)
        self.assertEqual(
            values[FEATURE_NAMES.index("added_qwen_system2")], 1.0)
        self.assertEqual(
            values[FEATURE_NAMES.index("added_qwen_self_consistency")],
            0.0)
        berlin_component = next(
            node for node in row["nodes"]
            if node.get("representative") == "Berlin")
        self.assertEqual(
            berlin_component["memory_evidence"]["qwen_system2_share"], 1.0)
        self.assertEqual(
            berlin_component["memory_evidence"][
                "qwen_self_consistency_silent"], 1.0)
        self.assertEqual(
            berlin_component["memory_evidence"][
                "gemma_independent_silent"], 1.0)
        self.assertIn({
            "source": "memory:qwen_recall",
            "target": "route:qwen:system2",
            "edge_type": "contains_route",
            "directed": True,
        }, row["edges"])
        system2_support = next(
            edge for edge in row["edges"]
            if edge["source"] == "route:qwen:system2"
            and edge["edge_type"] == "supports_component")
        self.assertEqual(system2_support["support_share"], 1.0)
        self.assertEqual(system2_support["within_route_rank"], 1.0)
        self.assertTrue(any(
            edge["source"] == "route:qwen:system2"
            and edge["edge_type"] == "silent_on_component"
            for edge in row["edges"]))

    def test_unavailable_route_is_not_treated_as_negative_evidence(self):
        source = graph()
        source["proposal_routes"]["qwen:system2"] = {
            "model_family": "qwen_recall", "available": False}
        row = build_hierarchical_row(source)
        london = next(
            action for action in row["actions"]
            if action["objects"] == ["London"])
        values = action_features(row, london)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertFalse(any(
            edge["source"] == "route:qwen:system2"
            and edge["edge_type"] == "silent_on_component"
            for edge in row["edges"]))

    def test_unavailable_numeric_dispersion_is_zero(self):
        source = graph()
        source["agents"]["qwen_recall"]["numeric_log_mad"] = None
        row = build_hierarchical_row(source)
        values = action_features(row, row["actions"][0])
        self.assertEqual(values[FEATURE_NAMES.index("numeric_row_dispersion")], 0.0)
        self.assertTrue(all(
            value == value for value in row_gate_features(row)))

    def test_subjects_never_cross_folds(self):
        rows = [
            {"SubjectEntity": "same", "Relation": "hasArea"},
            {"SubjectEntity": "same", "Relation": "hasCapacity"},
            {"SubjectEntity": "a", "Relation": "hasArea"},
            {"SubjectEntity": "b", "Relation": "hasCapacity"},
            {"SubjectEntity": "c", "Relation": "awardWonBy"},
            {"SubjectEntity": "d", "Relation": "personHasCityOfDeath"},
        ]
        folds = grouped_relation_folds(rows, 3, seed=7)
        self.assertEqual(
            folds[("same", "hasArea")],
            folds[("same", "hasCapacity")])

    def test_decoder_compares_actions_directly_with_keep(self):
        row = build_hierarchical_row(graph())
        x = [action_features(row, action) for action in row["actions"]]
        y = [
            0.0 if action["action_type"] == "KEEP"
            else (1.0 if action["objects"] == ["London"] else -1.0)
            for action in row["actions"]
        ]
        model = WeightedRidge(0.1).fit(x, y, [1.0] * len(x))
        objects, detail = decode_one(model, row)
        self.assertEqual(objects, ["London"])
        self.assertGreater(detail["predicted_advantage"], 0.0)

    def test_selector_round_trip_preserves_parameters(self):
        row = build_hierarchical_row(graph())
        action_x = [
            action_features(row, action) for action in row["actions"]]
        action_y = [
            0.0 if action["action_type"] == "KEEP" else -1.0
            for action in row["actions"]]
        action = WeightedRidge(1.0).fit(
            action_x, action_y, [1.0] * len(action_x))
        gate_x = [
            row_gate_features(row),
            [value + 0.01 for value in row_gate_features(row)],
        ]
        gate = WeightedLogistic(1.0).fit(
            gate_x, [0.0, 1.0], [1.0, 1.0])
        original = UnifiedSelector(action, gate)
        restored = UnifiedSelector.from_dict(original.to_dict())
        self.assertEqual(restored.parameter_count, original.parameter_count)


if __name__ == "__main__":
    unittest.main()
