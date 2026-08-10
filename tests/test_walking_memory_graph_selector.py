import unittest

from experiments.heterogeneous_agents.walking_memory_graph_selector import (
    expand_states,
    state_key,
    state_training_arrays,
    walk_with_chooser,
)
from experiments.heterogeneous_agents.unified_memory_action_graph import (
    build_hierarchical_row,
)
from tests.test_unified_memory_action_graph import graph


class WalkingMemoryGraphSelectorTest(unittest.TestCase):
    def test_state_expansion_is_unique_bounded_and_label_free(self):
        row = build_hierarchical_row(graph())
        states = expand_states(row, depth=2, max_states=4)
        keys = [
            state_key(state, state["incumbent_objects"])
            for state in states]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertLessEqual(len(states), 4)
        self.assertFalse(any("ObjectEntities" in state for state in states))

    def test_training_weights_sum_to_one_per_original_row(self):
        row = build_hierarchical_row(graph())
        gold = {
            ("Repeated Subject", "personHasCityOfDeath"): {
                "SubjectEntity": "Repeated Subject",
                "Relation": "personHasCityOfDeath",
                "ObjectEntities": [["London"]],
            }
        }
        arrays = state_training_arrays([row], gold)
        action_weights = arrays[2]
        gate_weights = arrays[5]
        self.assertAlmostEqual(sum(action_weights), 1.0)
        self.assertAlmostEqual(sum(gate_weights), 1.0)
        self.assertEqual(arrays[6]["row_action_weight_range"], [1.0, 1.0])

    def test_walk_stops_on_cycle(self):
        row = build_hierarchical_row(graph())

        def chooser(view):
            current = view["incumbent_objects"]
            target = ["London"] if current != ["London"] else ["Paris"]
            return target, {"selected_action": "REPLACE"}

        objects, trace = walk_with_chooser(row, chooser, max_steps=5)
        self.assertEqual(objects, ["London"])
        self.assertEqual(trace[-1]["stop_reason"], "cycle_or_no_change")
        self.assertEqual(len(trace), 2)

    def test_walk_stops_on_keep(self):
        row = build_hierarchical_row(graph())
        objects, trace = walk_with_chooser(
            row,
            lambda view: (
                view["incumbent_objects"], {"selected_action": "KEEP"}),
        )
        self.assertEqual(objects, ["Paris"])
        self.assertEqual(trace[-1]["stop_reason"], "keep")


if __name__ == "__main__":
    unittest.main()
