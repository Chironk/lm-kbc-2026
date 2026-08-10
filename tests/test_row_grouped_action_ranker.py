import unittest

import numpy as np

from experiments.heterogeneous_agents.row_grouped_action_ranker import (
    COMPACT_FEATURE_NAMES,
    PairwiseRanker,
    compact_action_features,
    oriented_pair_examples,
)
from experiments.heterogeneous_agents.unified_memory_action_graph import (
    FEATURE_NAMES,
)


class RowGroupedActionRankerTest(unittest.TestCase):
    def test_keep_is_explicit_and_has_no_review_leakage(self):
        values = [0.0] * len(FEATURE_NAMES)
        values[FEATURE_NAMES.index("action_keep")] = 1.0
        graph = {
            "SubjectEntity": "Example",
            "Relation": "hasCapacity",
            "relation_family": "numeric",
        }
        action = {
            "id": "action:0",
            "action_type": "KEEP",
            "_inference_features": values,
        }
        compact = compact_action_features(
            graph, action, {}, include_review=True)
        self.assertEqual(len(compact), len(COMPACT_FEATURE_NAMES))
        self.assertEqual(compact[0], 1.0)
        self.assertEqual(compact[-3:], [0.0, 0.0, 0.0])

    def test_neutral_pair_prefers_keep(self):
        keep = [0.0] * len(COMPACT_FEATURE_NAMES)
        alternative = [0.0] * len(COMPACT_FEATURE_NAMES)
        keep[0] = 1.0
        alternative[3] = 1.0
        x, y = oriented_pair_examples(
            [keep, alternative], [0.5, 0.5], keep_index=0)
        self.assertEqual(y, [1.0, 0.0])
        self.assertEqual(x[0][0], 1.0)
        self.assertEqual(x[0][3], -1.0)

    def test_pairwise_ranker_learns_group_preference(self):
        keep = [0.0] * len(COMPACT_FEATURE_NAMES)
        good = [0.0] * len(COMPACT_FEATURE_NAMES)
        bad = [0.0] * len(COMPACT_FEATURE_NAMES)
        keep[0] = 1.0
        good[3] = 1.0
        good[11] = 1.0
        bad[3] = 1.0
        bad[11] = -1.0
        groups = (
            [([keep, good, bad], [0.5, 1.0, 0.0], 0)] * 20
            + [([keep, bad], [0.5, 0.0], 0)] * 20
        )
        model = PairwiseRanker(l2=0.1).fit(groups)
        scores = model.scores([keep, good, bad])
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[0], scores[2])
        self.assertEqual(model.parameter_count, 27)
        self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
