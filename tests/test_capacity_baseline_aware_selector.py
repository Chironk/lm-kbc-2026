import numpy as np

from experiments.heterogeneous_agents.components.capacity_baseline_aware_selector import (
    FEATURE_NAMES,
    RidgeActionModel,
    _actions,
    _decode,
    action_features,
)
from experiments.heterogeneous_agents.components.capacity_proposal_graph import (
    ARM_ROUTES,
    augment_capacity_row,
)


def _graph():
    return {
        "SubjectEntity": "Example Stadium",
        "Relation": "hasCapacity",
        "baseline_objects": ["10000"],
        "candidates": [{
            "key": "numeric:10000",
            "item": "10000",
            "type": "numeric",
            "sources": {},
            "selected_by": {},
            "routes": {
                "qwen:self_consistency": {
                    "model_family": "qwen_recall",
                    "samples": 10,
                    "support": 7,
                    "support_rate": 0.7,
                    "selected": True,
                },
            },
            "route_summary": {},
        }],
        "proposal_routes": {},
    }


def _augmented():
    return augment_capacity_row(_graph(), {
        ARM_ROUTES["gemma_direct"]: [12000.0] * 3,
        ARM_ROUTES["gemma_magnitude"]: [12000.0, 13000.0, 14000.0],
        ARM_ROUTES["qwen_direct"]: [12000.0, 15000.0, 16000.0],
    })


def test_features_compare_candidate_and_incumbent_evidence():
    graph = _augmented()
    component = next(
        item for item in graph["relational_graph"]["components"]
        if "12000" in item["member_items"])
    features = action_features(graph, component, 12000.0, 10000.0)
    assert len(features) == len(FEATURE_NAMES)
    assert np.isfinite(features).all()
    assert features[FEATURE_NAMES.index("incumbent_legacy_support_sum")] > 0
    assert features[FEATURE_NAMES.index("candidate_proposal_views")] == 3


def test_candidate_heavy_row_has_total_weight_one():
    actions = _actions(_augmented(), {
        "SubjectEntity": "Example Stadium",
        "Relation": "hasCapacity",
        "ObjectEntities": [["12000"]],
    })
    assert actions
    assert abs(sum(action.weight for action in actions) - 1.0) < 1e-12


def test_closed_gate_preserves_external_production_incumbent():
    graph = _augmented()
    actions = _actions(graph, {
        "SubjectEntity": "Example Stadium",
        "Relation": "hasCapacity",
        "ObjectEntities": [["12000"]],
    })
    model = RidgeActionModel(1.0).fit(actions)
    # The current production prediction may differ from graph.baseline_objects.
    prediction, detail = _decode(
        model, graph, float("inf"), control=["10500"])
    assert prediction == ["10500"]
    assert detail["used_baseline"] is True
