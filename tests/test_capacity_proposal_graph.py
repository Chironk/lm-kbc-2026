from experiments.heterogeneous_agents.capacity_proposal_graph import (
    ARM_ROUTES,
    ROUTE,
    augment_capacity_row,
)
from experiments.heterogeneous_agents.capacity_proposal_selector import (
    proposal_action,
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
                    "samples": 3,
                    "support": 2,
                    "support_rate": 2 / 3,
                    "selected": True,
                },
            },
            "route_summary": {},
        }],
        "proposal_routes": {
            "qwen:self_consistency": {
                "model_family": "qwen_recall",
                "available": True,
                "n_samples": 3,
            },
        },
    }


def test_new_capacity_route_survives_component_construction():
    row = augment_capacity_row(_graph(), [12000.0, 12000.0, 12000.0])
    component = next(
        item for item in row["relational_graph"]["components"]
        if "12000" in item["member_items"])
    assert component["routes"][ROUTE]["max_support_rate"] == 1.0
    assert any(
        node.get("id") == f"route:{ROUTE}"
        for node in row["relational_graph"]["nodes"])


def test_unanimous_new_component_is_actionable():
    row = augment_capacity_row(_graph(), [12000.0, 12000.0, 12000.0])
    assert proposal_action(row, "new_unanimous") == "12000"


def test_majority_rule_rejects_singleton():
    row = augment_capacity_row(_graph(), [12000.0, 13000.0, 14000.0])
    assert proposal_action(row, "new_majority") is None


def test_exact_existing_surface_merges_instead_of_duplicating():
    row = augment_capacity_row(_graph(), [10000.0, 10000.0, 12000.0])
    assert sum(candidate["item"] == "10000"
               for candidate in row["candidates"]) == 1


def test_multiview_routes_survive_as_distinct_evidence():
    routes = {
        ARM_ROUTES["gemma_direct"]: [12000.0] * 3,
        ARM_ROUTES["gemma_magnitude"]: [12000.0, 13000.0, 14000.0],
        ARM_ROUTES["qwen_direct"]: [12000.0, 15000.0, 16000.0],
    }
    row = augment_capacity_row(_graph(), routes)
    component = next(
        item for item in row["relational_graph"]["components"]
        if "12000" in item["member_items"])
    assert set(ARM_ROUTES.values()) <= set(component["routes"])
