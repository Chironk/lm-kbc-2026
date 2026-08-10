from types import SimpleNamespace

from experiments.heterogeneous_agents.components.coherent_candidate_selector import (
    PAIRED_FEATURE_NAMES,
    PAIRED_HIERARCHICAL_FEATURE_NAMES,
    PAIRED_INTERACTION_SIGNAL_NAMES,
    SHARED_FEATURE_NAMES,
    _candidate_weights,
    _features,
)


def test_candidate_weights_balance_relation_then_row_then_candidate() -> None:
    rows = [
        SimpleNamespace(
            relation="awardWonBy", alternatives=[object(), object()]),
        SimpleNamespace(
            relation="hasArea", alternatives=[object()]),
        SimpleNamespace(
            relation="hasArea",
            alternatives=[object(), object(), object()]),
    ]
    weights = _candidate_weights(rows)
    assert len(weights) == 6
    # Within award, the only row's two candidates split its relation mass.
    assert abs(weights[0] - weights[1]) < 1e-12
    # Within area, each of two rows gets equal mass despite candidate count.
    assert abs(weights[2] - sum(weights[3:6])) < 1e-12
    # Both represented relations receive equal total mass.
    assert abs(sum(weights[:2]) - sum(weights[2:])) < 1e-12


def test_paired_features_compare_challenger_and_incumbent_truth() -> None:
    shared = [0.0] * len(SHARED_FEATURE_NAMES)
    for name, value in {
        "binary_qwen_margin": 0.2,
        "binary_gemma_margin": 0.3,
        "binary_mean_margin": 0.25,
        "binary_direction_agreement": 1.0,
        "added_joint_support": 0.8,
        "removed_joint_support": 0.2,
    }.items():
        shared[SHARED_FEATURE_NAMES.index(name)] = value
    truth = [
        0.8, 0.8, 0.8,
        0.2, 0.2, 0.2,
        0.9, 0.7, 0.3, 0.1,
        0.2, 0.2, 0.8, 0.3,
    ]
    action = SimpleNamespace(
        shared_features=tuple(shared),
        source=SimpleNamespace(
            truth_features=tuple(truth),
            action={"action_type": "REPLACE"},
        ),
    )
    row = SimpleNamespace(relation="hasArea")
    values = _features(
        row,
        action,
        hierarchical=False,
        include_truth=False,
        paired=True,
        relation_bias=False,
    )
    assert len(values) == len(PAIRED_FEATURE_NAMES)
    assert abs(values[2] - 0.6) < 1e-12
    assert abs(values[3] - 0.6) < 1e-12
    assert abs(values[4] - 0.6) < 1e-12
    hierarchical = _features(
        row,
        action,
        hierarchical=False,
        include_truth=False,
        paired=True,
        relation_bias="interactions",
    )
    assert len(hierarchical) == len(PAIRED_HIERARCHICAL_FEATURE_NAMES)
    assert len(hierarchical) == (
        len(PAIRED_FEATURE_NAMES)
        + 6
        + 6 * len(PAIRED_INTERACTION_SIGNAL_NAMES)
    )
