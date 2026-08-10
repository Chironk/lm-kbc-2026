from types import SimpleNamespace

from experiments.heterogeneous_agents.components.coherent_row_selector import (
    GATE_FEATURE_NAMES,
    HIERARCHICAL_FEATURE_NAMES,
    INTERACTION_SIGNAL_NAMES,
    SHARED_FEATURE_NAMES,
    TRUTH_HIERARCHICAL_FEATURE_NAMES,
    _canonical_output,
    _macro_row_weights,
    _pair_masses,
    _ranking_pairs,
)
from experiments.heterogeneous_agents.components.crossfit_action_utility_selector import (
    TRUTH_FEATURE_NAMES,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import (
    RELATIONS,
)


def test_canonical_output_is_set_identity_not_construction_history() -> None:
    assert _canonical_output(
        ["NASDAQ", "NYSE", "NASDAQ"],
        "companyTradesAtStockExchange",
    ) == _canonical_output(
        ["NYSE", "NASDAQ"],
        "companyTradesAtStockExchange",
    )


def test_macro_weights_give_each_relation_equal_total_mass() -> None:
    rows = [
        SimpleNamespace(relation="awardWonBy"),
        *[
            SimpleNamespace(relation="hasArea")
            for _ in range(10)
        ],
    ]
    weights = _macro_row_weights(rows)
    award = sum(
        value for row, value in zip(rows, weights, strict=True)
        if row.relation == "awardWonBy")
    area = sum(
        value for row, value in zip(rows, weights, strict=True)
        if row.relation == "hasArea")
    assert abs(award - area) < 1e-12


def test_hierarchical_schema_is_shared_plus_shrunken_deviations() -> None:
    assert len(HIERARCHICAL_FEATURE_NAMES) == (
        len(SHARED_FEATURE_NAMES)
        + len(RELATIONS) * len(INTERACTION_SIGNAL_NAMES)
    )
    assert len(GATE_FEATURE_NAMES) == (
        len(SHARED_FEATURE_NAMES) + len(RELATIONS) + 3)
    assert len(TRUTH_HIERARCHICAL_FEATURE_NAMES) == (
        len(HIERARCHICAL_FEATURE_NAMES) + len(TRUTH_FEATURE_NAMES))


def test_utility_pair_mass_preserves_row_weight_and_prioritizes_gap() -> None:
    row = SimpleNamespace(alternatives=[
        SimpleNamespace(delta=0.0),
        SimpleNamespace(delta=0.1),
        SimpleNamespace(delta=1.0),
    ])
    pairs = [(0, 1), (0, 2), (1, 2)]
    equal = _pair_masses(row, pairs, utility_weighted=False)
    weighted = _pair_masses(row, pairs, utility_weighted=True)
    assert abs(sum(equal) - 1.0) < 1e-12
    assert abs(sum(weighted) - 1.0) < 1e-12
    assert weighted[1] > weighted[0]
    assert weighted[2] > weighted[0]


def test_helpful_contrast_ignores_ordering_between_two_bad_actions() -> None:
    row = SimpleNamespace(alternatives=[
        SimpleNamespace(delta=-0.8),
        SimpleNamespace(delta=-0.1),
        SimpleNamespace(delta=0.2),
    ])
    assert _ranking_pairs(row, objective="ordinal") == [
        (0, 1), (0, 2), (1, 2)]
    assert _ranking_pairs(row, objective="helpful_contrast") == [
        (0, 2), (1, 2)]
