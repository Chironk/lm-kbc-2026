from experiments.heterogeneous_agents.components.crossfit_action_utility_selector import (
    _percentile_ranks,
)


def test_percentile_ranks_are_row_relative_and_deterministic() -> None:
    ranks = _percentile_ranks({"b": 2.0, "a": 2.0, "c": 5.0})
    assert ranks == {"a": -1.0, "b": 0.0, "c": 1.0}
    assert _percentile_ranks({}) == {}
    assert _percentile_ranks({"only": -100.0}) == {"only": 0.0}
