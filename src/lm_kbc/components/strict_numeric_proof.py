#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
from lm_kbc.components.capacity_graph_decoder import FAMILIES, TOLERANCE, _component_values, _format, capacity_options
from lm_kbc.core import ContractError

MIN_EXACT_FAMILY_FRACTION = 2.0 / 3.0

MIN_EXACT_FAMILY_ADVANTAGE = 2.0 / 3.0

EPSILON = 1e-12

def _coherent_component(
    graph: Mapping[str, Any], option: Mapping[str, Any],
) -> bool:
    values = _component_values(graph)
    members = [values.get(str(component)) for component in option["component_ids"]]
    if not members or any(value is None for value in members):
        return False
    numeric = [float(value) for value in members if value is not None]
    return all(
        abs(left - right) / max(abs(right), 1e-12) <= TOLERANCE + EPSILON
        for left in numeric for right in numeric
    )

def strict_numeric_eligible(
    graph: Mapping[str, Any], challenger: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> bool:
    """Return whether a scalar challenger satisfies the frozen proof."""
    if str(graph["Relation"]) not in {"hasArea", "hasCapacity"}:
        raise ContractError("strict numeric proof applied to nonnumeric relation")
    if not challenger or not incumbent:
        return False
    # Both hypotheses are positive, non-empty singleton numeric components.
    if not (
        math.isfinite(float(challenger["value"]))
        and float(challenger["value"]) > 0.0
        and math.isfinite(float(incumbent["value"]))
        and float(incumbent["value"]) > 0.0
    ):
        return False
    if not _coherent_component(graph, challenger):
        return False

    challenger_fraction = float(challenger["family_fraction"])
    incumbent_fraction = float(incumbent["family_fraction"])
    if challenger_fraction + EPSILON < MIN_EXACT_FAMILY_FRACTION:
        return False
    if (
        challenger_fraction - incumbent_fraction + EPSILON
        < MIN_EXACT_FAMILY_ADVANTAGE
    ):
        return False

    # For scalar events, a family is compatible exactly when at least one of
    # its generations lands in the candidate's 5% component.  Requiring all
    # three families prevents a two-family majority from overriding a third
    # family that generated only incompatible values.
    if any(float(challenger["rates"].get(family, 0.0)) <= 0.0
           for family in FAMILIES):
        return False
    return True

def decode_row(
    graph: Mapping[str, Any], incumbent_objects: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    options = capacity_options(graph, incumbent_objects)
    incumbent = next(option for option in options if option["is_incumbent"])
    eligible = [
        option for option in options
        if not option["is_incumbent"]
        and strict_numeric_eligible(graph, option, incumbent)
    ]
    selected = max(
        eligible,
        key=lambda option: (
            float(option["family_fraction"]),
            float(option["minimum_family_rate"]),
            float(option["mean_family_rate"]),
            float(option["total_event_rate"]),
            -abs(math.log(float(option["value"]) / float(incumbent["value"]))),
            -float(option["value"]),
        ),
        default=incumbent,
    )
    return [_format(float(selected["value"]))], {
        "changed": selected is not incumbent,
        "incumbent": [_format(float(incumbent["value"]))],
        "selected": [_format(float(selected["value"]))],
        "eligible_options": len(eligible),
        "incumbent_family_fraction": float(incumbent["family_fraction"]),
        "selected_family_fraction": float(selected["family_fraction"]),
        "selected_family_rates": dict(selected["rates"]),
        "cardinality_contract": "equal_nonempty_singletons",
    }
