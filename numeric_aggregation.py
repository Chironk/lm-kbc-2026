"""Production numeric aggregation policies.

These functions are deliberately independent of labeled data. They encode
only frozen aggregation rules selected by prior experiments. Experiment
harnesses may import this module; production must never import an archived
experiment module.
"""
from __future__ import annotations

from typing import List

from run_inference import (aggregate_median, is_pure_numeric_candidate,
                           parse_answer)


def strict_numeric_values(answers: List[str]) -> List[float]:
    """Return purity-gated numeric candidates in generation order."""
    values: List[float] = []
    for answer in answers:
        if not is_pure_numeric_candidate(answer):
            continue
        parsed = parse_answer(answer, is_numeric=True)
        if not parsed:
            continue
        try:
            values.append(float(parsed[0].replace(",", "")))
        except (AttributeError, TypeError, ValueError):
            continue
    return values


def aggregate_quantile(answers: List[str], quantile: float) -> List[str]:
    """Apply the project's frozen strict upward-quantile rule.

    This intentionally preserves the historical v0495/v0501 definition:
    sort strict candidates and select ``min(n - 1, int(quantile * n))``.
    With ten values and q=0.55 this is index five, not an interpolated
    statistical quantile. If no strict candidate survives, retain the
    production median fallback and its never-abstain behavior.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    values = sorted(strict_numeric_values(answers))
    if not values:
        return aggregate_median(answers)
    index = min(len(values) - 1, int(quantile * len(values)))
    value = values[index]
    return [str(int(value))] if value.is_integer() else [str(value)]
