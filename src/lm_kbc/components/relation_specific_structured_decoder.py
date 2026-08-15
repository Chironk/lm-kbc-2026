#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

from typing import Any, Mapping

CITY = "personHasCityOfDeath"

COMPANY = "companyTradesAtStockExchange"

def _prob(graph: Mapping[str, Any], agent: str, phase: str, label: str) -> float:
    commitment = graph["agents"][agent][phase]
    if not commitment.get("available"):
        return 0.0
    return float(commitment.get("probabilities", {}).get(label, 0.0))

def _support(node: Mapping[str, Any], agent: str) -> float:
    source = node.get("sources", {}).get(agent)
    return float(source.get("support_rate", 0.0)) if source else 0.0
