#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence
import numpy as np
from lm_kbc.core import ContractError, canonical_key
from lm_kbc.components.dual_model_validation import GEMMA, QWEN
from lm_kbc.components.heterogeneous_memory_selector import _key
from lm_kbc.components.relation_specific_structured_decoder import CITY, COMPANY, _prob, _support
from lm_kbc.components.route_aware_candidate_graph import ROUTE_GEMMA, ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2

ARMS = ("base_residual", "route_agreement", "route_full")

def _object_key(item: str, relation: str) -> str:
    return canonical_key(str(item), relation)

def _action_key(objects: Sequence[str], relation: str) -> tuple[str, ...]:
    return tuple(sorted({
        _object_key(str(item), relation) for item in objects
        if _object_key(str(item), relation)
    }))

def _dedupe_actions(
        actions: Sequence[Sequence[str]], relation: str,
) -> list[list[str]]:
    unique: dict[tuple[str, ...], list[str]] = {}
    for action in actions:
        values = list(dict.fromkeys(str(item) for item in action))
        unique.setdefault(_action_key(values, relation), values)
    return list(unique.values())

def _eligible_nodes(
        graph: Mapping[str, Any], arm: str,
) -> list[Mapping[str, Any]]:
    if arm not in ARMS:
        raise ContractError(f"unknown decoder arm {arm}")
    nodes = list(graph.get("candidates", []))
    if arm == "route_agreement":
        return [
            node for node in nodes
            if not node.get("route_summary", {}).get("system2_only", False)
        ]
    return nodes

def actions_for(
        graph: Mapping[str, Any], control: Sequence[str], arm: str,
) -> list[list[str]]:
    """Return predeclared, bounded edits around the incumbent."""
    relation = str(graph["Relation"])
    nodes = _eligible_nodes(graph, arm)
    if relation == CITY:
        return _dedupe_actions([
            list(control[:1]),
            [],
            *[[str(node["item"])] for node in nodes],
        ], relation)
    if relation != COMPANY:
        raise ContractError(f"unsupported relation {relation}")
    actions: list[list[str]] = [list(control), []]
    control_keys = set(_action_key(control, relation))
    for node in nodes:
        if str(node["key"]) not in control_keys:
            actions.append([*control, str(node["item"])])
    for item in control:
        key = _object_key(str(item), relation)
        actions.append([
            value for value in control
            if _object_key(str(value), relation) != key
        ])
    return _dedupe_actions(actions, relation)

def _node_by_key(
        graph: Mapping[str, Any], arm: str,
) -> dict[str, Mapping[str, Any]]:
    return {
        str(node["key"]): node for node in _eligible_nodes(graph, arm)
    }

def _route_values(node: Mapping[str, Any] | None) -> dict[str, float]:
    if node is None:
        return {
            "qwen_sc": 0.0, "system2": 0.0, "gemma": 0.0,
            "route_count": 0.0, "family_count": 0.0,
            "cross_model": 0.0, "within_qwen": 0.0,
            "all_three": 0.0, "system2_only": 0.0,
            "qwen_sc_only": 0.0, "gemma_only": 0.0,
        }
    routes = node.get("routes")
    if isinstance(routes, Mapping):
        names = set(routes)
        summary = node.get("route_summary", {})
        qwen_sc = float(ROUTE_QWEN_SC in names)
        system2 = float(ROUTE_QWEN_SYSTEM2 in names)
        gemma = float(ROUTE_GEMMA in names)
        return {
            "qwen_sc": qwen_sc,
            "system2": system2,
            "gemma": gemma,
            "route_count": min(1.0, len(names) / 3.0),
            "family_count": min(
                1.0, float(summary.get("model_family_count", 0)) / 2.0),
            "cross_model": float(
                summary.get("cross_model_agreement", False)),
            "within_qwen": float(
                summary.get("within_qwen_route_agreement", False)),
            "all_three": float(
                {ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2, ROUTE_GEMMA} <= names),
            "system2_only": float(summary.get("system2_only", False)),
            "qwen_sc_only": float(summary.get("qwen_sc_only", False)),
            "gemma_only": float(summary.get("gemma_only", False)),
        }
    sources = set(node.get("sources", {}))
    qwen_sc = float(QWEN in sources)
    gemma = float(GEMMA in sources)
    return {
        "qwen_sc": qwen_sc, "system2": 0.0, "gemma": gemma,
        "route_count": min(1.0, len(sources) / 3.0),
        "family_count": min(1.0, len(sources) / 2.0),
        "cross_model": float(qwen_sc and gemma),
        "within_qwen": 0.0, "all_three": 0.0, "system2_only": 0.0,
        "qwen_sc_only": float(sources == {QWEN}),
        "gemma_only": float(sources == {GEMMA}),
    }

def feature_names() -> list[str]:
    return [
        "relation_city", "control_empty", "action_empty",
        "control_size", "action_size", "size_delta",
        "noop", "add", "drop", "replace", "multi_edit",
        "overlap", "candidate_present",
        "qwen_support", "gemma_support", "qwen_selected",
        "gemma_selected", "qwen_sc_route", "system2_route",
        "gemma_route", "route_count", "model_family_count",
        "cross_model", "within_qwen", "all_three",
        "system2_only", "qwen_sc_only", "gemma_only",
        "qwen_none_rate", "gemma_none_rate",
        "qwen_exist_yes", "gemma_exist_yes",
        "qwen_exist_no", "gemma_exist_no",
        "qwen_card_zero", "gemma_card_zero",
        "qwen_card_one", "gemma_card_one",
        "qwen_card_many", "gemma_card_many",
        "candidate_count", "action_cardinality_gap",
    ]

def action_features(
        graph: Mapping[str, Any], control: Sequence[str],
        action: Sequence[str], arm: str,
) -> list[float]:
    relation = str(graph["Relation"])
    control_keys = set(_action_key(control, relation))
    action_keys = set(_action_key(action, relation))
    added = sorted(action_keys - control_keys)
    dropped = sorted(control_keys - action_keys)
    changed_key = added[0] if added else (dropped[0] if dropped else None)
    node = _node_by_key(graph, arm).get(changed_key) if changed_key else None
    route = _route_values(node)
    edit_count = len(added) + len(dropped)
    expected_cardinality = statistics.mean([
        _prob(graph, agent, "cardinality", "ONE")
        + 2.0 * _prob(graph, agent, "cardinality", "MANY")
        for agent in (QWEN, GEMMA)
    ])
    values = [
        float(relation == CITY),
        float(not control_keys), float(not action_keys),
        min(1.0, len(control_keys) / 5.0),
        min(1.0, len(action_keys) / 5.0),
        max(-1.0, min(1.0, (len(action_keys) - len(control_keys)) / 3.0)),
        float(edit_count == 0),
        float(bool(added) and not dropped),
        float(bool(dropped) and not added),
        float(bool(added) and bool(dropped) and edit_count == 2),
        float(edit_count > 2),
        len(control_keys & action_keys) / max(1, len(control_keys | action_keys)),
        float(node is not None),
        _support(node, QWEN) if node is not None else 0.0,
        _support(node, GEMMA) if node is not None else 0.0,
        float(node is not None and node.get(
            "selected_by", {}).get(QWEN, False)),
        float(node is not None and node.get(
            "selected_by", {}).get(GEMMA, False)),
        route["qwen_sc"], route["system2"], route["gemma"],
        route["route_count"], route["family_count"],
        route["cross_model"], route["within_qwen"], route["all_three"],
        route["system2_only"], route["qwen_sc_only"], route["gemma_only"],
        float(graph["agents"][QWEN]["none_rate"]),
        float(graph["agents"][GEMMA]["none_rate"]),
        _prob(graph, QWEN, "existence", "YES"),
        _prob(graph, GEMMA, "existence", "YES"),
        _prob(graph, QWEN, "existence", "NO"),
        _prob(graph, GEMMA, "existence", "NO"),
        _prob(graph, QWEN, "cardinality", "ZERO"),
        _prob(graph, GEMMA, "cardinality", "ZERO"),
        _prob(graph, QWEN, "cardinality", "ONE"),
        _prob(graph, GEMMA, "cardinality", "ONE"),
        _prob(graph, QWEN, "cardinality", "MANY"),
        _prob(graph, GEMMA, "cardinality", "MANY"),
        min(1.0, len(_eligible_nodes(graph, arm)) / 10.0),
        min(1.0, abs(len(action_keys) - expected_cardinality) / 4.0),
    ]
    if len(values) != len(feature_names()):
        raise AssertionError("residual action feature schema drift")
    return values

class ResidualRidge:
    """Weighted standardized ridge whose signed predictions are not clipped."""

    def __init__(self, names: Sequence[str], l2: float = 10.0):
        self.names = list(names)
        self.l2 = float(l2)
        self.mean = np.zeros(len(names), dtype=np.float64)
        self.scale = np.ones(len(names), dtype=np.float64)
        self.coefficients = np.zeros(len(names) + 1, dtype=np.float64)

    def fit(
        self, x: Sequence[Sequence[float]], y: Sequence[float],
        weights: Sequence[float],
    ) -> "ResidualRidge":
        matrix = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        if (matrix.ndim != 2 or matrix.shape[1] != len(self.names)
                or len(matrix) == 0):
            raise ContractError("invalid residual action matrix")
        if target.shape != weight.shape or target.shape != (len(matrix),):
            raise ContractError("invalid residual action targets")
        if np.any(weight <= 0) or not np.all(np.isfinite(matrix)):
            raise ContractError("invalid residual action weights/features")
        weight = weight * (len(weight) / weight.sum())
        self.mean = np.average(matrix, axis=0, weights=weight)
        variance = np.average(
            (matrix - self.mean) ** 2, axis=0, weights=weight)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = np.column_stack([
            np.ones(len(matrix)), (matrix - self.mean) / self.scale])
        root = np.sqrt(weight)[:, None]
        lhs = (design * root).T @ (design * root)
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.l2
        penalty[0, 0] = 0.0
        rhs = (design * root).T @ (target * root[:, 0])
        self.coefficients = np.linalg.solve(lhs + penalty, rhs)
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.names):
            raise ContractError("invalid residual prediction matrix")
        design = np.column_stack([
            np.ones(len(matrix)), (matrix - self.mean) / self.scale])
        return design @ self.coefficients

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "signed-standardized-residual-ridge-v1",
            "feature_names": self.names,
            "l2": self.l2,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
        }

def decode(
        model: ResidualRidge, graph: Mapping[str, Any],
        control: Sequence[str], arm: str, margin: float,
) -> tuple[list[str], dict[str, Any]]:
    actions = actions_for(graph, control, arm)
    estimates = model.predict([
        action_features(graph, control, action, arm) for action in actions])
    best = max(range(len(actions)), key=lambda index: (
        float(estimates[index]), -len(actions[index]),
        _action_key(actions[index], str(graph["Relation"]))))
    proposed = actions[best]
    estimated_delta = float(estimates[best])
    use_control = estimated_delta <= margin
    return (list(control) if use_control else list(proposed)), {
        "arm": arm,
        "control_objects": list(control),
        "proposed_objects": list(proposed),
        "estimated_f1_delta": estimated_delta,
        "guard_margin": float(margin),
        "used_control": use_control,
        "action_count": len(actions),
    }

def _prediction_rows(
        control_rows: Sequence[Mapping[str, Any]],
        replacements: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict[str, Any]]:
    return [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": list(replacements.get(
            _key(row), row.get("ObjectEntities", []))),
    } for row in control_rows]
