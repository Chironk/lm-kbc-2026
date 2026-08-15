#!/usr/bin/env python3
"""Frozen production helpers extracted from the historical research module.

Only symbols reached by the public inference and deterministic replay paths
are retained here. The complete pre-consolidation source is preserved in
the local recovery branch ``archive/pre-consolidation-20260814``.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence
import numpy as np
from evaluate import RELATION_TYPE
from lm_kbc.core import ContractError
from lm_kbc.components.dual_model_validation import GEMMA, QWEN

RELATIONS = tuple(sorted(RELATION_TYPE))

PROMPT_POLICIES = ("direct", "shared_cot5", "disjoint_cot5")

def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])

def _prob(commitment: Mapping[str, Any], label: str) -> tuple[float, float]:
    if not commitment.get("available"):
        return 0.0, 1.0
    probs = commitment.get("probabilities", {})
    return float(probs.get(label, 0.0)), 0.0

def feature_names() -> list[str]:
    base = [
        "intercept", "qwen_source", "gemma_source", "cross_model",
        "qwen_support", "gemma_support", "max_support", "mean_support",
        "qwen_selected", "gemma_selected", "candidate_count",
        "qwen_exist_yes", "gemma_exist_yes", "qwen_exist_missing",
        "gemma_exist_missing", "qwen_card_one", "gemma_card_one",
        "qwen_card_many", "gemma_card_many", "numeric_cross_tolerance",
        "numeric_qwen_distance", "numeric_gemma_distance",
    ]
    base += [f"relation={relation}" for relation in RELATIONS]
    base += [f"qwen_source*{relation}" for relation in RELATIONS]
    base += [f"gemma_source*{relation}" for relation in RELATIONS]
    base += [f"support*{relation}" for relation in RELATIONS]
    base += [f"prompt={policy}" for policy in PROMPT_POLICIES]
    return base

def _numeric_value(item: Any) -> float | None:
    try:
        value = float(str(item).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None

def _agent_numeric_center(graph: Mapping[str, Any], agent: str) -> float | None:
    values = []
    for node in graph["candidates"]:
        source = node["sources"].get(agent)
        value = _numeric_value(node["item"])
        if source and value is not None:
            values.extend([value] * max(1, int(source["support"])))
    return statistics.median(values) if values else None

def candidate_features(graph: Mapping[str, Any], node: Mapping[str, Any]) -> list[float]:
    relation = str(graph["Relation"])
    qsource, gsource = node["sources"].get(QWEN), node["sources"].get(GEMMA)
    qs, gs = (float(qsource["support_rate"]) if qsource else 0.0,
              float(gsource["support_rate"]) if gsource else 0.0)
    qyes, qmiss = _prob(graph["agents"][QWEN]["existence"], "YES")
    gyes, gmiss = _prob(graph["agents"][GEMMA]["existence"], "YES")
    qone, _ = _prob(graph["agents"][QWEN]["cardinality"], "ONE")
    gone, _ = _prob(graph["agents"][GEMMA]["cardinality"], "ONE")
    qmany, _ = _prob(graph["agents"][QWEN]["cardinality"], "MANY")
    gmany, _ = _prob(graph["agents"][GEMMA]["cardinality"], "MANY")
    value = _numeric_value(node["item"])
    qcenter, gcenter = _agent_numeric_center(graph, QWEN), _agent_numeric_center(graph, GEMMA)
    def distance(center: float | None) -> float:
        if value is None or center is None:
            return 0.0
        return min(1.0, abs(math.log(value / center)) / 5.0)
    numeric_agree = 0.0
    if value is not None and (qsource or gsource):
        other_agent = GEMMA if qsource and not gsource else QWEN if gsource and not qsource else None
        if qsource and gsource:
            numeric_agree = 1.0
        elif other_agent is not None:
            for other in graph["candidates"]:
                other_value = _numeric_value(other["item"])
                if (other_agent in other["sources"] and other_value is not None
                        and abs(other_value - value) / max(abs(value), 1e-12) <= 0.05):
                    numeric_agree = 1.0
                    break
    values = [
        1.0, float(qsource is not None), float(gsource is not None),
        float(qsource is not None and gsource is not None), qs, gs, max(qs, gs),
        (qs + gs) / max(1, int(qsource is not None) + int(gsource is not None)),
        float(node["selected_by"].get(QWEN, False)),
        float(node["selected_by"].get(GEMMA, False)),
        min(1.0, len(graph["candidates"]) / 20.0),
        qyes, gyes, qmiss, gmiss, qone, gone, qmany, gmany,
        numeric_agree, distance(qcenter), distance(gcenter),
    ]
    values += [float(relation == candidate) for candidate in RELATIONS]
    values += [float(qsource is not None and relation == candidate) for candidate in RELATIONS]
    values += [float(gsource is not None and relation == candidate) for candidate in RELATIONS]
    values += [max(qs, gs) if relation == candidate else 0.0 for candidate in RELATIONS]
    values += [float(graph.get("prompt_policy") == policy) for policy in PROMPT_POLICIES]
    if len(values) != len(feature_names()):
        raise AssertionError("candidate feature schema drift")
    return values

class LogisticCalibrator:
    """Small deterministic L2-regularized logistic calibrator (IRLS)."""

    def __init__(self, names: Sequence[str], l2: float = 2.0):
        self.names = list(names)
        self.l2 = float(l2)
        self.coefficients = np.zeros(len(self.names), dtype=np.float64)

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float],
            weights: Sequence[float] | None = None) -> "LogisticCalibrator":
        matrix = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.names) or len(target) != len(matrix):
            raise ContractError("invalid calibration matrix")
        if not len(target) or set(np.unique(target)) - {0.0, 1.0}:
            raise ContractError("logistic targets must be nonempty binary values")
        sample_weight = (np.ones(len(target), dtype=np.float64) if weights is None
                         else np.asarray(weights, dtype=np.float64))
        if sample_weight.shape != target.shape or np.any(sample_weight <= 0):
            raise ContractError("invalid calibration sample weights")
        beta = np.zeros(matrix.shape[1], dtype=np.float64)
        penalty = np.eye(matrix.shape[1], dtype=np.float64) * self.l2
        penalty[0, 0] = 0.0
        for _ in range(60):
            logits = np.clip(matrix @ beta, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            variance = np.maximum(probs * (1.0 - probs), 1e-6)
            effective = sample_weight * variance
            working = logits + (target - probs) / variance
            lhs = matrix.T @ (matrix * effective[:, None]) + penalty
            rhs = matrix.T @ (effective * working)
            try:
                updated = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            if np.max(np.abs(updated - beta)) < 1e-8:
                beta = updated
                break
            beta = updated
        if not np.all(np.isfinite(beta)):
            raise ContractError("non-finite calibration coefficients")
        self.coefficients = beta
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        logits = np.clip(matrix @ self.coefficients, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict:
        return {"feature_names": self.names, "l2": self.l2,
                "coefficients": self.coefficients.tolist()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogisticCalibrator":
        model = cls(value["feature_names"], float(value["l2"]))
        model.coefficients = np.asarray(value["coefficients"], dtype=np.float64)
        if model.coefficients.shape != (len(model.names),):
            raise ContractError("calibration coefficient shape mismatch")
        return model

def _weighted_median(items: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(items)
    total = sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= total / 2:
            return value
    return ordered[-1][0]

def _decode_numeric(graph: Mapping[str, Any], candidates: Sequence[tuple[dict, float]]) -> tuple[list[str], float]:
    usable = [(node, probability, _numeric_value(node["item"]))
              for node, probability in candidates]
    usable = [(node, probability, value) for node, probability, value in usable
              if value is not None]
    if not usable:
        return [], 0.0
    clusters = []
    for anchor_node, anchor_probability, anchor in usable:
        members = [(node, probability, value) for node, probability, value in usable
                   if abs(value - anchor) / max(abs(anchor), 1e-12) <= 0.05]
        score_value = sum(probability * (0.5 + 0.5 * max(
            source["support_rate"] for source in node["sources"].values()))
                          for node, probability, _ in members)
        clusters.append((score_value, members))
    cluster_score, best = max(clusters, key=lambda row: (row[0], -len(row[1])))
    value = _weighted_median([
        (number, max(1e-6, probability) * sum(
            source["support_rate"] for source in node["sources"].values()))
        for node, probability, number in best
    ])
    return [format(value, ".12g")], min(1.0, cluster_score)
