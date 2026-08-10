#!/usr/bin/env python3
"""Train-OOF city and company structured risk decoding.

The city head factorizes null detection from conditional candidate identity.
The company head scores complete candidate sets rather than thresholding
candidate probabilities independently.  Both heads are trained and gated with
the frozen five-fold training split.  Validation labels are opened only after
all prediction artifacts have been written.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluate import RELATION_TYPE, true_positives
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.explicit_cardinality_ablation import (
    ExplicitCardinalityModel,
    _prediction_rows as explicit_prediction_rows,
)
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    LogisticCalibrator,
    _fit_models,
    _key,
    _load_graph,
)


CITY = "personHasCityOfDeath"
COMPANY = "companyTradesAtStockExchange"
RELATIONS = (CITY, COMPANY)
DEFAULT_MARGINS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
CURRENT_CONTROL_MARGIN = 0.20


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _prob(graph: Mapping[str, Any], agent: str, phase: str, label: str) -> float:
    commitment = graph["agents"][agent][phase]
    if not commitment.get("available"):
        return 0.0
    return float(commitment.get("probabilities", {}).get(label, 0.0))


def _support(node: Mapping[str, Any], agent: str) -> float:
    source = node.get("sources", {}).get(agent)
    return float(source.get("support_rate", 0.0)) if source else 0.0


def _candidate_key(item: str, relation: str) -> str:
    return canonical_key(str(item), relation)


def _gold_aliases(row: Mapping[str, Any]) -> list[list[str]]:
    values = row.get("ObjectEntities", [])
    return [[str(item)] for item in values] if values and isinstance(values[0], str) else values


def _row_f1(
        objects: Sequence[str], gold: Mapping[str, Any], relation: str,
) -> float:
    aliases = _gold_aliases(gold)
    predictions = list(dict.fromkeys(str(item) for item in objects))
    tp = true_positives(
        predictions, aliases, RELATION_TYPE[relation], 0.05)
    precision = tp / len(predictions) if predictions else 1.0
    recall = tp / len(aliases) if aliases else 1.0
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0)


def city_null_feature_names() -> list[str]:
    return [
        "intercept",
        "qwen_none_rate", "gemma_none_rate",
        "qwen_exist_no", "gemma_exist_no",
        "qwen_exist_yes", "gemma_exist_yes",
        "qwen_card_zero", "gemma_card_zero",
        "qwen_card_one", "gemma_card_one",
        "candidate_count", "max_qwen_support", "max_gemma_support",
        "shared_candidate", "qwen_parse_failure_rate",
        "gemma_parse_failure_rate", "graph_baseline_empty",
    ]


def city_null_features(graph: Mapping[str, Any]) -> list[float]:
    nodes = graph["candidates"]
    qn = max(1.0, float(graph["agents"][QWEN]["n_samples"]))
    gn = max(1.0, float(graph["agents"][GEMMA]["n_samples"]))
    values = [
        1.0,
        float(graph["agents"][QWEN]["none_rate"]),
        float(graph["agents"][GEMMA]["none_rate"]),
        _prob(graph, QWEN, "existence", "NO"),
        _prob(graph, GEMMA, "existence", "NO"),
        _prob(graph, QWEN, "existence", "YES"),
        _prob(graph, GEMMA, "existence", "YES"),
        _prob(graph, QWEN, "cardinality", "ZERO"),
        _prob(graph, GEMMA, "cardinality", "ZERO"),
        _prob(graph, QWEN, "cardinality", "ONE"),
        _prob(graph, GEMMA, "cardinality", "ONE"),
        min(1.0, len(nodes) / 10.0),
        max((_support(node, QWEN) for node in nodes), default=0.0),
        max((_support(node, GEMMA) for node in nodes), default=0.0),
        float(any({QWEN, GEMMA} <= set(node["sources"]) for node in nodes)),
        min(1.0, float(graph["agents"][QWEN]["parse_failures"]) / qn),
        min(1.0, float(graph["agents"][GEMMA]["parse_failures"]) / gn),
        float(not graph.get("baseline_objects")),
    ]
    if len(values) != len(city_null_feature_names()):
        raise AssertionError("city null feature schema drift")
    return values


def city_candidate_feature_names() -> list[str]:
    return [
        "intercept", "qwen_source", "gemma_source", "cross_model",
        "qwen_support", "gemma_support", "max_support",
        "qwen_selected", "gemma_selected",
        "candidate_count", "support_rank", "support_gap",
        "qwen_exist_yes", "gemma_exist_yes",
        "qwen_card_one", "gemma_card_one",
        "qwen_none_rate", "gemma_none_rate",
    ]


def city_candidate_features(
        graph: Mapping[str, Any], node: Mapping[str, Any],
) -> list[float]:
    qsource = QWEN in node["sources"]
    gsource = GEMMA in node["sources"]
    support = max(_support(node, QWEN), _support(node, GEMMA))
    ordered = sorted(
        [max(_support(item, QWEN), _support(item, GEMMA))
         for item in graph["candidates"]],
        reverse=True)
    rank = next(
        (index for index, value in enumerate(ordered)
         if math.isclose(value, support)), len(ordered))
    next_support = ordered[rank + 1] if rank + 1 < len(ordered) else 0.0
    values = [
        1.0, float(qsource), float(gsource), float(qsource and gsource),
        _support(node, QWEN), _support(node, GEMMA), support,
        float(node["selected_by"].get(QWEN, False)),
        float(node["selected_by"].get(GEMMA, False)),
        min(1.0, len(graph["candidates"]) / 10.0),
        1.0 / (rank + 1.0),
        max(0.0, support - next_support),
        _prob(graph, QWEN, "existence", "YES"),
        _prob(graph, GEMMA, "existence", "YES"),
        _prob(graph, QWEN, "cardinality", "ONE"),
        _prob(graph, GEMMA, "cardinality", "ONE"),
        float(graph["agents"][QWEN]["none_rate"]),
        float(graph["agents"][GEMMA]["none_rate"]),
    ]
    if len(values) != len(city_candidate_feature_names()):
        raise AssertionError("city candidate feature schema drift")
    return values


class CityConditionalDecoder:
    def __init__(self, l2: float = 2.0):
        self.l2 = float(l2)
        self.null_model = LogisticCalibrator(
            city_null_feature_names(), l2=l2)
        self.candidate_model = LogisticCalibrator(
            city_candidate_feature_names(), l2=l2)

    def fit(
            self, graphs: Sequence[Mapping[str, Any]],
            gold: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> "CityConditionalDecoder":
        rows = [graph for graph in graphs if graph["Relation"] == CITY]
        self.null_model.fit(
            [city_null_features(graph) for graph in rows],
            [float(not gold[_key(graph)]["ObjectEntities"]) for graph in rows],
        )
        x, y, weights = [], [], []
        for graph in rows:
            target = gold[_key(graph)]
            if not target["ObjectEntities"] or not graph["candidates"]:
                continue
            row_weight = 1.0 / len(graph["candidates"])
            for node in graph["candidates"]:
                x.append(city_candidate_features(graph, node))
                y.append(float(true_positives(
                    [node["item"]], _gold_aliases(target), "string", 0.05) > 0))
                weights.append(row_weight)
        self.candidate_model.fit(x, y, weights)
        return self

    def action_utilities(
            self, graph: Mapping[str, Any],
    ) -> tuple[dict[tuple[str, ...], float], dict[str, Any]]:
        p_null = float(self.null_model.predict(
            [city_null_features(graph)])[0])
        probabilities = (
            self.candidate_model.predict([
                city_candidate_features(graph, node)
                for node in graph["candidates"]])
            if graph["candidates"] else np.asarray([], dtype=np.float64))
        actions: dict[tuple[str, ...], float] = {(): p_null}
        for node, conditional in zip(graph["candidates"], probabilities):
            actions[(str(node["item"]),)] = (
                (1.0 - p_null) * float(conditional))
        return actions, {
            "p_null": p_null,
            "conditional_candidate_probabilities": [
                {
                    "item": node["item"],
                    "probability": float(probability),
                }
                for node, probability in zip(
                    graph["candidates"], probabilities)
            ],
        }

    def decode(
            self, graph: Mapping[str, Any], control: Sequence[str],
            margin: float,
    ) -> tuple[list[str], dict[str, Any]]:
        actions, detail = self.action_utilities(graph)
        proposed = max(
            actions, key=lambda action: (actions[action], -len(action), action))
        control_tuple = tuple(control[:1])
        control_key = (
            _candidate_key(control_tuple[0], CITY) if control_tuple else None)
        matched_control = next((
            action for action in actions
            if len(action) == len(control_tuple)
            and (
                not action
                or _candidate_key(action[0], CITY) == control_key)
        ), None)
        control_utility = (
            actions[matched_control] if matched_control is not None else 0.0)
        proposed_utility = actions[proposed]
        use_control = control_utility + margin >= proposed_utility
        selected = control_tuple if use_control else proposed
        return list(selected), {
            **detail,
            "control_objects": list(control_tuple),
            "control_utility": control_utility,
            "proposed_objects": list(proposed),
            "proposed_utility": proposed_utility,
            "estimated_improvement": proposed_utility - control_utility,
            "guard_margin": float(margin),
            "used_control": use_control,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "city-null-conditional-candidate-v1",
            "l2": self.l2,
            "null_model": self.null_model.to_dict(),
            "conditional_candidate_model": self.candidate_model.to_dict(),
        }


class StandardizedRidge:
    def __init__(self, names: Sequence[str], l2: float = 5.0):
        self.names = list(names)
        self.l2 = float(l2)
        self.mean = np.zeros(len(names), dtype=np.float64)
        self.scale = np.ones(len(names), dtype=np.float64)
        self.coefficients = np.zeros(len(names) + 1, dtype=np.float64)

    def fit(
            self, x: Sequence[Sequence[float]], y: Sequence[float],
            weights: Sequence[float],
    ) -> "StandardizedRidge":
        matrix = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.names):
            raise ContractError("invalid company set-risk matrix")
        if weight.shape != target.shape or np.any(weight <= 0):
            raise ContractError("invalid company set-risk weights")
        weight = weight * (len(weight) / weight.sum())
        self.mean = np.average(matrix, axis=0, weights=weight)
        variance = np.average(
            (matrix - self.mean) ** 2, axis=0, weights=weight)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = np.column_stack([
            np.ones(len(matrix)),
            (matrix - self.mean) / self.scale,
        ])
        root = np.sqrt(weight)[:, None]
        lhs = (design * root).T @ (design * root)
        penalty = np.eye(design.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        rhs = (design * root).T @ (target * root[:, 0])
        self.coefficients = np.linalg.solve(lhs + penalty, rhs)
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        design = np.column_stack([
            np.ones(len(matrix)),
            (matrix - self.mean) / self.scale,
        ])
        return np.clip(design @ self.coefficients, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "standardized-company-set-risk-ridge-v1",
            "feature_names": self.names,
            "l2": self.l2,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
        }


def company_candidate_feature_names() -> list[str]:
    return [
        "intercept", "qwen_source", "gemma_source", "cross_model",
        "qwen_support", "gemma_support", "max_support",
        "qwen_selected", "gemma_selected", "candidate_count",
        "qwen_none_rate", "gemma_none_rate",
        "qwen_exist_yes", "gemma_exist_yes",
        "qwen_card_many", "gemma_card_many",
    ]


def company_candidate_features(
        graph: Mapping[str, Any], node: Mapping[str, Any],
) -> list[float]:
    qsource, gsource = QWEN in node["sources"], GEMMA in node["sources"]
    values = [
        1.0, float(qsource), float(gsource), float(qsource and gsource),
        _support(node, QWEN), _support(node, GEMMA),
        max(_support(node, QWEN), _support(node, GEMMA)),
        float(node["selected_by"].get(QWEN, False)),
        float(node["selected_by"].get(GEMMA, False)),
        min(1.0, len(graph["candidates"]) / 10.0),
        float(graph["agents"][QWEN]["none_rate"]),
        float(graph["agents"][GEMMA]["none_rate"]),
        _prob(graph, QWEN, "existence", "YES"),
        _prob(graph, GEMMA, "existence", "YES"),
        _prob(graph, QWEN, "cardinality", "MANY"),
        _prob(graph, GEMMA, "cardinality", "MANY"),
    ]
    return values


def company_set_feature_names() -> list[str]:
    return [
        "empty", "size", "expected_tp", "plugin_expected_f1",
        "mean_probability", "min_probability", "max_probability",
        "omitted_max_probability", "expected_cardinality",
        "size_cardinality_gap", "qwen_support_sum", "gemma_support_sum",
        "shared_count", "qwen_only_count", "gemma_only_count",
        "qwen_selected_count", "gemma_selected_count",
        "control_overlap", "added_vs_control", "dropped_vs_control",
        "is_probability_prefix", "candidate_count",
        "card_zero", "card_one", "card_many",
    ]


def _action_key(objects: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({_candidate_key(item, COMPANY) for item in objects}))


def company_actions(
        graph: Mapping[str, Any], probabilities: Sequence[float],
        control: Sequence[str],
) -> list[list[str]]:
    nodes = list(graph["candidates"])
    ranked = sorted(
        zip(nodes, probabilities),
        key=lambda pair: (-float(pair[1]), pair[0]["key"]))
    actions: dict[tuple[str, ...], list[str]] = {}

    def add(objects: Sequence[str]) -> None:
        values = list(dict.fromkeys(str(item) for item in objects))
        actions.setdefault(_action_key(values), values)

    add([])
    add(control)
    for node in nodes:
        add([node["item"]])
        add([*control, node["item"]])
    for item in control:
        add([value for value in control
             if _candidate_key(value, COMPANY) != _candidate_key(item, COMPANY)])
    for count in range(1, min(6, len(ranked)) + 1):
        prefix = [node["item"] for node, _ in ranked[:count]]
        add(prefix)
        add([*control, *prefix])
    return list(actions.values())


def company_set_features(
        graph: Mapping[str, Any], action: Sequence[str],
        probabilities: Sequence[float], control: Sequence[str],
        cardinality: Mapping[str, float],
) -> list[float]:
    nodes = list(graph["candidates"])
    by_key = {
        node["key"]: (node, float(probability))
        for node, probability in zip(nodes, probabilities)}
    keys = set(_action_key(action))
    selected = [by_key[key] for key in keys if key in by_key]
    selected_probabilities = [probability for _, probability in selected]
    omitted = [
        probability for key, (_, probability) in by_key.items()
        if key not in keys]
    expected_tp = sum(selected_probabilities)
    expected_cardinality = (
        float(cardinality["ONE"]) + 2.0 * float(cardinality["MANY"]))
    expected_cardinality = max(
        expected_cardinality, sum(probabilities))
    denominator = len(action) + expected_cardinality
    plugin_f1 = 2.0 * expected_tp / denominator if denominator else 1.0
    control_keys = set(_action_key(control))
    ranked_keys = [
        node["key"] for node, _ in sorted(
            zip(nodes, probabilities),
            key=lambda pair: (-float(pair[1]), pair[0]["key"]))]
    is_prefix = keys == set(ranked_keys[:len(keys)])
    values = [
        float(not action),
        min(1.0, len(action) / 6.0),
        min(1.0, expected_tp / 3.0),
        plugin_f1,
        statistics.mean(selected_probabilities) if selected_probabilities else 0.0,
        min(selected_probabilities) if selected_probabilities else 0.0,
        max(selected_probabilities) if selected_probabilities else 0.0,
        max(omitted, default=0.0),
        min(1.0, expected_cardinality / 4.0),
        min(1.0, abs(len(action) - expected_cardinality) / 4.0),
        min(1.0, sum(_support(node, QWEN) for node, _ in selected) / 3.0),
        min(1.0, sum(_support(node, GEMMA) for node, _ in selected) / 3.0),
        min(1.0, sum({QWEN, GEMMA} <= set(node["sources"])
                     for node, _ in selected) / 3.0),
        min(1.0, sum(set(node["sources"]) == {QWEN}
                     for node, _ in selected) / 3.0),
        min(1.0, sum(set(node["sources"]) == {GEMMA}
                     for node, _ in selected) / 3.0),
        min(1.0, sum(node["selected_by"].get(QWEN, False)
                     for node, _ in selected) / 3.0),
        min(1.0, sum(node["selected_by"].get(GEMMA, False)
                     for node, _ in selected) / 3.0),
        len(keys & control_keys) / max(1, len(keys | control_keys)),
        min(1.0, len(keys - control_keys) / 3.0),
        min(1.0, len(control_keys - keys) / 3.0),
        float(is_prefix),
        min(1.0, len(nodes) / 10.0),
        float(cardinality["ZERO"]),
        float(cardinality["ONE"]),
        float(cardinality["MANY"]),
    ]
    if len(values) != len(company_set_feature_names()):
        raise AssertionError("company set feature schema drift")
    return values


class CompanyStructuredDecoder:
    def __init__(self, candidate_l2: float = 2.0, set_l2: float = 5.0):
        self.candidate_l2 = float(candidate_l2)
        self.set_l2 = float(set_l2)
        self.candidate_model = LogisticCalibrator(
            company_candidate_feature_names(), l2=candidate_l2)
        self.set_model = StandardizedRidge(
            company_set_feature_names(), l2=set_l2)
        self.cardinality = ExplicitCardinalityModel(candidate_l2)

    def fit(
            self, graphs: Sequence[Mapping[str, Any]],
            gold: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> "CompanyStructuredDecoder":
        rows = [graph for graph in graphs if graph["Relation"] == COMPANY]
        x, y, weights = [], [], []
        for graph in rows:
            row_weight = 1.0 / max(1, len(graph["candidates"]))
            for node in graph["candidates"]:
                x.append(company_candidate_features(graph, node))
                y.append(float(true_positives(
                    [node["item"]], _gold_aliases(gold[_key(graph)]),
                    "string", 0.05) > 0))
                weights.append(row_weight)
        self.candidate_model.fit(x, y, weights)
        self.cardinality.fit(graphs, gold)
        sx, sy, sw = [], [], []
        for graph in rows:
            probabilities = (
                self.candidate_model.predict([
                    company_candidate_features(graph, node)
                    for node in graph["candidates"]])
                if graph["candidates"] else np.asarray([], dtype=np.float64))
            cardinality = self.cardinality.predict_one(graph)
            control = list(graph.get("baseline_objects", []))
            actions = company_actions(graph, probabilities, control)
            row_weight = 1.0 / len(actions)
            for action in actions:
                sx.append(company_set_features(
                    graph, action, probabilities, control, cardinality))
                sy.append(_row_f1(action, gold[_key(graph)], COMPANY))
                sw.append(row_weight)
        self.set_model.fit(sx, sy, sw)
        return self

    def decode(
            self, graph: Mapping[str, Any], control: Sequence[str],
            margin: float,
    ) -> tuple[list[str], dict[str, Any]]:
        probabilities = (
            self.candidate_model.predict([
                company_candidate_features(graph, node)
                for node in graph["candidates"]])
            if graph["candidates"] else np.asarray([], dtype=np.float64))
        cardinality = self.cardinality.predict_one(graph)
        actions = company_actions(graph, probabilities, control)
        features = [
            company_set_features(
                graph, action, probabilities, control, cardinality)
            for action in actions]
        utilities = self.set_model.predict(features)
        best = max(
            range(len(actions)),
            key=lambda index: (
                float(utilities[index]), -len(actions[index]),
                _action_key(actions[index])))
        control_key = _action_key(control)
        control_index = next(
            index for index, action in enumerate(actions)
            if _action_key(action) == control_key)
        improvement = float(utilities[best] - utilities[control_index])
        use_control = improvement <= margin
        selected = list(control) if use_control else actions[best]
        return selected, {
            "candidate_probabilities": [
                {"item": node["item"], "probability": float(probability)}
                for node, probability in zip(
                    graph["candidates"], probabilities)
            ],
            "cardinality_probabilities": cardinality,
            "control_objects": list(control),
            "control_utility": float(utilities[control_index]),
            "proposed_objects": actions[best],
            "proposed_utility": float(utilities[best]),
            "estimated_improvement": improvement,
            "guard_margin": float(margin),
            "used_control": use_control,
            "action_count": len(actions),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "company-structured-set-risk-v1",
            "candidate_model": self.candidate_model.to_dict(),
            "set_model": self.set_model.to_dict(),
            "cardinality_model": self.cardinality.to_dict(),
        }


def _control_oof(
        fit_graphs: Sequence[Mapping[str, Any]],
        holdout: Sequence[Mapping[str, Any]],
        gold: Mapping[tuple[str, str], Mapping[str, Any]],
        l2: float,
) -> dict[tuple[str, str], list[str]]:
    candidate, _ = _fit_models(
        fit_graphs, gold, l2, "row-agent-balanced")
    cardinality = ExplicitCardinalityModel(l2).fit(fit_graphs, gold)
    rows, _ = explicit_prediction_rows(
        holdout, candidate, cardinality, CURRENT_CONTROL_MARGIN)
    return {_key(row): list(row["ObjectEntities"]) for row in rows}


def _selection(
        fold_scores: Mapping[float, Mapping[int, float]],
        baseline: Mapping[int, float],
) -> tuple[float, bool, dict[str, Any]]:
    details = {}
    folds = sorted(baseline)
    for margin, scores in sorted(fold_scores.items()):
        deltas = [scores[fold] - baseline[fold] for fold in folds]
        details[str(margin)] = {
            "fold_scores": {str(fold): scores[fold] for fold in folds},
            "control_fold_scores": {
                str(fold): baseline[fold] for fold in folds},
            "paired_fold_deltas": {
                str(fold): delta for fold, delta in zip(folds, deltas)},
            "mean_paired_delta": statistics.mean(deltas),
            "standard_error": statistics.stdev(deltas) / math.sqrt(len(deltas)),
        }
    best_mean = max(
        fold_scores,
        key=lambda margin: (details[str(margin)]["mean_paired_delta"], margin))
    threshold = (
        details[str(best_mean)]["mean_paired_delta"]
        - details[str(best_mean)]["standard_error"])
    eligible = [
        margin for margin in fold_scores
        if details[str(margin)]["mean_paired_delta"] >= threshold - 1e-12
    ]
    selected = max(eligible)
    deltas = list(details[str(selected)]["paired_fold_deltas"].values())
    # Predeclared deployment gate: positive mean, at least three positive
    # folds, and no loss larger than one row in a roughly 20-row fold.
    enabled = (
        details[str(selected)]["mean_paired_delta"] > 1e-12
        and sum(delta > 1e-12 for delta in deltas) >= 3
        and min(deltas) >= -0.05 - 1e-12
    )
    for margin in fold_scores:
        details[str(margin)]["within_one_standard_error"] = margin in eligible
    return float(selected), enabled, {
        "rule": (
            "largest predeclared margin within one standard error of the "
            "best mean train-OOF margin; deploy only with positive mean, at "
            "least three positive folds, and no fold loss worse than 0.05"),
        "best_mean_margin": float(best_mean),
        "one_standard_error_threshold": float(threshold),
        "selected_margin": float(selected),
        "enabled": enabled,
        "margins": details,
    }


def _merge(
        control: Sequence[Mapping[str, Any]],
        replacements: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict]:
    return [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": list(replacements.get(
            _key(row), row.get("ObjectEntities", []))),
    } for row in control]


def run(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = _json(source / "plan/PLAN.json")
    train_path = source / "graphs/train_graph.jsonl"
    validation_path = source / "graphs/validation_graph.jsonl"
    all_train_graphs = _load_graph(train_path, expected_split="train")
    train_graphs = [
        graph for graph in all_train_graphs
        if graph.get("calibration_eligible", True) is not False]
    excluded_graphs = [
        graph for graph in all_train_graphs
        if graph.get("calibration_eligible", True) is False]
    if not train_graphs:
        raise ContractError("no calibration-eligible training graphs")
    validation_graphs = _load_graph(
        validation_path, expected_split="validation")
    train_gold = {
        _key(row): row for row in read_jsonl(Path(plan["train_gold"]))}
    fold_path = Path(plan["folds"])
    if sha256(fold_path) != plan["folds_sha256"]:
        raise ContractError("fold hash mismatch")
    folds = {_key(row): int(row["fold"]) for row in read_jsonl(fold_path)}
    if set(folds) != {_key(graph) for graph in all_train_graphs}:
        raise ContractError("fold manifest does not cover full training graph")
    margins = tuple(float(value) for value in args.guard_margins.split(","))
    if not margins or any(value < 0 or not math.isfinite(value)
                          for value in margins):
        raise ContractError("invalid guard margins")

    fold_scores = {
        relation: {margin: {} for margin in margins}
        for relation in RELATIONS}
    control_scores = {relation: {} for relation in RELATIONS}
    for fold in sorted(set(folds.values())):
        fit_graphs = [
            graph for graph in train_graphs if folds[_key(graph)] != fold]
        holdout_all = [
            graph for graph in train_graphs if folds[_key(graph)] == fold]
        holdout = [
            graph for graph in holdout_all if graph["Relation"] in RELATIONS]
        controls = _control_oof(
            fit_graphs, holdout_all, train_gold, args.candidate_l2)
        city = CityConditionalDecoder(args.candidate_l2).fit(
            fit_graphs, train_gold)
        company = CompanyStructuredDecoder(
            args.candidate_l2, args.set_l2).fit(fit_graphs, train_gold)
        for relation, decoder in ((CITY, city), (COMPANY, company)):
            relation_rows = [
                graph for graph in holdout if graph["Relation"] == relation]
            control_values = [
                _row_f1(
                    controls[_key(graph)], train_gold[_key(graph)], relation)
                for graph in relation_rows]
            control_scores[relation][fold] = statistics.mean(control_values)
            for margin in margins:
                values = []
                for graph in relation_rows:
                    objects, _ = decoder.decode(
                        graph, controls[_key(graph)], margin)
                    values.append(_row_f1(
                        objects, train_gold[_key(graph)], relation))
                fold_scores[relation][margin][fold] = statistics.mean(values)

    selected, enabled, selection = {}, {}, {}
    for relation in RELATIONS:
        margin, gate, details = _selection(
            fold_scores[relation], control_scores[relation])
        selected[relation], enabled[relation] = margin, gate
        selection[relation] = details

    city = CityConditionalDecoder(args.candidate_l2).fit(
        train_graphs, train_gold)
    company = CompanyStructuredDecoder(
        args.candidate_l2, args.set_l2).fit(train_graphs, train_gold)
    model_path = output / "structured_models.json"
    model_path.write_text(json.dumps({
        "schema": "city-company-structured-models-v1",
        "train_graph": str(train_path),
        "train_graph_sha256": sha256(train_path),
        "train_rows_total": len(all_train_graphs),
        "train_rows_calibration_eligible": len(train_graphs),
        "train_rows_excluded": len(excluded_graphs),
        "excluded_keys": [list(_key(graph)) for graph in excluded_graphs],
        "folds": str(fold_path),
        "folds_sha256": sha256(fold_path),
        "validation_labels_used_for_selection": False,
        "selected_margins": selected,
        "enabled_relations": enabled,
        "selection": selection,
        "city": city.to_dict(),
        "company": company.to_dict(),
    }, indent=2, sort_keys=True) + "\n")

    control_path = Path(args.control_predictions).resolve()
    control_rows = read_jsonl(control_path)
    control_by_key = {_key(row): row for row in control_rows}
    if set(control_by_key) != {_key(graph) for graph in validation_graphs}:
        raise ContractError("validation control does not cover graph")
    replacements, diagnostics = {}, []
    for graph in validation_graphs:
        relation = str(graph["Relation"])
        if relation not in RELATIONS or not enabled[relation]:
            continue
        decoder = city if relation == CITY else company
        objects, detail = decoder.decode(
            graph, control_by_key[_key(graph)]["ObjectEntities"],
            selected[relation])
        replacements[_key(graph)] = objects
        diagnostics.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": relation,
            **detail,
        })
    predictions = _merge(control_rows, replacements)
    prediction_path = output / "validation_stable_oof.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(
        output / "validation_stable_oof.diagnostics.jsonl", diagnostics)
    prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json").write_text(json.dumps({
            "schema": "city-company-structured-predictions-v1",
            "contains_labels": False,
            "gold_aware": False,
            "rows": len(predictions),
            "output_sha256": sha256(prediction_path),
            "source_graph": str(validation_path),
            "source_graph_sha256": sha256(validation_path),
            "control_predictions": str(control_path),
            "control_predictions_sha256": sha256(control_path),
            "models": str(model_path),
            "models_sha256": sha256(model_path),
            "selected_margins": selected,
            "enabled_relations": enabled,
            "validation_labels_used_for_decoding": False,
        }, indent=2, sort_keys=True) + "\n")

    # Validation labels are opened only after the frozen prediction artifact.
    validation_gold = read_jsonl(Path(plan["validation_gold"]))
    scores = {
        "control": score(control_rows, validation_gold),
        "stable_oof": score(predictions, validation_gold),
    }
    oracles = {}
    validation_gold_by_key = {_key(row): row for row in validation_gold}
    for relation, decoder in ((CITY, city), (COMPANY, company)):
        values = []
        for graph in validation_graphs:
            if graph["Relation"] != relation:
                continue
            control = control_by_key[_key(graph)]["ObjectEntities"]
            if relation == CITY:
                actions, _ = city.action_utilities(graph)
                candidates = [list(action) for action in actions]
                candidates.append(list(control))
            else:
                probabilities = (
                    company.candidate_model.predict([
                        company_candidate_features(graph, node)
                        for node in graph["candidates"]])
                    if graph["candidates"]
                    else np.asarray([], dtype=np.float64))
                candidates = company_actions(graph, probabilities, control)
            values.append(max(
                _row_f1(action, validation_gold_by_key[_key(graph)], relation)
                for action in candidates))
        oracles[relation] = statistics.mean(values)

    numeric_result_path = (
        source / "relation_specific_numeric_decoder/RESULT.json")
    numeric_result = (
        _json(numeric_result_path) if numeric_result_path.is_file() else {})
    numeric_capacity_selection = (
        numeric_result.get("selection", {}).get("hasCapacity", {}))
    numeric_capacity_best = str(
        numeric_result.get("best_mean_margins", {}).get("hasCapacity", 0.0))
    numeric_capacity_margin = numeric_capacity_selection.get(
        "margins", {}).get(numeric_capacity_best, {})
    numeric_capacity_baselines = numeric_capacity_margin.get(
        "baseline_fold_accuracy", {})
    capacity_train_control = (
        statistics.mean(float(value) for value in numeric_capacity_baselines.values())
        if numeric_capacity_baselines else None)
    capacity_validation_control = scores["control"]["hasCapacity"]
    capacity_audit = {
        "eligible_for_new_decoder": False,
        "reason": (
            "training OOF control and validation control accuracy are not "
            "production-matched; new capacity selection remains fail-closed"),
        "train_oof_control": capacity_train_control,
        "validation_control": capacity_validation_control,
        "absolute_control_gap": (
            abs(capacity_validation_control - capacity_train_control)
            if capacity_train_control is not None else None),
        "source_result": str(numeric_result_path),
        "source_result_sha256": (
            sha256(numeric_result_path) if numeric_result_path.is_file()
            else None),
    }
    review_audit = {
        "enabled": False,
        "reason": (
            "existing selective, residual, pairwise, and contrastive review "
            "ablations did not pass their train-only deployment gates"),
    }
    result = {
        "schema": "city-company-structured-decoder-ablation-v1",
        "development_only": True,
        "train_rows_total": len(all_train_graphs),
        "train_rows_calibration_eligible": len(train_graphs),
        "train_rows_excluded": len(excluded_graphs),
        "validation_labels_used_for_selection": False,
        "validation_labels_used_for_posthoc_evaluation": True,
        "scores": scores,
        "delta": (
            scores["stable_oof"]["*** All Relations ***"]
            - scores["control"]["*** All Relations ***"]),
        "selected_margins": selected,
        "enabled_relations": enabled,
        "selection": selection,
        "action_oracle_validation_nondeployable": oracles,
        "capacity_production_match_audit": capacity_audit,
        "semantic_review_gate": review_audit,
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# City and company structured risk decoding",
        "",
        "Validation labels were opened only after predictions were frozen.",
        "",
        "| policy | pooled | city | company | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in ("control", "stable_oof"):
        values = scores[policy]
        lines.append(
            f"| {policy} | {values['*** All Relations ***']:.9f} | "
            f"{values[CITY]:.6f} | {values[COMPANY]:.6f} | "
            f"{values['*** All Relations ***'] - scores['control']['*** All Relations ***']:+.9f} |")
    lines += [
        "",
        "| relation | selected margin | enabled | validation action oracle "
        "(nondeployable) |",
        "|---|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        lines.append(
            f"| {relation} | {selected[relation]:.3f} | "
            f"{enabled[relation]} | {oracles[relation]:.6f} |")
    lines += [
        "",
        "Capacity remains disabled by its production-match audit. Semantic "
        "cross-review remains disabled by the existing negative deployment "
        "gates.",
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(
        f"complete: pooled={scores['stable_oof']['*** All Relations ***']:.9f}; "
        f"enabled={enabled}; report={output / 'RESULT.md'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-predictions", required=True)
    parser.add_argument("--candidate-l2", type=float, default=2.0)
    parser.add_argument("--set-l2", type=float, default=5.0)
    parser.add_argument(
        "--guard-margins",
        default=",".join(str(value) for value in DEFAULT_MARGINS))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
