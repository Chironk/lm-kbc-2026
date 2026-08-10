#!/usr/bin/env python3
"""Leakage-safe downstream test of explicit proposal-route evidence.

This experiment predicts the *change in row F1* caused by a concrete edit to
the frozen incumbent.  It deliberately does not predict absolute candidate
correctness.  Three arms separate the effect of the decoder from the effect of
the new graph:

``base_residual``
    Old two-source graph and the baseline-relative residual decoder.
``route_agreement``
    Explicit Qwen-self-consistency/System-2/Gemma route features, excluding
    candidates proposed only by System-2.
``route_full``
    The full route graph, including System-2-only candidate identities.

Outer-fold controls are produced without outer-fold labels.  Controls used to
train an outer-fold residual model are themselves inner-fold predictions, so
the residual target never sees an incumbent fitted on its own label.  Arm and
margin selection use training OOF results only.  Validation labels are opened
only after every arm prediction and the train-selected prediction are written.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    _key,
    _load_graph,
)
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    CITY,
    COMPANY,
    RELATIONS,
    _control_oof,
    _prob,
    _row_f1,
    _selection,
    _support,
)
from experiments.heterogeneous_agents.components.route_aware_candidate_graph import (
    ROUTE_GEMMA,
    ROUTE_QWEN_SC,
    ROUTE_QWEN_SYSTEM2,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "production_matched_oof_20260723_v1")
DEFAULT_ROUTE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "route_aware_graph_20260723_v1")
DEFAULT_CONTROL = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "expanded_calibration_n1_20260723_v1/"
    "relation_specific_numeric_decoder/validation_stable_oof.jsonl")
ARMS = ("base_residual", "route_agreement", "route_full")
DEFAULT_MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


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


def fit_residual(
        graphs: Sequence[Mapping[str, Any]],
        controls: Mapping[tuple[str, str], Sequence[str]],
        gold: Mapping[tuple[str, str], Mapping[str, Any]],
        relation: str, arm: str, l2: float,
) -> ResidualRidge:
    x: list[list[float]] = []
    y: list[float] = []
    weights: list[float] = []
    for graph in graphs:
        if graph["Relation"] != relation:
            continue
        key = _key(graph)
        control = list(controls[key])
        actions = actions_for(graph, control, arm)
        control_f1 = _row_f1(control, gold[key], relation)
        row_weight = 1.0 / len(actions)
        for action in actions:
            x.append(action_features(graph, control, action, arm))
            y.append(_row_f1(action, gold[key], relation) - control_f1)
            weights.append(row_weight)
    return ResidualRidge(feature_names(), l2).fit(x, y, weights)


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


def _align(
        base: Sequence[Mapping[str, Any]],
        route: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], Mapping[str, Any]],
           dict[tuple[str, str], Mapping[str, Any]]]:
    base_by = {_key(row): row for row in base}
    route_by = {_key(row): row for row in route}
    if set(base_by) != set(route_by):
        raise ContractError("base and route graphs do not cover identical rows")
    for key in base_by:
        if (base_by[key].get("calibration_eligible", True)
                != route_by[key].get("calibration_eligible", True)):
            raise ContractError(f"calibration eligibility drift for {key}")
    return base_by, route_by


def _graphs_for_arm(
        arm: str, keys: Sequence[tuple[str, str]],
        base: Mapping[tuple[str, str], Mapping[str, Any]],
        route: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    source = base if arm == "base_residual" else route
    return [source[key] for key in keys]


def _nested_controls(
        eligible_keys: Sequence[tuple[str, str]], outer_fold: int,
        folds: Mapping[tuple[str, str], int],
        base: Mapping[tuple[str, str], Mapping[str, Any]],
        gold: Mapping[tuple[str, str], Mapping[str, Any]], l2: float,
) -> tuple[dict[tuple[str, str], list[str]],
           dict[tuple[str, str], list[str]]]:
    outer_fit = [key for key in eligible_keys if folds[key] != outer_fold]
    outer_hold = [key for key in eligible_keys if folds[key] == outer_fold]
    train_controls: dict[tuple[str, str], list[str]] = {}
    for inner_fold in sorted(set(folds[key] for key in outer_fit)):
        inner_hold = [key for key in outer_fit if folds[key] == inner_fold]
        inner_fit = [key for key in outer_fit if folds[key] != inner_fold]
        train_controls.update(_control_oof(
            [base[key] for key in inner_fit],
            [base[key] for key in inner_hold],
            gold, l2))
    hold_controls = _control_oof(
        [base[key] for key in outer_fit],
        [base[key] for key in outer_hold],
        gold, l2)
    if set(train_controls) != set(outer_fit):
        raise ContractError("nested controls miss outer-fit rows")
    if set(hold_controls) != set(outer_hold):
        raise ContractError("outer controls miss holdout rows")
    return train_controls, hold_controls


def _full_oof_controls(
        eligible_keys: Sequence[tuple[str, str]],
        folds: Mapping[tuple[str, str], int],
        base: Mapping[tuple[str, str], Mapping[str, Any]],
        gold: Mapping[tuple[str, str], Mapping[str, Any]], l2: float,
) -> dict[tuple[str, str], list[str]]:
    controls: dict[tuple[str, str], list[str]] = {}
    for fold in sorted(set(folds[key] for key in eligible_keys)):
        hold = [key for key in eligible_keys if folds[key] == fold]
        fit = [key for key in eligible_keys if folds[key] != fold]
        controls.update(_control_oof(
            [base[key] for key in fit], [base[key] for key in hold],
            gold, l2))
    if set(controls) != set(eligible_keys):
        raise ContractError("full OOF controls do not cover eligible training")
    return controls


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


def run(args: argparse.Namespace) -> int:
    base_root = Path(args.base_output_dir).resolve()
    route_root = Path(args.route_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = _json(base_root / "plan/PLAN.json")
    route_plan = _json(route_root / "plan/PLAN.json")
    if (plan["folds_sha256"] != route_plan["folds_sha256"]
            or plan["train_gold_sha256"] != route_plan["train_gold_sha256"]):
        raise ContractError("base/route plans do not share folds and train gold")

    base_train_path = base_root / "graphs/train_graph.jsonl"
    base_val_path = base_root / "graphs/validation_graph.jsonl"
    route_train_path = route_root / "graphs/train_graph.jsonl"
    route_val_path = route_root / "graphs/validation_graph.jsonl"
    base_train, route_train = _align(
        _load_graph(base_train_path, expected_split="train"),
        _load_graph(route_train_path, expected_split="train"))
    base_val, route_val = _align(
        _load_graph(base_val_path, expected_split="validation"),
        _load_graph(route_val_path, expected_split="validation"))

    train_gold = {
        _key(row): row for row in read_jsonl(Path(plan["train_gold"]))}
    fold_path = Path(plan["folds"])
    if sha256(fold_path) != plan["folds_sha256"]:
        raise ContractError("fold hash mismatch")
    folds = {_key(row): int(row["fold"]) for row in read_jsonl(fold_path)}
    if set(folds) != set(base_train):
        raise ContractError("fold map does not cover full train graph")
    eligible_keys = [
        key for key, graph in base_train.items()
        if graph.get("calibration_eligible", True) is not False]
    margins = tuple(float(item) for item in args.guard_margins.split(","))
    if (not margins or any(value < 0 or not math.isfinite(value)
                           for value in margins)):
        raise ContractError("invalid guard margins")

    fold_scores = {
        arm: {
            relation: {margin: {} for margin in margins}
            for relation in RELATIONS}
        for arm in ARMS}
    control_scores = {relation: {} for relation in RELATIONS}
    fold_diagnostics: list[dict[str, Any]] = []
    for outer_fold in sorted(set(folds.values())):
        train_controls, hold_controls = _nested_controls(
            eligible_keys, outer_fold, folds, base_train, train_gold,
            args.control_l2)
        fit_keys = [key for key in eligible_keys if folds[key] != outer_fold]
        hold_keys = [key for key in eligible_keys if folds[key] == outer_fold]
        for relation in RELATIONS:
            relation_hold = [
                key for key in hold_keys if key[1] == relation]
            control_scores[relation][outer_fold] = statistics.mean(
                _row_f1(hold_controls[key], train_gold[key], relation)
                for key in relation_hold)
        for arm in ARMS:
            fit_graphs = _graphs_for_arm(
                arm, fit_keys, base_train, route_train)
            hold_graphs = _graphs_for_arm(
                arm, hold_keys, base_train, route_train)
            for relation in RELATIONS:
                model = fit_residual(
                    fit_graphs, train_controls, train_gold, relation, arm,
                    args.residual_l2)
                relation_hold = [
                    graph for graph in hold_graphs
                    if graph["Relation"] == relation]
                for margin in margins:
                    values = []
                    changed = helpful = harmful = equal = 0
                    for graph in relation_hold:
                        key = _key(graph)
                        control = hold_controls[key]
                        objects, detail = decode(
                            model, graph, control, arm, margin)
                        before = _row_f1(control, train_gold[key], relation)
                        after = _row_f1(objects, train_gold[key], relation)
                        values.append(after)
                        if _action_key(objects, relation) != _action_key(
                                control, relation):
                            changed += 1
                            helpful += int(after > before + 1e-12)
                            harmful += int(after < before - 1e-12)
                            equal += int(abs(after - before) <= 1e-12)
                    fold_scores[arm][relation][margin][
                        outer_fold] = statistics.mean(values)
                    fold_diagnostics.append({
                        "arm": arm, "relation": relation,
                        "outer_fold": outer_fold, "margin": margin,
                        "changed": changed, "helpful": helpful,
                        "harmful": harmful, "equal": equal,
                    })

    selection: dict[str, dict[str, Any]] = {}
    selected_margin: dict[str, dict[str, float]] = {}
    enabled: dict[str, dict[str, bool]] = {}
    for arm in ARMS:
        selection[arm], selected_margin[arm], enabled[arm] = {}, {}, {}
        for relation in RELATIONS:
            margin, gate, detail = _selection(
                fold_scores[arm][relation], control_scores[relation])
            selected_margin[arm][relation] = margin
            enabled[arm][relation] = gate
            selection[arm][relation] = detail

    # Choose the graph arm from train OOF only. Ties prefer the simpler arm.
    chosen_arm: dict[str, str | None] = {}
    for relation in RELATIONS:
        candidates = [
            arm for arm in ARMS if enabled[arm][relation]]
        chosen_arm[relation] = (
            max(candidates, key=lambda arm: (
                selection[arm][relation]["margins"][
                    str(selected_margin[arm][relation])
                ]["mean_paired_delta"],
                -ARMS.index(arm)))
            if candidates else None)

    full_controls = _full_oof_controls(
        eligible_keys, folds, base_train, train_gold, args.control_l2)
    final_models: dict[str, dict[str, ResidualRidge]] = {}
    serialized: dict[str, Any] = {}
    for arm in ARMS:
        graphs = _graphs_for_arm(
            arm, eligible_keys, base_train, route_train)
        final_models[arm], serialized[arm] = {}, {}
        for relation in RELATIONS:
            model = fit_residual(
                graphs, full_controls, train_gold, relation, arm,
                args.residual_l2)
            final_models[arm][relation] = model
            serialized[arm][relation] = model.to_dict()

    model_path = output / "models.json"
    model_path.write_text(json.dumps({
        "schema": "baseline-relative-route-decoder-models-v1",
        "validation_labels_used_for_selection": False,
        "control_generation": (
            "nested five-fold OOF for outer evaluation; five-fold OOF "
            "incumbents for final residual fit"),
        "base_train_graph": str(base_train_path),
        "base_train_graph_sha256": sha256(base_train_path),
        "route_train_graph": str(route_train_path),
        "route_train_graph_sha256": sha256(route_train_path),
        "folds": str(fold_path), "folds_sha256": sha256(fold_path),
        "train_rows_calibration_eligible": len(eligible_keys),
        "guard_margins": list(margins),
        "selection": selection,
        "selected_margins": selected_margin,
        "enabled": enabled,
        "chosen_arm": chosen_arm,
        "models": serialized,
    }, indent=2, sort_keys=True) + "\n")

    control_path = Path(args.control_predictions).resolve()
    control_rows = read_jsonl(control_path)
    control_by = {_key(row): row for row in control_rows}
    if set(control_by) != set(base_val):
        raise ContractError("validation control does not cover graph")
    arm_predictions: dict[str, list[dict[str, Any]]] = {}
    validation_diagnostics: list[dict[str, Any]] = []
    for arm in ARMS:
        replacements: dict[tuple[str, str], list[str]] = {}
        source = base_val if arm == "base_residual" else route_val
        for key, graph in source.items():
            relation = key[1]
            if relation not in RELATIONS or not enabled[arm][relation]:
                continue
            objects, detail = decode(
                final_models[arm][relation], graph,
                control_by[key]["ObjectEntities"], arm,
                selected_margin[arm][relation])
            replacements[key] = objects
            validation_diagnostics.append({
                "SubjectEntity": key[0], "Relation": relation, **detail})
        rows = _prediction_rows(control_rows, replacements)
        path = output / f"validation_{arm}.jsonl"
        write_jsonl_atomic(path, rows)
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps({
                "schema": "baseline-relative-route-predictions-v1",
                "contains_labels": False, "gold_aware": False,
                "arm": arm, "rows": len(rows),
                "output_sha256": sha256(path),
                "control_predictions": str(control_path),
                "control_predictions_sha256": sha256(control_path),
                "models": str(model_path),
                "models_sha256": sha256(model_path),
                "validation_labels_used_for_decoding": False,
            }, indent=2, sort_keys=True) + "\n")
        arm_predictions[arm] = rows

    selected_replacements: dict[tuple[str, str], list[str]] = {}
    for key in base_val:
        relation = key[1]
        arm = chosen_arm.get(relation)
        if arm is None:
            continue
        source = base_val if arm == "base_residual" else route_val
        objects, _ = decode(
            final_models[arm][relation], source[key],
            control_by[key]["ObjectEntities"], arm,
            selected_margin[arm][relation])
        selected_replacements[key] = objects
    selected_rows = _prediction_rows(control_rows, selected_replacements)
    selected_path = output / "validation_train_selected.jsonl"
    write_jsonl_atomic(selected_path, selected_rows)
    write_jsonl_atomic(
        output / "validation_diagnostics.jsonl", validation_diagnostics)
    selected_path.with_suffix(
        selected_path.suffix + ".manifest.json").write_text(json.dumps({
            "schema": "train-selected-route-decoder-predictions-v1",
            "contains_labels": False, "gold_aware": False,
            "rows": len(selected_rows), "output_sha256": sha256(selected_path),
            "chosen_arm": chosen_arm,
            "selected_margins": selected_margin,
            "validation_labels_used_for_selection": False,
            "validation_labels_used_for_decoding": False,
        }, indent=2, sort_keys=True) + "\n")

    # Labels are intentionally opened only after every prediction is frozen.
    validation_gold = read_jsonl(Path(plan["validation_gold"]))
    scores = {"control": score(control_rows, validation_gold)}
    scores.update({
        arm: score(rows, validation_gold)
        for arm, rows in arm_predictions.items()})
    scores["train_selected"] = score(selected_rows, validation_gold)
    pooled_control = scores["control"]["*** All Relations ***"]
    result = {
        "schema": "baseline-relative-route-decoder-ablation-v1",
        "development_only": True,
        "validation_labels_used_for_selection": False,
        "validation_labels_used_for_posthoc_evaluation": True,
        "arms": {
            "base_residual": "old graph; residual decoder",
            "route_agreement": (
                "route graph/features; System-2-only nodes excluded"),
            "route_full": "route graph/features; System-2-only nodes included",
        },
        "chosen_arm": chosen_arm,
        "selected_margins": selected_margin,
        "enabled": enabled,
        "selection": selection,
        "scores": scores,
        "pooled_deltas": {
            name: values["*** All Relations ***"] - pooled_control
            for name, values in scores.items()},
        "route_downstream_oof_delta_vs_base": {
            relation: {
                arm: (
                    selection[arm][relation]["margins"][
                        str(selected_margin[arm][relation])
                    ]["mean_paired_delta"]
                    - selection["base_residual"][relation]["margins"][
                        str(selected_margin["base_residual"][relation])
                    ]["mean_paired_delta"])
                for arm in ("route_agreement", "route_full")}
            for relation in RELATIONS},
        "predictions": str(selected_path),
        "predictions_sha256": sha256(selected_path),
        "model_artifact": str(model_path),
        "model_artifact_sha256": sha256(model_path),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_jsonl_atomic(output / "fold_diagnostics.jsonl", fold_diagnostics)
    lines = [
        "# Baseline-relative route decoder ablation",
        "",
        "All arm/margin choices were frozen from nested train OOF evidence. "
        "Validation labels were opened only after predictions were written.",
        "",
        "| policy | pooled | city | company | pooled delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("control", *ARMS, "train_selected"):
        values = scores[name]
        lines.append(
            f"| {name} | {values['*** All Relations ***']:.9f} | "
            f"{values[CITY]:.6f} | {values[COMPANY]:.6f} | "
            f"{values['*** All Relations ***'] - pooled_control:+.9f} |")
    lines += [
        "",
        "| relation | arm | margin | enabled | mean train-OOF delta |",
        "|---|---|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        for arm in ARMS:
            margin = selected_margin[arm][relation]
            mean_delta = selection[arm][relation]["margins"][
                str(margin)]["mean_paired_delta"]
            lines.append(
                f"| {relation} | {arm} | {margin:.3f} | "
                f"{enabled[arm][relation]} | {mean_delta:+.6f} |")
    lines += [
        "",
        f"Train-selected arms: `{json.dumps(chosen_arm, sort_keys=True)}`.",
        "",
        "`route_agreement - base_residual` is the clean downstream test of "
        "route identity. `route_full` additionally tests whether exposing "
        "System-2-only identities is useful.",
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(
        f"complete: pooled={scores['train_selected']['*** All Relations ***']:.9f}; "
        f"chosen={chosen_arm}; report={output / 'RESULT.md'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-output-dir", default=str(DEFAULT_BASE))
    parser.add_argument(
        "--route-output-dir", default=str(DEFAULT_ROUTE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--control-predictions", default=str(DEFAULT_CONTROL))
    parser.add_argument("--control-l2", type=float, default=2.0)
    parser.add_argument("--residual-l2", type=float, default=10.0)
    parser.add_argument(
        "--guard-margins",
        default=",".join(str(value) for value in DEFAULT_MARGINS))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
