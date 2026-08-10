#!/usr/bin/env python3
"""Baseline-aware action model for the multiview capacity graph.

The first capacity proposal selector switched whenever Gemma repeated a new
number three times.  That improved train by 0.03 but reduced validation by
0.03 because repetition was treated as correctness and incumbent evidence was
ignored.  This replacement learns the quantity needed at deployment:

    utility(candidate) = official_f1(candidate) - official_f1(incumbent)

Only inference-legal graph features are used.  Candidate and incumbent route
support, cross-model/view agreement, proposal centrality, and numeric distance
are represented jointly.  Candidate-heavy rows receive total training weight
one.  Ridge strength and the switch margin are chosen with nested frozen-fold
predictions, and the final deployment gate requires positive train OOF gain,
at least three winning folds, and no harmful fold.

Validation predictions are written and hashed before validation labels are
opened.  This is a development experiment; it does not make the validation
set blind again after earlier experiments have inspected it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.components.capacity_proposal_graph import (
    ARM_ROUTES,
    RELATION,
)
from experiments.heterogeneous_agents.components.capacity_proposal_selector import (
    _number,
    _within,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    _key,
    _load_graph,
)
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    _row_f1,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "capacity_multiview_graph_20260725_v1/graphs/train_graph.jsonl")
DEFAULT_VALIDATION = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "capacity_multiview_confirmation_20260725_v1/graphs/"
    "validation_graph.jsonl")
DEFAULT_BASE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "targeted_company_gemma_n3_20260724_v1/component_decoder/"
    "validation_train_selected.jsonl")
DEFAULT_PLAN = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "capacity_multiview_graph_20260725_v1/plan/PLAN.json")

ROUTE_QWEN_SC = "qwen:self_consistency"
ROUTE_GEMMA_BASE = "gemma:independent"
ROUTE_SYSTEM2 = "qwen:system2"
ROUTE_GEMMA_DIRECT = ARM_ROUTES["gemma_direct"]
ROUTE_GEMMA_MAGNITUDE = ARM_ROUTES["gemma_magnitude"]
ROUTE_QWEN_DIRECT = ARM_ROUTES["qwen_direct"]
PROPOSAL_ROUTES = tuple(ARM_ROUTES.values())
LEGACY_ROUTES = (ROUTE_QWEN_SC, ROUTE_GEMMA_BASE, ROUTE_SYSTEM2)
ALL_ROUTES = PROPOSAL_ROUTES + LEGACY_ROUTES

ALPHAS = (1.0, 4.0, 16.0, 64.0)
MARGINS = (0.0, 0.02, 0.05, 0.10, 0.20)

FEATURE_NAMES = (
    "bias",
    "log_candidate",
    "signed_log_ratio",
    "absolute_log_ratio",
    "candidate_proposal_views",
    "candidate_model_families",
    "candidate_proposal_support_sum",
    "candidate_proposal_support_max",
    "candidate_legacy_support_sum",
    "candidate_all_support_sum",
    "candidate_route_count",
    "candidate_is_proposal_only",
    "candidate_cross_model",
    "candidate_gemma_direct",
    "candidate_gemma_magnitude",
    "candidate_qwen_direct",
    "incumbent_present",
    "incumbent_proposal_views",
    "incumbent_model_families",
    "incumbent_proposal_support_sum",
    "incumbent_legacy_support_sum",
    "incumbent_all_support_sum",
    "incumbent_route_count",
    "incumbent_cross_model",
    "proposal_support_advantage",
    "legacy_support_advantage",
    "all_support_advantage",
    "distance_from_proposal_log_median",
    "candidate_proposal_rank",
    "row_proposal_log_dispersion",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _route_support(component: Mapping[str, Any] | None, route: str) -> float:
    if component is None:
        return 0.0
    return float(component.get("routes", {}).get(
        route, {}).get("max_support_rate", 0.0))


def _component_value(component: Mapping[str, Any]) -> float | None:
    values = [
        value for item in component.get("member_items", [])
        if (value := _number(item)) is not None]
    if not values:
        return None
    representative = _number(component.get("representative"))
    return representative if representative is not None else statistics.median(values)


def _model_families(component: Mapping[str, Any] | None) -> int:
    if component is None:
        return 0
    routes = set(component.get("routes", {}))
    qwen = bool(routes & {
        ROUTE_QWEN_SC, ROUTE_SYSTEM2, ROUTE_QWEN_DIRECT})
    gemma = bool(routes & {
        ROUTE_GEMMA_BASE, ROUTE_GEMMA_DIRECT, ROUTE_GEMMA_MAGNITUDE})
    return int(qwen) + int(gemma)


def _component_for_value(
    graph: Mapping[str, Any], value: float,
) -> Mapping[str, Any] | None:
    choices = []
    for component in graph.get(
            "relational_graph", {}).get("components", []):
        candidate = _component_value(component)
        if candidate is None or not _within(candidate, value):
            continue
        choices.append((
            abs(math.log(candidate / value)),
            -sum(_route_support(component, route) for route in ALL_ROUTES),
            component,
        ))
    return min(choices, key=lambda item: item[:2])[2] if choices else None


def _proposal_components(
    graph: Mapping[str, Any], incumbent: float,
) -> list[tuple[Mapping[str, Any], float]]:
    output = []
    for component in graph.get(
            "relational_graph", {}).get("components", []):
        routes = set(component.get("routes", {}))
        if not routes.intersection(PROPOSAL_ROUTES):
            continue
        value = _component_value(component)
        if value is None or _within(value, incumbent):
            continue
        output.append((component, value))
    return output


def _summaries(component: Mapping[str, Any] | None) -> dict[str, float]:
    routes = set(component.get("routes", {})) if component else set()
    proposal = [_route_support(component, route) for route in PROPOSAL_ROUTES]
    legacy = [_route_support(component, route) for route in LEGACY_ROUTES]
    return {
        "proposal_views": float(sum(value > 0 for value in proposal)),
        "model_families": float(_model_families(component)),
        "proposal_sum": sum(proposal),
        "proposal_max": max(proposal, default=0.0),
        "legacy_sum": sum(legacy),
        "all_sum": sum(proposal) + sum(legacy),
        "route_count": float(len(routes)),
        "proposal_only": float(bool(routes) and routes <= set(PROPOSAL_ROUTES)),
        "cross_model": float(_model_families(component) == 2),
    }


def action_features(
    graph: Mapping[str, Any],
    component: Mapping[str, Any],
    candidate: float,
    incumbent: float,
) -> np.ndarray:
    """Return the fixed, inference-legal incumbent/challenger feature vector."""
    incumbent_component = _component_for_value(graph, incumbent)
    candidate_summary = _summaries(component)
    incumbent_summary = _summaries(incumbent_component)
    proposals = _proposal_components(graph, incumbent)
    proposal_values = sorted(value for _, value in proposals)
    logs = [math.log(value) for value in proposal_values]
    log_median = statistics.median(logs) if logs else math.log(candidate)
    dispersion = (
        statistics.pstdev(logs) if len(logs) > 1 else 0.0)
    rank = (
        proposal_values.index(candidate) / max(1, len(proposal_values) - 1)
        if candidate in proposal_values else 0.5)
    values = (
        1.0,
        math.log(candidate),
        math.log(candidate / incumbent),
        abs(math.log(candidate / incumbent)),
        candidate_summary["proposal_views"],
        candidate_summary["model_families"],
        candidate_summary["proposal_sum"],
        candidate_summary["proposal_max"],
        candidate_summary["legacy_sum"],
        candidate_summary["all_sum"],
        candidate_summary["route_count"],
        candidate_summary["proposal_only"],
        candidate_summary["cross_model"],
        _route_support(component, ROUTE_GEMMA_DIRECT),
        _route_support(component, ROUTE_GEMMA_MAGNITUDE),
        _route_support(component, ROUTE_QWEN_DIRECT),
        float(incumbent_component is not None),
        incumbent_summary["proposal_views"],
        incumbent_summary["model_families"],
        incumbent_summary["proposal_sum"],
        incumbent_summary["legacy_sum"],
        incumbent_summary["all_sum"],
        incumbent_summary["route_count"],
        incumbent_summary["cross_model"],
        candidate_summary["proposal_sum"] - incumbent_summary["proposal_sum"],
        candidate_summary["legacy_sum"] - incumbent_summary["legacy_sum"],
        candidate_summary["all_sum"] - incumbent_summary["all_sum"],
        abs(math.log(candidate) - log_median),
        rank,
        dispersion,
    )
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("capacity feature schema drift")
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ContractError("non-finite capacity action features")
    return vector


@dataclass
class Action:
    key: tuple[str, str]
    component_id: str
    value: float
    features: np.ndarray
    utility: float | None
    weight: float


def _actions(
    graph: Mapping[str, Any],
    gold: Mapping[str, Any] | None,
    control: Sequence[str] | None = None,
) -> list[Action]:
    incumbent_objects = (
        list(control) if control is not None
        else list(graph.get("baseline_objects", [])))
    incumbent_values = [
        value for item in incumbent_objects
        if (value := _number(item)) is not None]
    if len(incumbent_values) != 1:
        raise ContractError(f"{_key(graph)}: expected one numeric incumbent")
    incumbent = incumbent_values[0]
    proposals = _proposal_components(graph, incumbent)
    if not proposals:
        return []
    weight = 1.0 / len(proposals)
    before = (
        _row_f1(incumbent_objects, gold, RELATION)
        if gold is not None else None)
    output = []
    for component, value in proposals:
        utility = (
            _row_f1([format(value, ".15g")], gold, RELATION) - before
            if gold is not None and before is not None else None)
        output.append(Action(
            key=_key(graph),
            component_id=str(component["id"]),
            value=value,
            features=action_features(graph, component, value, incumbent),
            utility=utility,
            weight=weight,
        ))
    return output


class RidgeActionModel:
    """Small dependency-free weighted ridge model for signed switch utility."""

    def __init__(
        self, alpha: float, feature_names: Sequence[str] = FEATURE_NAMES,
    ):
        self.alpha = float(alpha)
        self.feature_names = tuple(str(name) for name in feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(
                self.feature_names):
            raise ContractError("invalid ridge feature schema")
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    def fit(self, actions: Sequence[Action]) -> "RidgeActionModel":
        if not actions:
            raise ContractError("cannot fit capacity model without actions")
        x = np.stack([action.features for action in actions])
        if x.shape[1] != len(self.feature_names):
            raise ContractError(
                "action feature width does not match ridge feature schema")
        y = np.asarray([action.utility for action in actions], dtype=np.float64)
        w = np.asarray([action.weight for action in actions], dtype=np.float64)
        if any(action.utility is None for action in actions):
            raise ContractError("training action lacks utility")
        self.center = np.average(x, axis=0, weights=w)
        self.scale = np.sqrt(np.average(
            (x - self.center) ** 2, axis=0, weights=w))
        # Preserve the explicit intercept and avoid exploding sparse features.
        self.center[0] = 0.0
        self.scale[0] = 1.0
        self.scale[self.scale < 1e-8] = 1.0
        z = (x - self.center) / self.scale
        root_w = np.sqrt(w)
        zw = z * root_w[:, None]
        yw = y * root_w
        penalty = np.eye(z.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(
            zw.T @ zw + penalty + np.eye(z.shape[1]) * 1e-10,
            zw.T @ yw,
        )
        return self

    def predict(self, features: np.ndarray) -> float:
        if self.center is None or self.scale is None or self.coef is None:
            raise ContractError("capacity action model is not fitted")
        return float(((features - self.center) / self.scale) @ self.coef)

    def to_dict(self) -> dict[str, Any]:
        if self.center is None or self.scale is None or self.coef is None:
            raise ContractError("capacity action model is not fitted")
        return {
            "schema": "capacity-baseline-aware-ridge-v1",
            "alpha": self.alpha,
            "feature_names": list(self.feature_names),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coef.tolist(),
        }


def _fit_actions(
    graphs: Sequence[Mapping[str, Any]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Action]:
    return [
        action for graph in graphs
        for action in _actions(graph, gold[_key(graph)])]


def _decode(
    model: RidgeActionModel,
    graph: Mapping[str, Any],
    margin: float,
    control: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    incumbent_objects = (
        list(control) if control is not None
        else list(graph["baseline_objects"]))
    actions = _actions(graph, None, incumbent_objects)
    scored = sorted([
        (model.predict(action.features), action)
        for action in actions
    ], key=lambda item: (item[0], -abs(item[1].features[2])), reverse=True)
    if not scored or scored[0][0] <= margin:
        return incumbent_objects, {
            "used_baseline": True,
            "candidate_count": len(scored),
            "estimated_improvement": scored[0][0] if scored else 0.0,
            "margin": margin,
            "scored_candidates": [{
                "value": action.value, "estimate": estimate,
                "component_id": action.component_id,
            } for estimate, action in scored],
        }
    estimate, selected = scored[0]
    return [format(selected.value, ".15g")], {
        "used_baseline": False,
        "candidate_count": len(scored),
        "estimated_improvement": estimate,
        "margin": margin,
        "selected_value": selected.value,
        "selected_component": selected.component_id,
        "scored_candidates": [{
            "value": action.value, "estimate": value,
            "component_id": action.component_id,
        } for value, action in scored],
    }


def _configuration_scores(
    graphs: Sequence[Mapping[str, Any]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    folds: Mapping[tuple[str, str], int],
    fold_ids: Sequence[int],
) -> dict[tuple[float, float], dict[str, Any]]:
    output = {}
    for alpha in ALPHAS:
        raw = {}
        for fold in fold_ids:
            fit = [row for row in graphs if folds[_key(row)] != fold]
            hold = [row for row in graphs if folds[_key(row)] == fold]
            model = RidgeActionModel(alpha).fit(_fit_actions(fit, gold))
            raw[fold] = []
            for graph in hold:
                proposal, detail = _decode(model, graph, float("-inf"))
                raw[fold].append((
                    _row_f1(proposal, gold[_key(graph)], RELATION)
                    - _row_f1(
                        list(graph["baseline_objects"]),
                        gold[_key(graph)], RELATION),
                    float(detail["estimated_improvement"]),
                ))
        for margin in MARGINS:
            fold_deltas = {
                str(fold): statistics.mean([
                    delta if estimate > margin else 0.0
                    for delta, estimate in raw[fold]])
                for fold in fold_ids}
            actions = sum(
                estimate > margin
                for fold in fold_ids for _, estimate in raw[fold])
            values = list(fold_deltas.values())
            output[(alpha, margin)] = {
                "alpha": alpha,
                "margin": margin,
                "fold_deltas": fold_deltas,
                "mean_paired_delta": statistics.mean(values),
                "winning_folds": sum(value > 1e-12 for value in values),
                "harmful_folds": sum(value < -1e-12 for value in values),
                "actions": actions,
            }
    return output


def _select(
    scores: Mapping[tuple[float, float], Mapping[str, Any]],
) -> tuple[float, float, bool]:
    # Configuration selection is conservative: gain first, then fewer
    # interventions, stronger regularization, and larger margin.
    key, record = max(scores.items(), key=lambda item: (
        float(item[1]["mean_paired_delta"]),
        -int(item[1]["actions"]),
        float(item[0][0]),
        float(item[0][1]),
    ))
    enabled = (
        float(record["mean_paired_delta"]) > 1e-12
        and int(record["winning_folds"]) >= max(2, len(
            record["fold_deltas"]) // 2)
        and int(record["harmful_folds"]) == 0
    )
    return key[0], key[1], enabled


def _validate_graph(path: Path, split: str) -> list[dict[str, Any]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = _json(manifest_path)
    if manifest.get("split") != split:
        raise ContractError(f"{path}: split mismatch")
    if manifest.get("contains_labels") or manifest.get("gold_aware"):
        raise ContractError(f"{path}: graph must be label-free")
    if manifest.get("output_sha256") != sha256(path):
        raise ContractError(f"{path}: graph hash mismatch")
    if set(manifest.get("proposal_routes", [])) != set(PROPOSAL_ROUTES):
        raise ContractError(f"{path}: incomplete proposal routes")
    rows = [
        row for row in _load_graph(path, expected_split=split)
        if str(row["Relation"]) == RELATION]
    if len(rows) != 100:
        raise ContractError(f"{path}: expected 100 capacity rows")
    return rows


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train_graph).resolve()
    validation_path = Path(args.validation_graph).resolve()
    base_path = Path(args.base_predictions).resolve()
    plan_path = Path(args.plan).resolve()
    train = _validate_graph(train_path, "train")
    validation = _validate_graph(validation_path, "validation")
    plan = _json(plan_path)
    train_gold_path = Path(plan["train_gold"])
    folds_path = Path(plan["folds"])
    if sha256(train_gold_path) != plan["train_gold_sha256"]:
        raise ContractError("train gold hash mismatch")
    if sha256(folds_path) != plan["folds_sha256"]:
        raise ContractError("fold hash mismatch")
    train_gold = {
        _key(row): row for row in read_jsonl(train_gold_path)}
    folds = {_key(row): int(row["fold"]) for row in read_jsonl(folds_path)}
    fold_ids = sorted({folds[_key(row)] for row in train})
    if fold_ids != list(range(5)):
        raise ContractError("expected frozen folds 0..4")

    # Nested cross-fitting: every outer row is decoded with a model and
    # configuration selected without that row or its fold.
    outer_rows = []
    outer_fold_deltas = {}
    for outer in fold_ids:
        fit = [row for row in train if folds[_key(row)] != outer]
        hold = [row for row in train if folds[_key(row)] == outer]
        inner_ids = [fold for fold in fold_ids if fold != outer]
        inner_scores = _configuration_scores(
            fit, train_gold, folds, inner_ids)
        alpha, margin, enabled = _select(inner_scores)
        model = RidgeActionModel(alpha).fit(_fit_actions(fit, train_gold))
        deltas = []
        for graph in hold:
            proposal, detail = _decode(
                model, graph, margin if enabled else float("inf"))
            before = _row_f1(
                list(graph["baseline_objects"]),
                train_gold[_key(graph)], RELATION)
            after = _row_f1(proposal, train_gold[_key(graph)], RELATION)
            delta = after - before
            deltas.append(delta)
            outer_rows.append({
                "SubjectEntity": graph["SubjectEntity"],
                "Relation": RELATION,
                "fold": outer,
                "alpha": alpha,
                "margin": margin,
                "inner_gate_enabled": enabled,
                "incumbent": list(graph["baseline_objects"]),
                "proposal": proposal,
                "delta": delta,
                **detail,
            })
        outer_fold_deltas[str(outer)] = statistics.mean(deltas)
    outer_values = list(outer_fold_deltas.values())
    deployment_enabled = (
        statistics.mean(outer_values) > 1e-12
        and sum(value > 1e-12 for value in outer_values) >= 3
        and not any(value < -1e-12 for value in outer_values)
    )

    full_scores = _configuration_scores(
        train, train_gold, folds, fold_ids)
    alpha, margin, full_gate = _select(full_scores)
    deployment_enabled = deployment_enabled and full_gate
    model = RidgeActionModel(alpha).fit(_fit_actions(train, train_gold))
    model_path = output / "MODEL.json"
    model_path.write_text(json.dumps({
        **model.to_dict(),
        "contains_labels": True,
        "gold_aware": True,
        "validation_labels_used_for_selection": False,
        "train_graph": str(train_path),
        "train_graph_sha256": sha256(train_path),
        "folds": str(folds_path),
        "folds_sha256": sha256(folds_path),
        "nested_oof_fold_deltas": outer_fold_deltas,
        "nested_oof_mean_delta": statistics.mean(outer_values),
        "deployment_enabled": deployment_enabled,
        "selected_margin": margin,
        "configuration_scores": {
            f"alpha={a:g}|margin={m:g}": record
            for (a, m), record in full_scores.items()},
    }, indent=2, sort_keys=True) + "\n")
    write_jsonl_atomic(output / "train_oof_diagnostics.jsonl", outer_rows)

    base = read_jsonl(base_path)
    base_by = {_key(row): row for row in base}
    replacements = {}
    ungated_replacements = {}
    validation_diagnostics = []
    for graph in validation:
        control = list(base_by[_key(graph)]["ObjectEntities"])
        proposal, detail = _decode(
            model, graph, margin if deployment_enabled else float("inf"),
            control)
        ungated, ungated_detail = _decode(model, graph, margin, control)
        replacements[_key(graph)] = proposal
        ungated_replacements[_key(graph)] = ungated
        validation_diagnostics.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": RELATION,
            "deployment_enabled": deployment_enabled,
            "incumbent": control,
            "proposal": proposal,
            "ungated_proposal": ungated,
            "ungated_estimated_improvement": (
                ungated_detail["estimated_improvement"]),
            **detail,
        })
    predictions = []
    for row in base:
        new = dict(row)
        if _key(row) in replacements:
            new["ObjectEntities"] = replacements[_key(row)]
        predictions.append(new)
    prediction_path = output / "validation_predictions.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    ungated_predictions = []
    for row in base:
        new = dict(row)
        if _key(row) in ungated_replacements:
            new["ObjectEntities"] = ungated_replacements[_key(row)]
        ungated_predictions.append(new)
    ungated_path = output / "validation_ungated_diagnostic.jsonl"
    write_jsonl_atomic(ungated_path, ungated_predictions)
    write_jsonl_atomic(
        output / "validation_diagnostics.jsonl", validation_diagnostics)
    prediction_manifest = {
        "schema": "capacity-baseline-aware-predictions-v1",
        "contains_labels": False,
        "gold_aware": False,
        "validation_labels_used_for_selection": False,
        "rows": len(predictions),
        "output_sha256": sha256(prediction_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "base_predictions": str(base_path),
        "base_predictions_sha256": sha256(base_path),
        "validation_graph": str(validation_path),
        "validation_graph_sha256": sha256(validation_path),
        "deployment_enabled": deployment_enabled,
    }
    prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json").write_text(
            json.dumps(prediction_manifest, indent=2, sort_keys=True) + "\n")

    # Only post-hoc evaluation occurs below this line.
    validation_gold_path = Path(plan["validation_gold"])
    if sha256(validation_gold_path) != plan["validation_gold_sha256"]:
        raise ContractError("validation gold hash mismatch")
    validation_gold = read_jsonl(validation_gold_path)
    base_scores = score(base, validation_gold)
    scores = score(predictions, validation_gold)
    ungated_scores = score(ungated_predictions, validation_gold)
    if not deployment_enabled and predictions != base:
        raise ContractError(
            "closed deployment gate failed to preserve production baseline")
    edits = [
        detail for detail in validation_diagnostics
        if detail["proposal"] != detail["incumbent"]]
    ungated_edits = [
        detail for detail in validation_diagnostics
        if detail["ungated_proposal"] != detail["incumbent"]]
    gold_by = {_key(row): row for row in validation_gold}
    helpful = harmful = neutral = 0
    for edit in edits:
        key = _key(edit)
        delta = (
            _row_f1(edit["proposal"], gold_by[key], RELATION)
            - _row_f1(edit["incumbent"], gold_by[key], RELATION))
        helpful += delta > 1e-12
        harmful += delta < -1e-12
        neutral += abs(delta) <= 1e-12
    ungated_helpful = ungated_harmful = ungated_neutral = 0
    for edit in ungated_edits:
        key = _key(edit)
        delta = (
            _row_f1(edit["ungated_proposal"], gold_by[key], RELATION)
            - _row_f1(edit["incumbent"], gold_by[key], RELATION))
        ungated_helpful += delta > 1e-12
        ungated_harmful += delta < -1e-12
        ungated_neutral += abs(delta) <= 1e-12
    result = {
        "schema": "capacity-baseline-aware-selector-result-v1",
        "development_only": True,
        "validation_labels_used_for_selection": False,
        "validation_labels_used_for_posthoc_evaluation": True,
        "deployment_enabled": deployment_enabled,
        "selected_alpha": alpha,
        "selected_margin": margin,
        "nested_oof_fold_deltas": outer_fold_deltas,
        "nested_oof_mean_delta": statistics.mean(outer_values),
        "base_scores": base_scores,
        "scores": scores,
        "ungated_diagnostic_scores": ungated_scores,
        "ungated_capacity_delta": (
            ungated_scores[RELATION] - base_scores[RELATION]),
        "ungated_validation_actions": len(ungated_edits),
        "ungated_helpful_actions": ungated_helpful,
        "ungated_harmful_actions": ungated_harmful,
        "ungated_neutral_actions": ungated_neutral,
        "capacity_delta": scores[RELATION] - base_scores[RELATION],
        "pooled_delta": (
            scores["*** All Relations ***"]
            - base_scores["*** All Relations ***"]),
        "validation_actions": len(edits),
        "helpful_actions": helpful,
        "harmful_actions": harmful,
        "neutral_actions": neutral,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "RESULT.md").write_text("\n".join([
        "# Baseline-aware capacity action selector",
        "",
        "The switch model was selected with nested training folds. Validation "
        "predictions were frozen before validation labels were opened.",
        "",
        f"- deployment enabled: **{deployment_enabled}**",
        f"- nested train OOF delta: "
        f"**{statistics.mean(outer_values):+.4f}**",
        f"- validation capacity delta: "
        f"**{result['capacity_delta']:+.4f}**",
        f"- validation pooled delta: **{result['pooled_delta']:+.6f}**",
        f"- actions: **{len(edits)}** "
        f"({helpful} helpful / {harmful} harmful / {neutral} neutral)",
        "",
        "This is a development result, not a blind-test estimate.",
    ]) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-graph", default=str(DEFAULT_TRAIN))
    parser.add_argument("--validation-graph", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--base-predictions", default=str(DEFAULT_BASE))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    return parser


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
