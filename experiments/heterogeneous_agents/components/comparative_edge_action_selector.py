#!/usr/bin/env python3
"""Complete-action selector over incumbent-anchored comparative graph edges.

The original full-candidate tournament compared scores normalized against
UNKNOWN across different option groups.  Those values are not cross-group
comparable.  This selector instead creates directed challenger->incumbent
edges from log P(challenger) - log P(incumbent) inside the *same* prompt
group.  The context normalizer therefore cancels exactly.

The directed edges augment the fixed compact complete-action representation.
One row remains one training group, KEEP remains an explicit action, and all
reported training predictions are subject-grouped out of fold.  Validation is
structurally absent until the fixed promotion gate passes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.full_candidate_tournament import (
    PROMPT_ARMS,
    _json,
)
from experiments.heterogeneous_agents.components.row_grouped_action_ranker import (
    COMPACT_FEATURE_NAMES,
    FIXED_L2,
    PRIMARY_ARM,
    _load,
    _relation_deltas,
    _utility,
    compact_action_features,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks
from experiments.heterogeneous_agents.components.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import _key


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_REVIEW_RUN = (
    RUNS / "baseline_conditioned_action_review_20260727_v2")
DEFAULT_TOURNAMENT_RUN = RUNS / "full_candidate_tournament_20260727_v1"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_OUTPUT = RUNS / "comparative_edge_action_selector_20260727_v1"
AGENTS = (QWEN, GEMMA)
ARMS = (
    "compact_control", "comparative_edges", "combined",
    "comparative_identity_gate", "combined_identity_gate",
    "comparative_sign_gate", "combined_sign_gate",
    "hierarchical_edge_sign_gate", "hierarchical_sign_gate",
)
PRIMARY = "hierarchical_sign_gate"
EPSILON = 1e-12

MIN_POOLED_DELTA = 0.005
MIN_POSITIVE_FOLDS = 4
MIN_RELATION_DELTA = -0.010

EDGE_FEATURE_NAMES = tuple(
    f"{prompt}_{agent}_{summary}"
    for prompt in PROMPT_ARMS
    for agent in ("qwen", "gemma")
    for summary in ("sum", "minimum")
) + (
    "edge_all_mean",
    "edge_all_minimum",
    "edge_positive_fraction",
    "edge_agent_disagreement",
    "edge_action_coverage",
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _validated_tournament(
    tournament_run: Path,
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[str, str], dict[tuple[str, str], dict[str, float]]],
    dict[str, Any],
]:
    plan_path = tournament_run / "plan/PLAN.json"
    plan = _json(plan_path)
    registry_path = Path(plan["registry"])
    if (
        plan.get("schema") != "full-candidate-tournament-plan-v1"
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or sha256(registry_path) != plan["registry_sha256"]
    ):
        raise ContractError("invalid comparative tournament plan")
    registry = {_key(row): row for row in read_jsonl(registry_path)}
    tasks: list[Mapping[str, Any]] = []
    responses: dict[str, Mapping[str, Any]] = {}
    for agent, job in plan["jobs"].items():
        task_path = Path(job["task_path"])
        response_path = Path(job["response_path"])
        manifest_path = response_path.with_suffix(
            response_path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise ContractError("missing comparative response manifest")
        manifest = _json(manifest_path)
        if (
            sha256(task_path) != job["task_sha256"]
            or manifest.get("task_sha256") != job["task_sha256"]
            or manifest.get("output_sha256") != sha256(response_path)
            or int(manifest.get("tasks", -1)) != int(job["tasks"])
            or manifest.get("agent_id") != agent
        ):
            raise ContractError("stale comparative response artifact")
        agent_tasks = read_jsonl(task_path)
        by_id = validate_tasks(agent_tasks, agent)
        agent_responses = read_jsonl(response_path)
        if len(agent_tasks) != len(agent_responses):
            raise ContractError("incomplete comparative responses")
        for response in agent_responses:
            task_id = str(response["task_id"])
            if task_id not in by_id or task_id in responses:
                raise ContractError("invalid comparative response id")
            validate_task_response(by_id[task_id], response)
            responses[task_id] = response
        tasks.extend(agent_tasks)

    accumulator: dict[
        tuple[str, str],
        dict[tuple[str, str], dict[str, list[float]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for task in tasks:
        if task["view"] != "incumbent3":
            continue
        key = str(task["subject"]), str(task["relation"])
        row = registry[key]
        incumbent_ids = {
            str(node["node_id"]) for node in row["nodes"]
            if bool(node["is_incumbent"])}
        group = set(map(str, task["group_node_ids"]))
        anchors = incumbent_ids & group
        if not anchors:
            raise ContractError(f"{key}: incumbent3 group lacks incumbent")
        probabilities = responses[str(task["task_id"])][
            "choice_probabilities"]
        signal = str(task["prompt_arm"]), str(task["agent_id"])
        for challenger in group - incumbent_ids:
            for incumbent in anchors:
                margin = (
                    math.log(max(float(probabilities[challenger]), EPSILON))
                    - math.log(max(float(probabilities[incumbent]), EPSILON)))
                accumulator[key][signal][challenger].append(margin)
    edges = {
        key: {
            signal: {
                node: float(np.mean(values))
                for node, values in nodes.items()
            }
            for signal, nodes in signals.items()
        }
        for key, signals in accumulator.items()
    }
    return registry, edges, plan


def _component_map(graph: Mapping[str, Any]) -> dict[str, str]:
    relation = str(graph["Relation"])
    output: dict[str, str] = {}
    for node in graph["nodes"]:
        if node.get("node_type") != "candidate_component":
            continue
        for value in [
            *node.get("member_items", []), node.get("representative", "")]:
            output[canonical_key(str(value), relation)] = str(node["id"])
    return output


def edge_action_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    tournament_registry: Mapping[str, Any] | None,
    edges: Mapping[tuple[str, str], Mapping[str, float]] | None,
) -> list[float]:
    """Summarize only within-group directed evidence for one complete action."""
    if tournament_registry is None or edges is None:
        return [0.0] * len(EDGE_FEATURE_NAMES)
    relation = str(graph["Relation"])
    component_by_surface = _component_map(graph)
    incumbent_ids = {
        str(node["node_id"]) for node in tournament_registry["nodes"]
        if bool(node["is_incumbent"])}
    selected_ids = {
        component_by_surface[key]
        for item in action["objects"]
        if (key := canonical_key(str(item), relation))
        in component_by_surface
    }
    added = selected_ids - incumbent_ids
    all_margins: list[float] = []
    values: list[float] = []
    q_means: list[float] = []
    g_means: list[float] = []
    covered = 0
    for prompt in PROMPT_ARMS:
        for agent in AGENTS:
            signal = prompt, agent
            margins = [
                float(edges.get(signal, {}).get(node, -12.0))
                for node in added]
            if not added:
                summary_sum = summary_min = 0.0
            else:
                covered += sum(
                    node in edges.get(signal, {}) for node in added)
                summary_sum = sum(margins)
                summary_min = min(margins)
                all_margins.extend(margins)
            values.extend([
                max(-24.0, min(24.0, summary_sum)) / 24.0,
                max(-12.0, min(12.0, summary_min)) / 12.0,
            ])
            (q_means if agent == QWEN else g_means).append(
                float(np.mean(margins)) if margins else 0.0)
    if all_margins:
        mean_margin = float(np.mean(all_margins))
        minimum_margin = min(all_margins)
        positive_fraction = float(np.mean(
            [margin > 0.0 for margin in all_margins]))
    else:
        mean_margin = minimum_margin = positive_fraction = 0.0
    values.extend([
        max(-12.0, min(12.0, mean_margin)) / 12.0,
        max(-12.0, min(12.0, minimum_margin)) / 12.0,
        positive_fraction,
        min(12.0, float(np.mean([
            abs(q - g) for q, g in zip(q_means, g_means, strict=True)
        ]))) / 12.0,
        covered / max(1, len(added) * len(PROMPT_ARMS) * len(AGENTS)),
    ])
    if len(values) != len(EDGE_FEATURE_NAMES):
        raise AssertionError("comparative edge feature schema drift")
    return values


def selector_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    tournament_registry: Mapping[str, Any] | None,
    edges: Mapping[tuple[str, str], Mapping[str, float]] | None,
    arm: str,
) -> list[float]:
    compact = compact_action_features(
        graph, action, review_evidence,
        include_review=arm in {
            "compact_control", "combined", "combined_identity_gate"})
    comparative = edge_action_features(
        graph, action, tournament_registry, edges)
    if arm == "compact_control":
        return compact
    if arm in {"comparative_edges", "comparative_identity_gate"}:
        # Retain action identity/family/geometry, but exclude model review.
        return compact[:len(COMPACT_FEATURE_NAMES) - 3] + comparative
    if arm in {"combined", "combined_identity_gate"}:
        return compact + comparative
    raise ContractError(f"unknown comparative selector arm: {arm}")


class ConditionalActionRanker:
    """Regularized conditional-logit model with dynamic feature schema."""

    def __init__(self, feature_names: Sequence[str], l2: float = FIXED_L2):
        self.feature_names = tuple(feature_names)
        self.l2 = float(l2)
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    def fit(
        self,
        groups: Sequence[
            tuple[Sequence[Sequence[float]], Sequence[float], int]],
    ) -> "ConditionalActionRanker":
        improvable = [
            max(map(float, utilities))
            > float(utilities[keep]) + EPSILON
            for _, utilities, keep in groups]
        counts = {
            value: sum(item == value for item in improvable)
            for value in (False, True)}
        if min(counts.values()) <= 0:
            raise ContractError("ranker requires both row classes")
        rows: list[np.ndarray] = []
        labels: list[float] = []
        weights: list[float] = []
        for (features, utilities, keep), positive in zip(
            groups, improvable, strict=True,
        ):
            pairs = []
            for left, right in itertools.combinations(
                range(len(features)), 2,
            ):
                delta = float(utilities[left]) - float(utilities[right])
                if abs(delta) <= EPSILON:
                    if keep not in {left, right}:
                        continue
                    winner = keep
                else:
                    winner = left if delta > 0.0 else right
                loser = right if winner == left else left
                difference = (
                    np.asarray(features[winner], dtype=np.float64)
                    - np.asarray(features[loser], dtype=np.float64))
                pairs.extend([(difference, 1.0), (-difference, 0.0)])
            if not pairs:
                continue
            weight = 0.5 / counts[positive] / len(pairs)
            for difference, label in pairs:
                rows.append(difference)
                labels.append(label)
                weights.append(weight)
        matrix = np.asarray(rows)
        target = np.asarray(labels)
        sample_weight = np.asarray(weights)
        sample_weight *= len(sample_weight) / sample_weight.sum()
        variance = np.average(matrix**2, axis=0, weights=sample_weight)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = matrix / self.scale
        beta = np.zeros(design.shape[1])
        penalty = np.eye(design.shape[1]) * self.l2
        for _ in range(100):
            logits = np.clip(design @ beta, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            curvature = np.maximum(
                probability * (1.0 - probability), 1e-8)
            gradient = (
                design.T @ (sample_weight * (probability - target))
                + penalty @ beta)
            hessian = (
                design.T @ (
                    design * (sample_weight * curvature)[:, None])
                + penalty)
            step = np.linalg.solve(hessian, gradient)
            beta -= step
            if float(np.max(np.abs(step))) < 1e-9:
                break
        self.coef = beta
        return self

    def scores(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        if self.scale is None or self.coef is None:
            raise RuntimeError("unfitted comparative action ranker")
        return np.asarray(features) / self.scale @ self.coef

    def to_dict(self) -> dict[str, Any]:
        if self.scale is None or self.coef is None:
            raise RuntimeError("unfitted comparative action ranker")
        return {
            "schema": "comparative-edge-conditional-action-ranker-v1",
            "feature_names": list(self.feature_names),
            "parameter_count": len(self.feature_names),
            "l2": self.l2,
            "scale": self.scale.tolist(),
            "coefficients": self.coef.tolist(),
        }


def _feature_names(arm: str) -> tuple[str, ...]:
    if arm == "compact_control":
        return COMPACT_FEATURE_NAMES
    base = COMPACT_FEATURE_NAMES[:-3]
    if arm in {"comparative_edges", "comparative_identity_gate"}:
        return base + EDGE_FEATURE_NAMES
    if arm in {"combined", "combined_identity_gate"}:
        return COMPACT_FEATURE_NAMES + EDGE_FEATURE_NAMES
    raise ContractError("unknown feature arm")


RELATIONS = (
    "awardWonBy", "companyTradesAtStockExchange",
    "countryLandBordersCountry", "hasArea", "hasCapacity",
    "personHasCityOfDeath",
)
SIGN_FEATURE_NAMES = (
    EDGE_FEATURE_NAMES
    + tuple(f"relation_{relation}" for relation in RELATIONS)
)
COMBINED_SIGN_FEATURE_NAMES = SIGN_FEATURE_NAMES + (
    "qwen_review_log_odds", "gemma_review_log_odds",
    "mean_review_uncertainty",
)


def _sign_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    tournament_registry: Mapping[str, Any] | None,
    edges: Mapping[tuple[str, str], Mapping[str, float]] | None,
    *,
    include_review: bool,
) -> list[float]:
    values = edge_action_features(
        graph, action, tournament_registry, edges)
    relation = str(graph["Relation"])
    values.extend(float(relation == item) for item in RELATIONS)
    if include_review:
        compact = compact_action_features(
            graph, action, review_evidence, include_review=True)
        values.extend(compact[-3:])
    return values


class UtilitySignGate:
    """Regularized benefit-versus-harm classifier.

    Utility-neutral substitutions are intentionally excluded: they contain no
    information about the sign of an accuracy change and previously dominated
    the KEEP loss.
    """

    def __init__(self, feature_names: Sequence[str], l2: float = FIXED_L2):
        self.feature_names = tuple(feature_names)
        self.l2 = float(l2)
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept = 0.0

    def fit(
        self, features: Sequence[Sequence[float]], labels: Sequence[float],
    ) -> "UtilitySignGate":
        matrix = np.asarray(features, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(self.feature_names)
            or set(np.unique(target)) != {0.0, 1.0}
        ):
            raise ContractError("invalid utility-sign training matrix")
        variance = np.mean(matrix**2, axis=0)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = np.column_stack([
            np.ones(len(matrix)), matrix / self.scale])
        beta = np.zeros(design.shape[1])
        penalty = np.eye(design.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        for _ in range(100):
            logits = np.clip(design @ beta, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            curvature = np.maximum(
                probability * (1.0 - probability), 1e-8)
            gradient = design.T @ (probability - target) + penalty @ beta
            hessian = (
                design.T @ (design * curvature[:, None]) + penalty)
            step = np.linalg.solve(hessian, gradient)
            beta -= step
            if float(np.max(np.abs(step))) < 1e-9:
                break
        self.intercept = float(beta[0])
        self.coef = beta[1:]
        return self

    def probability(self, features: Sequence[float]) -> float:
        if self.scale is None or self.coef is None:
            raise RuntimeError("unfitted utility-sign gate")
        logit = self.intercept + (
            np.asarray(features) / self.scale) @ self.coef
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))

    def to_dict(self) -> dict[str, Any]:
        if self.scale is None or self.coef is None:
            raise RuntimeError("unfitted utility-sign gate")
        return {
            "schema": "comparative-edge-utility-sign-gate-v1",
            "feature_names": list(self.feature_names),
            "parameter_count": len(self.feature_names) + 1,
            "l2": self.l2,
            "scale": self.scale.tolist(),
            "intercept": self.intercept,
            "coefficients": self.coef.tolist(),
            "neutral_training_policy": "excluded",
            "decision_boundary": 0.5,
        }


def _identity_action_indices(
    graph: Mapping[str, Any],
    tournament_registry: Mapping[str, Any] | None,
    edges: Mapping[tuple[str, str], Mapping[str, float]] | None,
) -> tuple[int, ...]:
    """Return KEEP plus one fixed-cardinality Qwen-submission challenger.

    Identity and gate are intentionally separated.  The challenger is chosen
    without labels by the strongest direct anchored edge collected in the
    final-submission prompt.  The learned model sees only KEEP versus this
    already-fixed identity.
    """
    actions = list(graph["actions"])
    keep = next(
        index for index, action in enumerate(actions)
        if action["action_type"] == "KEEP")
    incumbent_size = len(actions[keep]["objects"])
    feature_index = EDGE_FEATURE_NAMES.index("submission_qwen_sum")
    candidates = []
    for index, action in enumerate(actions):
        if index == keep or len(action["objects"]) != incumbent_size:
            continue
        features = edge_action_features(
            graph, action, tournament_registry, edges)
        if features[-1] <= 0.0:
            continue
        candidates.append((features[feature_index], -index, index))
    if not candidates:
        return (keep,)
    best = max(candidates)
    # A directed edge must actually prefer the challenger to its same-context
    # incumbent anchor.  Otherwise the graph has supplied rejection evidence,
    # not a weakly ranked alternative.
    if best[0] <= 0.0:
        return (keep,)
    return keep, best[2]


def _groups(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    tournament_registry: Mapping[tuple[str, str], Mapping[str, Any]],
    edges: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    arm: str,
) -> list[tuple[list[list[float]], list[float], int]]:
    output = []
    for graph in graphs:
        key = _key(graph)
        actions = list(graph["actions"])
        key = _key(graph)
        keep = next(
            index for index, action in enumerate(actions)
            if action["action_type"] == "KEEP")
        indices = (
            _identity_action_indices(
                graph, tournament_registry.get(key), edges.get(key))
            if arm.endswith("_identity_gate")
            else tuple(range(len(actions))))
        output.append((
            [
                selector_features(
                    graph, action, review_evidence,
                    tournament_registry.get(key), edges.get(key), arm)
                for index, action in enumerate(actions) if index in indices
            ],
            [
                _utility(graph, action, gold_by)
                for index, action in enumerate(actions) if index in indices
            ],
            indices.index(keep),
        ))
    return output


def _decode(
    model: ConditionalActionRanker,
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    tournament_registry: Mapping[tuple[str, str], Mapping[str, Any]],
    edges: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics = [], []
    for graph in graphs:
        key = _key(graph)
        actions = list(graph["actions"])
        indices = (
            _identity_action_indices(
                graph, tournament_registry.get(key), edges.get(key))
            if arm.endswith("_identity_gate")
            else tuple(range(len(actions))))
        selected_actions = [actions[index] for index in indices]
        features = [
            selector_features(
                graph, action, review_evidence,
                tournament_registry.get(key), edges.get(key), arm)
            for action in selected_actions]
        values = model.scores(features)
        keep_local = next(
            index for index, action in enumerate(selected_actions)
            if action["action_type"] == "KEEP")
        selected = max(
            range(len(selected_actions)),
            key=lambda index: (
                float(values[index]), index == keep_local, -index))
        action = selected_actions[selected]
        before = _utility(graph, selected_actions[keep_local], gold_by)
        after = _utility(graph, action, gold_by)
        best = max(_utility(graph, item, gold_by) for item in actions)
        predictions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": list(action["objects"]),
        })
        diagnostics.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "selected_action": action["id"],
            "changed": selected != keep_local,
            "selected_score": float(values[selected]),
            "keep_score": float(values[keep_local]),
            "utility_delta": after - before,
            "oracle_regret": best - after,
            "row_has_beneficial_action": best > before + EPSILON,
        })
    return predictions, diagnostics


def _fit_sign_gate(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    tournament_registry: Mapping[tuple[str, str], Mapping[str, Any]],
    edges: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    *,
    include_review: bool,
) -> UtilitySignGate:
    features, labels = [], []
    for graph in graphs:
        key = _key(graph)
        actions = list(graph["actions"])
        indices = _identity_action_indices(
            graph, tournament_registry.get(key), edges.get(key))
        if len(indices) != 2:
            continue
        keep, alternative = indices
        delta = (
            _utility(graph, actions[alternative], gold_by)
            - _utility(graph, actions[keep], gold_by))
        if abs(delta) <= EPSILON:
            continue
        features.append(_sign_features(
            graph, actions[alternative], review_evidence,
            tournament_registry.get(key), edges.get(key),
            include_review=include_review))
        labels.append(float(delta > 0.0))
    names = (
        COMBINED_SIGN_FEATURE_NAMES if include_review
        else SIGN_FEATURE_NAMES)
    return UtilitySignGate(names).fit(features, labels)


def _fit_relation_reliability(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    tournament_registry: Mapping[tuple[str, str], Mapping[str, Any]],
    edges: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
) -> dict[str, dict[str, Any]]:
    """Estimate whether the fixed comparative identity has positive net value.

    This is a hierarchical safety layer, not a relation-specific decoder.  It
    learns one sign per relation from the same complete-action utility target
    and defaults fail-closed when no utility-changing examples exist.
    """
    values: dict[str, list[float]] = defaultdict(list)
    for graph in graphs:
        key = _key(graph)
        actions = list(graph["actions"])
        indices = _identity_action_indices(
            graph, tournament_registry.get(key), edges.get(key))
        if len(indices) != 2:
            continue
        keep, alternative = indices
        delta = (
            _utility(graph, actions[alternative], gold_by)
            - _utility(graph, actions[keep], gold_by))
        if abs(delta) > EPSILON:
            values[str(graph["Relation"])].append(delta)
    return {
        relation: {
            "allowed": bool(
                values.get(relation)
                and sum(values[relation]) > 0.0),
            "utility_changing_rows": len(values.get(relation, [])),
            "helped": sum(
                value > 0.0 for value in values.get(relation, [])),
            "harmed": sum(
                value < 0.0 for value in values.get(relation, [])),
            "net_utility": sum(values.get(relation, [])),
        }
        for relation in RELATIONS
    }


def _decode_sign_gate(
    model: UtilitySignGate,
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    tournament_registry: Mapping[tuple[str, str], Mapping[str, Any]],
    edges: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    *,
    include_review: bool,
    relation_reliability: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics = [], []
    for graph in graphs:
        key = _key(graph)
        actions = list(graph["actions"])
        indices = _identity_action_indices(
            graph, tournament_registry.get(key), edges.get(key))
        keep = indices[0]
        probability = 0.0
        selected = keep
        if len(indices) == 2:
            alternative = indices[1]
            probability = model.probability(_sign_features(
                graph, actions[alternative], review_evidence,
                tournament_registry.get(key), edges.get(key),
                include_review=include_review))
            relation_allowed = (
                True if relation_reliability is None
                else bool(relation_reliability[
                    str(graph["Relation"])]["allowed"]))
            if probability > 0.5 and relation_allowed:
                selected = alternative
        action = actions[selected]
        before = _utility(graph, actions[keep], gold_by)
        after = _utility(graph, action, gold_by)
        best = max(_utility(graph, item, gold_by) for item in actions)
        predictions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": list(action["objects"]),
        })
        diagnostics.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "selected_action": action["id"],
            "changed": selected != keep,
            "benefit_probability": probability,
            "relation_allowed": (
                True if relation_reliability is None
                else bool(relation_reliability[
                    str(graph["Relation"])]["allowed"])),
            "utility_delta": after - before,
            "oracle_regret": best - after,
            "row_has_beneficial_action": best > before + EPSILON,
        })
    return predictions, diagnostics


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    review_run = Path(args.review_run).resolve()
    tournament_run = Path(args.tournament_run).resolve()
    gold_path = Path(args.gold).resolve()
    review_plan, graphs, control_rows, gold_by, folds, review_evidence = _load(
        review_run, gold_path)
    control_by = {_key(row): row for row in control_rows}
    tournament_registry, edges, tournament_plan = _validated_tournament(
        tournament_run)
    if not set(tournament_registry).issubset({_key(row) for row in graphs}):
        raise ContractError("tournament rows not covered by action graph")

    arm_results: dict[str, Any] = {}
    prediction_artifacts: dict[str, list[dict[str, Any]]] = {}
    diagnostic_artifacts: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        oof: dict[tuple[str, str], dict[str, Any]] = {}
        diagnostics: list[dict[str, Any]] = []
        fold_results = []
        for outer in sorted(set(folds.values())):
            fit = [row for row in graphs if folds[_key(row)] != outer]
            hold = [row for row in graphs if folds[_key(row)] == outer]
            if arm.endswith("_sign_gate"):
                include_review = (
                    arm.startswith("combined")
                    or arm == "hierarchical_sign_gate")
                model = _fit_sign_gate(
                    fit, gold_by, review_evidence,
                    tournament_registry, edges,
                    include_review=include_review)
                relation_reliability = (
                    _fit_relation_reliability(
                        fit, gold_by, tournament_registry, edges)
                    if arm.startswith("hierarchical") else None)
                predictions, detail = _decode_sign_gate(
                    model, hold, gold_by, review_evidence,
                    tournament_registry, edges,
                    include_review=include_review,
                    relation_reliability=relation_reliability)
            else:
                model = ConditionalActionRanker(
                    _feature_names(arm)).fit(_groups(
                        fit, gold_by, review_evidence,
                        tournament_registry, edges, arm))
                predictions, detail = _decode(
                    model, hold, gold_by, review_evidence,
                    tournament_registry, edges, arm)
            for row in predictions:
                oof[_key(row)] = row
            diagnostics.extend(
                {**item, "outer_fold": outer, "arm": arm}
                for item in detail)
            hold_gold = [gold_by[_key(row)] for row in hold]
            hold_control = [control_by[_key(row)] for row in hold]
            control_value = score(
                hold_control, hold_gold)["*** All Relations ***"]
            selected_value = score(
                predictions, hold_gold)["*** All Relations ***"]
            fold_results.append({
                "fold": outer,
                "control": control_value,
                "selected": selected_value,
                "delta": selected_value - control_value,
                "changed": sum(item["changed"] for item in detail),
                "helped": sum(item["utility_delta"] > EPSILON for item in detail),
                "harmed": sum(item["utility_delta"] < -EPSILON for item in detail),
            })
        if set(oof) != {_key(row) for row in graphs}:
            raise ContractError("comparative OOF coverage failure")
        ordered = [oof[_key(row)] for row in graphs]
        ordered_gold = [gold_by[_key(row)] for row in graphs]
        ordered_control = [control_by[_key(row)] for row in graphs]
        control_scores = score(ordered_control, ordered_gold)
        selected_scores = score(ordered, ordered_gold)
        changed = [item for item in diagnostics if item["changed"]]
        arm_results[arm] = {
            "control_scores": control_scores,
            "selected_scores": selected_scores,
            "incremental_delta": (
                selected_scores["*** All Relations ***"]
                - control_scores["*** All Relations ***"]),
            "relation_deltas": _relation_deltas(
                selected_scores, control_scores),
            "changed_rows": len(changed),
            "helped_rows": sum(
                item["utility_delta"] > EPSILON for item in changed),
            "harmed_rows": sum(
                item["utility_delta"] < -EPSILON for item in changed),
            "neutral_changed_rows": sum(
                abs(item["utility_delta"]) <= EPSILON for item in changed),
            "positive_folds": sum(item["delta"] > 0.0 for item in fold_results),
            "folds": fold_results,
        }
        prediction_artifacts[arm] = ordered
        diagnostic_artifacts[arm] = diagnostics

    primary = arm_results[PRIMARY]
    passed = (
        primary["incremental_delta"] >= MIN_POOLED_DELTA
        and primary["positive_folds"] >= MIN_POSITIVE_FOLDS
        and min(primary["relation_deltas"].values()) >= MIN_RELATION_DELTA
        and primary["helped_rows"] > primary["harmed_rows"])

    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for arm in ARMS:
        prediction_path = output / f"analysis/{arm}_OOF_PREDICTIONS.jsonl"
        diagnostic_path = output / f"analysis/{arm}_OOF_DIAGNOSTICS.jsonl"
        write_jsonl_atomic(prediction_path, prediction_artifacts[arm])
        write_jsonl_atomic(diagnostic_path, diagnostic_artifacts[arm])
        artifacts[arm] = {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
        }

    final_relation_reliability = None
    if PRIMARY.endswith("_sign_gate"):
        final_model = _fit_sign_gate(
            graphs, gold_by, review_evidence,
            tournament_registry, edges,
            include_review=(
                PRIMARY.startswith("combined")
                or PRIMARY == "hierarchical_sign_gate"))
        if PRIMARY.startswith("hierarchical"):
            final_relation_reliability = _fit_relation_reliability(
                graphs, gold_by, tournament_registry, edges)
    else:
        final_model = ConditionalActionRanker(
            _feature_names(PRIMARY)).fit(_groups(
                graphs, gold_by, review_evidence,
                tournament_registry, edges, PRIMARY))
    model_path = output / "analysis/TRAIN_FIT_MODEL.json"
    _write_json(model_path, {
        **final_model.to_dict(),
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "promotion_gate_passed": passed,
        "relation_reliability": final_relation_reliability,
    })
    result = {
        "schema": "comparative-edge-action-selector-result-v1",
        "development_only": True,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "primary_arm": PRIMARY,
        "fixed_l2": FIXED_L2,
        "rows": len(graphs),
        "actions": sum(len(row["actions"]) for row in graphs),
        "tournament_rows": len(tournament_registry),
        "edge_rule": "same-group log P(challenger)-log P(incumbent)",
        "arms": arm_results,
        "artifacts": artifacts,
        "source_hashes": {
            "review_plan": sha256(review_run / "plan/PLAN.json"),
            "tournament_plan": sha256(tournament_run / "plan/PLAN.json"),
            "gold": sha256(gold_path),
        },
        "deployment_gate": {
            "passed": passed,
            "minimum_incremental_delta": MIN_POOLED_DELTA,
            "minimum_positive_folds": MIN_POSITIVE_FOLDS,
            "minimum_relation_delta": MIN_RELATION_DELTA,
        },
    }
    _write_json(output / "RESULT.json", result)
    lines = [
        "# Comparative edge complete-action selector", "",
        "Uses only within-group challenger/incumbent log-odds edges.", "",
        "| arm | OOF score | delta | changed | helped | harmed | positive folds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in arm_results.items():
        lines.append(
            f"| {arm} | "
            f"{item['selected_scores']['*** All Relations ***']:.6f} | "
            f"{item['incremental_delta']:+.6f} | "
            f"{item['changed_rows']} | {item['helped_rows']} | "
            f"{item['harmed_rows']} | {item['positive_folds']}/5 |")
    lines += ["", f"Promotion gate: **{passed}**", ""]
    (output / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps({
        "arms": {
            arm: {
                "score": item["selected_scores"]["*** All Relations ***"],
                "delta": item["incremental_delta"],
                "changed": item["changed_rows"],
                "helped": item["helped_rows"],
                "harmed": item["harmed_rows"],
                "positive_folds": item["positive_folds"],
            }
            for arm, item in arm_results.items()
        },
        "gate_passed": passed,
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--review-run", default=str(DEFAULT_REVIEW_RUN))
    value.add_argument("--tournament-run", default=str(DEFAULT_TOURNAMENT_RUN))
    value.add_argument("--gold", default=str(DEFAULT_GOLD))
    value.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
