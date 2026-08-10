#!/usr/bin/env python3
"""Complete-action selector over connected within-row preference graphs.

The full candidate tournament asks Qwen and Gemma to compare small anonymous
groups of candidates.  Probabilities from *different* groups cannot be
compared directly because each group has a different softmax normalizer.
Within one group, however,

    log P(i | group) - log P(j | group)

is a valid directed pairwise margin: the group normalizer cancels exactly.

This module retains every such candidate/candidate margin from both tournament
views.  ``global3`` supplies challenger/challenger edges and ``incumbent3``
supplies the star edges that connect those local groups to the incumbent.  A
regularized graph-Laplacian solve produces one latent score per component for
each frozen model/prompt channel.  Complete legal actions are then compared by
the sum of their component scores.  A separately cross-fitted utility-sign
gate decides whether the fixed graph-selected alternative should replace the
exact registered SOTA incumbent.

All model outputs are frozen and label-free.  Labels are opened only for
subject-grouped five-fold fitting and evaluation.  Validation is structurally
absent from this experiment.
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
from experiments.heterogeneous_agents.comparative_edge_action_selector import (
    EPSILON,
    RELATIONS,
    UtilitySignGate,
    _decode_sign_gate as _decode_incumbent_gate,
    _fit_relation_reliability as _fit_incumbent_reliability,
    _fit_sign_gate as _fit_incumbent_gate,
    _validated_tournament as _validated_incumbent_edges,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.full_candidate_tournament import (
    PROMPT_ARMS,
    _json,
)
from experiments.heterogeneous_agents.row_grouped_action_ranker import (
    FIXED_L2,
    _load,
    _relation_deltas,
    _utility,
    compact_action_features,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks
from experiments.heterogeneous_agents.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
)
from experiments.heterogeneous_agents.unified_memory_action_graph import _key


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_REVIEW_RUN = (
    RUNS / "baseline_conditioned_action_review_20260727_v2")
DEFAULT_TOURNAMENT_RUN = RUNS / "full_candidate_tournament_20260727_v1"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_OUTPUT = RUNS / "pairwise_preference_graph_selector_20260727_v1"

AGENTS = (QWEN, GEMMA)
CHANNELS = tuple(
    (prompt, agent) for prompt in PROMPT_ARMS for agent in AGENTS)
IDENTITY_RULES = ("qwen_submission", "ensemble_mean", "robust_minimum")
ARMS = tuple(
    f"{identity}_{decoder}"
    for identity in IDENTITY_RULES
    for decoder in (
        "raw", "hierarchical_raw", "sign_gate",
        "hierarchical_sign_gate",
    )
)
CASCADE_ARM = "incumbent_gate_then_pairwise_graph"
ARMS = (*ARMS, CASCADE_ARM)
PRIMARY = CASCADE_ARM

# These are predeclared promotion requirements, not values selected after
# observing the output of this experiment.
MIN_INCREMENTAL_DELTA = 0.015
MIN_POSITIVE_FOLDS = 4
MIN_RELATION_DELTA = -0.010

GRAPH_RIDGE = 0.05
MAX_MARGIN = 12.0

GRAPH_ACTION_FEATURE_NAMES = tuple(
    f"{prompt}_{'qwen' if agent == QWEN else 'gemma'}_action_delta"
    for prompt, agent in CHANNELS
) + (
    "graph_delta_mean",
    "graph_delta_minimum",
    "graph_delta_maximum",
    "graph_positive_fraction",
    "graph_qwen_mean",
    "graph_gemma_mean",
    "graph_model_disagreement",
    "graph_selected_component_coverage",
)
GATE_FEATURE_NAMES = (
    GRAPH_ACTION_FEATURE_NAMES
    + tuple(f"relation_{relation}" for relation in RELATIONS)
    + (
        "qwen_review_log_odds",
        "gemma_review_log_odds",
        "mean_review_uncertainty",
    )
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _validated_pairwise_observations(
    tournament_run: Path,
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[
        tuple[str, str],
        dict[tuple[str, str], list[tuple[str, str, float]]],
    ],
    dict[str, Any],
]:
    """Load frozen responses and retain every legal same-group pair margin."""
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
        raise ContractError("invalid pairwise tournament plan")
    registry = {_key(row): row for row in read_jsonl(registry_path)}

    observations: dict[
        tuple[str, str],
        dict[tuple[str, str], list[tuple[str, str, float]]],
    ] = defaultdict(lambda: defaultdict(list))
    task_count = 0
    for agent, job in plan["jobs"].items():
        task_path = Path(job["task_path"])
        response_path = Path(job["response_path"])
        manifest_path = response_path.with_suffix(
            response_path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise ContractError("missing pairwise response manifest")
        manifest = _json(manifest_path)
        if (
            sha256(task_path) != job["task_sha256"]
            or manifest.get("task_sha256") != job["task_sha256"]
            or manifest.get("output_sha256") != sha256(response_path)
            or int(manifest.get("tasks", -1)) != int(job["tasks"])
            or manifest.get("agent_id") != agent
        ):
            raise ContractError("stale pairwise response artifact")
        tasks = read_jsonl(task_path)
        tasks_by_id = validate_tasks(tasks, agent)
        responses = read_jsonl(response_path)
        if len(tasks) != len(responses):
            raise ContractError("incomplete pairwise responses")
        responses_by_id = {}
        for response in responses:
            task_id = str(response["task_id"])
            if task_id not in tasks_by_id or task_id in responses_by_id:
                raise ContractError("invalid pairwise response id")
            validate_task_response(tasks_by_id[task_id], response)
            responses_by_id[task_id] = response
        for task in tasks:
            key = str(task["subject"]), str(task["relation"])
            if key not in registry:
                raise ContractError(f"unregistered tournament row: {key}")
            probabilities = responses_by_id[str(task["task_id"])][
                "choice_probabilities"]
            node_ids = tuple(map(str, task["group_node_ids"]))
            channel = str(task["prompt_arm"]), str(task["agent_id"])
            for left, right in itertools.combinations(node_ids, 2):
                margin = (
                    math.log(max(float(probabilities[left]), EPSILON))
                    - math.log(max(float(probabilities[right]), EPSILON)))
                observations[key][channel].append((
                    left, right,
                    max(-MAX_MARGIN, min(MAX_MARGIN, margin)),
                ))
            task_count += 1
    if task_count != sum(int(job["tasks"]) for job in plan["jobs"].values()):
        raise ContractError("pairwise task coverage failure")
    return registry, {
        key: dict(channels) for key, channels in observations.items()
    }, plan


def solve_preference_scores(
    node_ids: Sequence[str],
    observations: Sequence[tuple[str, str, float]],
    incumbent_ids: Sequence[str],
    *,
    ridge: float = GRAPH_RIDGE,
) -> dict[str, float]:
    """Solve weighted log-odds edge differences with an incumbent anchor.

    Repeated observations are retained as repeated equal-weight edges.  The
    ridge makes disconnected or weakly constrained inputs deterministic.  The
    final translation sets the mean incumbent score to zero; translations do
    not affect any pairwise margin or fixed-cardinality action comparison.
    """
    nodes = tuple(dict.fromkeys(map(str, node_ids)))
    if not nodes:
        return {}
    index = {node: position for position, node in enumerate(nodes)}
    laplacian = np.eye(len(nodes), dtype=np.float64) * float(ridge)
    target = np.zeros(len(nodes), dtype=np.float64)
    for left, right, margin in observations:
        if left not in index or right not in index or left == right:
            raise ContractError("invalid preference edge endpoint")
        i, j = index[left], index[right]
        laplacian[i, i] += 1.0
        laplacian[j, j] += 1.0
        laplacian[i, j] -= 1.0
        laplacian[j, i] -= 1.0
        target[i] += float(margin)
        target[j] -= float(margin)
    values = np.linalg.solve(laplacian, target)
    anchors = [values[index[node]] for node in incumbent_ids if node in index]
    offset = float(np.mean(anchors)) if anchors else float(np.mean(values))
    return {
        node: float(values[position] - offset)
        for node, position in index.items()
    }


def _row_channel_scores(
    registry: Mapping[str, Any],
    observations: Mapping[
        tuple[str, str], Sequence[tuple[str, str, float]]],
) -> dict[tuple[str, str], dict[str, float]]:
    node_ids = [str(node["node_id"]) for node in registry["nodes"]]
    incumbents = [
        str(node["node_id"]) for node in registry["nodes"]
        if bool(node["is_incumbent"])]
    return {
        channel: solve_preference_scores(
            node_ids, observations.get(channel, ()), incumbents)
        for channel in CHANNELS
    }


def build_preference_graphs(
    registry: Mapping[tuple[str, str], Mapping[str, Any]],
    observations: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Sequence[tuple[str, str, float]]],
    ],
) -> dict[
    tuple[str, str], dict[tuple[str, str], dict[str, float]],
]:
    return {
        key: _row_channel_scores(row, observations.get(key, {}))
        for key, row in registry.items()
    }


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


def _action_component_ids(
    graph: Mapping[str, Any], action: Mapping[str, Any],
) -> tuple[str, ...] | None:
    component_by_surface = _component_map(graph)
    relation = str(graph["Relation"])
    selected = []
    for item in action["objects"]:
        key = canonical_key(str(item), relation)
        if key not in component_by_surface:
            return None
        selected.append(component_by_surface[key])
    return tuple(selected)


def graph_action_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    channel_scores: Mapping[tuple[str, str], Mapping[str, float]] | None,
) -> list[float]:
    actions = list(graph["actions"])
    keep = next(item for item in actions if item["action_type"] == "KEEP")
    selected = _action_component_ids(graph, action)
    incumbent = _action_component_ids(graph, keep)
    if channel_scores is None or selected is None or incumbent is None:
        return [0.0] * len(GRAPH_ACTION_FEATURE_NAMES)
    deltas = []
    covered = 0
    for channel in CHANNELS:
        scores = channel_scores.get(channel, {})
        covered += sum(node in scores for node in selected)
        delta = (
            sum(float(scores.get(node, 0.0)) for node in selected)
            - sum(float(scores.get(node, 0.0)) for node in incumbent))
        deltas.append(max(-24.0, min(24.0, delta)) / 24.0)
    qwen = [
        deltas[index] for index, (_, agent) in enumerate(CHANNELS)
        if agent == QWEN]
    gemma = [
        deltas[index] for index, (_, agent) in enumerate(CHANNELS)
        if agent == GEMMA]
    values = [
        *deltas,
        float(np.mean(deltas)) if deltas else 0.0,
        min(deltas) if deltas else 0.0,
        max(deltas) if deltas else 0.0,
        float(np.mean([value > 0.0 for value in deltas]))
        if deltas else 0.0,
        float(np.mean(qwen)) if qwen else 0.0,
        float(np.mean(gemma)) if gemma else 0.0,
        abs(float(np.mean(qwen)) - float(np.mean(gemma)))
        if qwen and gemma else 0.0,
        covered / max(1, len(selected) * len(CHANNELS)),
    ]
    if len(values) != len(GRAPH_ACTION_FEATURE_NAMES):
        raise AssertionError("pairwise graph feature schema drift")
    return values


def _identity_value(
    features: Sequence[float], identity_rule: str,
) -> float:
    channel_values = list(features[:len(CHANNELS)])
    if identity_rule == "qwen_submission":
        return channel_values[
            CHANNELS.index(("submission", QWEN))]
    if identity_rule == "ensemble_mean":
        return float(np.mean(channel_values))
    if identity_rule == "robust_minimum":
        # Minimum of the two model means is a conservative cross-memory rule.
        qwen = [
            channel_values[index]
            for index, (_, agent) in enumerate(CHANNELS)
            if agent == QWEN]
        gemma = [
            channel_values[index]
            for index, (_, agent) in enumerate(CHANNELS)
            if agent == GEMMA]
        return min(float(np.mean(qwen)), float(np.mean(gemma)))
    raise ContractError(f"unknown graph identity rule: {identity_rule}")


def _identity_action_indices(
    graph: Mapping[str, Any],
    channel_scores: Mapping[tuple[str, str], Mapping[str, float]] | None,
    identity_rule: str,
) -> tuple[int, ...]:
    """Choose one fixed-cardinality alternative without looking at labels."""
    actions = list(graph["actions"])
    keep = next(
        index for index, action in enumerate(actions)
        if action["action_type"] == "KEEP")
    cardinality = len(actions[keep]["objects"])
    candidates = []
    for index, action in enumerate(actions):
        if index == keep or len(action["objects"]) != cardinality:
            continue
        features = graph_action_features(graph, action, channel_scores)
        if features[-1] <= 0.0:
            continue
        value = _identity_value(features, identity_rule)
        candidates.append((value, -index, index))
    if not candidates:
        return (keep,)
    best = max(candidates)
    if best[0] <= 0.0:
        return (keep,)
    return keep, best[2]


def _gate_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    channel_scores: Mapping[tuple[str, str], Mapping[str, float]] | None,
) -> list[float]:
    values = graph_action_features(graph, action, channel_scores)
    relation = str(graph["Relation"])
    values.extend(float(relation == item) for item in RELATIONS)
    compact = compact_action_features(
        graph, action, review_evidence, include_review=True)
    values.extend(compact[-3:])
    return values


def _fit_gate(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    preference_graphs: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    identity_rule: str,
) -> UtilitySignGate:
    features, labels = [], []
    for graph in graphs:
        key = _key(graph)
        actions = list(graph["actions"])
        indices = _identity_action_indices(
            graph, preference_graphs.get(key), identity_rule)
        if len(indices) != 2:
            continue
        keep, alternative = indices
        delta = (
            _utility(graph, actions[alternative], gold_by)
            - _utility(graph, actions[keep], gold_by))
        if abs(delta) <= EPSILON:
            continue
        features.append(_gate_features(
            graph, actions[alternative], review_evidence,
            preference_graphs.get(key)))
        labels.append(float(delta > 0.0))
    return UtilitySignGate(GATE_FEATURE_NAMES, l2=FIXED_L2).fit(
        features, labels)


def _fit_relation_reliability(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    preference_graphs: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    identity_rule: str,
) -> dict[str, dict[str, Any]]:
    deltas: dict[str, list[float]] = defaultdict(list)
    for graph in graphs:
        key = _key(graph)
        actions = list(graph["actions"])
        indices = _identity_action_indices(
            graph, preference_graphs.get(key), identity_rule)
        if len(indices) != 2:
            continue
        keep, alternative = indices
        delta = (
            _utility(graph, actions[alternative], gold_by)
            - _utility(graph, actions[keep], gold_by))
        if abs(delta) > EPSILON:
            deltas[str(graph["Relation"])].append(delta)
    return {
        relation: {
            "allowed": bool(
                deltas.get(relation) and sum(deltas[relation]) > 0.0),
            "utility_changing_rows": len(deltas.get(relation, [])),
            "helped": sum(value > 0.0 for value in deltas.get(relation, [])),
            "harmed": sum(value < 0.0 for value in deltas.get(relation, [])),
            "net_utility": sum(deltas.get(relation, [])),
        }
        for relation in RELATIONS
    }


def _decode(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    review_evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    preference_graphs: Mapping[
        tuple[str, str],
        Mapping[tuple[str, str], Mapping[str, float]]],
    identity_rule: str,
    gate: UtilitySignGate | None,
    relation_reliability: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics = [], []
    for graph in graphs:
        key = _key(graph)
        actions = list(graph["actions"])
        indices = _identity_action_indices(
            graph, preference_graphs.get(key), identity_rule)
        keep = indices[0]
        alternative = indices[1] if len(indices) == 2 else keep
        probability = 1.0 if alternative != keep else 0.0
        relation_allowed = (
            True if relation_reliability is None
            else bool(relation_reliability[
                str(graph["Relation"])]["allowed"]))
        selected = alternative
        if gate is not None and alternative != keep:
            probability = gate.probability(_gate_features(
                graph, actions[alternative], review_evidence,
                preference_graphs.get(key)))
            if probability <= 0.5:
                selected = keep
        if not relation_allowed:
            selected = keep
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
            "identity_rule": identity_rule,
            "selected_action": action["id"],
            "alternative_action": actions[alternative]["id"],
            "changed": selected != keep,
            "benefit_probability": probability,
            "relation_allowed": relation_allowed,
            "utility_delta": after - before,
            "oracle_regret": best - after,
            "row_has_beneficial_action": best > before + EPSILON,
        })
    return predictions, diagnostics


def _cascade_outputs(
    old_predictions: Sequence[Mapping[str, Any]],
    old_diagnostics: Sequence[Mapping[str, Any]],
    new_predictions: Sequence[Mapping[str, Any]],
    new_diagnostics: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prefer the established precision selector, then expand its abstentions."""
    if not (
        len(old_predictions) == len(old_diagnostics)
        == len(new_predictions) == len(new_diagnostics)
    ):
        raise ContractError("cascade inputs are not aligned")
    predictions, diagnostics = [], []
    for old_row, old_detail, new_row, new_detail in zip(
        old_predictions, old_diagnostics,
        new_predictions, new_diagnostics, strict=True,
    ):
        if _key(old_row) != _key(new_row):
            raise ContractError("cascade row-order mismatch")
        use_old = bool(old_detail["changed"])
        selected_row = old_row if use_old else new_row
        selected_detail = old_detail if use_old else new_detail
        predictions.append(dict(selected_row))
        diagnostics.append({
            **dict(selected_detail),
            "cascade_stage": (
                "incumbent_edge_precision" if use_old
                else "pairwise_graph_recall"),
            "old_selector_changed": use_old,
            "new_selector_changed": bool(new_detail["changed"]),
        })
    return predictions, diagnostics


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    review_run = Path(args.review_run).resolve()
    tournament_run = Path(args.tournament_run).resolve()
    gold_path = Path(args.gold).resolve()
    _, graphs, control_rows, gold_by, folds, review_evidence = _load(
        review_run, gold_path)
    control_by = {_key(row): row for row in control_rows}
    registry, observations, _ = (
        _validated_pairwise_observations(tournament_run))
    incumbent_registry, incumbent_edges, _ = _validated_incumbent_edges(
        tournament_run)
    if set(incumbent_registry) != set(registry):
        raise ContractError("incumbent and pairwise registries disagree")
    if not set(registry).issubset({_key(row) for row in graphs}):
        raise ContractError("tournament rows not covered by action graph")
    preference_graphs = build_preference_graphs(registry, observations)

    results: dict[str, Any] = {}
    prediction_artifacts: dict[str, list[dict[str, Any]]] = {}
    diagnostic_artifacts: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        if arm == CASCADE_ARM:
            identity_rule = "qwen_submission"
            decoder = "cascade"
        else:
            identity_rule = next(
                item for item in IDENTITY_RULES if arm.startswith(item))
            decoder = arm[len(identity_rule) + 1:]
        oof: dict[tuple[str, str], dict[str, Any]] = {}
        diagnostics: list[dict[str, Any]] = []
        fold_results = []
        for outer in sorted(set(folds.values())):
            fit = [row for row in graphs if folds[_key(row)] != outer]
            hold = [row for row in graphs if folds[_key(row)] == outer]
            if decoder == "cascade":
                old_gate = _fit_incumbent_gate(
                    fit, gold_by, review_evidence,
                    incumbent_registry, incumbent_edges,
                    include_review=True)
                old_reliability = _fit_incumbent_reliability(
                    fit, gold_by, incumbent_registry, incumbent_edges)
                old_predictions, old_detail = _decode_incumbent_gate(
                    old_gate, hold, gold_by, review_evidence,
                    incumbent_registry, incumbent_edges,
                    include_review=True,
                    relation_reliability=old_reliability)
                new_reliability = _fit_relation_reliability(
                    fit, gold_by, preference_graphs, identity_rule)
                new_predictions, new_detail = _decode(
                    hold, gold_by, review_evidence, preference_graphs,
                    identity_rule, None, new_reliability)
                predictions, detail = _cascade_outputs(
                    old_predictions, old_detail,
                    new_predictions, new_detail)
            else:
                gate = (
                    None if decoder in {"raw", "hierarchical_raw"}
                    else _fit_gate(
                        fit, gold_by, review_evidence,
                        preference_graphs, identity_rule))
                reliability = (
                    _fit_relation_reliability(
                        fit, gold_by, preference_graphs, identity_rule)
                    if decoder in {
                        "hierarchical_raw", "hierarchical_sign_gate"}
                    else None)
                predictions, detail = _decode(
                    hold, gold_by, review_evidence, preference_graphs,
                    identity_rule, gate, reliability)
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
                "helped": sum(
                    item["utility_delta"] > EPSILON for item in detail),
                "harmed": sum(
                    item["utility_delta"] < -EPSILON for item in detail),
            })
        if set(oof) != {_key(row) for row in graphs}:
            raise ContractError("pairwise graph OOF coverage failure")
        ordered = [oof[_key(row)] for row in graphs]
        ordered_gold = [gold_by[_key(row)] for row in graphs]
        ordered_control = [control_by[_key(row)] for row in graphs]
        control_scores = score(ordered_control, ordered_gold)
        selected_scores = score(ordered, ordered_gold)
        changed = [item for item in diagnostics if item["changed"]]
        results[arm] = {
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
            "positive_folds": sum(
                item["delta"] > 0.0 for item in fold_results),
            "folds": fold_results,
        }
        prediction_artifacts[arm] = ordered
        diagnostic_artifacts[arm] = diagnostics

    primary = results[PRIMARY]
    passed = (
        primary["incremental_delta"] >= MIN_INCREMENTAL_DELTA
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

    primary_identity = "qwen_submission"
    final_incumbent_gate = _fit_incumbent_gate(
        graphs, gold_by, review_evidence,
        incumbent_registry, incumbent_edges,
        include_review=True)
    final_incumbent_reliability = _fit_incumbent_reliability(
        graphs, gold_by, incumbent_registry, incumbent_edges)
    final_reliability = _fit_relation_reliability(
        graphs, gold_by, preference_graphs, primary_identity)
    model_path = output / "analysis/TRAIN_FIT_MODEL.json"
    _write_json(model_path, {
        "schema": "pairwise-preference-graph-cascade-model-v1",
        "identity_rule": primary_identity,
        "row_level_gate": "incumbent_edge_precision_stage_only",
        "incumbent_edge_gate": final_incumbent_gate.to_dict(),
        "incumbent_relation_reliability": final_incumbent_reliability,
        "graph_ridge": GRAPH_RIDGE,
        "relation_reliability": final_reliability,
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "promotion_gate_passed": passed,
    })

    observation_count = sum(
        len(channel)
        for row in observations.values()
        for channel in row.values())
    result = {
        "schema": "pairwise-preference-graph-selector-result-v1",
        "development_only": True,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "primary_arm": PRIMARY,
        "rows": len(graphs),
        "actions": sum(len(row["actions"]) for row in graphs),
        "tournament_rows": len(registry),
        "pairwise_observations": observation_count,
        "graph_ridge": GRAPH_RIDGE,
        "edge_rule": "same-group log P(i)-log P(j), candidate pairs only",
        "graph_solver": "regularized weighted least-squares Laplacian",
        "arms": results,
        "artifacts": artifacts,
        "source_hashes": {
            "review_plan": sha256(review_run / "plan/PLAN.json"),
            "tournament_plan": sha256(tournament_run / "plan/PLAN.json"),
            "gold": sha256(gold_path),
        },
        "deployment_gate": {
            "passed": passed,
            "minimum_incremental_delta": MIN_INCREMENTAL_DELTA,
            "minimum_positive_folds": MIN_POSITIVE_FOLDS,
            "minimum_relation_delta": MIN_RELATION_DELTA,
        },
    }
    _write_json(output / "RESULT.json", result)
    lines = [
        "# Pairwise preference graph complete-action selector", "",
        "All scores are subject-grouped five-fold OOF against the exact "
        "registered SOTA train control.", "",
        "| arm | OOF score | delta | changed | helped | harmed | folds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in results.items():
        lines.append(
            f"| {arm} | "
            f"{item['selected_scores']['*** All Relations ***']:.6f} | "
            f"{item['incremental_delta']:+.6f} | "
            f"{item['changed_rows']} | {item['helped_rows']} | "
            f"{item['harmed_rows']} | {item['positive_folds']}/5 |")
    lines += [
        "",
        f"Predeclared promotion gate: **{passed}**",
        "",
        f"Pairwise observations consumed: **{observation_count}**.",
        "",
    ]
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
            for arm, item in results.items()
        },
        "gate_passed": passed,
        "pairwise_observations": observation_count,
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
