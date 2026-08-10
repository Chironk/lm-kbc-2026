#!/usr/bin/env python3
"""Baseline-conditioned review of complete heterogeneous-memory actions.

The candidate tournament established that contextual candidate truth scores
do not safely imply that replacing the current answer improves official F1.
This experiment asks the inference-legal question directly: for every legal
complete-output action, Qwen and Gemma compare the exact current SOTA output
against the alternative output.

Gold is absent from task construction and model inference.  During analysis,
train labels supervise the signed official row-F1 delta of each action.  The
selector is evaluated with the exact subject-grouped folds registered by the
current SOTA pipeline.  Every held-out row is decoded by a selector that did
not train or select hyperparameters on that row or subject.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    RELATION_QUESTIONS,
    balanced_choice_codebooks,
    read_jsonl,
    sha256,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.full_candidate_tournament import (
    RELATION_GUIDANCE,
    _graph_manifest,
)
from experiments.heterogeneous_agents.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    DEFAULT_TRAIN_OOF,
    validate_registered_predictions,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks
from experiments.heterogeneous_agents.unified_memory_action_graph import (
    FEATURE_NAMES,
    RELATIONS,
    _key,
    _row_f1,
    action_features,
    build_hierarchical_row,
)


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/graphs/train_graph.jsonl")
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_AGENTS = (
    ROOT / "experiments/heterogeneous_agents/agents_qwen_gemma_n1_frozen.json")
DEFAULT_OUTPUT = RUNS / "baseline_conditioned_action_review_20260727_v2"

CHOICES = ("KEEP_CURRENT", "USE_ALTERNATIVE", "UNCERTAIN")
EVIDENCE_ARMS = ("qwen", "gemma", "mean", "dual")
DECISION_MODES = ("direct_delta", "beneficial_hurdle")
L2_GRID = (0.1, 1.0, 10.0, 100.0)
EPSILON = 1e-8
MIN_INCREMENTAL_DELTA = 0.005
MIN_RELATION_DELTA = -0.02

REVIEW_FEATURE_NAMES = (
    "relation_award", "relation_company", "relation_borders",
    "relation_area", "relation_capacity", "relation_city",
    "qwen_use", "qwen_keep", "qwen_uncertain", "qwen_log_odds",
    "gemma_use", "gemma_keep", "gemma_uncertain", "gemma_log_odds",
    "review_min_use", "review_mean_log_odds", "review_agreement",
)
ALL_FEATURE_NAMES = tuple(FEATURE_NAMES) + REVIEW_FEATURE_NAMES
_UTILITY_CACHE: dict[tuple[tuple[str, str], tuple[str, ...]], float] = {}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        dict(value), indent=2, sort_keys=True) + "\n")


def _format_objects(objects: Sequence[str]) -> str:
    if not objects:
        return "None"
    return "[" + "; ".join(
        json.dumps(str(item), ensure_ascii=False) for item in objects) + "]"


def comparison_prompt(
    subject: str, relation: str, current: Sequence[str],
    alternative: Sequence[str], codebook: Mapping[str, str], *,
    alternative_first: bool,
) -> str:
    """Build a source-blind, order-balanced complete-output comparison."""
    first_name, first_objects = (
        ("ALTERNATIVE", alternative) if alternative_first
        else ("CURRENT", current))
    second_name, second_objects = (
        ("CURRENT", current) if alternative_first
        else ("ALTERNATIVE", alternative))
    codes = "; ".join(
        f"{codebook[choice]} = {choice}" for choice in CHOICES)
    return (
        "Act as the final editor of a closed-book knowledge-base answer. "
        "Compare the two COMPLETE outputs using only your own factual memory. "
        "Choose USE_ALTERNATIVE only when the alternative is more accurate "
        "under the exact relation requirement. Missing a true item and adding "
        "a false item can both make a list output worse. Do not prefer an "
        "answer merely because it is new, more detailed, or appears second. "
        "Choose UNCERTAIN when you cannot reliably tell which complete output "
        "is better.\n"
        f"SUBJECT: {subject}\n"
        f"RELATION: {relation}\n"
        f"QUESTION: {RELATION_QUESTIONS[relation].format(subject=subject)}\n"
        f"RELATION REQUIREMENT: {RELATION_GUIDANCE[relation]}\n"
        f"{first_name} OUTPUT: {_format_objects(first_objects)}\n"
        f"{second_name} OUTPUT: {_format_objects(second_objects)}\n"
        f"Choose exactly one code: {codes}.\n"
        "Return only the code.\nCODE:"
    )


def _task(
    graph: Mapping[str, Any], action: Mapping[str, Any], agent: str,
) -> dict[str, Any]:
    key = (
        f"{graph['SubjectEntity']}\x1f{graph['Relation']}\x1f"
        f"{action['id']}")
    codebooks = balanced_choice_codebooks(
        CHOICES, "baseline-conditioned-action-review-v1", agent, key)
    variants = []
    # One order-balanced variant per codebook keeps choice-token balance while
    # avoiding a six-sequence activation batch on 11 GiB Gemma workers.
    for index, codebook in enumerate(codebooks):
        variants.append({
            "choice_codes": dict(codebook),
            "prompt": comparison_prompt(
                str(graph["SubjectEntity"]), str(graph["Relation"]),
                graph["incumbent_objects"], action["objects"], codebook,
                alternative_first=bool(index % 2)),
        })
    task_id = (
        f"{agent}::baseline_action::{graph['_source']['input_index']}::"
        f"{action['id']}")
    return {
        "task_id": task_id,
        "agent_id": agent,
        "subject": graph["SubjectEntity"],
        "relation": graph["Relation"],
        "phase": "baseline_conditioned_action_review",
        "mode": "choice",
        "prompt": variants[0]["prompt"],
        "choices": list(CHOICES),
        "choice_codes": variants[0]["choice_codes"],
        "choice_variants": variants,
        "candidate_key": action["id"],
        "candidate_item": _format_objects(action["objects"]),
        "excluded_proposer_agents": [],
        "action_id": action["id"],
        "action_type": action["action_type"],
        "current_objects": list(graph["incumbent_objects"]),
        "alternative_objects": list(action["objects"]),
        "contains_labels": False,
        "gold_aware": False,
        "prompt_exposes_incumbent": True,
        "prompt_masks_provenance": True,
    }


def _build_graphs(
    graph_path: Path, control_path: Path,
) -> list[dict[str, Any]]:
    source = read_jsonl(graph_path)
    control = {_key(row): row for row in read_jsonl(control_path)}
    if {_key(row) for row in source} != set(control):
        raise ContractError("source graph and exact SOTA control mismatch")
    graphs = []
    for row in source:
        copied = dict(row)
        copied["baseline_objects"] = [
            str(item) for item in control[_key(row)]["ObjectEntities"]]
        graph = build_hierarchical_row(copied)
        for action in graph["actions"]:
            action_features(graph, action)
        graphs.append(graph)
    return graphs


def _serializable_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in graph.items()
        if key != "_source"
    }


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    graph_path = Path(args.graph).resolve()
    _graph_manifest(graph_path)
    control_path = Path(args.control).resolve()
    control_manifest = validate_registered_predictions(
        control_path, pipeline_id=COMPETITION_PIPELINE_ID, split="train")
    graphs = _build_graphs(graph_path, control_path)
    registry_path = output / "plan/ACTIONS.jsonl"
    write_jsonl_atomic(
        registry_path, [_serializable_graph(row) for row in graphs])

    jobs = {}
    for agent in (QWEN, GEMMA):
        tasks = [
            _task(graph, action, agent)
            for graph in graphs
            for action in graph["actions"]
            if action["action_type"] != "KEEP"
        ]
        task_path = output / f"plan/tasks/{agent}.jsonl"
        smoke_path = output / f"plan/smoke/{agent}.jsonl"
        write_jsonl_atomic(task_path, tasks)
        smoke = []
        seen = set()
        for task in tasks:
            relation = str(task["relation"])
            if relation not in seen:
                smoke.append(task)
                seen.add(relation)
        write_jsonl_atomic(smoke_path, smoke[:6])
        jobs[agent] = {
            "tasks": len(tasks),
            "task_path": str(task_path),
            "task_sha256": sha256(task_path),
            "smoke_path": str(smoke_path),
            "smoke_sha256": sha256(smoke_path),
            "response_path": str(output / f"responses/{agent}.jsonl"),
        }

    folds_path = Path(control_manifest["folds"]).resolve()
    plan = {
        "schema": "baseline-conditioned-action-review-plan-v1",
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "starting_predictions": str(control_path),
        "starting_predictions_sha256": sha256(control_path),
        "source_graph": str(graph_path),
        "source_graph_sha256": sha256(graph_path),
        "folds": str(folds_path),
        "folds_sha256": sha256(folds_path),
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "rows": len(graphs),
        "actions": sum(len(row["actions"]) for row in graphs),
        "review_actions": sum(len(row["actions"]) - 1 for row in graphs),
        "action_rule": (
            "compare each complete legal graph action against the exact "
            "registered SOTA incumbent"),
        "prompt_exposes_incumbent": True,
        "prompt_masks_provenance": True,
        "balanced_choice_codes": True,
        "alternating_output_order": True,
        "jobs": jobs,
        "agents": str(Path(args.agents).resolve()),
        "evidence_arms": list(EVIDENCE_ARMS),
        "decision_modes": list(DECISION_MODES),
        "l2_grid": list(L2_GRID),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(output / "plan/PLAN.json", plan)
    print(json.dumps({
        "rows": plan["rows"],
        "actions": plan["actions"],
        "review_actions": plan["review_actions"],
        "jobs": {agent: job["tasks"] for agent, job in jobs.items()},
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


def _validated_responses(
    plan: Mapping[str, Any],
) -> dict[tuple[tuple[str, str], str, str], Mapping[str, float]]:
    result = {}
    for agent, job in plan["jobs"].items():
        task_path = Path(job["task_path"])
        response_path = Path(job["response_path"])
        manifest_path = response_path.with_suffix(
            response_path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise ContractError(f"missing response manifest: {agent}")
        manifest = _json(manifest_path)
        if (
            sha256(task_path) != job["task_sha256"]
            or manifest.get("task_sha256") != job["task_sha256"]
            or manifest.get("output_sha256") != sha256(response_path)
            or manifest.get("agent_id") != agent
            or int(manifest.get("tasks", -1)) != int(job["tasks"])
        ):
            raise ContractError(f"stale action-review responses: {agent}")
        tasks = read_jsonl(task_path)
        by_id = validate_tasks(tasks, agent)
        responses = read_jsonl(response_path)
        if len(responses) != len(tasks):
            raise ContractError(f"incomplete action-review responses: {agent}")
        for response in responses:
            task_id = str(response["task_id"])
            if task_id not in by_id:
                raise ContractError("unknown action-review response")
            task = by_id[task_id]
            validate_task_response(task, response)
            key = (
                (str(task["subject"]), str(task["relation"])),
                str(task["action_id"]), agent)
            if key in result:
                raise ContractError("duplicate action-review evidence")
            result[key] = {
                choice: float(response["choice_probabilities"][choice])
                for choice in CHOICES}
    return result


def _review_features(
    relation: str, q: Mapping[str, float], g: Mapping[str, float],
    arm: str,
) -> list[float]:
    zero = {choice: 0.0 for choice in CHOICES}
    if arm == "qwen":
        g = zero
    elif arm == "gemma":
        q = zero
    elif arm == "mean":
        averaged = {
            choice: (float(q[choice]) + float(g[choice])) / 2.0
            for choice in CHOICES}
        q, g = averaged, zero
    elif arm != "dual":
        raise ContractError(f"unknown review arm: {arm}")

    def values(item: Mapping[str, float]) -> tuple[float, ...]:
        use = float(item["USE_ALTERNATIVE"])
        keep = float(item["KEEP_CURRENT"])
        uncertain = float(item["UNCERTAIN"])
        odds = math.log(max(use, EPSILON)) - math.log(max(keep, EPSILON))
        return use, keep, uncertain, max(-12.0, min(12.0, odds)) / 12.0

    qv, gv = values(q), values(g)
    q_use, g_use = qv[0], gv[0]
    active_use = [q_use] if arm in {"qwen", "mean"} else (
        [g_use] if arm == "gemma" else [q_use, g_use])
    active_odds = [qv[3]] if arm in {"qwen", "mean"} else (
        [gv[3]] if arm == "gemma" else [qv[3], gv[3]])
    relation_values = [float(relation == item) for item in RELATIONS]
    return [
        *relation_values, *qv, *gv,
        min(active_use),
        statistics.mean(active_odds),
        float(
            (q_use > float(q["KEEP_CURRENT"]))
            == (g_use > float(g["KEEP_CURRENT"])))
        if arm == "dual" else 0.0,
    ]


def _feature(
    graph: Mapping[str, Any], action: Mapping[str, Any],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    arm: str,
) -> list[float]:
    key = _key(graph), str(action["id"])
    q = evidence[key + (QWEN,)]
    g = evidence[key + (GEMMA,)]
    values = [
        *action["_inference_features"],
        *_review_features(str(graph["Relation"]), q, g, arm),
    ]
    if len(values) != len(ALL_FEATURE_NAMES):
        raise AssertionError("baseline action-review feature schema drift")
    if not all(math.isfinite(value) for value in values):
        raise ContractError("non-finite baseline action-review feature")
    return values


def _utility(
    graph: Mapping[str, Any], objects: Sequence[str],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    """Memoize the comparatively expensive official row utility."""
    key = _key(graph)
    cache_key = key, tuple(map(str, objects))
    if cache_key not in _UTILITY_CACHE:
        _UTILITY_CACHE[cache_key] = _row_f1(
            cache_key[1], gold_by[key], key[1])
    return _UTILITY_CACHE[cache_key]


class Ridge:
    """Small deterministic weighted ridge over signed action utility."""

    def __init__(self, l2: float):
        self.l2 = float(l2)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    def fit(
        self, x: Sequence[Sequence[float]], y: Sequence[float],
        weight: Sequence[float],
    ) -> "Ridge":
        matrix = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        weights = np.asarray(weight, dtype=np.float64)
        if (
            matrix.shape != (len(target), len(ALL_FEATURE_NAMES))
            or weights.shape != target.shape or len(target) < 2
            or np.any(weights <= 0)
        ):
            raise ValueError("invalid baseline action ridge arrays")
        weights *= len(weights) / weights.sum()
        self.mean = np.average(matrix, axis=0, weights=weights)
        variance = np.average(
            (matrix - self.mean) ** 2, axis=0, weights=weights)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = np.column_stack(
            [np.ones(len(target)), (matrix - self.mean) / self.scale])
        root = np.sqrt(weights)[:, None]
        penalty = np.eye(design.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(
            (design * root).T @ (design * root) + penalty,
            (design * root).T @ (target * root[:, 0]))
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("ridge is not fitted")
        matrix = np.asarray(x, dtype=np.float64)
        design = np.column_stack(
            [np.ones(len(matrix)), (matrix - self.mean) / self.scale])
        return np.clip(design @ self.coef, -1.0, 1.0)


class Logistic:
    """Weighted beneficial-action classifier with a fixed 0.5 boundary."""

    def __init__(self, l2: float):
        self.l2 = float(l2)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    def fit(
        self, x: Sequence[Sequence[float]], y: Sequence[float],
        weight: Sequence[float],
    ) -> "Logistic":
        matrix = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        weights = np.asarray(weight, dtype=np.float64)
        if (
            matrix.shape != (len(target), len(ALL_FEATURE_NAMES))
            or weights.shape != target.shape
            or set(np.unique(target)) != {0.0, 1.0}
        ):
            raise ValueError("invalid baseline action logistic arrays")
        # Equal total weight for beneficial and non-beneficial actions, while
        # retaining row balance inside each class.
        for label in (0.0, 1.0):
            mask = target == label
            weights[mask] *= 0.5 / weights[mask].sum()
        weights *= len(weights) / weights.sum()
        self.mean = np.average(matrix, axis=0, weights=weights)
        variance = np.average(
            (matrix - self.mean) ** 2, axis=0, weights=weights)
        self.scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
        design = np.column_stack(
            [np.ones(len(target)), (matrix - self.mean) / self.scale])
        beta = np.zeros(design.shape[1], dtype=np.float64)
        penalty = np.eye(design.shape[1]) * self.l2
        penalty[0, 0] = 0.0
        for _ in range(100):
            logits = np.clip(design @ beta, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            curvature = np.maximum(
                probability * (1.0 - probability), 1e-8)
            gradient = design.T @ (
                weights * (probability - target)) + penalty @ beta
            hessian = (
                design.T @ (
                    design * (weights * curvature)[:, None]) + penalty)
            step = np.linalg.solve(hessian, gradient)
            beta -= step
            if float(np.max(np.abs(step))) < 1e-9:
                break
        self.coef = beta
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("logistic is not fitted")
        matrix = np.asarray(x, dtype=np.float64)
        design = np.column_stack(
            [np.ones(len(matrix)), (matrix - self.mean) / self.scale])
        logits = np.clip(design @ self.coef, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


def _training_arrays(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    arm: str,
) -> tuple[list[list[float]], list[float], list[float]]:
    x, y, weights = [], [], []
    for graph in graphs:
        baseline = _utility(
            graph, graph["incumbent_objects"], gold_by)
        alternatives = [
            action for action in graph["actions"]
            if action["action_type"] != "KEEP"]
        row_weight = 1.0 / max(len(alternatives), 1)
        for action in alternatives:
            x.append(_feature(graph, action, evidence, arm))
            y.append(_utility(graph, action["objects"], gold_by) - baseline)
            weights.append(row_weight)
    return x, y, weights


def _fit(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    arm: str, l2: float,
) -> tuple[Ridge, Logistic]:
    x, y, weights = _training_arrays(graphs, gold_by, evidence, arm)
    ridge = Ridge(l2).fit(x, y, weights)
    beneficial = [float(value > 1e-12) for value in y]
    logistic = Logistic(l2).fit(x, beneficial, weights)
    return ridge, logistic


def _decode_one(
    models: tuple[Ridge, Logistic], graph: Mapping[str, Any],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    arm: str, mode: str,
) -> tuple[list[str], dict[str, Any]]:
    alternatives = [
        action for action in graph["actions"]
        if action["action_type"] != "KEEP"]
    if not alternatives:
        return list(graph["incumbent_objects"]), {
            "selected_action": "KEEP", "changed": False,
            "predicted_delta": 0.0, "beneficial_probability": 0.0}
    features = [
        _feature(graph, action, evidence, arm) for action in alternatives]
    deltas = models[0].predict(features)
    beneficial = models[1].predict(features)
    allowed = [
        (
            float(deltas[index]) > 0.0
            and (
                mode == "direct_delta"
                or float(beneficial[index]) > 0.5
            )
        )
        for index in range(len(alternatives))
    ]
    eligible = [index for index, value in enumerate(allowed) if value]
    if not eligible:
        return list(graph["incumbent_objects"]), {
            "selected_action": "KEEP", "changed": False,
            "predicted_delta": 0.0,
            "beneficial_probability": max(map(float, beneficial)),
        }
    best = max(
        eligible,
        key=lambda index: (
            float(deltas[index]), float(beneficial[index]),
            -len(alternatives[index]["objects"]), -index))
    action = alternatives[best]
    return list(action["objects"]), {
        "selected_action": action["action_type"],
        "action_id": action["id"],
        "changed": True,
        "predicted_delta": float(deltas[best]),
        "beneficial_probability": float(beneficial[best]),
    }


def _predictions(
    models: tuple[Ridge, Logistic],
    graphs: Sequence[Mapping[str, Any]],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
    arm: str, mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, diagnostics = [], []
    for graph in graphs:
        objects, detail = _decode_one(
            models, graph, evidence, arm, mode)
        rows.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": objects,
        })
        diagnostics.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "arm": arm, "mode": mode, **detail,
        })
    return rows, diagnostics


def _macro(
    predictions: Sequence[Mapping[str, Any]],
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    return score(
        list(predictions), [gold_by[_key(row)] for row in graphs],
    )["*** All Relations ***"]


def _choose_inner(
    fit_graphs: Sequence[Mapping[str, Any]],
    folds: Mapping[tuple[str, str], int],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    evidence: Mapping[
        tuple[tuple[str, str], str, str], Mapping[str, float]],
) -> tuple[tuple[str, str, float], list[dict[str, Any]]]:
    present = sorted({folds[_key(row)] for row in fit_graphs})
    records = []
    for arm in EVIDENCE_ARMS:
        for l2 in L2_GRID:
            fold_cache = {}
            for inner in present:
                train = [
                    row for row in fit_graphs if folds[_key(row)] != inner]
                hold = [
                    row for row in fit_graphs if folds[_key(row)] == inner]
                models = _fit(train, gold_by, evidence, arm, l2)
                fold_cache[inner] = models, hold
            for mode in DECISION_MODES:
                deltas = []
                changes = 0
                for inner in present:
                    models, hold = fold_cache[inner]
                    predictions, diagnostics = _predictions(
                        models, hold, evidence, arm, mode)
                    control = [{
                        "SubjectEntity": row["SubjectEntity"],
                        "Relation": row["Relation"],
                        "ObjectEntities": row["incumbent_objects"],
                    } for row in hold]
                    deltas.append(
                        _macro(predictions, hold, gold_by)
                        - _macro(control, hold, gold_by))
                    changes += sum(row["changed"] for row in diagnostics)
                records.append({
                    "config": [arm, mode, l2],
                    "fold_deltas": deltas,
                    "mean_delta": statistics.mean(deltas),
                    "changed": changes,
                })
    selected = max(
        records,
        key=lambda row: (
            row["mean_delta"], -row["changed"], row["config"]))
    return (
        str(selected["config"][0]),
        str(selected["config"][1]),
        float(selected["config"][2]),
    ), records


def _relation_deltas(
    selected: Mapping[str, float], control: Mapping[str, float],
) -> dict[str, float]:
    return {
        relation: selected[relation] - control[relation]
        for relation in RELATIONS}


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    graph_path = Path(plan["source_graph"])
    control_path = Path(plan["starting_predictions"])
    registry_path = Path(plan["registry"])
    folds_path = Path(plan["folds"])
    if (
        plan.get("schema")
        != "baseline-conditioned-action-review-plan-v1"
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("starting_pipeline_id") != COMPETITION_PIPELINE_ID
        or plan.get("implementation_sha256")
        != sha256(Path(__file__).resolve())
        or sha256(graph_path) != plan["source_graph_sha256"]
        or sha256(control_path) != plan["starting_predictions_sha256"]
        or sha256(registry_path) != plan["registry_sha256"]
        or sha256(folds_path) != plan["folds_sha256"]
    ):
        raise ContractError("invalid baseline-conditioned review plan")
    validate_registered_predictions(
        control_path, pipeline_id=COMPETITION_PIPELINE_ID, split="train")
    evidence = _validated_responses(plan)
    raw_graphs = read_jsonl(graph_path)
    control = {_key(row): row for row in read_jsonl(control_path)}
    graphs = []
    for row in raw_graphs:
        copied = dict(row)
        copied["baseline_objects"] = control[_key(row)]["ObjectEntities"]
        graph = build_hierarchical_row(copied)
        for action in graph["actions"]:
            action_features(graph, action)
        graphs.append(graph)
    gold_rows = read_jsonl(Path(args.gold).resolve())
    gold_by = {_key(row): row for row in gold_rows}
    if {_key(row) for row in graphs} != set(gold_by):
        raise ContractError("baseline review graph/gold mismatch")
    folds = {
        _key(row): int(row["fold"]) for row in read_jsonl(folds_path)}
    if set(folds) != set(gold_by):
        raise ContractError("baseline review fold coverage mismatch")

    oof, diagnostics, selections = {}, [], []
    for outer in sorted(set(folds.values())):
        fit = [row for row in graphs if folds[_key(row)] != outer]
        hold = [row for row in graphs if folds[_key(row)] == outer]
        config, inner = _choose_inner(
            fit, folds, gold_by, evidence)
        arm, mode, l2 = config
        models = _fit(fit, gold_by, evidence, arm, l2)
        predictions, detail = _predictions(
            models, hold, evidence, arm, mode)
        for row in predictions:
            oof[_key(row)] = row
        diagnostics.extend({
            **row, "outer_fold": outer, "l2": l2
        } for row in detail)
        selections.append({
            "fold": outer,
            "fit_rows": len(fit),
            "hold_rows": len(hold),
            "selected_config": [arm, mode, l2],
            "inner_results": inner,
        })
    if set(oof) != set(gold_by):
        raise ContractError("baseline action nested OOF coverage failure")
    ordered = [oof[_key(row)] for row in gold_rows]
    control_rows = [control[_key(row)] for row in gold_rows]
    selected_scores = score(ordered, gold_rows)
    control_scores = score(control_rows, gold_rows)
    delta = (
        selected_scores["*** All Relations ***"]
        - control_scores["*** All Relations ***"])
    relation_deltas = _relation_deltas(selected_scores, control_scores)
    passed = (
        delta >= MIN_INCREMENTAL_DELTA
        and min(relation_deltas.values()) >= MIN_RELATION_DELTA)

    oracle = []
    for graph in graphs:
        relation = str(graph["Relation"])
        best = max(
            graph["actions"],
            key=lambda action: (
                _utility(graph, action["objects"], gold_by),
                action["action_type"] == "KEEP",
                -len(action["objects"])))
        oracle.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": relation,
            "ObjectEntities": best["objects"],
        })
    oracle_scores = score(oracle, gold_rows)

    prediction_path = output / "analysis/TRAIN_OOF_PREDICTIONS.jsonl"
    diagnostic_path = output / "analysis/TRAIN_OOF_DIAGNOSTICS.jsonl"
    write_jsonl_atomic(prediction_path, ordered)
    write_jsonl_atomic(diagnostic_path, diagnostics)
    helped = harmed = 0
    for graph in graphs:
        before = _utility(graph, graph["incumbent_objects"], gold_by)
        after = _utility(
            graph, oof[_key(graph)]["ObjectEntities"], gold_by)
        helped += after > before + 1e-12
        harmed += after + 1e-12 < before
    result = {
        "schema": "baseline-conditioned-action-review-result-v1",
        "development_only": True,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "control_scores": control_scores,
        "selected_scores": selected_scores,
        "oracle_action_scores": oracle_scores,
        "incremental_delta": delta,
        "relation_deltas": relation_deltas,
        "changed_rows": sum(row["changed"] for row in diagnostics),
        "helped_rows": helped,
        "harmed_rows": harmed,
        "fold_selections": selections,
        "selector_parameter_count_per_fold": (
            2 * (len(ALL_FEATURE_NAMES) + 1)),
        "deployment_gate": {
            "passed": passed,
            "minimum_incremental_delta": MIN_INCREMENTAL_DELTA,
            "minimum_relation_delta": MIN_RELATION_DELTA,
        },
        "artifacts": {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
        },
    }
    _write_json(output / "analysis/RESULT.json", result)
    lines = [
        "# Baseline-conditioned complete-action review", "",
        "Train-only, exact-SOTA, nested subject-grouped OOF audit. "
        "Validation was not opened.", "",
        f"- Exact SOTA control: "
        f"**{control_scores['*** All Relations ***']:.9f}**",
        f"- Reviewed action selector: "
        f"**{selected_scores['*** All Relations ***']:.9f}**",
        f"- Incremental delta: **{delta:+.9f}**",
        f"- Legal-action oracle: "
        f"**{oracle_scores['*** All Relations ***']:.9f}**",
        f"- Changed/helped/harmed rows: "
        f"**{result['changed_rows']} / {helped} / {harmed}**",
        f"- Promotion gate: **{passed}**", "",
        "## Relation deltas", "",
        "| relation | delta |", "|---|---:|",
    ]
    lines.extend(
        f"| {relation} | {value:+.6f} |"
        for relation, value in relation_deltas.items())
    lines.extend(["", "## Fold selections", ""])
    lines.extend(
        f"- fold {item['fold']}: `"
        f"{' / '.join(map(str, item['selected_config']))}`"
        for item in selections)
    (output / "analysis/RESULT.md").write_text(
        "\n".join(lines) + "\n")
    print(json.dumps({
        "control": control_scores["*** All Relations ***"],
        "selected": selected_scores["*** All Relations ***"],
        "incremental_delta": delta,
        "oracle": oracle_scores["*** All Relations ***"],
        "changed": result["changed_rows"],
        "helped": helped,
        "harmed": harmed,
        "gate_passed": passed,
        "output": str(output / "analysis"),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("prepare", "analyze"))
    value.add_argument("--graph", default=str(DEFAULT_GRAPH))
    value.add_argument("--control", default=str(DEFAULT_TRAIN_OOF))
    value.add_argument("--gold", default=str(DEFAULT_GOLD))
    value.add_argument("--agents", default=str(DEFAULT_AGENTS))
    value.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return value


def main() -> int:
    args = parser().parse_args()
    return prepare(args) if args.command == "prepare" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
