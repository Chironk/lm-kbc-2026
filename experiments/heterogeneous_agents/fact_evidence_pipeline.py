#!/usr/bin/env python3
"""Relation-specific candidate supply and independent verification pipeline.

This module targets the measured bottleneck in the heterogeneous-memory
system: fact-level evidence.  It deliberately separates two questions:

1. SUPPLY: did a relation-specific route place a gold-compatible fact in the
   graph?
2. SELECTION: when a useful alternative is present, can inference-legal
   evidence distinguish a helpful switch from a harmful one?

Capacity is the first adapter because it has the largest measured reachable
error block.  The contracts are intentionally reusable: proposal routes,
action inventory, review axes, fold gates, and manifests are explicit.

Train workflow:

    prepare-proposals -> GPU responses -> build-graph -> audit-supply
      -> prepare-verification -> GPU responses -> ledger -> fit

Validation preparation is absent by design until ``fit/GATE.json`` passes.
No command in this version accepts validation labels or validation graphs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate import try_parse_number
from experiments.heterogeneous_agents.capacity_baseline_aware_selector import (
    RidgeActionModel,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    balanced_choice_codebooks,
    canonical_key,
    proposal_parse_status,
    read_jsonl,
    sha256,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.heterogeneous_memory_selector import (
    _key,
    _load_graph,
)
from experiments.heterogeneous_agents.relational_candidate_graph import (
    _component_for_prediction,
    augment_relational_graph,
    component_actions,
)
from experiments.heterogeneous_agents.relation_specific_structured_decoder import (
    _row_f1,
)
from experiments.heterogeneous_agents.route_aware_candidate_graph import (
    _summarize_routes,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
RELATION = "hasCapacity"
AGENTS = (QWEN, GEMMA)
OTHER_AGENT = {QWEN: GEMMA, GEMMA: QWEN}

DEFAULT_GRAPH = (
    RUNS / "capacity_multiview_graph_20260725_v1/graphs/train_graph.jsonl")
DEFAULT_INCUMBENTS = (
    RUNS / "july_component_oof_parity_20260726_v1/"
    "train_oof_selected.jsonl")
DEFAULT_FOLDS = (
    RUNS / "expanded_calibration_n1_20260723_v1/plan/FOLDS.jsonl")
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_AGENTS = (
    ROOT / "experiments/heterogeneous_agents/"
    "agents_qwen_gemma_n1_frozen.json")

PROPOSAL_ARMS = (
    "qwen_identity",
    "qwen_configuration",
    "qwen_exact_memory",
    "gemma_identity",
    "gemma_configuration",
    "gemma_exact_memory",
)
ARM_AGENT = {
    arm: QWEN if arm.startswith("qwen_") else GEMMA
    for arm in PROPOSAL_ARMS
}
ARM_ROUTE = {
    arm: f"{ARM_AGENT[arm].split('_')[0]}:capacity_{arm.split('_', 1)[1]}"
    for arm in PROPOSAL_ARMS
}
PROPOSAL_TEMPERATURE = 0.2
PROPOSAL_MAX_NEW_TOKENS = 40

REVIEW_AXES = ("exact_memory", "identity", "configuration")
REVIEW_CHOICES = ("KEEP_BASELINE", "USE_ALTERNATIVE", "UNKNOWN")
# Ten component hypotheses preserve all 38 currently reachable capacity
# failures on the frozen training graph.  The old flat four-action cap
# preserved only 25.  This is a label-free runtime constant; the preservation
# audit is reported separately and remains gold-aware/nondeployable.
MAX_COMPONENTS_PER_ROW = 10

ALPHAS = (0.25, 1.0, 4.0, 16.0, 64.0)
MARGINS = (0.0, 0.02, 0.05, 0.10, 0.20)
MIN_SUPPLY_ROWS = 8
MIN_REVIEW_AUROC = 0.65
MIN_OOF_DELTA = 0.01
MIN_INCREMENTAL_DELTA = 0.005
MIN_WINS = 3
FOLD_FLOOR = -0.01
MIN_CHANGED_ACTIONS = 10

GRAPH_FEATURE_NAMES = (
    "bias",
    "log_incumbent",
    "log_challenger",
    "signed_log_ratio",
    "absolute_log_ratio",
    "challenger_support",
    "incumbent_support",
    "support_advantage",
    "challenger_route_count",
    "incumbent_route_count",
    "challenger_model_count",
    "incumbent_model_count",
    "challenger_new_route",
    "incumbent_new_route",
    "challenger_surface_support",
    "incumbent_surface_support",
    "surface_support_advantage",
    "challenger_surface_route_count",
    "incumbent_surface_route_count",
    "challenger_surface_model_count",
    "incumbent_surface_model_count",
    "surface_is_component_representative",
    "component_unique_surface_count",
    "surface_distance_from_component_median",
    "component_relative_span",
)
REVIEW_FEATURE_NAMES = tuple(
    f"{agent}:{axis}:{name}"
    for agent in AGENTS
    for axis in REVIEW_AXES
    for name in ("available", "keep", "use", "unknown", "signed", "independent")
) + (
    "independent_signed_mean",
    "independent_signed_min",
    "independent_signed_max",
    "independent_positive_count",
    "all_signed_mean",
    "agent_disagreement",
)
FULL_FEATURE_NAMES = GRAPH_FEATURE_NAMES + REVIEW_FEATURE_NAMES


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _number(value: Any) -> float | None:
    parsed = try_parse_number(str(value))
    if parsed is None or not math.isfinite(parsed) or parsed <= 0:
        return None
    return float(parsed)


def _within(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) / scale <= 0.05 + 1e-12


def _gold_number(row: Mapping[str, Any]) -> float | None:
    for group in row.get("ObjectEntities", []):
        for alias in group if isinstance(group, list) else [group]:
            value = _number(alias)
            if value is not None:
                return value
    return None


def proposal_prompt(arm: str, subject: str) -> str:
    if arm not in PROPOSAL_ARMS:
        raise ContractError(f"unknown proposal arm: {arm}")
    view = arm.split("_", 1)[1]
    if view == "identity":
        cue = (
            "Resolve the exact named venue, including renamings or similarly "
            "named venues, before recalling its capacity.")
    elif view == "configuration":
        cue = (
            "Use the venue's normal current spectator configuration. Exclude "
            "record attendance, temporary event layouts, area, and year.")
    else:
        cue = (
            "Retrieve the exact published capacity from factual memory; do "
            "not estimate it from the venue type or generic size.")
    return (
        f"What is the maximum spectator capacity of {subject}?\n"
        f"MEMORY CUE: {cue}\n"
        "Return exactly one line and no explanation:\n"
        "ANSWER: <single number>"
    )


def build_proposal_tasks(
    graphs: Sequence[Mapping[str, Any]], seed: int,
) -> dict[str, list[dict[str, Any]]]:
    tasks = {agent: [] for agent in AGENTS}
    capacity = [row for row in graphs if row["Relation"] == RELATION]
    for arm in PROPOSAL_ARMS:
        agent = ARM_AGENT[arm]
        for index, graph in enumerate(capacity):
            subject = str(graph["SubjectEntity"])
            tasks[agent].append({
                "task_id": f"{agent}::fact_supply::{arm}::{index}",
                "agent_id": agent,
                "subject": subject,
                "relation": RELATION,
                "phase": "fact_supply",
                "mode": "generate",
                "arm": arm,
                "route": ARM_ROUTE[arm],
                "input_index": index,
                "prompt": proposal_prompt(arm, subject),
                "n_samples": 1,
                "temperature": PROPOSAL_TEMPERATURE,
                "max_new_tokens": PROPOSAL_MAX_NEW_TOKENS,
                "seed": seed,
                "contains_labels": False,
                "gold_aware": False,
            })
    return tasks


def _manifest_for_graph(path: Path, split: str = "train") -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = _json(manifest_path)
    if (manifest.get("split") != split
            or manifest.get("contains_labels")
            or manifest.get("gold_aware")
            or manifest.get("output_sha256") != sha256(path)):
        raise ContractError(f"{path}: graph is not a clean frozen {split} graph")
    return manifest


def prepare_proposals(args: argparse.Namespace) -> int:
    graph_path = Path(args.train_graph).resolve()
    output = Path(args.output_dir).resolve()
    _manifest_for_graph(graph_path)
    graphs = _load_graph(graph_path, expected_split="train")
    capacity = [row for row in graphs if row["Relation"] == RELATION]
    tasks = build_proposal_tasks(graphs, args.seed)
    jobs = {}
    for agent, rows in tasks.items():
        task_path = output / f"proposal/plan/tasks/{agent}.jsonl"
        smoke_path = output / f"proposal/plan/smoke/{agent}.jsonl"
        write_jsonl_atomic(task_path, rows)
        write_jsonl_atomic(smoke_path, rows[:3])
        validate_tasks(rows, agent)
        jobs[agent] = {
            "tasks": len(rows),
            "task_path": str(task_path),
            "task_sha256": sha256(task_path),
            "smoke_path": str(smoke_path),
            "smoke_sha256": sha256(smoke_path),
            "response_path": str(
                output / f"proposal/responses/{agent}.jsonl"),
        }
    plan = {
        "schema": "fact-evidence-proposal-plan-v1",
        "split": "train",
        "relation": RELATION,
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "rows": len(capacity),
        "arms": list(PROPOSAL_ARMS),
        "arm_agent": ARM_AGENT,
        "arm_route": ARM_ROUTE,
        "n_samples_per_arm": 1,
        "purpose": "increase distinct fact supply, not self-consistency",
        "train_graph": str(graph_path),
        "train_graph_sha256": sha256(graph_path),
        "agents": str(Path(args.agents).resolve()),
        "seed": args.seed,
        "jobs": jobs,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(output / "proposal/plan/PLAN.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def _validated_responses(
    plan: Mapping[str, Any], phase: str,
) -> tuple[dict[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    responses: dict[str, Mapping[str, Any]] = {}
    tasks: list[Mapping[str, Any]] = []
    for agent, job in plan["jobs"].items():
        task_path = Path(job["task_path"])
        response_path = Path(job["response_path"])
        if sha256(task_path) != job["task_sha256"]:
            raise ContractError(f"{agent}: {phase} task hash mismatch")
        manifest = _json(
            response_path.with_suffix(response_path.suffix + ".manifest.json"))
        if manifest.get("contains_labels") or manifest.get("gold_aware"):
            raise ContractError(f"{agent}: gold-aware {phase} response rejected")
        if (manifest.get("task_sha256") != job["task_sha256"]
                or manifest.get("output_sha256") != sha256(response_path)):
            raise ContractError(f"{agent}: {phase} response manifest mismatch")
        agent_tasks = read_jsonl(task_path)
        agent_responses = {
            str(row["task_id"]): row for row in read_jsonl(response_path)}
        if {str(row["task_id"]) for row in agent_tasks} != set(agent_responses):
            raise ContractError(f"{agent}: incomplete {phase} response coverage")
        for task in agent_tasks:
            validate_task_response(task, agent_responses[str(task["task_id"])])
            if str(task["task_id"]) in responses:
                raise ContractError(f"duplicate {phase} task id")
            responses[str(task["task_id"])] = agent_responses[
                str(task["task_id"])]
        tasks.extend(agent_tasks)
    return responses, tasks


def _add_route_candidate(
    graph: Mapping[str, Any], value: float, route: str, agent: str,
) -> dict[str, Any]:
    row = copy.deepcopy(graph)
    candidates = list(row.get("candidates", []))
    match = next((
        candidate for candidate in candidates
        if (parsed := _number(candidate.get("item"))) is not None
        and abs(parsed - value) / max(parsed, value, 1.0) <= 1e-12
    ), None)
    if match is None:
        item = format(value, ".15g")
        match = {
            "key": canonical_key(item, RELATION),
            "item": item,
            "type": "numeric",
            "sources": {},
            "selected_by": {QWEN: False, GEMMA: False},
            "routes": {},
            "route_summary": {},
        }
        candidates.append(match)
    match.setdefault("routes", {})[route] = {
        "model_family": agent,
        "route_type": "relation-specific-fact-supply",
        "support": 1,
        "samples": 1,
        "support_rate": 1.0,
        "selected": True,
    }
    match["route_summary"] = _summarize_routes(match["routes"])
    row["candidates"] = candidates
    row.setdefault("proposal_routes", {})[route] = {
        "available": True,
        "model_family": agent,
        "n_samples": 1,
        "route_type": "relation-specific-fact-supply",
    }
    return augment_relational_graph(row)


def build_graph(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _json(output / "proposal/plan/PLAN.json")
    if plan.get("schema") != "fact-evidence-proposal-plan-v1":
        raise ContractError("proposal plan schema mismatch")
    if plan.get("implementation_sha256") != sha256(Path(__file__).resolve()):
        raise ContractError("implementation changed after proposal preparation")
    graph_path = Path(plan["train_graph"])
    _manifest_for_graph(graph_path)
    responses, tasks = _validated_responses(plan, "proposal")
    by_index: dict[int, list[tuple[str, str, float]]] = {}
    parse_failures = 0
    for task in tasks:
        response = responses[str(task["task_id"])]
        generations = list(response.get("generations", []))
        status, items = proposal_parse_status(
            str(generations[0]) if generations else "", RELATION)
        value = _number(items[0]) if items else None
        if value is None:
            parse_failures += 1
            continue
        by_index.setdefault(int(task["input_index"]), []).append((
            str(task["route"]), str(task["agent_id"]), value))
    graphs = _load_graph(graph_path, expected_split="train")
    capacity_index = 0
    augmented = []
    before_candidates = after_candidates = 0
    for graph in graphs:
        if graph["Relation"] != RELATION:
            augmented.append(copy.deepcopy(graph))
            continue
        row = copy.deepcopy(graph)
        before_candidates += len(row.get("candidates", []))
        for route, agent, value in by_index.get(capacity_index, []):
            row = _add_route_candidate(row, value, route, agent)
        after_candidates += len(row.get("candidates", []))
        augmented.append(row)
        capacity_index += 1
    target = output / "proposal/graphs/train_graph.jsonl"
    write_jsonl_atomic(target, augmented)
    manifest = {
        "schema": "heterogeneous-memory-graph-manifest-v1",
        "split": "train",
        "contains_labels": False,
        "gold_aware": False,
        "rows": len(augmented),
        "output_sha256": sha256(target),
        "source_graph": str(graph_path),
        "source_graph_sha256": sha256(graph_path),
        "proposal_plan": str(output / "proposal/plan/PLAN.json"),
        "proposal_plan_sha256": sha256(output / "proposal/plan/PLAN.json"),
        "proposal_response_tasks": len(tasks),
        "parse_failures": parse_failures,
        "candidate_delta": after_candidates - before_candidates,
        "parameter_count_delta": 0,
        "validation_graph_created": False,
    }
    _write_json(target.with_suffix(target.suffix + ".manifest.json"), manifest)
    _write_json(output / "proposal/BUILD.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _component_numbers(graph: Mapping[str, Any]) -> list[float]:
    values = []
    for component in graph.get("relational_graph", {}).get("components", []):
        value = _number(component.get("representative"))
        if value is not None:
            values.append(value)
    return values


def _surface_numbers(graph: Mapping[str, Any]) -> list[float]:
    return [
        value for candidate in graph.get("candidates", [])
        if (value := _number(candidate.get("item"))) is not None]


def audit_supply(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _json(output / "proposal/plan/PLAN.json")
    source_path = Path(plan["train_graph"])
    augmented_path = output / "proposal/graphs/train_graph.jsonl"
    _manifest_for_graph(source_path)
    _manifest_for_graph(augmented_path)
    gold_path = Path(args.train_gold).resolve()
    source = {
        _key(row): row for row in _load_graph(
            source_path, expected_split="train")
        if row["Relation"] == RELATION}
    augmented = {
        _key(row): row for row in _load_graph(
            augmented_path, expected_split="train")
        if row["Relation"] == RELATION}
    gold = {_key(row): row for row in read_jsonl(gold_path)}
    if set(source) != set(augmented) or not set(source) <= set(gold):
        raise ContractError("supply audit coverage mismatch")
    responses, tasks = _validated_responses(plan, "proposal")
    arm_diagnostics: dict[str, Counter] = {
        arm: Counter() for arm in PROPOSAL_ARMS}
    for task in tasks:
        arm = str(task["arm"])
        key = str(task["subject"]), str(task["relation"])
        diagnostic = arm_diagnostics[arm]
        diagnostic["tasks"] += 1
        generations = list(
            responses[str(task["task_id"])].get("generations", []))
        _, items = proposal_parse_status(
            str(generations[0]) if generations else "", RELATION)
        value = _number(items[0]) if items else None
        if value is None:
            diagnostic["parse_failures"] += 1
            continue
        diagnostic["parsed"] += 1
        target = _gold_number(gold[key])
        if target is None or not _within(value, target):
            continue
        diagnostic["gold_hits"] += 1
        source_had_surface = any(
            _within(candidate, target)
            for candidate in _surface_numbers(source[key]))
        if not source_had_surface:
            diagnostic["gold_hits_on_source_missing_rows"] += 1
    ledger = []
    newly_surface_covered = 0
    newly_actionable_covered = 0
    lost_actionable = 0
    for key, row in source.items():
        target = _gold_number(gold[key])
        source_surface = bool(
            target is not None
            and any(_within(value, target) for value in _surface_numbers(row)))
        augmented_surface = bool(
            target is not None
            and any(_within(value, target)
                    for value in _surface_numbers(augmented[key])))
        source_actionable = bool(
            target is not None
            and any(_within(value, target)
                    for value in _component_numbers(row)))
        augmented_actionable = bool(
            target is not None
            and any(_within(value, target)
                    for value in _component_numbers(augmented[key])))
        newly_surface_covered += int(
            augmented_surface and not source_surface)
        newly_actionable_covered += int(
            augmented_actionable and not source_actionable)
        lost_actionable += int(
            source_actionable and not augmented_actionable)
        ledger.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "gold_value": target,
            "source_surface_supply": source_surface,
            "augmented_surface_supply": augmented_surface,
            "new_surface_covered": augmented_surface and not source_surface,
            "source_actionable_supply": source_actionable,
            "augmented_actionable_supply": augmented_actionable,
            "new_actionable_covered": (
                augmented_actionable and not source_actionable),
            "actionable_supply_lost": (
                source_actionable and not augmented_actionable),
            "contains_labels": True,
            "gold_aware": True,
        })
    result = {
        "schema": "fact-evidence-supply-audit-v2",
        "contains_labels": True,
        "gold_aware": True,
        "deployable": False,
        "rows": len(ledger),
        "source_surface_covered": sum(
            row["source_surface_supply"] for row in ledger),
        "augmented_surface_covered": sum(
            row["augmented_surface_supply"] for row in ledger),
        "new_surface_covered": newly_surface_covered,
        "source_actionable_covered": sum(
            row["source_actionable_supply"] for row in ledger),
        "augmented_actionable_covered": sum(
            row["augmented_actionable_supply"] for row in ledger),
        "new_actionable_covered": newly_actionable_covered,
        "lost_actionable": lost_actionable,
        "arm_diagnostics": {
            arm: dict(values)
            for arm, values in arm_diagnostics.items()},
        "gate_threshold": MIN_SUPPLY_ROWS,
        "supply_gate_passed": (
            newly_surface_covered >= MIN_SUPPLY_ROWS
            and newly_actionable_covered >= MIN_SUPPLY_ROWS
            and lost_actionable == 0),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "validation_opened": False,
    }
    ledger_path = output / "proposal/audit/SUPPLY_LEDGER.jsonl"
    write_jsonl_atomic(ledger_path, ledger)
    result["ledger"] = str(ledger_path)
    result["ledger_sha256"] = sha256(ledger_path)
    _write_json(output / "proposal/audit/RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _incumbent_map(path: Path) -> dict[tuple[str, str], list[str]]:
    manifest = _json(path.with_suffix(path.suffix + ".manifest.json"))
    if (manifest.get("schema") != "component-aware-oof-incumbents-v1"
            or not manifest.get("oof_model_excludes_row")
            or manifest.get("output_sha256") != sha256(path)
            or manifest.get("split") != "train"):
        raise ContractError("production-strength OOF incumbent contract failed")
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ContractError(f"duplicate OOF incumbent: {key}")
        result[key] = [str(value) for value in row["ObjectEntities"]]
    return result


def audit_baseline(args: argparse.Namespace) -> int:
    """Freeze the supply-versus-selection diagnosis before new GPU evidence."""
    output = Path(args.output_dir).resolve()
    graph_path = Path(args.train_graph).resolve()
    incumbent_path = Path(args.train_incumbents).resolve()
    gold_path = Path(args.train_gold).resolve()
    _manifest_for_graph(graph_path)
    incumbents = _incumbent_map(incumbent_path)
    gold = {_key(row): row for row in read_jsonl(gold_path)}
    rows = []
    counts = Counter()
    baseline_scores, component_oracle_scores, surface_oracle_scores = [], [], []
    for graph in _load_graph(graph_path, expected_split="train"):
        if graph["Relation"] != RELATION:
            continue
        key = _key(graph)
        if key not in incumbents or key not in gold:
            raise ContractError(f"baseline audit coverage missing {key}")
        baseline = incumbents[key]
        baseline_f1 = _row_f1(baseline, gold[key], RELATION)
        alternatives = [baseline] + component_actions(graph, baseline)
        scored = [
            (float(_row_f1(action, gold[key], RELATION)), list(action))
            for action in alternatives]
        oracle_f1, oracle_action = max(
            scored, key=lambda item: (
                item[0], -len(item[1]), tuple(map(str, item[1]))))
        surface_actions = [baseline]
        for component in graph["relational_graph"]["components"]:
            surface_actions.extend(
                [[str(row["surface"])]
                 for row in _component_surface_rows(graph, component)])
        surface_scored = [
            (float(_row_f1(action, gold[key], RELATION)), list(action))
            for action in surface_actions]
        surface_oracle_f1, surface_oracle_action = max(
            surface_scored, key=lambda item: (
                item[0], -len(item[1]), tuple(map(str, item[1]))))
        target = _gold_number(gold[key])
        surface_supply = bool(
            target is not None
            and any(_within(value, target)
                    for value in _surface_numbers(graph)))
        actionable_supply = bool(
            target is not None
            and any(_within(value, target)
                    for value in _component_numbers(graph)))
        if baseline_f1 >= 1.0 - 1e-12:
            failure = "current_correct"
        elif surface_oracle_f1 > baseline_f1 + 1e-12:
            failure = "selection_failure"
        elif surface_supply and not actionable_supply:
            failure = "representation_failure"
        elif not surface_supply:
            failure = "supply_missing"
        else:
            failure = "candidate_present_no_action_gain"
        counts[failure] += 1
        baseline_scores.append(float(baseline_f1))
        component_oracle_scores.append(float(oracle_f1))
        surface_oracle_scores.append(float(surface_oracle_f1))
        rows.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "baseline": baseline,
            "baseline_f1": baseline_f1,
            "component_oracle": oracle_action,
            "component_oracle_f1": oracle_f1,
            "surface_oracle": surface_oracle_action,
            "surface_oracle_f1": surface_oracle_f1,
            "available_gain": surface_oracle_f1 - baseline_f1,
            "gold_compatible_surface_available": surface_supply,
            "gold_compatible_actionable_component_available": (
                actionable_supply),
            "failure_class": failure,
            "contains_labels": True,
            "gold_aware": True,
        })
    ledger_path = output / "baseline_audit/FAILURE_LEDGER.jsonl"
    write_jsonl_atomic(ledger_path, rows)
    result = {
        "schema": "fact-evidence-baseline-audit-v1",
        "contains_labels": True,
        "gold_aware": True,
        "deployable": False,
        "relation": RELATION,
        "rows": len(rows),
        "failure_classes": dict(counts),
        "mean_baseline_f1": statistics.mean(baseline_scores),
        "mean_component_action_oracle_f1": statistics.mean(
            component_oracle_scores),
        "mean_surface_action_oracle_f1": statistics.mean(
            surface_oracle_scores),
        "mean_available_gain": (
            statistics.mean(surface_oracle_scores)
            - statistics.mean(baseline_scores)),
        "ledger": str(ledger_path),
        "ledger_sha256": sha256(ledger_path),
        "train_graph": str(graph_path),
        "train_graph_sha256": sha256(graph_path),
        "train_incumbents": str(incumbent_path),
        "train_incumbents_sha256": sha256(incumbent_path),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "validation_opened": False,
    }
    _write_json(output / "baseline_audit/RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _component_stats(
    graph: Mapping[str, Any], value: str,
) -> tuple[float, int, set[str], bool]:
    component = _component_for_prediction(graph, value)
    if component is None:
        return 0.0, 0, set(), False
    routes = component.get("routes", {})
    support = sum(
        float(route.get("max_support_rate", 0.0))
        for route in routes.values())
    # Component route summaries intentionally contain only aggregate support.
    # Recover model family from the stable route namespace when the detailed
    # candidate-level ``model_family`` field is not propagated.
    agents = set()
    for route_name, route in routes.items():
        family = route.get("model_family")
        if family:
            agents.add(str(family))
        elif str(route_name).startswith("qwen:"):
            agents.add(QWEN)
        elif str(route_name).startswith("gemma:"):
            agents.add(GEMMA)
    new = any(route in ARM_ROUTE.values() for route in routes)
    return support, len(routes), agents, new


def _surface_key(value: Any) -> str | None:
    number = _number(value)
    return f"{number:.12g}" if number is not None else None


def _surface_stats(
    graph: Mapping[str, Any], value: Any,
) -> tuple[float, int, set[str], bool]:
    """Return evidence attached to one numeric surface, not its component."""
    key = _surface_key(value)
    if key is None:
        return 0.0, 0, set(), False
    route_support: dict[str, float] = {}
    agents: set[str] = set()
    for candidate in graph.get("candidates", []):
        if _surface_key(candidate.get("item")) != key:
            continue
        for route_name, route in candidate.get("routes", {}).items():
            name = str(route_name)
            route_support[name] = max(
                route_support.get(name, 0.0),
                float(route.get("support_rate", 0.0)))
            family = route.get("model_family")
            if family:
                agents.add(str(family))
            elif name.startswith("qwen:"):
                agents.add(QWEN)
            elif name.startswith("gemma:"):
                agents.add(GEMMA)
    return (
        sum(route_support.values()), len(route_support), agents,
        any(route in ARM_ROUTE.values() for route in route_support))


def _component_surface_rows(
    graph: Mapping[str, Any], component: Mapping[str, Any],
) -> list[dict[str, Any]]:
    unique: dict[str, float] = {}
    for item in component.get("member_items", []):
        number = _number(item)
        key = _surface_key(item)
        if number is not None and key is not None:
            unique.setdefault(key, number)
    if not unique:
        return []
    numbers = sorted(unique.values())
    median = statistics.median(numbers)
    relative_span = (
        (max(numbers) - min(numbers)) / max(median, 1.0)
        if len(numbers) > 1 else 0.0)
    representative_key = _surface_key(component.get("representative"))
    rows = []
    for surface, number in sorted(unique.items(), key=lambda item: item[1]):
        support, route_count, proposers, new = _surface_stats(graph, surface)
        rows.append({
            "surface": surface,
            "surface_number": number,
            "surface_support": support,
            "surface_route_count": route_count,
            "surface_proposer_agents": sorted(proposers),
            "surface_new_route": new,
            "surface_is_component_representative": (
                surface == representative_key),
            "component_unique_surface_count": len(unique),
            "surface_distance_from_component_median": (
                abs(number - median) / max(median, 1.0)),
            "component_relative_span": relative_span,
        })
    return rows


def _action_inventory(
    graphs: Sequence[Mapping[str, Any]],
    incumbents: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict[str, Any]]:
    rows = []
    for graph in graphs:
        if graph["Relation"] != RELATION:
            continue
        key = _key(graph)
        if key not in incumbents:
            raise ContractError(f"missing OOF incumbent: {key}")
        baseline = list(incumbents[key])
        base_number = _number(baseline[0]) if baseline else None
        components = []
        for component in graph["relational_graph"]["components"]:
            representative = str(component["representative"])
            challenger = _number(representative)
            if challenger is None or (
                    base_number is not None
                    and _within(base_number, challenger)):
                continue
            support, routes, proposers, new = _component_stats(
                graph, representative)
            components.append((
                -int(new), -support,
                abs(math.log(challenger / base_number))
                if base_number else 0.0,
                representative, component, support, routes, proposers, new))
        selected_components = sorted(
            components, key=lambda row: row[:4])[:MAX_COMPONENTS_PER_ROW]
        candidates = []
        for (_, _, _, representative, component, support, routes,
             component_proposers, new) in selected_components:
            for surface in _component_surface_rows(graph, component):
                challenger = float(surface["surface_number"])
                if (base_number is not None
                        and _within(base_number, challenger)):
                    continue
                # Reviewer independence follows the exact surface provenance.
                # Fall back to component provenance only for legacy graphs
                # whose surface nodes lack model-family metadata.
                proposers = (
                    set(surface["surface_proposer_agents"])
                    or set(component_proposers))
                candidates.append({
                    "alternative": [str(surface["surface"])],
                    "component_id": str(component["id"]),
                    "component_representative": representative,
                    "challenger_support": support,
                    "challenger_route_count": routes,
                    "proposer_agents": sorted(proposers),
                    "challenger_new_route": new,
                    **surface,
                })
        for candidate in candidates:
            identity = hashlib.sha256(
                f"{key[0]}\x1f{key[1]}\x1f{baseline}\x1f"
                f"{candidate['alternative']}".encode()).hexdigest()[:16]
            rows.append({
                "schema": "fact-evidence-surface-action-v2",
                "SubjectEntity": key[0],
                "Relation": key[1],
                "baseline": baseline,
                **candidate,
                "action_id": identity,
                "contains_labels": False,
                "gold_aware": False,
            })
    return rows


def _render(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def verification_prompt(
    *, subject: str, baseline: Sequence[str], alternative: Sequence[str],
    axis: str, codebook: Mapping[str, str], alternative_first: bool,
) -> str:
    if axis not in REVIEW_AXES or set(codebook) != set(REVIEW_CHOICES):
        raise ContractError("verification prompt contract mismatch")
    if axis == "exact_memory":
        cue = (
            "Choose only when you recognize the exact published fact from "
            "memory; generic plausibility is insufficient.")
    elif axis == "identity":
        cue = (
            "Resolve the exact named venue and possible renamings; reject a "
            "capacity belonging to another similarly named venue.")
    else:
        cue = (
            "Use normal current maximum spectator capacity, excluding record "
            "attendance and temporary configurations.")
    first = alternative if alternative_first else baseline
    second = baseline if alternative_first else alternative
    first_choice = (
        "USE_ALTERNATIVE" if alternative_first else "KEEP_BASELINE")
    second_choice = (
        "KEEP_BASELINE" if alternative_first else "USE_ALTERNATIVE")
    return (
        "Use only closed-book factual memory. Compare two anonymous proposed "
        "answers symmetrically. If you cannot distinguish them factually, "
        "choose UNKNOWN. Do not infer from proposer identity, confidence, "
        "frequency, or generic venue size.\n"
        f"SUBJECT: {subject}\nRELATION: {RELATION}\n"
        f"VERIFICATION AXIS: {cue}\n"
        f"OUTPUT A: {_render(first)}\nOUTPUT B: {_render(second)}\n"
        f"Choose exactly one code: {codebook[first_choice]} = OUTPUT A, "
        f"{codebook[second_choice]} = OUTPUT B, "
        f"{codebook['UNKNOWN']} = UNKNOWN.\nCODE:"
    )


def verification_task(
    action: Mapping[str, Any], agent: str, axis: str,
) -> dict[str, Any]:
    codebooks = balanced_choice_codebooks(
        REVIEW_CHOICES, "fact-evidence-verification-v1",
        action["SubjectEntity"], action["Relation"], action["action_id"],
        agent, axis)
    variants = []
    for codebook in codebooks:
        for alternative_first in (False, True):
            variants.append({
                "choice_codes": dict(codebook),
                "prompt": verification_prompt(
                    subject=str(action["SubjectEntity"]),
                    baseline=action["baseline"],
                    alternative=action["alternative"],
                    axis=axis,
                    codebook=codebook,
                    alternative_first=alternative_first,
                ),
            })
    independent = agent not in set(action["proposer_agents"])
    return {
        "task_id": (
            f"{agent}::fact_verify::{action['action_id']}::{axis}"),
        "agent_id": agent,
        "subject": action["SubjectEntity"],
        "relation": action["Relation"],
        "phase": "fact_verification",
        "mode": "choice",
        "prompt": variants[0]["prompt"],
        "choices": list(REVIEW_CHOICES),
        "choice_codes": variants[0]["choice_codes"],
        "choice_variants": variants,
        "candidate_key": f"{action['action_id']}::{axis}",
        "candidate_item": _render(action["alternative"]),
        "excluded_proposer_agents": list(action["proposer_agents"]),
        "action_id": action["action_id"],
        "axis": axis,
        "reviewer_is_independent": independent,
        "contains_labels": False,
        "gold_aware": False,
        "prompt_masks_provenance": True,
        "prompt_masks_incumbency": True,
    }


def prepare_verification(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    source_graph = Path(args.train_graph).resolve()
    augmented_graph = output / "proposal/graphs/train_graph.jsonl"
    supply_result_path = output / "proposal/audit/RESULT.json"
    supply_gate = None
    if augmented_graph.is_file():
        if not supply_result_path.is_file():
            raise ContractError(
                "augmented proposal graph requires completed supply audit")
        supply_gate = _json(supply_result_path)
        if (supply_gate.get("schema") != "fact-evidence-supply-audit-v2"
                or supply_gate.get("validation_opened")):
            raise ContractError("invalid supply audit contract")
    use_augmented = bool(
        supply_gate is not None and supply_gate.get("supply_gate_passed"))
    graph_path = augmented_graph if use_augmented else source_graph
    _manifest_for_graph(graph_path)
    incumbents_path = Path(args.train_incumbents).resolve()
    incumbents = _incumbent_map(incumbents_path)
    actions = _action_inventory(
        _load_graph(graph_path, expected_split="train"), incumbents)
    action_path = output / "verification/plan/actions.jsonl"
    write_jsonl_atomic(action_path, actions)
    jobs = {}
    for agent in AGENTS:
        tasks = []
        for action in actions:
            proposers = set(action["proposer_agents"])
            reviewers = (
                [OTHER_AGENT[next(iter(proposers))]]
                if len(proposers) == 1 and next(iter(proposers)) in OTHER_AGENT
                else list(AGENTS))
            if agent not in reviewers:
                continue
            tasks.extend(
                verification_task(action, agent, axis)
                for axis in REVIEW_AXES)
        task_path = output / f"verification/plan/tasks/{agent}.jsonl"
        smoke_path = output / f"verification/plan/smoke/{agent}.jsonl"
        write_jsonl_atomic(task_path, tasks)
        write_jsonl_atomic(smoke_path, tasks[:3])
        validate_tasks(tasks, agent)
        jobs[agent] = {
            "tasks": len(tasks),
            "task_path": str(task_path),
            "task_sha256": sha256(task_path),
            "smoke_path": str(smoke_path),
            "smoke_sha256": sha256(smoke_path),
            "response_path": str(
                output / f"verification/responses/{agent}.jsonl"),
        }
    plan = {
        "schema": "fact-evidence-verification-plan-v2",
        "split": "train",
        "relation": RELATION,
        "contains_labels": False,
        "gold_aware": True,
        "selection_uses_train_labels": True,
        "oof_model_excludes_row": True,
        "deployable": False,
        "validation_opened": False,
        "actions": len(actions),
        "max_components_per_row": MAX_COMPONENTS_PER_ROW,
        "surface_preserving": True,
        "review_axes": list(REVIEW_AXES),
        "independent_review_policy": (
            "single-origin candidates reviewed only by other model; "
            "shared-origin candidates reviewed by both"),
        "graph": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "graph_selection": (
            "augmented_supply_gate_passed"
            if use_augmented else "source_supply_gate_failed"
            if supply_gate is not None else "source_no_augmented_graph"),
        "supply_gate": (
            str(supply_result_path) if supply_gate is not None else None),
        "supply_gate_sha256": (
            sha256(supply_result_path) if supply_gate is not None else None),
        "supply_gate_passed": (
            bool(supply_gate.get("supply_gate_passed"))
            if supply_gate is not None else None),
        "train_incumbents": str(incumbents_path),
        "train_incumbents_sha256": sha256(incumbents_path),
        "inventory": str(action_path),
        "inventory_sha256": sha256(action_path),
        "jobs": jobs,
        "agents": str(Path(args.agents).resolve()),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(output / "verification/plan/PLAN.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def ledger(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _json(output / "verification/plan/PLAN.json")
    gold_path = Path(args.train_gold).resolve()
    folds_path = Path(args.folds).resolve()
    gold = {_key(row): row for row in read_jsonl(gold_path)}
    folds = {_key(row): int(row["fold"]) for row in read_jsonl(folds_path)}
    rows, counts = [], Counter()
    for action in read_jsonl(Path(plan["inventory"])):
        key = _key(action)
        if key not in gold or key not in folds:
            raise ContractError(f"ledger supervision missing {key}")
        before = _row_f1(action["baseline"], gold[key], RELATION)
        after = _row_f1(action["alternative"], gold[key], RELATION)
        utility = after - before
        outcome = (
            "helpful" if utility > 1e-12
            else "harmful" if utility < -1e-12 else "neutral")
        counts[outcome] += 1
        rows.append({
            **action,
            "schema": "fact-evidence-action-ledger-row-v1",
            "fold": folds[key],
            "before_f1": before,
            "after_f1": after,
            "utility": utility,
            "outcome": outcome,
            "gold": gold[key]["ObjectEntities"],
            "contains_labels": True,
            "gold_aware": True,
        })
    path = output / "verification/ledger/actions.jsonl"
    write_jsonl_atomic(path, rows)
    helpful_keys = {
        _key(row) for row in rows if float(row["utility"]) > 1e-12}
    best_gain = {}
    for row in rows:
        key = _key(row)
        best_gain[key] = max(
            best_gain.get(key, 0.0), float(row["utility"]))
    relation_rows = [
        row for row in gold.values() if row["Relation"] == RELATION]
    incumbents = _incumbent_map(Path(plan["train_incumbents"]))
    incumbent_mean = statistics.mean(
        _row_f1(
            incumbents[_key(row)], row, RELATION)
        for row in relation_rows)
    inventory_oracle_delta = sum(
        max(0.0, best_gain.get(_key(row), 0.0))
        for row in relation_rows) / max(1, len(relation_rows))
    result = {
        "schema": "fact-evidence-action-ledger-v1",
        "contains_labels": True,
        "gold_aware": True,
        "deployable": False,
        "rows": len(rows),
        "outcomes": dict(counts),
        "rows_with_helpful_action": len(helpful_keys),
        "incumbent_mean_f1": incumbent_mean,
        "inventory_action_oracle_delta": inventory_oracle_delta,
        "inventory_action_oracle_mean_f1": (
            incumbent_mean + inventory_oracle_delta),
        "output": str(path),
        "output_sha256": sha256(path),
        "inventory_sha256": plan["inventory_sha256"],
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "folds": str(folds_path),
        "folds_sha256": sha256(folds_path),
        "validation_opened": False,
    }
    _write_json(output / "verification/ledger/RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _route_features(
    action: Mapping[str, Any], graph: Mapping[str, Any],
) -> np.ndarray:
    incumbent = _number(action["baseline"][0]) if action["baseline"] else None
    challenger = _number(action["alternative"][0])
    if challenger is None:
        raise ContractError("capacity action challenger is not numeric")
    cs, cr, ca, cn = _component_stats(graph, action["alternative"][0])
    if incumbent is None:
        ins, inr, ina, inn = 0.0, 0, set(), False
        log_ratio = 0.0
    else:
        ins, inr, ina, inn = _component_stats(graph, action["baseline"][0])
        log_ratio = math.log(challenger / incumbent)
    css, csr, csa, _ = _surface_stats(
        graph, action["alternative"][0])
    if incumbent is None:
        iss, isr, isa = 0.0, 0, set()
    else:
        iss, isr, isa, _ = _surface_stats(graph, action["baseline"][0])
    values = (
        1.0,
        math.log1p(incumbent or 0.0) / 15.0,
        math.log1p(challenger) / 15.0,
        max(-2.0, min(2.0, log_ratio)) / 2.0,
        min(2.0, abs(log_ratio)) / 2.0,
        min(6.0, cs) / 6.0,
        min(6.0, ins) / 6.0,
        max(-6.0, min(6.0, cs - ins)) / 6.0,
        min(10, cr) / 10.0,
        min(10, inr) / 10.0,
        min(2, len(ca)) / 2.0,
        min(2, len(ina)) / 2.0,
        float(cn),
        float(inn),
        min(6.0, css) / 6.0,
        min(6.0, iss) / 6.0,
        max(-6.0, min(6.0, css - iss)) / 6.0,
        min(10, csr) / 10.0,
        min(10, isr) / 10.0,
        min(2, len(csa)) / 2.0,
        min(2, len(isa)) / 2.0,
        float(action.get(
            "surface_is_component_representative", False)),
        min(7, int(action.get(
            "component_unique_surface_count", 1))) / 7.0,
        min(0.10, float(action.get(
            "surface_distance_from_component_median", 0.0))) / 0.10,
        min(0.10, float(action.get(
            "component_relative_span", 0.0))) / 0.10,
    )
    if len(values) != len(GRAPH_FEATURE_NAMES):
        raise AssertionError("graph feature schema drift")
    return np.asarray(values, dtype=np.float64)


def _review_map(
    output: Path, plan: Mapping[str, Any],
) -> dict[tuple[str, str, str], Mapping[str, float]]:
    responses, tasks = _validated_responses(plan, "verification")
    result = {}
    for task in tasks:
        response = responses[str(task["task_id"])]
        probabilities = {
            choice: float(response["choice_probabilities"][choice])
            for choice in REVIEW_CHOICES}
        if (any(not math.isfinite(value) or value < 0
                for value in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1.0,
                                    abs_tol=1e-6, rel_tol=1e-6)):
            raise ContractError(f"{task['task_id']}: invalid review probability")
        result[(
            str(task["action_id"]), str(task["agent_id"]),
            str(task["axis"]))] = {
                **probabilities,
                "independent": bool(task["reviewer_is_independent"]),
            }
    return result


def _review_features(
    action: Mapping[str, Any],
    reviews: Mapping[tuple[str, str, str], Mapping[str, float]],
) -> np.ndarray:
    values = []
    independent_signed = []
    all_signed = []
    per_agent = {agent: [] for agent in AGENTS}
    for agent in AGENTS:
        for axis in REVIEW_AXES:
            row = reviews.get((str(action["action_id"]), agent, axis))
            if row is None:
                values.extend((0.0,) * 6)
                continue
            keep = float(row["KEEP_BASELINE"])
            use = float(row["USE_ALTERNATIVE"])
            unknown = float(row["UNKNOWN"])
            signed = use - keep
            independent = bool(row["independent"])
            values.extend((1.0, keep, use, unknown, signed, float(independent)))
            all_signed.append(signed)
            per_agent[agent].append(signed)
            if independent:
                independent_signed.append(signed)
    source = independent_signed or all_signed or [0.0]
    agent_means = [
        statistics.mean(per_agent[agent])
        for agent in AGENTS if per_agent[agent]]
    values.extend((
        statistics.mean(source),
        min(source),
        max(source),
        float(sum(value > 0 for value in source)),
        statistics.mean(all_signed or [0.0]),
        abs(agent_means[0] - agent_means[1])
        if len(agent_means) == 2 else 0.0,
    ))
    vector = np.asarray(values, dtype=np.float64)
    if len(vector) != len(REVIEW_FEATURE_NAMES):
        raise AssertionError("review feature schema drift")
    return vector


def _records(
    output: Path,
    reviews: Mapping[tuple[str, str, str], Mapping[str, float]] | None,
) -> list[dict[str, Any]]:
    plan = _json(output / "verification/plan/PLAN.json")
    graphs = {
        _key(row): row for row in _load_graph(
            Path(plan["graph"]), expected_split="train")}
    rows = []
    for action in read_jsonl(output / "verification/ledger/actions.jsonl"):
        graph = graphs[_key(action)]
        features = _route_features(action, graph)
        if reviews is not None:
            features = np.concatenate([
                features, _review_features(action, reviews)])
        rows.append({**action, "features": features})
    per_row = Counter(_key(row) for row in rows)
    for row in rows:
        row["weight"] = 1.0 / per_row[_key(row)]
    return rows


def _fit_model(
    rows: Sequence[Mapping[str, Any]], alpha: float,
    names: Sequence[str],
) -> RidgeActionModel:
    class Record:
        def __init__(self, row: Mapping[str, Any]):
            self.features = row["features"]
            self.utility = float(row["utility"])
            self.weight = float(row["weight"])
    return RidgeActionModel(alpha, names).fit([Record(row) for row in rows])


def _decode(
    model: RidgeActionModel, rows: Sequence[Mapping[str, Any]],
    margin: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_key(row), []).append(row)
    details = []
    for key, actions in grouped.items():
        best = max(actions, key=lambda row: (
            model.predict(row["features"]), str(row["action_id"])))
        estimate = float(model.predict(best["features"]))
        selected = best if estimate > margin else None
        details.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "fold": int(best["fold"]),
            "action_id": (
                str(selected["action_id"]) if selected is not None else None),
            "estimated_utility": estimate,
            "utility": float(selected["utility"]) if selected else 0.0,
            "baseline": list(best["baseline"]),
            "alternative": (
                list(selected["alternative"]) if selected else None),
            "component_id": (
                str(selected["component_id"]) if selected else None),
        })
    return details


def _configuration(
    rows: Sequence[Mapping[str, Any]], folds: Sequence[int],
    names: Sequence[str],
    evaluation_folds: Mapping[tuple[str, str], int] | None = None,
) -> dict[str, Any]:
    candidates = []
    for alpha in ALPHAS:
        predictions = {}
        for fold in folds:
            fit_rows = [row for row in rows if int(row["fold"]) != fold]
            hold = [row for row in rows if int(row["fold"]) == fold]
            model = _fit_model(fit_rows, alpha, names)
            predictions[fold] = (
                model, hold)
        for margin in MARGINS:
            values, changed = [], 0
            fold_deltas = {}
            for fold in folds:
                model, hold = predictions[fold]
                details = _decode(model, hold, margin)
                denominator = (
                    sum(value == fold for value in evaluation_folds.values())
                    if evaluation_folds is not None else len(details))
                delta = (
                    sum(float(row["utility"]) for row in details)
                    / max(1, denominator))
                fold_deltas[str(fold)] = delta
                values.append(delta)
                changed += sum(row["action_id"] is not None for row in details)
            candidates.append({
                "alpha": alpha,
                "margin": margin,
                "fold_deltas": fold_deltas,
                "mean_delta": statistics.mean(values),
                "wins": sum(value > 1e-12 for value in values),
                "worst": min(values),
                "changed": changed,
            })
    return max(candidates, key=lambda row: (
        row["mean_delta"], row["wins"], row["worst"],
        -row["changed"], row["margin"], row["alpha"]))


def _nested(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str],
    evaluation_folds: Mapping[tuple[str, str], int] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    folds = sorted({int(row["fold"]) for row in rows})
    fold_deltas, details = {}, []
    for outer in folds:
        fit_rows = [row for row in rows if int(row["fold"]) != outer]
        hold = [row for row in rows if int(row["fold"]) == outer]
        inner = [fold for fold in folds if fold != outer]
        config = _configuration(
            fit_rows, inner, names, evaluation_folds)
        model = _fit_model(fit_rows, float(config["alpha"]), names)
        decoded = _decode(model, hold, float(config["margin"]))
        denominator = (
            sum(value == outer for value in evaluation_folds.values())
            if evaluation_folds is not None else len(decoded))
        fold_deltas[str(outer)] = (
            sum(float(row["utility"]) for row in decoded)
            / max(1, denominator))
        details.extend({
            **row,
            "outer_fold": outer,
            "alpha": config["alpha"],
            "margin": config["margin"],
        } for row in decoded)
    return fold_deltas, details


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


def fit(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _json(output / "verification/plan/PLAN.json")
    if plan.get("implementation_sha256") != sha256(Path(__file__).resolve()):
        raise ContractError("implementation changed after verification prepare")
    reviews = _review_map(output, plan)
    graph_rows = _records(output, None)
    full_rows = _records(output, reviews)
    ledger_result = _json(output / "verification/ledger/RESULT.json")
    fold_table = {
        _key(row): int(row["fold"])
        for row in read_jsonl(Path(ledger_result["folds"]))}
    evaluation_keys = {
        _key(row) for row in _load_graph(
            Path(plan["graph"]), expected_split="train")
        if row["Relation"] == RELATION}
    evaluation_folds = {
        key: fold_table[key] for key in evaluation_keys}
    graph_folds, graph_details = _nested(
        graph_rows, GRAPH_FEATURE_NAMES, evaluation_folds)
    full_folds, full_details = _nested(
        full_rows, FULL_FEATURE_NAMES, evaluation_folds)
    graph_mean = statistics.mean(graph_folds.values())
    full_mean = statistics.mean(full_folds.values())
    changed = sum(
        left["action_id"] != right["action_id"]
        for left, right in zip(
            sorted(graph_details, key=lambda row: (
                row["SubjectEntity"], row["Relation"])),
            sorted(full_details, key=lambda row: (
                row["SubjectEntity"], row["Relation"]))))
    labels = [int(float(row["utility"]) > 1e-12) for row in full_rows]
    independent_scores = [
        float(_review_features(row, reviews)[-6])
        for row in full_rows]
    review_auroc = _auroc(labels, independent_scores)
    wins = sum(value > 1e-12 for value in full_folds.values())
    incremental = full_mean - graph_mean
    passed = bool(
        review_auroc is not None
        and review_auroc >= MIN_REVIEW_AUROC
        and full_mean >= MIN_OOF_DELTA
        and incremental >= MIN_INCREMENTAL_DELTA
        and wins >= MIN_WINS
        and min(full_folds.values()) >= FOLD_FLOOR
        and changed >= MIN_CHANGED_ACTIONS)
    config = _configuration(
        full_rows, sorted({int(row["fold"]) for row in full_rows}),
        FULL_FEATURE_NAMES, evaluation_folds)
    model = _fit_model(
        full_rows, float(config["alpha"]), FULL_FEATURE_NAMES)
    fit_dir = output / "verification/fit"
    model_path = fit_dir / "MODEL.json"
    _write_json(model_path, {
        **model.to_dict(),
        "schema": "fact-evidence-action-model-v1",
        "contains_labels": True,
        "gold_aware": True,
        "deployable": passed,
        "selected_margin": config["margin"],
        "feature_names": list(FULL_FEATURE_NAMES),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    })
    write_jsonl_atomic(fit_dir / "GRAPH_ONLY_OOF.jsonl", graph_details)
    write_jsonl_atomic(fit_dir / "FULL_OOF.jsonl", full_details)
    gate = {
        "schema": "fact-evidence-train-gate-v1",
        "passed": passed,
        "deployable": passed,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "thresholds": {
            "minimum_helpful_vs_rest_auroc": MIN_REVIEW_AUROC,
            "minimum_full_oof_delta": MIN_OOF_DELTA,
            "minimum_incremental_delta": MIN_INCREMENTAL_DELTA,
            "minimum_winning_folds": MIN_WINS,
            "fold_floor": FOLD_FLOOR,
            "minimum_changed_actions": MIN_CHANGED_ACTIONS,
        },
        "helpful_vs_rest_auroc": review_auroc,
        "graph_only_fold_deltas": graph_folds,
        "graph_only_mean_delta": graph_mean,
        "full_fold_deltas": full_folds,
        "full_mean_delta": full_mean,
        "incremental_review_delta": incremental,
        "winning_folds": wins,
        "worst_fold": min(full_folds.values()),
        "changed_oof_rows_vs_graph_only": changed,
        "evaluated_rows": len(evaluation_folds),
        "rows_without_eligible_action": (
            len(evaluation_folds)
            - len({_key(row) for row in full_rows})),
        "selection": config,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(fit_dir / "GATE.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("audit-baseline", audit_baseline),
        ("prepare-proposals", prepare_proposals),
        ("build-graph", build_graph),
        ("audit-supply", audit_supply),
        ("prepare-verification", prepare_verification),
        ("ledger", ledger),
        ("fit", fit),
    ):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--train-graph", default=str(DEFAULT_GRAPH))
        command.add_argument(
            "--train-incumbents", default=str(DEFAULT_INCUMBENTS))
        command.add_argument("--train-gold", default=str(DEFAULT_GOLD))
        command.add_argument("--folds", default=str(DEFAULT_FOLDS))
        command.add_argument("--agents", default=str(DEFAULT_AGENTS))
        command.add_argument("--seed", type=int, default=20260728)
        command.set_defaults(func=function)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
