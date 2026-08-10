#!/usr/bin/env python3
"""Broad candidate-conditioned factual evidence for the memory graph.

This experiment asks both heterogeneous parametric memories whether each
actionable graph component is factually true.  It is deliberately staged:

1. ``prepare`` writes label-free, forced-binary GPU tasks for every relation;
2. ``analyze`` validates response provenance and measures raw candidate
   discrimination on the labeled training graph;
3. only a passing discrimination result may be consumed by a later selector.

Validation is not an input and no relation-specific threshold is fitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.core import (
    ContractError,
    RELATION_QUESTIONS,
    SINGLE_RELATIONS,
    balanced_choice_codebooks,
    read_jsonl,
    sha256,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.fact_evidence_pipeline import _auroc
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    _row_f1,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks
from experiments.heterogeneous_agents.components.unified_memory_action_graph import (
    RELATIONS,
    _key,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "targeted_company_gemma_n3_20260724_v1/graphs/train_graph.jsonl")
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_AGENTS = (
    ROOT / "experiments/heterogeneous_agents/"
    "agents_qwen_gemma_n1_frozen.json")
DEFAULT_OUTPUT = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "candidate_truth_evidence_20260727_v1")

AGENTS = (QWEN, GEMMA)
CHOICES = ("TRUE", "FALSE")
OVERALL_AUROC_GATE = 0.65
MIN_RELATION_AUROC = 0.60
MIN_PASSING_RELATIONS = 3
MIN_ACTION_DELTA = 0.01


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        dict(value), indent=2, sort_keys=True) + "\n")


def _graph_manifest(
    path: Path, *, expected_split: str = "train",
) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ContractError(f"missing graph manifest: {manifest_path}")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != "heterogeneous-memory-graph-manifest-v1"
        or manifest.get("split") != expected_split
        or manifest.get("contains_labels")
        or manifest.get("gold_aware")
        or manifest.get("output_sha256") != sha256(path)
    ):
        raise ContractError("candidate graph is not certified label-free")
    return manifest


def _component_proposers(
    graph: Mapping[str, Any], component: Mapping[str, Any],
) -> list[str]:
    output = set()
    for route in component.get("routes", {}):
        metadata = graph.get("proposal_routes", {}).get(route, {})
        family = str(metadata.get(
            "model_family",
            GEMMA if str(route).startswith("gemma:") else QWEN))
        if family not in AGENTS:
            raise ContractError(f"unknown proposer family {family!r}")
        output.add(family)
    return sorted(output)


def component_inventory(
    graphs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return every actionable component without consulting labels."""
    output = []
    seen = set()
    for graph in graphs:
        relation = str(graph["Relation"])
        if relation not in RELATIONS:
            raise ContractError(f"unsupported relation {relation!r}")
        components = graph.get("relational_graph", {}).get("components", [])
        for component in components:
            representative = str(component["representative"])
            identity = hashlib.sha256(
                f"{graph['SubjectEntity']}\x1f{relation}\x1f"
                f"{component['id']}\x1f{representative}".encode()
            ).hexdigest()[:20]
            if identity in seen:
                raise ContractError("duplicate component verification id")
            seen.add(identity)
            output.append({
                "schema": "candidate-truth-component-v1",
                "component_key": identity,
                "SubjectEntity": str(graph["SubjectEntity"]),
                "Relation": relation,
                "component_id": str(component["id"]),
                "candidate": representative,
                "member_items": [
                    str(item) for item in component.get("member_items", [])],
                "proposer_agents": _component_proposers(graph, component),
                "contains_labels": False,
                "gold_aware": False,
            })
    return output


def truth_prompt(
    *, subject: str, relation: str, candidate: str,
    codebook: Mapping[str, str],
) -> str:
    if relation not in RELATION_QUESTIONS or set(codebook) != set(CHOICES):
        raise ContractError("candidate truth prompt contract mismatch")
    membership = (
        "For a relation that can have several answers, judge only whether "
        "this candidate is a true member; do not require it to be the complete "
        "answer set. "
        if relation not in {"hasArea", "hasCapacity", "personHasCityOfDeath"}
        else "")
    return (
        "Use only closed-book factual memory. You must decide whether the "
        "specific candidate correctly completes the subject-relation fact, "
        "even if uncertain. TRUE requires factual recognition; generic "
        "plausibility, name similarity, proposer confidence, and repetition "
        "are not evidence. "
        f"{membership}\n"
        f"SUBJECT: {subject}\n"
        f"RELATION: {relation}\n"
        f"QUESTION: {RELATION_QUESTIONS[relation].format(subject=subject)}\n"
        f"CANDIDATE: {json.dumps(candidate, ensure_ascii=False)}\n"
        f"Choose exactly one code: {codebook['TRUE']} = TRUE; "
        f"{codebook['FALSE']} = FALSE. Return only the code.\nCODE:"
    )


def truth_task(
    component: Mapping[str, Any], agent: str,
) -> dict[str, Any]:
    codebooks = balanced_choice_codebooks(
        CHOICES, "candidate-truth-evidence-v1", agent,
        component["component_key"])
    variants = [{
        "choice_codes": dict(codebook),
        "prompt": truth_prompt(
            subject=str(component["SubjectEntity"]),
            relation=str(component["Relation"]),
            candidate=str(component["candidate"]),
            codebook=codebook,
        ),
    } for codebook in codebooks]
    return {
        "task_id": (
            f"{agent}::candidate_truth::{component['component_key']}"),
        "agent_id": agent,
        "subject": component["SubjectEntity"],
        "relation": component["Relation"],
        "phase": "candidate_truth_verification",
        "mode": "choice",
        "prompt": variants[0]["prompt"],
        "choices": list(CHOICES),
        "choice_codes": variants[0]["choice_codes"],
        "choice_variants": variants,
        "candidate_key": component["component_key"],
        "candidate_item": str(component["candidate"]),
        "component_id": component["component_id"],
        "proposer_agents": list(component["proposer_agents"]),
        "reviewer_is_independent": (
            agent not in set(component["proposer_agents"])),
        "contains_labels": False,
        "gold_aware": False,
        "prompt_masks_provenance": True,
        "prompt_masks_incumbency": True,
        "forced_binary": True,
    }


def prepare(args: argparse.Namespace) -> int:
    split = str(args.split)
    graph_argument = args.graph or args.train_graph
    graph_path = Path(graph_argument).resolve()
    _graph_manifest(graph_path, expected_split=split)
    graphs = read_jsonl(graph_path)
    inventory = component_inventory(graphs)
    output = Path(args.output_dir).resolve()
    inventory_path = output / "plan/COMPONENTS.jsonl"
    write_jsonl_atomic(inventory_path, inventory)
    jobs = {}
    for agent in AGENTS:
        tasks = [truth_task(component, agent) for component in inventory]
        task_path = output / f"plan/tasks/{agent}.jsonl"
        smoke_path = output / f"plan/smoke/{agent}.jsonl"
        write_jsonl_atomic(task_path, tasks)
        # Cover all relation families in the smoke when possible.
        smoke = []
        for relation in RELATIONS:
            match = next((
                task for task in tasks if task["relation"] == relation), None)
            if match is not None:
                smoke.append(match)
        write_jsonl_atomic(smoke_path, smoke)
        validate_tasks(tasks, agent)
        jobs[agent] = {
            "tasks": len(tasks),
            "task_path": str(task_path),
            "task_sha256": sha256(task_path),
            "smoke_path": str(smoke_path),
            "smoke_sha256": sha256(smoke_path),
            "response_path": str(output / f"responses/{agent}.jsonl"),
        }
    counts = Counter(row["Relation"] for row in inventory)
    agents_path = Path(args.agents).resolve()
    plan = {
        "schema": "candidate-truth-evidence-plan-v1",
        "split": split,
        "contains_labels": False,
        "gold_aware": False,
        "deployable": False,
        "validation_opened": split == "validation",
        "validation_labels_used": False,
        "forced_binary": True,
        "unknown_choice_available": False,
        "relations": list(RELATIONS),
        "components": len(inventory),
        "components_by_relation": dict(counts),
        "inventory": str(inventory_path),
        "inventory_sha256": sha256(inventory_path),
        "graph": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "agents": str(agents_path),
        "agents_sha256": sha256(agents_path),
        "jobs": jobs,
        "discrimination_gate": {
            "minimum_overall_auroc": OVERALL_AUROC_GATE,
            "minimum_relation_auroc": MIN_RELATION_AUROC,
            "minimum_passing_relations": MIN_PASSING_RELATIONS,
            "minimum_raw_action_delta": MIN_ACTION_DELTA,
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    if split == "train":
        # Backward-compatible provenance fields used by the existing train
        # audit and its frozen downstream consumers.
        plan["train_graph"] = str(graph_path)
        plan["train_graph_sha256"] = sha256(graph_path)
    _write_json(output / "plan/PLAN.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def _validated_responses(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    responses, tasks = {}, []
    for agent, job in plan["jobs"].items():
        task_path = Path(job["task_path"])
        response_path = Path(job["response_path"])
        if sha256(task_path) != job["task_sha256"]:
            raise ContractError(f"{agent}: task hash mismatch")
        manifest_path = response_path.with_suffix(
            response_path.suffix + ".manifest.json")
        manifest = _json(manifest_path)
        if (
            manifest.get("task_sha256") != job["task_sha256"]
            or manifest.get("output_sha256") != sha256(response_path)
            or manifest.get("agent_id") != agent
            or int(manifest.get("tasks", -1)) != int(job["tasks"])
            or manifest.get("contains_labels")
            or manifest.get("gold_aware")
        ):
            raise ContractError(f"{agent}: response manifest mismatch")
        agent_tasks = read_jsonl(task_path)
        by_id = validate_tasks(agent_tasks, agent)
        agent_responses = read_jsonl(response_path)
        if len(agent_responses) != len(agent_tasks):
            raise ContractError(f"{agent}: incomplete response coverage")
        for response in agent_responses:
            task_id = str(response["task_id"])
            if task_id not in by_id or task_id in responses:
                raise ContractError(f"{agent}: invalid response id {task_id}")
            validate_task_response(by_id[task_id], response)
            responses[task_id] = response
        tasks.extend(agent_tasks)
    return responses, tasks


def _truth_scores(
    responses: Mapping[str, Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    per_component: dict[str, dict[str, float]] = defaultdict(dict)
    for task in tasks:
        response = responses[str(task["task_id"])]
        probabilities = response["choice_probabilities"]
        true_probability = float(probabilities["TRUE"])
        false_probability = float(probabilities["FALSE"])
        if (
            not math.isfinite(true_probability)
            or not math.isfinite(false_probability)
            or abs(true_probability + false_probability - 1.0) > 1e-6
        ):
            raise ContractError("invalid binary truth probabilities")
        per_component[str(task["candidate_key"])][
            str(task["agent_id"])] = true_probability
    output = {}
    for component_key, values in per_component.items():
        if set(values) != set(AGENTS):
            raise ContractError(
                f"{component_key}: missing heterogeneous truth score")
        output[component_key] = {
            QWEN: values[QWEN],
            GEMMA: values[GEMMA],
            "mean": statistics.mean(values.values()),
            "minimum": min(values.values()),
            "maximum": max(values.values()),
            "agreement": 1.0 - abs(values[QWEN] - values[GEMMA]),
        }
    return output


def _component_labels(
    inventory: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, float]:
    labels = {}
    for component in inventory:
        key = _key(component)
        if key not in gold_by:
            raise ContractError(f"missing gold row for {key}")
        labels[str(component["component_key"])] = float(
            _row_f1(
                [str(component["candidate"])],
                gold_by[key],
                str(component["Relation"]),
            ) > 0.0)
    return labels


def _raw_action_audit(
    inventory: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, float]],
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    candidates_by_key: dict[
        tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for component in inventory:
        candidates_by_key[_key(component)].append(component)
    utilities, changed = [], 0
    relation_utilities: dict[str, list[float]] = defaultdict(list)
    helped = harmed = neutral = 0
    for graph in graphs:
        key = _key(graph)
        relation = str(graph["Relation"])
        incumbent = [str(item) for item in graph.get("baseline_objects", [])]
        base_f1 = _row_f1(incumbent, gold_by[key], relation)
        candidates = candidates_by_key.get(key, [])
        if not candidates:
            utilities.append(0.0)
            relation_utilities[relation].append(0.0)
            continue
        ranked = sorted(candidates, key=lambda item: (
            float(scores[str(item["component_key"])][arm]),
            str(item["component_key"])), reverse=True)
        if relation in SINGLE_RELATIONS:
            proposed = (
                [str(ranked[0]["candidate"])]
                if float(scores[str(
                    ranked[0]["component_key"])][arm]) > 0.5 else [])
        else:
            proposed = [
                str(item["candidate"]) for item in ranked
                if float(scores[str(item["component_key"])][arm]) > 0.5
            ]
        # This is deliberately a fixed posterior decision boundary, not a
        # threshold selected against labels or validation.  Non-nullable
        # relations retain the incumbent when no candidate clears it.
        if not proposed and relation not in {
            "companyTradesAtStockExchange",
            "countryLandBordersCountry",
            "personHasCityOfDeath",
        }:
            proposed = incumbent
        if sorted(proposed) == sorted(incumbent):
            utilities.append(0.0)
            relation_utilities[relation].append(0.0)
            continue
        after_f1 = _row_f1(proposed, gold_by[key], relation)
        utility = after_f1 - base_f1
        utilities.append(utility)
        relation_utilities[relation].append(utility)
        changed += 1
        helped += utility > 1e-12
        harmed += utility < -1e-12
        neutral += abs(utility) <= 1e-12
    return {
        "mean_delta": statistics.mean(utilities),
        "changed": changed,
        "helped": helped,
        "harmed": harmed,
        "neutral": neutral,
        "relation_deltas": {
            relation: (
                statistics.mean(relation_utilities[relation])
                if relation_utilities[relation] else 0.0
            )
            for relation in RELATIONS
        },
    }


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    if (
        plan.get("schema") != "candidate-truth-evidence-plan-v1"
        or plan.get("validation_opened")
        or plan.get("contains_labels")
        or sha256(Path(plan["inventory"])) != plan["inventory_sha256"]
        or sha256(Path(plan["train_graph"])) != plan["train_graph_sha256"]
        or sha256(Path(plan["agents"])) != plan["agents_sha256"]
        or sha256(Path(plan["implementation"]))
        != plan["implementation_sha256"]
    ):
        raise ContractError("candidate truth plan contract failed")
    responses, tasks = _validated_responses(plan)
    inventory = read_jsonl(Path(plan["inventory"]))
    graphs = read_jsonl(Path(plan["train_graph"]))
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    gold_by = {_key(row): row for row in gold_rows}
    labels = _component_labels(inventory, gold_by)
    scores = _truth_scores(responses, tasks)
    for component in inventory:
        component_scores = scores[str(component["component_key"])]
        proposers = set(component["proposer_agents"])
        if proposers == {QWEN}:
            independent = component_scores[GEMMA]
        elif proposers == {GEMMA}:
            independent = component_scores[QWEN]
        else:
            independent = component_scores["mean"]
        component_scores["independent"] = independent
    arms = (QWEN, GEMMA, "mean", "minimum", "independent")
    arm_results = {}
    for arm in arms:
        all_scores = [
            float(scores[str(row["component_key"])][arm])
            for row in inventory]
        all_labels = [
            float(labels[str(row["component_key"])])
            for row in inventory]
        relation_aurocs = {}
        for relation in RELATIONS:
            selected = [
                row for row in inventory if row["Relation"] == relation]
            relation_aurocs[relation] = _auroc(
                [float(labels[str(row["component_key"])])
                 for row in selected],
                [float(scores[str(row["component_key"])][arm])
                 for row in selected],
            )
        raw_action = _raw_action_audit(
            inventory, scores, graphs, gold_by, arm)
        passing_relations = sum(
            value is not None and value >= MIN_RELATION_AUROC
            for value in relation_aurocs.values())
        overall = _auroc(all_labels, all_scores)
        arm_results[arm] = {
            "overall_auroc": overall,
            "relation_aurocs": relation_aurocs,
            "passing_relations": passing_relations,
            "raw_fixed_boundary_action_audit": raw_action,
            "gate_passed": (
                overall is not None
                and overall >= OVERALL_AUROC_GATE
                and passing_relations >= MIN_PASSING_RELATIONS
                and raw_action["mean_delta"] >= MIN_ACTION_DELTA
            ),
        }
    selected_arm = max(arms, key=lambda arm: (
        bool(arm_results[arm]["gate_passed"]),
        float(
            arm_results[arm]["overall_auroc"]
            if arm_results[arm]["overall_auroc"] is not None else -1.0
        ),
        arm,
    ))
    result = {
        "schema": "candidate-truth-evidence-result-v1",
        "development_only": True,
        "contains_labels": True,
        "gold_aware": True,
        "deployable": False,
        "validation_opened": False,
        "components": len(inventory),
        "positive_components": int(sum(labels.values())),
        "negative_components": int(len(labels) - sum(labels.values())),
        "arms": arm_results,
        "selected_arm": selected_arm,
        "discrimination_gate_passed": bool(
            arm_results[selected_arm]["gate_passed"]),
        "next_stage": (
            "fit_action_aware_selector"
            if arm_results[selected_arm]["gate_passed"]
            else "reject_candidate_truth_evidence"),
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
    }
    result_path = output / "analysis/RESULT.json"
    _write_json(result_path, result)
    score_rows = []
    for component in inventory:
        component_key = str(component["component_key"])
        score_rows.append({
            **component,
            "schema": "candidate-truth-evidence-scored-component-v1",
            "truth_scores": scores[component_key],
            "correct": bool(labels[component_key]),
            "contains_labels": True,
            "gold_aware": True,
        })
    score_path = output / "analysis/SCORED_COMPONENTS.jsonl"
    write_jsonl_atomic(score_path, score_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--train-graph", default=str(DEFAULT_GRAPH))
    prepare_parser.add_argument(
        "--graph", default=None,
        help="generic split graph; overrides --train-graph")
    prepare_parser.add_argument(
        "--split", choices=("train", "validation"), default="train")
    prepare_parser.add_argument("--agents", default=str(DEFAULT_AGENTS))
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    prepare_parser.set_defaults(func=prepare)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--train-gold", default=str(DEFAULT_GOLD))
    analyze_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    analyze_parser.set_defaults(func=analyze)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
