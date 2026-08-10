#!/usr/bin/env python3
"""Numeric candidate-supply pilot for hasArea and hasCapacity (train only).

The dominant numeric bottleneck is SUPPLY: on the current graphs roughly half
of all wrong numeric rows have no gold-compatible candidate at all, so no
selector can recover them.  This pilot tests whether new relation-specific,
compact elicitation routes add correct numeric candidates.

Design (predeclared before any GPU work):

* every train hasArea/hasCapacity row is sampled (200 rows) so router
  selectivity can be evaluated offline instead of being baked into the pilot;
* three arms, all with compact numeric-only outputs (the project's measured
  lesson is that reasoning scaffolds hurt numeric recall):
  - ``gemma_direct``     relation-specific direct question, N=3;
  - ``gemma_magnitude``  order-of-magnitude anchor then estimate, N=3;
  - ``qwen_direct``      the same direct question through Qwen, N=3;
* per-row label-free uncertainty features are frozen at prepare time so the
  audit can report what an adaptive router would have captured;
* ``audit`` opens train gold only after responses are frozen and is explicitly
  gold-aware and nondeployable.

Go/no-go, declared here and in the plan manifest: the pilot is material only
if a single arm newly covers >= 8 previously supply-missing train rows at N=3,
or the union of arms newly covers >= 12.  Otherwise numeric supply is treated
as knowledge-bound and no selector is fitted on this evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate import try_parse_number
from experiments.heterogeneous_agents.core import (
    ContractError,
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "targeted_company_gemma_n3_20260724_v1")
DEFAULT_CONFIG = (
    ROOT / "experiments/heterogeneous_agents/"
    "agents_qwen_gemma_n1_frozen.json")
NUMERIC_RELATIONS = ("hasArea", "hasCapacity")
ARMS = ("gemma_direct", "gemma_magnitude", "qwen_direct")
ARM_AGENT = {
    "gemma_direct": GEMMA,
    "gemma_magnitude": GEMMA,
    "qwen_direct": QWEN,
}
N_SAMPLES = 3
TEMPERATURE = 0.8
MAX_NEW_TOKENS = {"gemma_direct": 48, "gemma_magnitude": 64, "qwen_direct": 48}
GO_SINGLE_ARM_MIN_NEW_ROWS = 8
GO_UNION_MIN_NEW_ROWS = 12

DIRECT_QUESTION = {
    "hasArea": (
        "What is the total surface area of {subject} in square kilometres? "
        "Give the total area of the exact named entity itself, not a larger "
        "country or region containing it."),
    "hasCapacity": (
        "What is the maximum spectator capacity of {subject}? Give the "
        "current official maximum capacity of the venue (seated plus "
        "standing where applicable), not a historical record attendance."),
}

DIRECT_CONTRACT = (
    "Answer from memory. Respond with exactly one line and no explanation:\n"
    "ANSWER: <single number>")

MAGNITUDE_CONTRACT = (
    "Answer from memory. First commit to the order of magnitude, then give "
    "your estimate. Respond with exactly two lines and no explanation:\n"
    "MAGNITUDE: 10^<k>\n"
    "ANSWER: <single number>")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def arm_prompt(arm: str, subject: str, relation: str) -> str:
    question = DIRECT_QUESTION[relation].format(subject=subject)
    contract = (
        MAGNITUDE_CONTRACT if arm == "gemma_magnitude" else DIRECT_CONTRACT)
    return f"{question}\n{contract}"


def _node_numeric(node: Mapping[str, Any]) -> float | None:
    value = try_parse_number(str(node.get("item", "")))
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _within_tolerance(value: float, reference: float) -> bool:
    scale = max(abs(value), abs(reference))
    return scale > 0 and abs(value - reference) / scale <= 0.05 + 1e-12


def uncertainty_features(graph: Mapping[str, Any]) -> dict[str, float]:
    """Label-free row features frozen at prepare time for router analysis."""
    values = []
    qwen_top = 0.0
    gemma_values = []
    for node in graph.get("candidates", []):
        value = _node_numeric(node)
        if value is None:
            continue
        values.append(value)
        routes = node.get("routes", node.get("sources", {}))
        for route_name, route in routes.items():
            rate = float(route.get("support_rate", 0.0))
            if "qwen" in str(route_name):
                qwen_top = max(qwen_top, rate)
            if "gemma" in str(route_name):
                gemma_values.append(value)
    components = graph.get("relational_graph", {}).get("components", [])
    incumbent = [
        try_parse_number(str(item))
        for item in graph.get("baseline_objects", [])]
    incumbent = [value for value in incumbent if value is not None]
    gemma_distinct = bool(
        gemma_values and incumbent
        and not any(_within_tolerance(value, incumbent[0])
                    for value in gemma_values))
    if len(values) >= 2:
        logs = sorted(math.log(value) for value in values)
        spread = logs[-1] - logs[0]
    else:
        spread = 0.0
    return {
        "candidate_count": float(len(values)),
        "component_count": float(len(components)),
        "qwen_top_support_rate": qwen_top,
        "gemma_distinct_from_incumbent": float(gemma_distinct),
        "log_value_spread": spread,
    }


def build_pilot_tasks(
        graphs: Sequence[Mapping[str, Any]], *, seed: int,
        arms: Sequence[str] = ARMS,
) -> dict[str, list[dict[str, Any]]]:
    tasks: dict[str, list[dict[str, Any]]] = {GEMMA: [], QWEN: []}
    for arm in arms:
        agent = ARM_AGENT[arm]
        for index, graph in enumerate(graphs):
            subject, relation = _key(graph)
            tasks[agent].append({
                "task_id": f"{agent}::{arm}::{index}::proposal",
                "agent_id": agent,
                "subject": subject,
                "relation": relation,
                "phase": "propose",
                "mode": "generate",
                "arm": arm,
                "input_index": index,
                "prompt": arm_prompt(arm, subject, relation),
                "n_samples": N_SAMPLES,
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS[arm],
                "pilot_seed": seed,
            })
    return tasks


def prepare(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    split = args.split
    arms = tuple(args.arms.split(","))
    if any(arm not in ARMS for arm in arms):
        raise ContractError(f"unknown arm in {arms!r}")
    plan_dir = output / "plan"
    task_dir = plan_dir / "tasks"
    smoke_dir = plan_dir / "smoke"
    task_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    train_graph_path = source / f"graphs/{split}_graph.jsonl"
    graphs = [
        graph for graph in _load_graph(train_graph_path, expected_split=split)
        if graph["Relation"] in NUMERIC_RELATIONS]
    if not graphs:
        raise ContractError(f"no numeric {split} rows found in source graph")
    rows = []
    for index, graph in enumerate(graphs):
        subject, relation = _key(graph)
        rows.append({
            "SubjectEntity": subject,
            "Relation": relation,
            "input_index": index,
            "uncertainty_features": uncertainty_features(graph),
        })
    inputs_path = plan_dir / "inputs.jsonl"
    write_jsonl_atomic(inputs_path, rows)
    tasks = build_pilot_tasks(graphs, seed=args.seed, arms=arms)
    source_plan = _json(source / "plan/PLAN.json")
    jobs = {}
    for agent, agent_tasks in tasks.items():
        if not agent_tasks:
            continue
        task_path = task_dir / f"{agent}.jsonl"
        write_jsonl_atomic(task_path, agent_tasks)
        smoke, seen = [], set()
        for task in agent_tasks:
            key = task["arm"], task["relation"]
            if key not in seen:
                smoke.append(task)
                seen.add(key)
        smoke_path = smoke_dir / f"{agent}.jsonl"
        write_jsonl_atomic(smoke_path, smoke)
        jobs[agent] = {
            "tasks": len(agent_tasks),
            "task_path": str(task_path),
            "task_sha256": sha256(task_path),
            "response_path": str(output / "responses" / f"{agent}.jsonl"),
            "smoke_path": str(smoke_path),
            "smoke_sha256": sha256(smoke_path),
        }
    record = {
        "schema": "numeric-supply-pilot-plan-v1",
        "development_only": True,
        "split": split,
        "purpose": (
            "test whether relation-specific compact elicitation adds correct "
            "numeric candidates on supply-missing rows"),
        "source": str(source),
        "source_train_graph_sha256": sha256(train_graph_path),
        "source_graph_path": str(train_graph_path),
        "relations": list(NUMERIC_RELATIONS),
        "arms": list(arms),
        "arm_agent": {arm: ARM_AGENT[arm] for arm in arms},
        "n_samples": N_SAMPLES,
        "temperature": TEMPERATURE,
        "max_new_tokens": dict(MAX_NEW_TOKENS),
        "rows": len(rows),
        "inputs": str(inputs_path),
        "inputs_sha256": sha256(inputs_path),
        "jobs": jobs,
        "seed": args.seed,
        "train_gold": source_plan["train_gold"],
        "train_gold_sha256": source_plan["train_gold_sha256"],
        "validation_gold": source_plan.get("validation_gold"),
        "validation_gold_sha256": source_plan.get("validation_gold_sha256"),
        "audit_gold_key": ("train_gold" if split == "train"
                          else "validation_gold"),
        "labels_in_model_tasks": False,
        "routing": "all numeric train rows; router selectivity analyzed offline",
        "predeclared_go_no_go": {
            "single_arm_min_new_rows": GO_SINGLE_ARM_MIN_NEW_ROWS,
            "union_min_new_rows": GO_UNION_MIN_NEW_ROWS,
            "measured_on": "previously supply-missing train rows at N=3",
        },
    }
    (plan_dir / "PLAN.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"numeric supply pilot prepared: {len(rows)} rows; "
        + ", ".join(f"{agent}={job['tasks']} tasks"
                    for agent, job in jobs.items()))
    print(f"plan={plan_dir / 'PLAN.json'}")
    return 0


def _validated_responses(
        plan: Mapping[str, Any], agent: str) -> dict[str, Mapping[str, Any]]:
    job = plan["jobs"][agent]
    task_path = Path(job["task_path"])
    if sha256(task_path) != job["task_sha256"]:
        raise ContractError(f"{agent}: task file changed after prepare")
    tasks = read_jsonl(task_path)
    response_path = Path(job["response_path"])
    manifest = _json(
        response_path.with_suffix(response_path.suffix + ".manifest.json"))
    if manifest.get("output_sha256") != sha256(response_path):
        raise ContractError(f"{agent}: response output hash mismatch")
    if manifest.get("task_sha256") != job["task_sha256"]:
        raise ContractError(f"{agent}: response task hash mismatch")
    responses = {
        str(row.get("task_id")): row for row in read_jsonl(response_path)}
    task_by = {str(task["task_id"]): task for task in tasks}
    if set(responses) != set(task_by):
        raise ContractError(f"{agent}: response coverage mismatch")
    for task_id, task in task_by.items():
        validate_task_response(task, responses[task_id])
    return responses


def _sample_values(generations: Sequence[str], relation: str) -> list[float | None]:
    values: list[float | None] = []
    for text in generations:
        _, items = proposal_parse_status(str(text), relation)
        value = try_parse_number(str(items[0])) if items else None
        if value is not None and (not math.isfinite(value) or value <= 0):
            value = None
        values.append(float(value) if value is not None else None)
    return values


def _gold_value(gold: Mapping[str, Any]) -> float | None:
    for alias_group in gold.get("ObjectEntities", []):
        aliases = alias_group if isinstance(alias_group, list) else [alias_group]
        for alias in aliases:
            value = try_parse_number(str(alias))
            if value is not None and math.isfinite(value):
                return float(value)
    return None


def audit(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _json(output / "plan/PLAN.json")
    if plan.get("schema") != "numeric-supply-pilot-plan-v1":
        raise ContractError("unsupported numeric supply pilot plan")
    active_arms = tuple(plan.get("arms", ARMS))
    source = Path(plan["source"])
    split = plan.get("split", "train")
    train_graph_path = Path(plan.get(
        "source_graph_path", source / f"graphs/{split}_graph.jsonl"))
    if sha256(train_graph_path) != plan["source_train_graph_sha256"]:
        raise ContractError("source graph changed since prepare")
    graphs = {
        _key(graph): graph
        for graph in _load_graph(train_graph_path, expected_split=split)
        if graph["Relation"] in NUMERIC_RELATIONS}
    inputs = read_jsonl(Path(plan["inputs"]))
    gold_key = plan.get("audit_gold_key", "train_gold")
    gold_path = Path(plan[gold_key])
    if sha256(gold_path) != plan[f"{gold_key}_sha256"]:
        raise ContractError("gold changed since prepare")
    gold_by_key = {_key(row): row for row in read_jsonl(gold_path)}
    responses: dict[str, Mapping[str, Any]] = {}
    for agent in sorted(plan["jobs"]):
        responses.update(_validated_responses(plan, agent))

    ledger = []
    for row in inputs:
        key = str(row["SubjectEntity"]), str(row["Relation"])
        relation = key[1]
        graph = graphs[key]
        gold = gold_by_key[key]
        gold_value = _gold_value(gold)
        existing = [
            value for node in graph.get("candidates", [])
            if (value := _node_numeric(node)) is not None]
        existing_supply = (
            gold_value is not None
            and any(_within_tolerance(value, gold_value) for value in existing))
        arms: dict[str, Any] = {}
        for arm in active_arms:
            agent = ARM_AGENT[arm]
            task_id = f"{agent}::{arm}::{row['input_index']}::proposal"
            generations = list(
                responses[task_id].get("generations", []))
            values = _sample_values(generations, relation)
            prefix_hits = {}
            for prefix in range(1, N_SAMPLES + 1):
                window = [value for value in values[:prefix] if value is not None]
                prefix_hits[str(prefix)] = bool(
                    gold_value is not None
                    and any(_within_tolerance(value, gold_value)
                            for value in window))
            parsed = [value for value in values if value is not None]
            new_false_surfaces = len({
                round(value, 6) for value in parsed
                if not any(_within_tolerance(value, other)
                           for other in existing)
                and not (gold_value is not None
                         and _within_tolerance(value, gold_value))})
            arms[arm] = {
                "values": values,
                "parse_failures": sum(value is None for value in values),
                "gold_hit_at_prefix": prefix_hits,
                "new_false_surfaces": new_false_surfaces,
            }
        ledger.append({
            "SubjectEntity": key[0],
            "Relation": relation,
            "gold_value": gold_value,
            "existing_supply": existing_supply,
            "uncertainty_features": row["uncertainty_features"],
            "arms": arms,
        })

    def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        missing = [row for row in rows if not row["existing_supply"]]
        result: dict[str, Any] = {
            "rows": len(rows),
            "previously_missing_supply_rows": len(missing),
        }
        union_new = set()
        for arm in active_arms:
            newly = [
                row for row in missing
                if row["arms"][arm]["gold_hit_at_prefix"][str(N_SAMPLES)]]
            union_new.update(_key(row) for row in newly)
            result[arm] = {
                "newly_covered_missing_rows": len(newly),
                "newly_covered_by_prefix": {
                    str(prefix): sum(
                        row["arms"][arm]["gold_hit_at_prefix"][str(prefix)]
                        for row in missing)
                    for prefix in range(1, N_SAMPLES + 1)},
                "gold_hit_any_row": sum(
                    row["arms"][arm]["gold_hit_at_prefix"][str(N_SAMPLES)]
                    for row in rows),
                "mean_new_false_surfaces_per_row": statistics.mean(
                    row["arms"][arm]["new_false_surfaces"] for row in rows)
                    if rows else 0.0,
                "parse_failure_rate": (
                    sum(row["arms"][arm]["parse_failures"] for row in rows)
                    / (N_SAMPLES * len(rows))) if rows else 0.0,
            }
        result["union_newly_covered_missing_rows"] = len(union_new)
        return result

    reports = {"overall": _summary(ledger)}
    for relation in NUMERIC_RELATIONS:
        reports[relation] = _summary(
            [row for row in ledger if row["Relation"] == relation])

    gate = plan["predeclared_go_no_go"]
    best_single = max(
        reports["overall"][arm]["newly_covered_missing_rows"] for arm in active_arms)
    union_new_rows = reports["overall"]["union_newly_covered_missing_rows"]
    go = (best_single >= int(gate["single_arm_min_new_rows"])
          or union_new_rows >= int(gate["union_min_new_rows"]))

    router_capture = {}
    missing_new_keys = set()
    for row in ledger:
        if not row["existing_supply"] and any(
                row["arms"][arm]["gold_hit_at_prefix"][str(N_SAMPLES)]
                for arm in active_arms):
            missing_new_keys.add(_key(row))
    for feature in ("component_count", "log_value_spread",
                    "gemma_distinct_from_incumbent"):
        ordered = sorted(
            ledger, key=lambda row: -float(
                row["uncertainty_features"][feature]))
        top_half = {_key(row) for row in ordered[:len(ordered) // 2]}
        captured = len(missing_new_keys & top_half)
        router_capture[feature] = {
            "top_half_capture": captured,
            "total_new_rows": len(missing_new_keys),
        }

    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = analysis_dir / "supply_ledger.jsonl"
    write_jsonl_atomic(ledger_path, ledger)
    result = {
        "schema": "numeric-supply-pilot-result-v1",
        "gold_aware": True,
        "contains_labels": True,
        "deployable": False,
        "reports": reports,
        "predeclared_gate": gate,
        "best_single_arm_new_rows": best_single,
        "union_new_rows": union_new_rows,
        "gate_passed": go,
        "router_capture_top_half": router_capture,
        "ledger": str(ledger_path),
        "ledger_sha256": sha256(ledger_path),
    }
    (analysis_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    ledger_path.with_suffix(ledger_path.suffix + ".manifest.json").write_text(
        json.dumps({
            "schema": "numeric-supply-pilot-ledger-manifest-v1",
            "contains_labels": True,
            "gold_aware": True,
            "deployable": False,
            "rows": len(ledger),
            "output_sha256": sha256(ledger_path),
        }, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Numeric supply pilot (train, gold-aware audit)",
        "",
        "Gold labels were opened only after all responses were frozen. This",
        "artifact is diagnostic and nondeployable.",
        "",
        f"- Gate (predeclared): single arm >= {gate['single_arm_min_new_rows']} "
        f"or union >= {gate['union_min_new_rows']} newly covered "
        "supply-missing rows",
        f"- Best single arm: **{best_single}**; union: **{union_new_rows}**",
        f"- **GATE {'PASSED' if go else 'FAILED'}**",
        "",
        "| scope | missing rows | " + " | ".join(active_arms) + " | union |",
        "|---|---:|" + "---:|" * (len(active_arms) + 1),
    ]
    for scope in ("overall", *NUMERIC_RELATIONS):
        report = reports[scope]
        lines.append(
            f"| {scope} | {report['previously_missing_supply_rows']} | "
            + " | ".join(
                str(report[arm]["newly_covered_missing_rows"])
                for arm in active_arms)
            + f" | {report['union_newly_covered_missing_rows']} |")
    lines += ["", "| scope | arm | false surfaces/row | parse failure rate |",
              "|---|---|---:|---:|"]
    for scope in ("overall", *NUMERIC_RELATIONS):
        for arm in active_arms:
            report = reports[scope][arm]
            lines.append(
                f"| {scope} | {arm} | "
                f"{report['mean_new_false_surfaces_per_row']:.2f} | "
                f"{report['parse_failure_rate']:.3f} |")
    (analysis_dir / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(
        f"supply audit: best_single={best_single} union={union_new_rows} "
        f"gate={'PASSED' if go else 'FAILED'}")
    print(f"report={analysis_dir / 'RESULT.md'}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--source-output-dir", default=str(DEFAULT_SOURCE))
    prep.add_argument("--seed", type=int, default=20260727)
    prep.add_argument("--split", choices=("train", "validation"),
                      default="train")
    prep.add_argument("--arms", default=",".join(ARMS),
                      help="comma-separated subset of arms to generate")
    prep.set_defaults(func=prepare)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--output-dir", required=True)
    audit_parser.set_defaults(func=audit)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.func(arguments))
