#!/usr/bin/env python3
"""Matched zero-shot versus synthetic-CoT ablation for Ministral.

The completed zero-shot N=10 arm from ``sampling_reasoning_ablation`` is the
immutable control.  This experiment generates exactly one new arm with the
same checkpoint, task IDs, subjects, seed, temperature, sample count, visible
reasoning budget, and answer-token budget.  The only intended prompt changes
are:

* five deterministic, relation-matched, target-excluded SyntheticCoT examples;
* demonstration reasoning is visible in those examples; and
* the agent role records synthetic-CoT recall rather than direct recall.

Analysis is training-only and uses nested prefixes N={1,3,5,10}.  It reports
two distinct outcomes:

1. Ministral by itself (standalone prediction F1 and candidate oracle);
2. Ministral added to the frozen Qwen/Gemma candidate graph and exact
   competition-pipeline OOF incumbent (combined oracle and residual F1).

Validation is structurally absent.  Candidate oracles are gold-aware,
nondeployable diagnostics; they never become prediction selectors.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.assemble_and_audit import (
    oracle_rows,
    score,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    build_agent_tasks,
    load_agent_config,
    load_synthetic_by_relation,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.ministral_candidate_supply import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    MINISTRAL,
    _agent,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks
from experiments.heterogeneous_agents.sampling_reasoning_ablation import (
    N_MAX,
    N_VALUES,
    POOLED,
    _evaluate_prefix,
    _key,
    _prefix_evidence,
    _response_rows,
    _source_rows,
    _validate_plan as _validate_sampling_plan,
)
from experiments.heterogeneous_agents.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    compose_competition_train_oof,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_OUTPUT = RUNS / "ministral_cot_ablation_20260729_v1"
DEFAULT_CONTROL = RUNS / "sampling_reasoning_ablation_20260729_v1"
DEFAULT_SYNTHETIC = ROOT / "data/synthetic_cot_faithful.jsonl"
DEFAULT_GOLD = ROOT / "data/train.jsonl"

PLAN_SCHEMA = "ministral-cot-ablation-plan-v1"
RESULT_SCHEMA = "ministral-cot-ablation-result-v1"
ARM_NAME = "ministral_cot5_cap20_n10"
CONTROL_ARM = "ministral_cap20_n10"
SEED = 20260729
SYNTHETIC_SHOTS = 5
REASONING_WORDS = 20
PRIMARY_POLICY = "two_thirds"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _cot_config(base: Mapping[str, Any]) -> dict[str, Any]:
    """Return a config whose only agent-policy change is Ministral CoT-5."""
    value = copy.deepcopy(base)
    agent = _agent(value, MINISTRAL)
    if (
        agent.get("model") != EXPECTED_MODEL
        or agent.get("revision") != EXPECTED_REVISION
        or int(agent.get("synthetic_shots", -1)) != 0
        or agent.get("proposal_output") != "bounded_reasoning"
        or int(agent.get("proposal_reasoning_words", -1)) != REASONING_WORDS
    ):
        raise ContractError("base Ministral arm is not the pinned zero-shot control")
    for item in value["agents"]:
        if item["id"] != MINISTRAL:
            continue
        item["role"] = "synthetic_cot_recall"
        item["synthetic_shots"] = SYNTHETIC_SHOTS
        item["demonstration_reasoning"] = True
    return value


def _smoke(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected, seen = [], set()
    for task in tasks:
        relation = str(task["relation"])
        if relation not in seen:
            selected.append(dict(task))
            seen.add(relation)
    if len(selected) != 6:
        raise ContractError("expected one proposal smoke task per relation")
    return selected


def _task_parity(
    control_tasks: Sequence[Mapping[str, Any]],
    cot_tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail unless prompt demonstrations are the only operational difference."""
    if (
        not cot_tasks
        or len(control_tasks) != len(cot_tasks)
    ):
        raise ContractError("control/CoT task coverage mismatch")
    control_by = {str(task["task_id"]): task for task in control_tasks}
    cot_by = {str(task["task_id"]): task for task in cot_tasks}
    if set(control_by) != set(cot_by):
        raise ContractError("control/CoT task IDs differ")
    frozen_fields = (
        "agent_id",
        "subject",
        "relation",
        "input_index",
        "phase",
        "mode",
        "n_samples",
        "temperature",
        "max_new_tokens",
        "proposal_output",
    )
    prompt_changes = 0
    for task_id, cot in cot_by.items():
        control = control_by[task_id]
        for field in frozen_fields:
            if cot.get(field) != control.get(field):
                raise ContractError(
                    f"{task_id}: non-prompt field changed: {field}")
        shots = list(cot.get("shot_subjects", []))
        if (
            len(shots) != SYNTHETIC_SHOTS
            or len(shots) != len(set(shots))
            or str(cot["subject"]) in shots
            or control.get("shot_subjects")
        ):
            raise ContractError(f"{task_id}: invalid matched shot assignment")
        prompt = str(cot.get("prompt", ""))
        if (
            "PRIVATE RECALL DEMONSTRATIONS:" not in prompt
            or prompt.count("QUESTION:") < SYNTHETIC_SHOTS
            or f"at most {REASONING_WORDS} words" not in prompt
        ):
            raise ContractError(f"{task_id}: CoT prompt contract missing")
        if prompt == str(control.get("prompt", "")):
            raise ContractError(f"{task_id}: CoT prompt equals zero-shot prompt")
        prompt_changes += 1
    return {
        "paired_tasks": len(cot_tasks),
        "prompt_changes": prompt_changes,
        "frozen_fields": list(frozen_fields),
        "shots_per_task": SYNTHETIC_SHOTS,
        "target_excluded": True,
        "subject_distinct": True,
    }


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    control = Path(args.control_run).resolve()
    synthetic_path = Path(args.synthetic_cot).resolve()
    control_plan = _validate_sampling_plan(control)
    control_arm = control_plan["arms"][CONTROL_ARM]
    control_config_path = Path(control_arm["config"])
    base_config = _json(control_config_path)
    cot_config = _cot_config(base_config)
    config_path = output / f"plan/configs/{ARM_NAME}.json"
    _write_json(config_path, cot_config)
    loaded = load_agent_config(config_path)
    agent = _agent(loaded, MINISTRAL)

    input_path = Path(control_plan["input_rows"])
    inputs = read_jsonl(input_path)
    if len(inputs) != 477 or any(row.get("ObjectEntities") for row in inputs):
        raise ContractError("control inputs are not 477 label-free rows")
    synthetic = load_synthetic_by_relation(synthetic_path)
    tasks = [
        task for task in build_agent_tasks(
            inputs,
            agent,
            synthetic,
            seed=SEED,
            n_proposals=N_MAX,
        )
        if task["phase"] == "propose"
    ]
    if len(tasks) != 477:
        raise ContractError(f"expected 477 CoT proposal tasks, got {len(tasks)}")
    validate_tasks(tasks, MINISTRAL)
    control_tasks = read_jsonl(Path(control_arm["tasks"]))
    parity = _task_parity(control_tasks, tasks)

    task_path = output / f"plan/tasks/{ARM_NAME}.jsonl"
    smoke_path = output / f"plan/smoke/{ARM_NAME}.jsonl"
    response_path = output / f"responses/{ARM_NAME}.jsonl"
    smoke_response_path = output / f"smoke_responses/{ARM_NAME}.jsonl"
    write_jsonl_atomic(task_path, tasks)
    write_jsonl_atomic(smoke_path, _smoke(tasks))

    control_response = Path(control_arm["responses"])
    control_manifest = _manifest_path(control_response)
    if not control_response.is_file() or not control_manifest.is_file():
        raise ContractError("completed zero-shot N=10 control is missing")
    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "split": "train",
        "validation_opened": False,
        "validation_labels_used": False,
        "selection_target": "matched_train_only_ablation",
        "seed": SEED,
        "n_values": list(N_VALUES),
        "n_samples": N_MAX,
        "reasoning_word_budget": REASONING_WORDS,
        "synthetic_shots": SYNTHETIC_SHOTS,
        "demonstration_reasoning": True,
        "prompt_policy": "relation_matched_target_excluded_cot5",
        "control_run": str(control),
        "control_plan": str(control / "plan/PLAN.json"),
        "control_plan_sha256": sha256(control / "plan/PLAN.json"),
        "control_arm": CONTROL_ARM,
        "control_config": str(control_config_path),
        "control_config_sha256": sha256(control_config_path),
        "control_tasks": control_arm["tasks"],
        "control_tasks_sha256": sha256(Path(control_arm["tasks"])),
        "control_responses": str(control_response),
        "control_responses_sha256": sha256(control_response),
        "control_response_manifest": str(control_manifest),
        "control_response_manifest_sha256": sha256(control_manifest),
        "source_graph": control_plan["source_graph"],
        "source_graph_sha256": sha256(Path(control_plan["source_graph"])),
        "input_rows": str(input_path),
        "input_rows_sha256": sha256(input_path),
        "synthetic_cot": str(synthetic_path),
        "synthetic_cot_sha256": sha256(synthetic_path),
        "cot_config": str(config_path),
        "cot_config_sha256": sha256(config_path),
        "cot_tasks": str(task_path),
        "cot_tasks_sha256": sha256(task_path),
        "cot_smoke_tasks": str(smoke_path),
        "cot_smoke_tasks_sha256": sha256(smoke_path),
        "cot_responses": str(response_path),
        "cot_smoke_responses": str(smoke_response_path),
        "rows": len(tasks),
        "proposal_generations": len(tasks) * N_MAX,
        "task_parity": parity,
        "model": agent["model"],
        "revision": agent["revision"],
        "verified_parameter_total": loaded["verified_parameter_total"],
        "parameter_cap": loaded["parameter_cap"],
        "parameter_headroom": loaded["declared_parameter_headroom"],
        "analysis_policy": {
            "sample_axis": "nested_prefixes_from_matched_n10_runs",
            "primary_support_rule": PRIMARY_POLICY,
            "standalone_endpoint": "model_prediction_and_candidate_oracle",
            "integrated_endpoint":
                "frozen_qwen_gemma_graph_and_competition_oof_incumbent",
            "validation_selection": False,
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    plan_path = output / "plan/PLAN.json"
    _write_json(plan_path, plan)
    print(json.dumps({
        "plan": str(plan_path),
        "rows": len(tasks),
        "new_proposal_generations": len(tasks) * N_MAX,
        "reused_zero_shot_generations": len(tasks) * N_MAX,
        "shots_per_task": SYNTHETIC_SHOTS,
        "parameter_total": loaded["verified_parameter_total"],
        "parameter_headroom": loaded["declared_parameter_headroom"],
    }, indent=2, sort_keys=True))
    return 0


def _validate_plan(output: Path, *, require_responses: bool) -> dict[str, Any]:
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or tuple(plan.get("n_values", [])) != N_VALUES
        or int(plan.get("n_samples", -1)) != N_MAX
        or int(plan.get("synthetic_shots", -1)) != SYNTHETIC_SHOTS
        or int(plan.get("reasoning_word_budget", -1)) != REASONING_WORDS
    ):
        raise ContractError("invalid Ministral CoT ablation plan")
    frozen = (
        "control_plan",
        "control_config",
        "control_tasks",
        "control_responses",
        "control_response_manifest",
        "source_graph",
        "input_rows",
        "synthetic_cot",
        "cot_config",
        "cot_tasks",
        "cot_smoke_tasks",
        "implementation",
    )
    for field in frozen:
        if sha256(Path(plan[field])) != plan[f"{field}_sha256"]:
            raise ContractError(f"frozen plan artifact changed: {field}")
    control_plan = _validate_sampling_plan(Path(plan["control_run"]))
    if (
        control_plan["arms"][CONTROL_ARM]["responses"]
        != plan["control_responses"]
    ):
        raise ContractError("control arm path drift")
    cot_tasks = read_jsonl(Path(plan["cot_tasks"]))
    validate_tasks(cot_tasks, MINISTRAL)
    _task_parity(read_jsonl(Path(plan["control_tasks"])), cot_tasks)
    if require_responses:
        arm = {
            "name": ARM_NAME,
            "agent_id": MINISTRAL,
            "reasoning_word_cap": REASONING_WORDS,
            "n_samples": N_MAX,
            "seed": SEED,
            "config": plan["cot_config"],
            "config_sha256": plan["cot_config_sha256"],
            "tasks": plan["cot_tasks"],
            "tasks_sha256": plan["cot_tasks_sha256"],
            "responses": plan["cot_responses"],
            "rows": plan["rows"],
        }
        _response_rows(arm)
    return plan


def _standalone_oracle(
    evidence: Mapping[tuple[str, str], Mapping[str, Any]],
    source_order: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, float]:
    candidates, gold = [], []
    for source in source_order:
        key = _key(source)
        candidates.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "candidates": [
                {"item": item}
                for item in evidence[key]["displays"].values()
            ],
        })
        gold.append(dict(gold_by[key]))
    return score(oracle_rows(candidates, gold), gold)


def _evaluate_condition(
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    incumbent_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gold_by = {_key(row): row for row in gold_rows}
    result = {}
    for n in N_VALUES:
        evidence, telemetry = _prefix_evidence(
            responses, n, REASONING_WORDS)
        evaluated = _evaluate_prefix(
            n=n,
            evidence=evidence,
            telemetry=telemetry,
            source_rows=source_rows,
            incumbent_rows=incumbent_rows,
            gold_rows=gold_rows,
        )
        evaluated["standalone_candidate_oracle_scores"] = _standalone_oracle(
            evidence, source_rows, gold_by)
        result[str(n)] = evaluated
    return result


def _difference(
    cot: Mapping[str, float],
    control: Mapping[str, float],
) -> dict[str, float]:
    if set(cot) != set(control):
        raise ContractError("score relation sets differ")
    return {key: float(cot[key]) - float(control[key]) for key in cot}


def _paired(
    control: Mapping[str, Any],
    cot: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    for n in N_VALUES:
        key = str(n)
        left, right = control[key], cot[key]
        left_unique = {
            tuple(item) for item in left["unique_correct_rows_beyond_source_graph"]
        }
        right_unique = {
            tuple(item) for item in right["unique_correct_rows_beyond_source_graph"]
        }
        left_policy = left["policies"][PRIMARY_POLICY]
        right_policy = right["policies"][PRIMARY_POLICY]
        result[key] = {
            "standalone_prediction_delta": _difference(
                right_policy["standalone_scores"],
                left_policy["standalone_scores"],
            ),
            "integrated_residual_delta": _difference(
                right_policy["residual_scores"],
                left_policy["residual_scores"],
            ),
            "standalone_candidate_oracle_delta": _difference(
                right["standalone_candidate_oracle_scores"],
                left["standalone_candidate_oracle_scores"],
            ),
            "combined_candidate_oracle_delta": _difference(
                right["combined_candidate_oracle_scores"],
                left["combined_candidate_oracle_scores"],
            ),
            "candidate_truth_auroc_delta": (
                float(right["candidate_truth_auroc"])
                - float(left["candidate_truth_auroc"])
            ),
            "parse_failure_rate_delta": (
                float(right["telemetry"]["parse_failure_rate"])
                - float(left["telemetry"]["parse_failure_rate"])
            ),
            "control_unique_correct_rows": len(left_unique),
            "cot_unique_correct_rows": len(right_unique),
            "unique_correct_rows_gained": [
                list(item) for item in sorted(right_unique - left_unique)
            ],
            "unique_correct_rows_lost": [
                list(item) for item in sorted(left_unique - right_unique)
            ],
            "unique_correct_rows_shared": len(left_unique & right_unique),
        }
    return result


def _markdown(result: Mapping[str, Any]) -> str:
    control = result["conditions"]["zero_shot"]
    cot = result["conditions"]["cot5"]
    paired = result["paired"]
    lines = [
        "# Matched Ministral synthetic-CoT ablation",
        "",
        "Training-only OOF audit. Validation was not opened. The zero-shot "
        "control and CoT-5 arm use the same checkpoint, seed, sample count, "
        "reasoning budget, temperature, subjects, and task IDs.",
        "",
        f"- Competition-pipeline OOF anchor: "
        f"**{result['incumbent_scores'][POOLED]:.6f}**",
        f"- Frozen Qwen/Gemma source-graph oracle: "
        f"**{result['source_candidate_oracle_scores'][POOLED]:.6f}**",
        f"- Synthetic demonstrations per CoT task: **{SYNTHETIC_SHOTS}**",
        "",
        "## Standalone Ministral",
        "",
        "| N | zero-shot F1 | CoT-5 F1 | delta | zero-shot oracle | "
        "CoT-5 oracle | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in N_VALUES:
        key = str(n)
        z = control[key]
        c = cot[key]
        zf = z["policies"][PRIMARY_POLICY]["standalone_scores"][POOLED]
        cf = c["policies"][PRIMARY_POLICY]["standalone_scores"][POOLED]
        zo = z["standalone_candidate_oracle_scores"][POOLED]
        co = c["standalone_candidate_oracle_scores"][POOLED]
        lines.append(
            f"| {n} | {zf:.4f} | {cf:.4f} | {cf-zf:+.4f} | "
            f"{zo:.4f} | {co:.4f} | {co-zo:+.4f} |")
    lines.extend([
        "",
        "## Added to the frozen Qwen/Gemma system",
        "",
        "| N | zero-shot residual F1 | CoT-5 residual F1 | delta | "
        "zero-shot combined oracle | CoT-5 combined oracle | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for n in N_VALUES:
        key = str(n)
        z = control[key]
        c = cot[key]
        zf = z["policies"][PRIMARY_POLICY]["residual_scores"][POOLED]
        cf = c["policies"][PRIMARY_POLICY]["residual_scores"][POOLED]
        zo = z["combined_candidate_oracle_scores"][POOLED]
        co = c["combined_candidate_oracle_scores"][POOLED]
        lines.append(
            f"| {n} | {zf:.4f} | {cf:.4f} | {cf-zf:+.4f} | "
            f"{zo:.4f} | {co:.4f} | {co-zo:+.4f} |")
    n10 = paired[str(N_MAX)]
    lines.extend([
        "",
        "## N=10 complementarity",
        "",
        f"- Zero-shot unique correct rows beyond source graph: "
        f"**{n10['control_unique_correct_rows']}**",
        f"- CoT-5 unique correct rows beyond source graph: "
        f"**{n10['cot_unique_correct_rows']}**",
        f"- CoT-only correct rows: "
        f"**{len(n10['unique_correct_rows_gained'])}**",
        f"- Correct rows lost relative to zero-shot: "
        f"**{len(n10['unique_correct_rows_lost'])}**",
        f"- Candidate-truth AUROC delta: "
        f"**{n10['candidate_truth_auroc_delta']:+.6f}**",
        f"- Parse-failure-rate delta: "
        f"**{n10['parse_failure_rate_delta']:+.6f}**",
        "",
        "The residual policy is predeclared and identical across conditions: "
        "strings add candidates with at least ceil(2N/3) support to the exact "
        "OOF incumbent; numeric rows replace it only for one uniquely supported "
        "top candidate. Candidate oracles are gold-aware and nondeployable.",
        "",
    ])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output, require_responses=True)
    control_plan = _validate_sampling_plan(Path(plan["control_run"]))
    control_arm = control_plan["arms"][CONTROL_ARM]
    control_responses = _response_rows(control_arm)
    cot_arm = {
        "name": ARM_NAME,
        "agent_id": MINISTRAL,
        "reasoning_word_cap": REASONING_WORDS,
        "n_samples": N_MAX,
        "seed": SEED,
        "config": plan["cot_config"],
        "config_sha256": plan["cot_config_sha256"],
        "tasks": plan["cot_tasks"],
        "tasks_sha256": plan["cot_tasks_sha256"],
        "responses": plan["cot_responses"],
        "rows": plan["rows"],
    }
    cot_responses = _response_rows(cot_arm)
    source_rows = _source_rows(Path(plan["source_graph"]))
    incumbent_rows, incumbent_detail = compose_competition_train_oof()
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    if len(gold_rows) != 477:
        raise ContractError("expected 477 training gold rows")
    incumbent_scores = score(incumbent_rows, gold_rows)
    source_oracle_scores = score(
        oracle_rows(source_rows, gold_rows), gold_rows)
    conditions = {
        "zero_shot": _evaluate_condition(
            control_responses,
            source_rows=source_rows,
            incumbent_rows=incumbent_rows,
            gold_rows=gold_rows,
        ),
        "cot5": _evaluate_condition(
            cot_responses,
            source_rows=source_rows,
            incumbent_rows=incumbent_rows,
            gold_rows=gold_rows,
        ),
    }
    paired = _paired(conditions["zero_shot"], conditions["cot5"])
    n10 = paired[str(N_MAX)]
    result = {
        "schema": RESULT_SCHEMA,
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "split": "train",
        "validation_opened": False,
        "validation_labels_used": False,
        "selection_must_be_frozen_before_validation": True,
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "incumbent_pipeline": COMPETITION_PIPELINE_ID,
        "incumbent_detail": incumbent_detail,
        "incumbent_scores": incumbent_scores,
        "source_candidate_oracle_scores": source_oracle_scores,
        "conditions": conditions,
        "paired": paired,
        "n10_summary": {
            "standalone_f1_delta":
                n10["standalone_prediction_delta"][POOLED],
            "standalone_oracle_delta":
                n10["standalone_candidate_oracle_delta"][POOLED],
            "integrated_residual_f1_delta":
                n10["integrated_residual_delta"][POOLED],
            "combined_oracle_delta":
                n10["combined_candidate_oracle_delta"][POOLED],
            "candidate_truth_auroc_delta":
                n10["candidate_truth_auroc_delta"],
            "parse_failure_rate_delta":
                n10["parse_failure_rate_delta"],
            "cot_only_correct_rows":
                len(n10["unique_correct_rows_gained"]),
            "zero_only_correct_rows":
                len(n10["unique_correct_rows_lost"]),
        },
    }
    analysis = output / "analysis"
    _write_json(analysis / "RESULT.json", result)
    (analysis / "RESULT.md").write_text(_markdown(result))
    print(json.dumps({
        "result": str(analysis / "RESULT.md"),
        **result["n10_summary"],
    }, indent=2, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_path = output / "plan/PLAN.json"
    if not plan_path.is_file():
        print(json.dumps({"prepared": False}, indent=2))
        return 1
    plan = _validate_plan(output, require_responses=False)
    response = Path(plan["cot_responses"])
    rows = len(read_jsonl(response)) if response.is_file() else 0
    manifest = _manifest_path(response)
    result = output / "analysis/RESULT.json"
    print(json.dumps({
        "prepared": True,
        "rows_expected": int(plan["rows"]),
        "rows_complete": rows,
        "response_manifest": manifest.is_file(),
        "analysis_complete": result.is_file(),
        "resumable": rows < int(plan["rows"]),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    prep.add_argument("--control-run", default=str(DEFAULT_CONTROL))
    prep.add_argument("--synthetic-cot", default=str(DEFAULT_SYNTHETIC))
    prep.set_defaults(function=prepare)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    analysis.add_argument("--train-gold", default=str(DEFAULT_GOLD))
    analysis.set_defaults(function=analyze)
    stat = subparsers.add_parser("status")
    stat.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    stat.set_defaults(function=status)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
