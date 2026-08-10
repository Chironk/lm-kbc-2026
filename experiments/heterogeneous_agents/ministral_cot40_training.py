#!/usr/bin/env python3
"""Train-only completion of the Ministral CoT/reasoning-budget factorial.

The repository already contains two immutable controls:

* SyntheticCoT-5, requested 20-word reasoning, N=10; and
* zero-shot, requested 40-word reasoning, N=3.

This experiment generates the one missing arm needed to test their
interaction: SyntheticCoT-5, requested 40-word reasoning, N=10.  One ordered
N=10 response file yields nested N={1,3,5,10} prefixes without further model
calls.  Preparation is label-free; analysis is training-only.  Validation is
structurally absent and must not be opened until a setting is frozen.
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
from experiments.heterogeneous_agents.ministral_cot_ablation import (
    ARM_NAME as COT20_ARM,
    DEFAULT_OUTPUT as DEFAULT_COT20,
    _validate_plan as _validate_cot20_plan,
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
DEFAULT_OUTPUT = RUNS / "ministral_cot40_training_20260729_v1"
DEFAULT_SAMPLING = RUNS / "sampling_reasoning_ablation_20260729_v1"
DEFAULT_SYNTHETIC = ROOT / "data/synthetic_cot_faithful.jsonl"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
SOTA_IMPLEMENTATION = HERE / "sota_pipeline.py"

PLAN_SCHEMA = "ministral-cot40-training-plan-v1"
RESULT_SCHEMA = "ministral-cot40-training-result-v1"
ARM_NAME = "ministral_cot5_cap40_n10"
ZERO40_ARM = "ministral_cap40_n3"
SEED = 20260729
SYNTHETIC_SHOTS = 5
REASONING_WORDS = 40
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


def _cot40_config(base: Mapping[str, Any]) -> dict[str, Any]:
    """Add relation-matched SyntheticCoT while preserving the 40-word arm."""
    value = copy.deepcopy(base)
    agent = _agent(value, MINISTRAL)
    if (
        agent.get("model") != EXPECTED_MODEL
        or agent.get("revision") != EXPECTED_REVISION
        or int(agent.get("synthetic_shots", -1)) != 0
        or agent.get("proposal_output") != "bounded_reasoning"
        or int(agent.get("proposal_reasoning_words", -1))
        != REASONING_WORDS
    ):
        raise ContractError("base arm is not the pinned zero-shot cap-40 arm")
    for item in value["agents"]:
        if item["id"] == MINISTRAL:
            item["role"] = "synthetic_cot_recall"
            item["synthetic_shots"] = SYNTHETIC_SHOTS
            item["demonstration_reasoning"] = True
    return value


def _select_smoke(
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected, seen = [], set()
    for task in tasks:
        relation = str(task["relation"])
        if relation not in seen:
            selected.append(dict(task))
            seen.add(relation)
    if len(selected) != 6:
        raise ContractError("expected one smoke task per relation")
    return selected


def _task_map(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result = {str(task["task_id"]): task for task in tasks}
    if len(result) != len(tasks):
        raise ContractError("duplicate task IDs")
    return result


def _task_parity(
    zero40_tasks: Sequence[Mapping[str, Any]],
    cot20_tasks: Sequence[Mapping[str, Any]],
    cot40_tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Enforce a matched factorial rather than three loosely related runs."""
    zero_by = _task_map(zero40_tasks)
    cot20_by = _task_map(cot20_tasks)
    cot40_by = _task_map(cot40_tasks)
    if (
        not cot40_by
        or set(zero_by) != set(cot40_by)
        or set(cot20_by) != set(cot40_by)
    ):
        raise ContractError("control/CoT task coverage mismatch")

    common = (
        "agent_id",
        "subject",
        "relation",
        "input_index",
        "phase",
        "mode",
        "temperature",
        "proposal_output",
    )
    for task_id, task in cot40_by.items():
        zero = zero_by[task_id]
        cot20 = cot20_by[task_id]
        for field in common:
            if task.get(field) != zero.get(field):
                raise ContractError(
                    f"{task_id}: cap-40 zero-shot mismatch: {field}")
            if task.get(field) != cot20.get(field):
                raise ContractError(
                    f"{task_id}: CoT-20 mismatch: {field}")
        if (
            int(task.get("n_samples", -1)) != N_MAX
            or int(cot20.get("n_samples", -1)) != N_MAX
            or int(zero.get("n_samples", -1)) != 3
        ):
            raise ContractError(f"{task_id}: sample-count contract drift")
        if (
            task.get("max_new_tokens") != zero.get("max_new_tokens")
            or task.get("max_new_tokens") == cot20.get("max_new_tokens")
        ):
            raise ContractError(
                f"{task_id}: cap-40 answer-token budget mismatch")
        shots = [str(value) for value in task.get("shot_subjects", [])]
        cot20_shots = [
            str(value) for value in cot20.get("shot_subjects", [])]
        if (
            len(shots) != SYNTHETIC_SHOTS
            or len(set(shots)) != SYNTHETIC_SHOTS
            or str(task["subject"]) in shots
            or shots != cot20_shots
            or zero.get("shot_subjects")
        ):
            raise ContractError(f"{task_id}: synthetic-shot mismatch")
        prompt = str(task.get("prompt", ""))
        if (
            "PRIVATE RECALL DEMONSTRATIONS:" not in prompt
            or prompt.count("QUESTION:") < SYNTHETIC_SHOTS
            or f"at most {REASONING_WORDS} words" not in prompt
        ):
            raise ContractError(f"{task_id}: CoT-40 prompt contract missing")
        if prompt == str(zero.get("prompt", "")):
            raise ContractError(f"{task_id}: CoT prompt equals zero-shot")
    return {
        "paired_tasks": len(cot40_by),
        "nested_prefixes": list(N_VALUES),
        "shots_per_task": SYNTHETIC_SHOTS,
        "shot_assignment_matches_cot20": True,
        "sampling_matches_cot20": True,
        "reasoning_budget_matches_zero40": True,
        "target_excluded": True,
    }


def _arm(
    *,
    name: str,
    config: Path,
    tasks: Path,
    responses: Path,
    rows: int,
    n_samples: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "agent_id": MINISTRAL,
        "reasoning_word_cap": REASONING_WORDS,
        "n_samples": n_samples,
        "seed": SEED,
        "config": str(config),
        "config_sha256": sha256(config),
        "tasks": str(tasks),
        "tasks_sha256": sha256(tasks),
        "responses": str(responses),
        "rows": rows,
    }


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    sampling = Path(args.sampling_run).resolve()
    cot20 = Path(args.cot20_run).resolve()
    synthetic_path = Path(args.synthetic_cot).resolve()
    sampling_plan = _validate_sampling_plan(sampling)
    cot20_plan = _validate_cot20_plan(cot20, require_responses=True)
    zero_arm = sampling_plan["arms"][ZERO40_ARM]

    base_path = Path(zero_arm["config"])
    config_value = _cot40_config(_json(base_path))
    config_path = output / f"plan/configs/{ARM_NAME}.json"
    _write_json(config_path, config_value)
    loaded = load_agent_config(config_path)
    agent = _agent(loaded, MINISTRAL)

    input_path = Path(sampling_plan["input_rows"])
    inputs = read_jsonl(input_path)
    if len(inputs) != 477 or any(row.get("ObjectEntities") for row in inputs):
        raise ContractError("expected 477 label-free training inputs")
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
    tasks.sort(key=lambda task: (
        int(task["max_new_tokens"]),
        str(task["relation"]),
        int(task["input_index"]),
    ))
    if len(tasks) != 477:
        raise ContractError(f"expected 477 tasks, got {len(tasks)}")
    validate_tasks(tasks, MINISTRAL)
    zero_tasks = read_jsonl(Path(zero_arm["tasks"]))
    cot20_tasks = read_jsonl(Path(cot20_plan["cot_tasks"]))
    parity = _task_parity(zero_tasks, cot20_tasks, tasks)

    tasks_path = output / f"plan/tasks/{ARM_NAME}.jsonl"
    smoke_path = output / f"plan/smoke/{ARM_NAME}.jsonl"
    responses_path = output / f"responses/{ARM_NAME}.jsonl"
    smoke_responses = output / f"smoke_responses/{ARM_NAME}.jsonl"
    write_jsonl_atomic(tasks_path, tasks)
    write_jsonl_atomic(smoke_path, _select_smoke(tasks))

    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "split": "train",
        "validation_opened": False,
        "validation_labels_used": False,
        "selection_target": "train_only_freeze_then_single_validation",
        "seed": SEED,
        "n_values": list(N_VALUES),
        "n_samples": N_MAX,
        "reasoning_word_budget": REASONING_WORDS,
        "reasoning_budget_is_requested_not_hard_truncated": True,
        "synthetic_shots": SYNTHETIC_SHOTS,
        "prompt_policy": "relation_matched_target_excluded_cot5",
        "sampling_run": str(sampling),
        "sampling_plan": str(sampling / "plan/PLAN.json"),
        "sampling_plan_sha256": sha256(sampling / "plan/PLAN.json"),
        "cot20_run": str(cot20),
        "cot20_plan": str(cot20 / "plan/PLAN.json"),
        "cot20_plan_sha256": sha256(cot20 / "plan/PLAN.json"),
        "zero40_arm": ZERO40_ARM,
        "zero40_config": zero_arm["config"],
        "zero40_config_sha256": zero_arm["config_sha256"],
        "zero40_tasks": zero_arm["tasks"],
        "zero40_tasks_sha256": zero_arm["tasks_sha256"],
        "zero40_responses": zero_arm["responses"],
        "zero40_responses_sha256": sha256(Path(zero_arm["responses"])),
        "zero40_response_manifest": str(
            _manifest_path(Path(zero_arm["responses"]))),
        "zero40_response_manifest_sha256": sha256(
            _manifest_path(Path(zero_arm["responses"]))),
        "cot20_config": cot20_plan["cot_config"],
        "cot20_config_sha256": cot20_plan["cot_config_sha256"],
        "cot20_tasks": cot20_plan["cot_tasks"],
        "cot20_tasks_sha256": cot20_plan["cot_tasks_sha256"],
        "cot20_responses": cot20_plan["cot_responses"],
        "cot20_responses_sha256": sha256(
            Path(cot20_plan["cot_responses"])),
        "cot20_response_manifest": str(
            _manifest_path(Path(cot20_plan["cot_responses"]))),
        "cot20_response_manifest_sha256": sha256(
            _manifest_path(Path(cot20_plan["cot_responses"]))),
        "source_graph": sampling_plan["source_graph"],
        "source_graph_sha256": sha256(Path(sampling_plan["source_graph"])),
        "incumbent_implementation": str(SOTA_IMPLEMENTATION),
        "incumbent_implementation_sha256": sha256(SOTA_IMPLEMENTATION),
        "input_rows": str(input_path),
        "input_rows_sha256": sha256(input_path),
        "synthetic_cot": str(synthetic_path),
        "synthetic_cot_sha256": sha256(synthetic_path),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "tasks": str(tasks_path),
        "tasks_sha256": sha256(tasks_path),
        "smoke_tasks": str(smoke_path),
        "smoke_tasks_sha256": sha256(smoke_path),
        "responses": str(responses_path),
        "smoke_responses": str(smoke_responses),
        "rows": len(tasks),
        "proposal_generations": len(tasks) * N_MAX,
        "task_parity": parity,
        "model": agent["model"],
        "revision": agent["revision"],
        "verified_parameter_total": loaded["verified_parameter_total"],
        "parameter_cap": loaded["parameter_cap"],
        "parameter_headroom": loaded["declared_parameter_headroom"],
        "analysis_policy": {
            "sample_axis": "nested_prefixes_n1_n3_n5_n10",
            "primary_support_rule": PRIMARY_POLICY,
            "factorial_comparisons": [
                "cot40_vs_cot20_at_matched_nested_N",
                "cot40_vs_zero40_at_N1_and_N3",
            ],
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
        "reused_cot20_generations": len(tasks) * N_MAX,
        "reused_zero40_generations": len(tasks) * 3,
        "nested_prefixes": list(N_VALUES),
        "parameter_total": loaded["verified_parameter_total"],
        "parameter_headroom": loaded["declared_parameter_headroom"],
    }, indent=2, sort_keys=True))
    return 0


def _validate_plan(
    output: Path,
    *,
    require_responses: bool,
) -> dict[str, Any]:
    path = output / "plan/PLAN.json"
    plan = _json(path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or tuple(plan.get("n_values", [])) != N_VALUES
        or int(plan.get("n_samples", -1)) != N_MAX
        or int(plan.get("reasoning_word_budget", -1))
        != REASONING_WORDS
        or int(plan.get("synthetic_shots", -1)) != SYNTHETIC_SHOTS
    ):
        raise ContractError("invalid Ministral CoT-40 training plan")
    frozen = (
        "sampling_plan",
        "cot20_plan",
        "zero40_config",
        "zero40_tasks",
        "zero40_responses",
        "zero40_response_manifest",
        "cot20_config",
        "cot20_tasks",
        "cot20_responses",
        "cot20_response_manifest",
        "source_graph",
        "incumbent_implementation",
        "input_rows",
        "synthetic_cot",
        "config",
        "tasks",
        "smoke_tasks",
        "implementation",
    )
    for field in frozen:
        if sha256(Path(plan[field])) != plan[f"{field}_sha256"]:
            raise ContractError(f"frozen plan artifact changed: {field}")
    sampling = _validate_sampling_plan(Path(plan["sampling_run"]))
    cot20 = _validate_cot20_plan(
        Path(plan["cot20_run"]), require_responses=True)
    if sampling["arms"][ZERO40_ARM]["responses"] != plan[
        "zero40_responses"
    ]:
        raise ContractError("zero-shot cap-40 control path drift")
    tasks = read_jsonl(Path(plan["tasks"]))
    validate_tasks(tasks, MINISTRAL)
    _task_parity(
        read_jsonl(Path(plan["zero40_tasks"])),
        read_jsonl(Path(plan["cot20_tasks"])),
        tasks,
    )
    if require_responses:
        _response_rows(_arm(
            name=ARM_NAME,
            config=Path(plan["config"]),
            tasks=Path(plan["tasks"]),
            responses=Path(plan["responses"]),
            rows=int(plan["rows"]),
            n_samples=N_MAX,
        ))
    return plan


def _standalone_oracle(
    evidence: Mapping[tuple[str, str], Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, float]:
    candidates, gold = [], []
    for source in source_rows:
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


def _evaluate(
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    n_values: Sequence[int],
    reasoning_words: int,
    source_rows: Sequence[Mapping[str, Any]],
    incumbent_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gold_by = {_key(row): row for row in gold_rows}
    result = {}
    for n in n_values:
        evidence, telemetry = _prefix_evidence(
            responses, int(n), reasoning_words)
        item = _evaluate_prefix(
            n=int(n),
            evidence=evidence,
            telemetry=telemetry,
            source_rows=source_rows,
            incumbent_rows=incumbent_rows,
            gold_rows=gold_rows,
        )
        item["standalone_candidate_oracle_scores"] = _standalone_oracle(
            evidence, source_rows, gold_by)
        result[str(n)] = item
    return result


def _score_delta(
    right: Mapping[str, float],
    left: Mapping[str, float],
) -> dict[str, float]:
    if set(left) != set(right):
        raise ContractError("score relation sets differ")
    return {key: float(right[key]) - float(left[key]) for key in left}


def _comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    n_values: Sequence[int],
) -> dict[str, Any]:
    result = {}
    for n in n_values:
        key = str(n)
        l_item, r_item = left[key], right[key]
        l_policy = l_item["policies"][PRIMARY_POLICY]
        r_policy = r_item["policies"][PRIMARY_POLICY]
        l_unique = {
            tuple(value)
            for value in l_item["unique_correct_rows_beyond_source_graph"]
        }
        r_unique = {
            tuple(value)
            for value in r_item["unique_correct_rows_beyond_source_graph"]
        }
        result[key] = {
            "standalone_f1_delta": _score_delta(
                r_policy["standalone_scores"],
                l_policy["standalone_scores"],
            ),
            "residual_f1_delta": _score_delta(
                r_policy["residual_scores"],
                l_policy["residual_scores"],
            ),
            "standalone_oracle_delta": _score_delta(
                r_item["standalone_candidate_oracle_scores"],
                l_item["standalone_candidate_oracle_scores"],
            ),
            "combined_oracle_delta": _score_delta(
                r_item["combined_candidate_oracle_scores"],
                l_item["combined_candidate_oracle_scores"],
            ),
            "candidate_truth_auroc_delta": (
                float(r_item["candidate_truth_auroc"])
                - float(l_item["candidate_truth_auroc"])
            ),
            "parse_failure_rate_delta": (
                float(r_item["telemetry"]["parse_failure_rate"])
                - float(l_item["telemetry"]["parse_failure_rate"])
            ),
            "right_only_correct_rows": [
                list(value) for value in sorted(r_unique - l_unique)
            ],
            "left_only_correct_rows": [
                list(value) for value in sorted(l_unique - r_unique)
            ],
            "shared_unique_correct_rows": len(l_unique & r_unique),
        }
    return result


def _markdown(result: Mapping[str, Any]) -> str:
    c20 = result["conditions"]["cot20"]
    c40 = result["conditions"]["cot40"]
    z40 = result["conditions"]["zero40"]
    lines = [
        "# Ministral CoT × requested-reasoning training ablation",
        "",
        "Training-only audit; validation was structurally absent. CoT-20 and "
        "CoT-40 use the same checkpoint, subjects, SyntheticCoT examples, "
        "seed, temperature, and ordered N=10 sampling stream. The word budget "
        "is requested in the prompt and measured, not hard-truncated.",
        "",
        f"- Competition-pipeline OOF anchor: "
        f"**{result['incumbent_scores'][POOLED]:.6f}**",
        f"- Frozen source-graph oracle: "
        f"**{result['source_candidate_oracle_scores'][POOLED]:.6f}**",
        "",
        "## Matched SyntheticCoT-5: requested 20 vs 40 words",
        "",
        "| N | CoT-20 standalone F1 | CoT-40 standalone F1 | delta | "
        "CoT-20 residual F1 | CoT-40 residual F1 | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in N_VALUES:
        key = str(n)
        l = c20[key]["policies"][PRIMARY_POLICY]
        r = c40[key]["policies"][PRIMARY_POLICY]
        ls, rs = l["standalone_scores"][POOLED], r[
            "standalone_scores"][POOLED]
        lr, rr = l["residual_scores"][POOLED], r["residual_scores"][POOLED]
        lines.append(
            f"| {n} | {ls:.4f} | {rs:.4f} | {rs-ls:+.4f} | "
            f"{lr:.4f} | {rr:.4f} | {rr-lr:+.4f} |")
    lines.extend([
        "",
        "## Matched requested 40 words: zero-shot vs SyntheticCoT-5",
        "",
        "| N | zero-shot standalone F1 | CoT-5 standalone F1 | delta | "
        "zero-shot residual F1 | CoT-5 residual F1 | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for n in (1, 3):
        key = str(n)
        l = z40[key]["policies"][PRIMARY_POLICY]
        r = c40[key]["policies"][PRIMARY_POLICY]
        ls, rs = l["standalone_scores"][POOLED], r[
            "standalone_scores"][POOLED]
        lr, rr = l["residual_scores"][POOLED], r["residual_scores"][POOLED]
        lines.append(
            f"| {n} | {ls:.4f} | {rs:.4f} | {rs-ls:+.4f} | "
            f"{lr:.4f} | {rr:.4f} | {rr-lr:+.4f} |")
    lines.extend([
        "",
        "## Decision rule",
        "",
        "The train-only screen exposes every support rule and nested prefix. "
        "No validation setting is chosen here. A final N/support policy must "
        "be frozen from these training diagnostics before one validation "
        "confirmation.",
        "",
    ])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output, require_responses=True)
    sampling = _validate_sampling_plan(Path(plan["sampling_run"]))
    cot20 = _validate_cot20_plan(
        Path(plan["cot20_run"]), require_responses=True)
    zero_arm = sampling["arms"][ZERO40_ARM]
    zero_responses = _response_rows(zero_arm)
    cot20_responses = _response_rows({
        "name": COT20_ARM,
        "agent_id": MINISTRAL,
        "reasoning_word_cap": 20,
        "n_samples": N_MAX,
        "seed": SEED,
        "config": cot20["cot_config"],
        "config_sha256": cot20["cot_config_sha256"],
        "tasks": cot20["cot_tasks"],
        "tasks_sha256": cot20["cot_tasks_sha256"],
        "responses": cot20["cot_responses"],
        "rows": cot20["rows"],
    })
    cot40_responses = _response_rows(_arm(
        name=ARM_NAME,
        config=Path(plan["config"]),
        tasks=Path(plan["tasks"]),
        responses=Path(plan["responses"]),
        rows=int(plan["rows"]),
        n_samples=N_MAX,
    ))
    source_rows = _source_rows(Path(plan["source_graph"]))
    incumbent_rows, incumbent_detail = compose_competition_train_oof()
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    if len(gold_rows) != 477:
        raise ContractError("expected 477 training gold rows")
    conditions = {
        "zero40": _evaluate(
            zero_responses,
            n_values=(1, 3),
            reasoning_words=REASONING_WORDS,
            source_rows=source_rows,
            incumbent_rows=incumbent_rows,
            gold_rows=gold_rows,
        ),
        "cot20": _evaluate(
            cot20_responses,
            n_values=N_VALUES,
            reasoning_words=20,
            source_rows=source_rows,
            incumbent_rows=incumbent_rows,
            gold_rows=gold_rows,
        ),
        "cot40": _evaluate(
            cot40_responses,
            n_values=N_VALUES,
            reasoning_words=REASONING_WORDS,
            source_rows=source_rows,
            incumbent_rows=incumbent_rows,
            gold_rows=gold_rows,
        ),
    }
    comparisons = {
        "cot40_minus_cot20": _comparison(
            conditions["cot20"], conditions["cot40"], n_values=N_VALUES),
        "cot40_minus_zero40": _comparison(
            conditions["zero40"], conditions["cot40"], n_values=(1, 3)),
    }
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
        "incumbent_scores": score(incumbent_rows, gold_rows),
        "source_candidate_oracle_scores": score(
            oracle_rows(source_rows, gold_rows), gold_rows),
        "conditions": conditions,
        "comparisons": comparisons,
    }
    analysis = output / "analysis"
    _write_json(analysis / "RESULT.json", result)
    (analysis / "RESULT.md").write_text(_markdown(result))
    summary = {}
    for n in N_VALUES:
        item = comparisons["cot40_minus_cot20"][str(n)]
        summary[f"n{n}_cot40_minus_cot20_standalone"] = item[
            "standalone_f1_delta"][POOLED]
        summary[f"n{n}_cot40_minus_cot20_residual"] = item[
            "residual_f1_delta"][POOLED]
    print(json.dumps({
        "result": str(analysis / "RESULT.md"),
        **summary,
    }, indent=2, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_path = output / "plan/PLAN.json"
    if not plan_path.is_file():
        print(json.dumps({"prepared": False}, indent=2))
        return 1
    plan = _validate_plan(output, require_responses=False)
    response = Path(plan["responses"])
    rows = len(read_jsonl(response)) if response.is_file() else 0
    print(json.dumps({
        "prepared": True,
        "rows_expected": int(plan["rows"]),
        "rows_complete": rows,
        "response_manifest": _manifest_path(response).is_file(),
        "analysis_complete": (output / "analysis/RESULT.json").is_file(),
        "resumable": rows < int(plan["rows"]),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    prep.add_argument("--sampling-run", default=str(DEFAULT_SAMPLING))
    prep.add_argument("--cot20-run", default=str(DEFAULT_COT20))
    prep.add_argument("--synthetic-cot", default=str(DEFAULT_SYNTHETIC))
    prep.set_defaults(function=prepare)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    analysis.add_argument("--train-gold", default=str(DEFAULT_GOLD))
    analysis.set_defaults(function=analyze)
    stat = commands.add_parser("status")
    stat.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    stat.set_defaults(function=status)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
