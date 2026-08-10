#!/usr/bin/env python3
"""Paired sampling-count and visible-reasoning ablation.

The experiment has two orthogonal axes:

* at the frozen 20-word contract, generate Nmax=10 once and evaluate the
  nested prefixes N={1,3,5,10};
* at fixed N=3, compare visible-reasoning caps {20,40,80}.

Only proposal tasks are generated.  Preparation is label-free.  Analysis opens
the training labels and anchors every residual prediction to the canonical
competition-pipeline subject-grouped OOF artifact.  Validation is structurally
absent so the selected setting can be frozen before one later confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence

from evaluate import RELATION_TYPE
from experiments.heterogeneous_agents.assemble_and_audit import (
    oracle_rows,
    score,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    build_agent_tasks,
    canonical_key,
    load_agent_config,
    load_synthetic_by_relation,
    proposal_parse_status,
    read_jsonl,
    sha256,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.dual_model_validation import (
    GEMMA,
    QWEN,
    _agent,
    apply_prompt_policy,
)
from experiments.heterogeneous_agents.ministral_candidate_supply import (
    EXPECTED_MODEL as MINISTRAL_MODEL,
    EXPECTED_REVISION as MINISTRAL_REVISION,
    MINISTRAL,
    _has_correct,
    _items,
    _source_manifest,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks
from experiments.heterogeneous_agents.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    compose_competition_train_oof,
)
from experiments.heterogeneous_agents.wide_candidate_discrimination import (
    _binary_auroc,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_OUTPUT = RUNS / "sampling_reasoning_ablation_20260729_v1"
DEFAULT_CONFIG = ROOT / "configs/final/portfolio_supply.json"
DEFAULT_SOURCE_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/graphs/train_graph.jsonl"
)
DEFAULT_SYNTHETIC = ROOT / "data/synthetic_cot_faithful.jsonl"
DEFAULT_GOLD = ROOT / "data/train.jsonl"

PLAN_SCHEMA = "sampling-reasoning-ablation-plan-v1"
RESULT_SCHEMA = "sampling-reasoning-ablation-result-v1"
SEED = 20260729
N_VALUES = (1, 3, 5, 10)
N_MAX = max(N_VALUES)
REASONING_CAPS = (20, 40, 80)
CAP_N = 3
NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}
POOLED = "*** All Relations ***"
REASONING_LINE = re.compile(
    r"^[ \t]*REASONING[ \t]*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Arm:
    name: str
    agent_id: str
    reasoning_word_cap: int
    n_samples: int
    axis: str


def arms() -> tuple[Arm, ...]:
    result = []
    for agent_id, short in (
        (GEMMA, "gemma"),
        (MINISTRAL, "ministral"),
    ):
        result.append(Arm(
            f"{short}_cap20_n10", agent_id, 20, N_MAX, "sample_count"))
        for cap in REASONING_CAPS[1:]:
            result.append(Arm(
                f"{short}_cap{cap}_n3",
                agent_id,
                cap,
                CAP_N,
                "reasoning_cap",
            ))
    return tuple(result)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def _token_caps(reasoning_cap: int) -> dict[str, int]:
    """Leave answer room while increasing only the requested rationale cap."""
    values = {
        20: {"default": 80, "awardWonBy": 224},
        40: {"default": 112, "awardWonBy": 256},
        80: {"default": 160, "awardWonBy": 320},
    }
    if reasoning_cap not in values:
        raise ContractError(f"unsupported reasoning cap: {reasoning_cap}")
    return values[reasoning_cap]


def _variant_config(
    base: Mapping[str, Any], arm: Arm,
) -> dict[str, Any]:
    value = copy.deepcopy(base)
    matches = [
        agent for agent in value.get("agents", [])
        if str(agent.get("id")) == arm.agent_id
    ]
    if len(matches) != 1:
        raise ContractError(f"config does not contain {arm.agent_id}")
    matches[0]["proposal_output"] = "bounded_reasoning"
    matches[0]["proposal_reasoning_words"] = arm.reasoning_word_cap
    matches[0]["proposal_max_new_tokens"] = _token_caps(
        arm.reasoning_word_cap)
    return value


def _source_rows(path: Path) -> list[dict[str, Any]]:
    _source_manifest(path)
    rows = read_jsonl(path)
    keys = [_key(row) for row in rows]
    if len(rows) != 477 or len(set(keys)) != len(keys):
        raise ContractError(
            f"expected 477 unique source-graph rows, got {len(rows)}")
    return rows


def _select_smoke(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result, seen = [], set()
    for task in tasks:
        relation = str(task["relation"])
        if relation not in seen:
            result.append(dict(task))
            seen.add(relation)
    if len(result) != len(set(RELATION_TYPE)):
        raise ContractError("smoke does not cover every relation")
    return result


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    source_path = Path(args.source_graph).resolve()
    config_path = Path(args.agents).resolve()
    synthetic_path = Path(args.synthetic_cot).resolve()
    source_rows = _source_rows(source_path)
    base_config = _json(config_path)
    synthetic = load_synthetic_by_relation(synthetic_path)
    inputs = [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": [],
    } for row in source_rows]
    input_path = output / "plan/INPUT_ROWS.jsonl"
    write_jsonl_atomic(input_path, inputs)

    plan_arms = {}
    for arm in arms():
        config_value = _variant_config(base_config, arm)
        variant_path = output / f"plan/configs/{arm.name}.json"
        _write_json(variant_path, config_value)
        config = load_agent_config(variant_path)
        agent = _agent(config, arm.agent_id)
        tasks = build_agent_tasks(
            inputs,
            agent,
            synthetic,
            seed=args.seed,
            n_proposals=arm.n_samples,
        )
        tasks = [task for task in tasks if task["phase"] == "propose"]
        if arm.agent_id == GEMMA:
            apply_prompt_policy(
                tasks,
                gemma_agent=agent,
                qwen_agent=_agent(config, QWEN),
                synthetic=synthetic,
                seed=args.seed,
            )
        for task in tasks:
            task["ablation_arm"] = arm.name
            task["reasoning_word_cap"] = arm.reasoning_word_cap
            if int(task["n_samples"]) != arm.n_samples:
                raise ContractError(f"{arm.name}: task sample-count drift")
            expected_phrase = (
                f"at most {arm.reasoning_word_cap} words")
            if expected_phrase not in str(task["prompt"]):
                raise ContractError(
                    f"{arm.name}: prompt lacks frozen reasoning cap")
            if str(task["subject"]) in {
                str(item) for item in task.get("shot_subjects", [])
            }:
                raise ContractError(f"{arm.name}: target-shot leakage")
        tasks.sort(key=lambda task: (
            int(task["max_new_tokens"]),
            str(task["relation"]),
            int(task["input_index"]),
        ))
        if len(tasks) != len(inputs):
            raise ContractError(f"{arm.name}: incomplete proposal plan")
        validate_tasks(tasks, arm.agent_id)
        task_path = output / f"plan/tasks/{arm.name}.jsonl"
        smoke_path = output / f"plan/smoke/{arm.name}.jsonl"
        response_path = output / f"responses/{arm.name}.jsonl"
        smoke_response_path = output / f"smoke_responses/{arm.name}.jsonl"
        write_jsonl_atomic(task_path, tasks)
        write_jsonl_atomic(smoke_path, _select_smoke(tasks))
        plan_arms[arm.name] = {
            **asdict(arm),
            "seed": int(args.seed),
            "config": str(variant_path),
            "config_sha256": sha256(variant_path),
            "tasks": str(task_path),
            "tasks_sha256": sha256(task_path),
            "smoke_tasks": str(smoke_path),
            "smoke_tasks_sha256": sha256(smoke_path),
            "responses": str(response_path),
            "smoke_responses": str(smoke_response_path),
            "rows": len(tasks),
            "proposal_generations": len(tasks) * arm.n_samples,
            "max_new_tokens": _token_caps(arm.reasoning_word_cap),
        }

    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "split": "train",
        "validation_opened": False,
        "validation_labels_used": False,
        "selection_target": "train_only_then_freeze_before_validation",
        "seed": int(args.seed),
        "n_values": list(N_VALUES),
        "n_max": N_MAX,
        "reasoning_caps": list(REASONING_CAPS),
        "reasoning_cap_n": CAP_N,
        "source_graph": str(source_path),
        "source_graph_sha256": sha256(source_path),
        "input_rows": str(input_path),
        "input_rows_sha256": sha256(input_path),
        "base_agents": str(config_path),
        "base_agents_sha256": sha256(config_path),
        "synthetic_cot": str(synthetic_path),
        "synthetic_cot_sha256": sha256(synthetic_path),
        "arms": plan_arms,
        "analysis_policy": {
            "sample_axis": "nested_prefixes_from_one_ordered_n10_run",
            "reasoning_axis": "fixed_n3_with_caps_20_40_80",
            "primary_support_rule": "ceil(2*N/3)",
            "string_residual_action": "add_supported_candidates_to_oof_incumbent",
            "numeric_residual_action": (
                "replace_incumbent_only_for_one_unique_supported_top_vote"),
            "tie_break": "smaller_N_then_smaller_reasoning_cap",
            "validation_selection": False,
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    plan_path = output / "plan/PLAN.json"
    _write_json(plan_path, plan)
    print(json.dumps({
        "output": str(output),
        "plan": str(plan_path),
        "arms": {
            name: {
                "rows": value["rows"],
                "proposal_generations": value["proposal_generations"],
                "reasoning_word_cap": value["reasoning_word_cap"],
                "n_samples": value["n_samples"],
            }
            for name, value in plan_arms.items()
        },
        "total_proposal_generations": sum(
            int(value["proposal_generations"])
            for value in plan_arms.values()
        ),
    }, indent=2, sort_keys=True))
    return 0


def _validate_plan(output: Path) -> dict[str, Any]:
    path = output / "plan/PLAN.json"
    plan = _json(path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or tuple(plan.get("n_values", [])) != N_VALUES
        or tuple(plan.get("reasoning_caps", [])) != REASONING_CAPS
    ):
        raise ContractError("invalid sampling/reasoning ablation plan")
    for field in ("source_graph", "input_rows", "base_agents", "synthetic_cot"):
        if sha256(Path(plan[field])) != plan[f"{field}_sha256"]:
            raise ContractError(f"plan input changed: {field}")
    expected_arms = {arm.name: arm for arm in arms()}
    if set(plan.get("arms", {})) != set(expected_arms):
        raise ContractError("plan arm set changed")
    for name, arm in expected_arms.items():
        value = plan["arms"][name]
        if (
            value["agent_id"] != arm.agent_id
            or int(value["reasoning_word_cap"]) != arm.reasoning_word_cap
            or int(value["n_samples"]) != arm.n_samples
            or int(value["seed"]) != int(plan["seed"])
            or sha256(Path(value["config"])) != value["config_sha256"]
            or sha256(Path(value["tasks"])) != value["tasks_sha256"]
        ):
            raise ContractError(f"plan arm changed: {name}")
    return plan


def _response_rows(
    arm: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    response_path = Path(arm["responses"])
    manifest_path = response_path.with_suffix(
        response_path.suffix + ".manifest.json")
    if not response_path.is_file() or not manifest_path.is_file():
        raise ContractError(f"missing responses for {arm['name']}")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != "heterogeneous-agent-responses-v1"
        or manifest.get("agent_id") != arm["agent_id"]
        or manifest.get("task_sha256") != arm["tasks_sha256"]
        or manifest.get("agent_config_sha256") != arm["config_sha256"]
        or manifest.get("output_sha256") != sha256(response_path)
        or int(manifest.get("seed", -1)) != int(arm["seed"])
        or int(manifest.get("tasks", -1)) != int(arm["rows"])
    ):
        raise ContractError(f"stale response manifest for {arm['name']}")
    if arm["agent_id"] == MINISTRAL and (
        manifest.get("model") != MINISTRAL_MODEL
        or manifest.get("revision") != MINISTRAL_REVISION
    ):
        raise ContractError("foreign Ministral checkpoint")
    tasks = read_jsonl(Path(arm["tasks"]))
    task_by_id = validate_tasks(tasks, str(arm["agent_id"]))
    responses = read_jsonl(response_path)
    response_by_id = {
        str(row["task_id"]): row for row in responses}
    if (
        len(response_by_id) != len(responses)
        or set(response_by_id) != set(task_by_id)
    ):
        raise ContractError(f"incomplete responses for {arm['name']}")
    result = {}
    for task_id, task in task_by_id.items():
        response = response_by_id[task_id]
        validate_task_response(task, response)
        generations = response.get("generations", [])
        if len(generations) != int(arm["n_samples"]):
            raise ContractError(
                f"{task_id}: expected {arm['n_samples']} generations")
        key = str(task["subject"]), str(task["relation"])
        if key in result:
            raise ContractError(f"duplicate response key: {key}")
        result[key] = response
    return result


def _reasoning_words(text: str) -> int | None:
    match = REASONING_LINE.search(str(text))
    return len(match.group(1).split()) if match else None


def _prefix_evidence(
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
    n: int,
    reasoning_cap: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    evidence = {}
    statuses = Counter()
    reasoning_counts = []
    output_tokens = []
    wall_seconds = 0.0
    for key, response in responses.items():
        relation = key[1]
        counts: Counter[str] = Counter()
        displays = {}
        generation_sets = []
        for generation in list(response["generations"])[:n]:
            status, items = proposal_parse_status(str(generation), relation)
            statuses[status] += 1
            current = set()
            for item in items:
                canonical = canonical_key(str(item), relation)
                if canonical:
                    current.add(canonical)
                    displays.setdefault(canonical, str(item))
            for canonical in current:
                counts[canonical] += 1
            generation_sets.append(frozenset(current))
            words = _reasoning_words(str(generation))
            if words is not None:
                reasoning_counts.append(words)
        telemetry = response.get("generation_telemetry", {})
        token_values = telemetry.get("output_tokens", [])
        if isinstance(token_values, list):
            output_tokens.extend(
                int(value) for value in token_values[:n])
        # Full-arm wall time is meaningful only at the full generated prefix.
        if n == len(response["generations"]):
            wall_seconds += float(telemetry.get("batch_wall_seconds", 0.0))
        evidence[key] = {
            "counts": counts,
            "displays": displays,
            "sets": generation_sets,
        }
    total = sum(statuses.values())
    parse_failures = total - statuses["parsed_nonempty"] - statuses[
        "explicit_none"]
    return evidence, {
        "generations": total,
        "parse_statuses": dict(statuses),
        "parse_failure_rate": parse_failures / total if total else 1.0,
        "reasoning_line_rate": (
            len(reasoning_counts) / total if total else 0.0),
        "reasoning_cap_violation_rate": (
            sum(value > reasoning_cap for value in reasoning_counts)
            / len(reasoning_counts)
            if reasoning_counts else None),
        "mean_reasoning_words": (
            statistics.mean(reasoning_counts) if reasoning_counts else None),
        "max_reasoning_words": (
            max(reasoning_counts) if reasoning_counts else None),
        "mean_output_tokens": (
            statistics.mean(output_tokens) if output_tokens else None),
        "telemetry_wall_seconds": wall_seconds or None,
    }


def _candidate_graph(
    base: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    relation = str(base["Relation"])
    candidates = {}
    for item in _items(base):
        canonical = canonical_key(item, relation)
        if canonical:
            candidates.setdefault(canonical, {"item": item})
    for canonical, item in evidence["displays"].items():
        candidates.setdefault(canonical, {"item": item})
    return {
        "SubjectEntity": base["SubjectEntity"],
        "Relation": relation,
        "candidates": list(candidates.values()),
    }


def _accepted(
    evidence: Mapping[str, Any], threshold: int,
) -> list[str]:
    return [
        evidence["displays"][canonical]
        for canonical, count in evidence["counts"].items()
        if count >= threshold
    ]


def _residual_prediction(
    incumbent: Mapping[str, Any],
    evidence: Mapping[str, Any],
    threshold: int,
) -> dict[str, Any]:
    relation = str(incumbent["Relation"])
    base_items = [str(item) for item in incumbent.get("ObjectEntities", [])]
    counts = evidence["counts"]
    displays = evidence["displays"]
    if relation in NUMERIC_RELATIONS:
        if counts:
            highest = max(counts.values())
            winners = [
                canonical for canonical, count in counts.items()
                if count == highest and count >= threshold
            ]
            selected = [displays[winners[0]]] if len(winners) == 1 else base_items
        else:
            selected = base_items
    else:
        by_key = {}
        for item in base_items:
            canonical = canonical_key(item, relation)
            if canonical:
                by_key.setdefault(canonical, item)
        for item in _accepted(evidence, threshold):
            canonical = canonical_key(item, relation)
            if canonical:
                by_key.setdefault(canonical, item)
        selected = list(by_key.values())
    return {
        "SubjectEntity": incumbent["SubjectEntity"],
        "Relation": relation,
        "ObjectEntities": selected,
    }


def _canonical_items(
    items: Sequence[str], relation: str,
) -> frozenset[str]:
    return frozenset(
        canonical for canonical in (
            canonical_key(str(item), relation) for item in items
        ) if canonical
    )


def _thresholds(n: int) -> dict[str, int]:
    return {
        "any": 1,
        "repeat2": min(2, n),
        "majority": n // 2 + 1,
        "two_thirds": math.ceil(2 * n / 3),
        "unanimous": n,
    }


def _evaluate_prefix(
    *,
    n: int,
    evidence: Mapping[tuple[str, str], Mapping[str, Any]],
    telemetry: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    incumbent_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_by = {_key(row): row for row in source_rows}
    incumbent_by = {_key(row): row for row in incumbent_rows}
    gold_by = {_key(row): row for row in gold_rows}
    expected = set(source_by)
    if (
        set(evidence) != expected
        or set(incumbent_by) != expected
        or set(gold_by) != expected
    ):
        raise ContractError("sampling ablation split coverage mismatch")
    combined = [
        _candidate_graph(source_by[_key(row)], evidence[_key(row)])
        for row in source_rows
    ]
    ordered_gold = [gold_by[_key(row)] for row in source_rows]
    oracle_score = score(
        oracle_rows(combined, ordered_gold), ordered_gold)

    labels, support_rates = [], []
    new_total = new_correct = 0
    unique_correct_rows = []
    unique_relations = Counter()
    for row in source_rows:
        key = _key(row)
        relation = key[1]
        base_keys = {
            canonical_key(item, relation) for item in _items(row)
            if canonical_key(item, relation)
        }
        base_has_correct = _has_correct(
            _items(row), gold_by[key], relation)
        proposal_has_correct = False
        for canonical, count in evidence[key]["counts"].items():
            item = evidence[key]["displays"][canonical]
            correct = _has_correct([item], gold_by[key], relation)
            labels.append(correct)
            support_rates.append(count / n)
            proposal_has_correct = proposal_has_correct or correct
            if canonical not in base_keys:
                new_total += 1
                new_correct += int(correct)
        if proposal_has_correct and not base_has_correct:
            unique_correct_rows.append([key[0], relation])
            unique_relations[relation] += 1

    policies = {}
    for name, threshold in _thresholds(n).items():
        standalone = []
        residual = []
        for row in source_rows:
            key = _key(row)
            objects = _accepted(evidence[key], threshold)
            standalone.append({
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": objects,
            })
            residual.append(_residual_prediction(
                incumbent_by[key], evidence[key], threshold))
        policies[name] = {
            "support_threshold": threshold,
            "standalone_scores": score(standalone, ordered_gold),
            "residual_scores": score(residual, ordered_gold),
            "residual_changed_rows": sum(
                _canonical_items(
                    row["ObjectEntities"], str(row["Relation"]))
                != _canonical_items(
                    incumbent_by[_key(row)]["ObjectEntities"],
                    str(row["Relation"]),
                )
                for row in residual
            ),
        }
    return {
        "n": n,
        "telemetry": dict(telemetry),
        "combined_candidate_oracle_scores": oracle_score,
        "candidate_truth_auroc": _binary_auroc(labels, support_rates),
        "proposal_candidates": len(labels),
        "positive_candidates": sum(labels),
        "new_candidates": new_total,
        "new_correct_candidates": new_correct,
        "new_candidate_precision": (
            new_correct / new_total if new_total else None),
        "unique_correct_rows_beyond_source_graph": unique_correct_rows,
        "unique_correct_row_count": len(unique_correct_rows),
        "unique_correct_relation_counts": dict(unique_relations),
        "policies": policies,
    }


def _markdown(
    result: Mapping[str, Any],
    missing: Sequence[str],
) -> str:
    lines = [
        "# Sampling-count and visible-reasoning ablation",
        "",
        "Training-only development audit. Validation was not opened.",
        "",
        f"- Competition-pipeline OOF anchor: "
        f"**{result['incumbent_scores'][POOLED]:.6f}**",
        f"- Source candidate oracle: "
        f"**{result['source_candidate_oracle_scores'][POOLED]:.6f}**",
        f"- Complete arms: **{len(result['arms'])}/{len(arms())}**",
    ]
    if missing:
        lines.append(f"- Missing arms: **{', '.join(missing)}**")
    lines.extend([
        "",
        "## N ablation at a 20-word cap",
        "",
        "| model | N | residual F1 | delta | candidate oracle | AUROC | "
        "new precision | parse fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    anchor = float(result["incumbent_scores"][POOLED])
    for model in ("gemma", "ministral"):
        arm = result["arms"].get(f"{model}_cap20_n10")
        if not arm:
            continue
        for n in N_VALUES:
            prefix = arm["prefixes"][str(n)]
            primary = prefix["policies"]["two_thirds"][
                "residual_scores"][POOLED]
            lines.append(
                f"| {model.title()} | {n} | {primary:.4f} | "
                f"{primary - anchor:+.4f} | "
                f"{prefix['combined_candidate_oracle_scores'][POOLED]:.4f} | "
                f"{_fmt(prefix['candidate_truth_auroc'])} | "
                f"{_fmt(prefix['new_candidate_precision'])} | "
                f"{prefix['telemetry']['parse_failure_rate']:.3f} |")
    lines.extend([
        "",
        "## Reasoning-cap ablation at N=3",
        "",
        "| model | word cap | residual F1 | delta | candidate oracle | "
        "AUROC | mean observed words | parse fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for model in ("gemma", "ministral"):
        for cap in REASONING_CAPS:
            name = (
                f"{model}_cap20_n10" if cap == 20
                else f"{model}_cap{cap}_n3"
            )
            arm = result["arms"].get(name)
            if not arm:
                continue
            prefix = arm["prefixes"][str(CAP_N)]
            primary = prefix["policies"]["two_thirds"][
                "residual_scores"][POOLED]
            lines.append(
                f"| {model.title()} | {cap} | {primary:.4f} | "
                f"{primary - anchor:+.4f} | "
                f"{prefix['combined_candidate_oracle_scores'][POOLED]:.4f} | "
                f"{_fmt(prefix['candidate_truth_auroc'])} | "
                f"{_fmt(prefix['telemetry']['mean_reasoning_words'], 1)} | "
                f"{prefix['telemetry']['parse_failure_rate']:.3f} |")
    lines.extend([
        "",
        "The primary policy is predeclared: strings add candidates supported "
        "by at least ceil(2N/3) samples to the exact OOF incumbent; numeric "
        "rows replace the incumbent only for one unique top-voted candidate "
        "meeting the same boundary. Oracle values are gold-aware, "
        "nondeployable coverage diagnostics.",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 3) -> str:
    return "---" if value is None else f"{float(value):.{digits}f}"


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    source_rows = _source_rows(Path(plan["source_graph"]))
    incumbent_rows, incumbent_detail = compose_competition_train_oof()
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    if len(gold_rows) != 477:
        raise ContractError("expected 477 training gold rows")
    incumbent_scores = score(incumbent_rows, gold_rows)
    source_oracle_scores = score(
        oracle_rows(source_rows, gold_rows), gold_rows)

    results = {}
    missing = []
    for arm in arms():
        arm_plan = plan["arms"][arm.name]
        response_path = Path(arm_plan["responses"])
        manifest_path = response_path.with_suffix(
            response_path.suffix + ".manifest.json")
        if (
            args.allow_partial
            and (not response_path.is_file() or not manifest_path.is_file())
        ):
            missing.append(arm.name)
            continue
        responses = _response_rows(arm_plan)
        prefix_ns = N_VALUES if arm.axis == "sample_count" else (CAP_N,)
        prefixes = {}
        for n in prefix_ns:
            evidence, telemetry = _prefix_evidence(
                responses, n, arm.reasoning_word_cap)
            prefixes[str(n)] = _evaluate_prefix(
                n=n,
                evidence=evidence,
                telemetry=telemetry,
                source_rows=source_rows,
                incumbent_rows=incumbent_rows,
                gold_rows=gold_rows,
            )
        results[arm.name] = {
            **dict(arm_plan),
            "responses_sha256": sha256(Path(arm_plan["responses"])),
            "prefixes": prefixes,
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
        "complete": not missing,
        "missing_arms": missing,
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "incumbent_pipeline": COMPETITION_PIPELINE_ID,
        "incumbent_detail": incumbent_detail,
        "incumbent_scores": incumbent_scores,
        "source_candidate_oracle_scores": source_oracle_scores,
        "arms": results,
    }
    analysis_dir = output / "analysis"
    _write_json(analysis_dir / "RESULT.json", result)
    (analysis_dir / "RESULT.md").write_text(
        _markdown(result, missing) + "\n")
    print(json.dumps({
        "complete": not missing,
        "completed_arms": len(results),
        "missing_arms": missing,
        "incumbent_f1": incumbent_scores[POOLED],
        "result": str(analysis_dir / "RESULT.md"),
    }, indent=2, sort_keys=True))
    return 0 if not missing else 3


def status(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    values = {}
    for name, arm in plan["arms"].items():
        response = Path(arm["responses"])
        manifest = response.with_suffix(response.suffix + ".manifest.json")
        completed = 0
        if response.is_file():
            completed = len(read_jsonl(response))
        values[name] = {
            "completed_rows": completed,
            "expected_rows": arm["rows"],
            "manifest": manifest.is_file(),
            "complete": completed == arm["rows"] and manifest.is_file(),
        }
    print(json.dumps(values, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name in ("prepare", "analyze", "status"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
        if name == "prepare":
            sub.add_argument("--source-graph", default=str(DEFAULT_SOURCE_GRAPH))
            sub.add_argument("--agents", default=str(DEFAULT_CONFIG))
            sub.add_argument("--synthetic-cot", default=str(DEFAULT_SYNTHETIC))
            sub.add_argument("--seed", type=int, default=SEED)
        elif name == "analyze":
            sub.add_argument("--train-gold", default=str(DEFAULT_GOLD))
            sub.add_argument("--allow-partial", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    return {
        "prepare": prepare,
        "analyze": analyze,
        "status": status,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
