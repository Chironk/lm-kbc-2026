#!/usr/bin/env python3
"""Prepare and compare a paired direct-vs-synthetic Gemma prompt ablation.

The existing 2026-07-20 portfolio pilot is the immutable control. Preparation
rebuilds the exact Qwen tasks as a hash check, changes only Gemma's proposal
prompt, reuses the exact Qwen responses, and seeds Gemma's unchanged blind
existence/cardinality responses. Consequently, the GPU runner executes only
the 130 changed Gemma proposal tasks.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate import RELATION_TYPE, true_positives
from experiments.heterogeneous_agents.assemble_and_audit import (
    _gold_aliases, _key, assemble_graphs, load_responses, oracle_rows,
    portfolio_diagnostics, prediction_rows, score,
)
from experiments.heterogeneous_agents.core import (
    ContractError, build_agent_tasks, load_agent_config,
    load_synthetic_by_relation, proposal_prompt, read_jsonl, sha256, stable_seed,
    validate_inputs,
    validate_task_response, write_jsonl_atomic,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = (ROOT / "experiments/heterogeneous_agents/runs" /
                "portfolio_pilot_20260720_v1")
DEFAULT_CONFIG = Path(__file__).with_name("agents_qwen_gemma_synthetic.json")
DEFAULT_SYNTHETIC = ROOT / "data/synthetic_cot_faithful.jsonl"
QWEN = "qwen_recall"
GEMMA = "gemma_independent"


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _agents_by_id(config: Mapping[str, Any]) -> dict[str, dict]:
    return {agent["id"]: agent for agent in config["agents"]}


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ContractError(f"refusing to overwrite non-matching artifact: {destination}")
        return
    shutil.copy2(source, destination)
    if sha256(destination) != sha256(source):
        raise ContractError(f"copy hash mismatch: {source} -> {destination}")


def _assert_paired_tasks(control: Sequence[dict], treatment: Sequence[dict]) -> dict:
    """Prove that only Gemma proposal prompt demonstrations/style changed."""
    if len(control) != len(treatment):
        raise ContractError("Gemma control/treatment task counts differ")
    changed_proposals = 0
    reused_commitments = 0
    relation_shot_counts: dict[str, list[int]] = defaultdict(list)
    for old, new in zip(control, treatment):
        if old["task_id"] != new["task_id"]:
            raise ContractError("Gemma task order/id changed")
        if old["phase"] != "propose":
            if old != new:
                raise ContractError(
                    f"non-proposal task changed in prompt ablation: {old['task_id']}")
            reused_commitments += 1
            continue
        ignored = {"prompt", "shot_subjects"}
        old_fixed = {key: value for key, value in old.items() if key not in ignored}
        new_fixed = {key: value for key, value in new.items() if key not in ignored}
        if old_fixed != new_fixed:
            raise ContractError(
                f"non-prompt proposal settings changed: {old['task_id']}")
        if old.get("shot_subjects"):
            raise ContractError(f"control unexpectedly has shots: {old['task_id']}")
        shots = new.get("shot_subjects", [])
        if not shots or new["subject"] in shots or len(shots) != len(set(shots)):
            raise ContractError(f"invalid target-excluded treatment shots: {new['task_id']}")
        if old["prompt"] == new["prompt"]:
            raise ContractError(f"proposal prompt did not change: {old['task_id']}")
        relation_shot_counts[new["relation"]].append(len(shots))
        changed_proposals += 1
    return {
        "changed_proposal_tasks": changed_proposals,
        "byte_identical_reused_commitment_tasks": reused_commitments,
        "shot_counts_by_relation": {
            relation: dict(Counter(counts))
            for relation, counts in sorted(relation_shot_counts.items())
        },
    }


def _minimum_overlap_shots(pool: Sequence[dict], *, subject: str, relation: str,
                           count: int, seed: int,
                           reference_subjects: Sequence[str]) -> list[dict]:
    """Draw target/subject-distinct shots with minimum reference-set overlap."""
    candidates = [row for row in pool if row.get("SubjectEntity") != subject]
    random.Random(stable_seed(
        "gemma-disjoint-shots-v1", seed, subject, relation)).shuffle(candidates)
    by_subject = {}
    for row in candidates:
        shot_subject = row.get("SubjectEntity")
        if isinstance(shot_subject, str) and shot_subject not in by_subject:
            by_subject[shot_subject] = row
    reference = set(reference_subjects)
    disjoint = [row for shot_subject, row in by_subject.items()
                if shot_subject not in reference]
    overlap = [row for shot_subject, row in by_subject.items()
               if shot_subject in reference]
    return (disjoint + overlap)[:count]


def _apply_disjoint_shots(gemma_tasks: Sequence[dict], qwen_tasks: Sequence[dict],
                          gemma_agent: Mapping[str, Any],
                          synthetic: Mapping[str, Sequence[dict]], *, seed: int) -> dict:
    qwen_proposals = {task["input_index"]: task for task in qwen_tasks
                      if task["phase"] == "propose"}
    overlaps = Counter()
    expected_overlaps = Counter()
    for task in gemma_tasks:
        if task["phase"] != "propose":
            continue
        reference = qwen_proposals[task["input_index"]].get("shot_subjects", [])
        shots = _minimum_overlap_shots(
            synthetic.get(task["relation"], []), subject=task["subject"],
            relation=task["relation"], count=gemma_agent["synthetic_shots"],
            seed=seed, reference_subjects=reference)
        if len(shots) != gemma_agent["synthetic_shots"]:
            raise ContractError(f"insufficient disjoint-shot pool: {task['task_id']}")
        subjects = [shot["SubjectEntity"] for shot in shots]
        overlap = len(set(subjects) & set(reference))
        eligible_subjects = {
            row.get("SubjectEntity") for row in synthetic.get(task["relation"], [])
            if isinstance(row.get("SubjectEntity"), str)
            and row.get("SubjectEntity") != task["subject"]
        }
        available_disjoint = len(eligible_subjects - set(reference))
        expected = max(0, gemma_agent["synthetic_shots"] - available_disjoint)
        if overlap != expected:
            raise ContractError(
                f"shot overlap is not minimal for {task['task_id']}: "
                f"actual={overlap}, minimum={expected}")
        task["prompt"] = proposal_prompt(
            gemma_agent, task["subject"], task["relation"], shots)
        task["shot_subjects"] = subjects
        overlaps[f"{task['relation']}::{overlap}"] += 1
        expected_overlaps[f"{task['relation']}::{expected}"] += 1
    return {"actual_overlap_counts": dict(overlaps),
            "minimum_possible_overlap_counts": dict(expected_overlaps)}


def prepare(args: argparse.Namespace) -> int:
    base = Path(args.base_run).resolve()
    output = Path(args.output_dir).resolve()
    plan_out = output / "plan"
    responses_out = output / "responses"
    config_path = Path(args.agents).resolve()
    synthetic_path = Path(args.synthetic_cot).resolve()
    base_plan = _json(base / "plan/PLAN.json")
    if base_plan.get("seed") != args.seed or base_plan.get("n_proposals") != args.n_proposals:
        raise ContractError(
            "requested seed/N does not match the frozen control pilot: "
            f"control seed={base_plan.get('seed')} N={base_plan.get('n_proposals')}")

    config = load_agent_config(config_path)
    agents = _agents_by_id(config)
    if set(agents) != {QWEN, GEMMA}:
        raise ContractError("ablation config must contain exactly Qwen and Gemma")
    if (agents[GEMMA]["role"], agents[GEMMA]["synthetic_shots"]) != (
            "synthetic_cot_recall", 5):
        raise ContractError("Gemma treatment must be synthetic_cot_recall with five shots")

    inputs = read_jsonl(base / "plan/input_subset.jsonl")
    validate_inputs(inputs)
    if sha256(base / "plan/input_subset.jsonl") != base_plan["input_subset_sha256"]:
        raise ContractError("frozen control input subset hash changed")
    synthetic = load_synthetic_by_relation(synthetic_path)
    plan_out.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(plan_out / "input_subset.jsonl", inputs)

    task_paths = {}
    treatment_tasks = {}
    for agent_id in (QWEN, GEMMA):
        tasks = build_agent_tasks(
            inputs, agents[agent_id], synthetic,
            seed=args.seed, n_proposals=args.n_proposals)
        treatment_tasks[agent_id] = tasks

    diversity_diagnostics = None
    if args.shot_policy == "disjoint":
        diversity_diagnostics = _apply_disjoint_shots(
            treatment_tasks[GEMMA], treatment_tasks[QWEN], agents[GEMMA],
            synthetic, seed=args.seed)

    # Write only after the optional treatment rewrite so hashes bind the exact
    # prompts that the model will receive.
    for agent_id in (QWEN, GEMMA):
        path = plan_out / "tasks" / f"{agent_id}.jsonl"
        write_jsonl_atomic(path, treatment_tasks[agent_id])
        task_paths[agent_id] = {
            "path": str(path), "sha256": sha256(path),
            "tasks": len(treatment_tasks[agent_id]),
        }

    base_qwen_task = base / "plan/tasks/qwen_recall.jsonl"
    if sha256(plan_out / "tasks/qwen_recall.jsonl") != sha256(base_qwen_task):
        raise ContractError("rebuilt Qwen tasks differ from the frozen control")
    control_gemma_tasks = read_jsonl(base / "plan/tasks/gemma_independent.jsonl")
    paired = _assert_paired_tasks(control_gemma_tasks, treatment_tasks[GEMMA])

    # Reuse Qwen byte-for-byte. Its model is not loaded in this ablation.
    for suffix in ("", ".manifest.json"):
        _copy_verified(
            base / f"responses/qwen_recall.jsonl{suffix}",
            responses_out / f"qwen_recall.jsonl{suffix}")

    # Seed only unchanged Gemma commitment responses. The resumable runner
    # will see exactly 130 pending proposal tasks and will write a fresh final
    # manifest after completing them.
    gemma_output = responses_out / "gemma_independent.jsonl"
    task_by_id = validate_tasks(treatment_tasks[GEMMA], GEMMA)
    if gemma_output.exists():
        existing = read_jsonl(gemma_output)
        for row in existing:
            task_id = row.get("task_id")
            if task_id not in task_by_id:
                raise ContractError(f"stale treatment response: {task_id}")
            validate_task_response(task_by_id[task_id], row)
    else:
        control_responses = read_jsonl(base / "responses/gemma_independent.jsonl")
        commitments = [row for row in control_responses if row.get("phase") != "propose"]
        for row in commitments:
            validate_task_response(task_by_id[row["task_id"]], row)
        write_jsonl_atomic(gemma_output, commitments)

    manifest = {
        "schema": "gemma-synthetic-prompt-ablation-v1",
        "base_run": str(base),
        "base_plan_sha256": sha256(base / "plan/PLAN.json"),
        "input_subset_sha256": sha256(plan_out / "input_subset.jsonl"),
        "agent_config": str(config_path),
        "agent_config_sha256": sha256(config_path),
        "synthetic_cot": str(synthetic_path),
        "synthetic_cot_sha256": sha256(synthetic_path),
        "seed": args.seed,
        "n_proposals": args.n_proposals,
        "rows": len(inputs),
        "task_files": task_paths,
        "paired_invariants": paired,
        "qwen_tasks_rebuilt_byte_identically": True,
        "qwen_responses_reused_byte_identically": True,
        "gemma_blind_commitments_reused": True,
        "only_gemma_proposals_require_gpu": True,
        "treatment": "Gemma synthetic_cot_recall with five target-excluded shots",
        "control": "Gemma independent_direct_recall with zero shots",
        "shot_policy": args.shot_policy,
        "shot_diversity_diagnostics": diversity_diagnostics,
    }
    (plan_out / "PLAN.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completed = len(read_jsonl(gemma_output))
    total = len(treatment_tasks[GEMMA])
    print(f"paired Gemma prompt ablation prepared: {len(inputs)} rows")
    print(f"Qwen: reused exactly; no Qwen GPU run needed")
    print(f"Gemma tasks: {completed} complete, {total - completed} pending")
    print(f"plan: {plan_out / 'PLAN.json'}")
    return 0


def _coverage(graphs: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]],
              agent_id: str) -> dict[str, float]:
    gold_by_key = {_key(row): row for row in gold}
    counts = Counter()
    hits = Counter()
    for graph in graphs:
        relation = graph["Relation"]
        aliases = _gold_aliases(gold_by_key[_key(graph)])
        if not aliases:
            continue
        counts[relation] += 1
        candidates = [node["item"] for node in graph["candidates"]
                      if agent_id in node["proposer_agents"]]
        if true_positives(candidates, aliases, RELATION_TYPE[relation], 0.05) > 0:
            hits[relation] += 1
    counts["*** All Relations ***"] = sum(counts.values())
    hits["*** All Relations ***"] = sum(hits.values())
    return {relation: hits[relation] / count for relation, count in counts.items() if count}


def _delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {key: right[key] - left[key] for key in left.keys() & right.keys()}


def _format_table(title: str, control: Mapping[str, float],
                  treatment: Mapping[str, float]) -> list[str]:
    relations = ["*** All Relations ***", "countryLandBordersCountry",
                 "companyTradesAtStockExchange", "personHasCityOfDeath",
                 "hasArea", "hasCapacity", "awardWonBy"]
    lines = [f"## {title}", "", "| relation | direct | synthetic | delta |",
             "|---|---:|---:|---:|"]
    for relation in relations:
        if relation in control and relation in treatment:
            lines.append(
                f"| {relation} | {control[relation]:.4f} | "
                f"{treatment[relation]:.4f} | "
                f"{treatment[relation] - control[relation]:+.4f} |")
    lines.append("")
    return lines


def compare(args: argparse.Namespace) -> int:
    base = Path(args.base_run).resolve()
    treatment = Path(args.output_dir).resolve()
    output = treatment / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    base_config = load_agent_config(Path(args.base_agents).resolve())
    treatment_config = load_agent_config(Path(args.agents).resolve())
    base_agents_by_id = _agents_by_id(base_config)
    treatment_agents_by_id = _agents_by_id(treatment_config)
    pair_ids = [QWEN, GEMMA]
    base_agents = [base_agents_by_id[agent_id] for agent_id in pair_ids]
    treatment_agents = [treatment_agents_by_id[agent_id] for agent_id in pair_ids]

    base_inputs_path = base / "plan/input_subset.jsonl"
    treatment_inputs_path = treatment / "plan/input_subset.jsonl"
    if sha256(base_inputs_path) != sha256(treatment_inputs_path):
        raise ContractError("control and treatment input subsets differ")
    inputs = read_jsonl(base_inputs_path)
    base_all_responses = load_responses(base / "responses", base_config["agents"])
    treatment_responses = load_responses(treatment / "responses", treatment_agents)
    if (sha256(base / "responses/qwen_recall.jsonl") !=
            sha256(treatment / "responses/qwen_recall.jsonl")):
        raise ContractError("Qwen response artifact was not reused byte-identically")
    base_responses = {agent_id: base_all_responses[agent_id] for agent_id in pair_ids}
    base_graphs = assemble_graphs(inputs, base_agents, base_responses)
    treatment_graphs = assemble_graphs(inputs, treatment_agents, treatment_responses)

    gold_all = read_jsonl(Path(args.gold).resolve())
    gold_by_key = {_key(row): row for row in gold_all}
    gold = [gold_by_key[_key(row)] for row in inputs]
    qwen_base = score(prediction_rows(base_graphs, f"agent:{QWEN}", pair_ids), gold)
    qwen_treatment = score(
        prediction_rows(treatment_graphs, f"agent:{QWEN}", pair_ids), gold)
    if qwen_base != qwen_treatment:
        raise ContractError("Qwen control score changed in paired treatment")
    gemma_direct = score(prediction_rows(base_graphs, f"agent:{GEMMA}", pair_ids), gold)
    gemma_synthetic = score(
        prediction_rows(treatment_graphs, f"agent:{GEMMA}", pair_ids), gold)
    direct_oracle = score(oracle_rows(base_graphs, gold), gold)
    synthetic_oracle = score(oracle_rows(treatment_graphs, gold), gold)
    direct_consensus = score(
        prediction_rows(base_graphs, "heterogeneous_proposal_consensus", pair_ids), gold)
    synthetic_consensus = score(
        prediction_rows(treatment_graphs, "heterogeneous_proposal_consensus", pair_ids), gold)
    direct_coverage = _coverage(base_graphs, gold, GEMMA)
    synthetic_coverage = _coverage(treatment_graphs, gold, GEMMA)

    treatment_plan = _json(treatment / "plan/PLAN.json")
    result = {
        "schema": "gemma-synthetic-prompt-comparison-v1",
        "paired_contract": {
            "same_input_subset": True,
            "same_qwen_tasks": True,
            "same_qwen_responses": True,
            "same_gemma_blind_commitments": True,
            "same_gemma_checkpoint_precision_seed_n_temperature": True,
            "changed_factor": (
                "Gemma proposal prompt: direct/0-shot -> synthetic-CoT/5-shot; "
                f"shot_policy={treatment_plan.get('shot_policy', 'same')}")
        },
        "scores": {
            "qwen_fixed": qwen_base,
            "gemma_direct": gemma_direct,
            "gemma_synthetic": gemma_synthetic,
            "qwen_gemma_direct_union_oracle": direct_oracle,
            "qwen_gemma_synthetic_union_oracle": synthetic_oracle,
            "qwen_gemma_direct_consensus": direct_consensus,
            "qwen_gemma_synthetic_consensus": synthetic_consensus,
        },
        "deltas": {
            "gemma_solo": _delta(gemma_direct, gemma_synthetic),
            "qwen_gemma_union_oracle": _delta(direct_oracle, synthetic_oracle),
            "qwen_gemma_consensus": _delta(direct_consensus, synthetic_consensus),
            "gemma_nonempty_gold_candidate_coverage": _delta(
                direct_coverage, synthetic_coverage),
        },
        "candidate_coverage": {
            "gemma_direct": direct_coverage,
            "gemma_synthetic": synthetic_coverage,
        },
        "diagnostics": {
            "direct_pair": portfolio_diagnostics(base_graphs, gold, pair_ids),
            "synthetic_pair": portfolio_diagnostics(treatment_graphs, gold, pair_ids),
        },
    }
    (output / "COMPARISON.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    pooled = "*** All Relations ***"
    lines = [
        "# Paired Gemma synthetic-prompt ablation", "",
        "This is a 130-row labeled-train development audit, not a blind-test result. ",
        "Qwen is reused byte-for-byte; only Gemma proposal prompts are regenerated.", "",
    ]
    lines += _format_table("Gemma solo macro-F1", gemma_direct, gemma_synthetic)
    lines += _format_table(
        "Qwen + Gemma candidate-union oracle", direct_oracle, synthetic_oracle)
    lines += _format_table(
        "Qwen + Gemma conservative consensus", direct_consensus, synthetic_consensus)
    lines += _format_table(
        "Gemma candidate coverage on nonempty gold rows",
        direct_coverage, synthetic_coverage)
    oracle_delta = synthetic_oracle[pooled] - direct_oracle[pooled]
    solo_delta = gemma_synthetic[pooled] - gemma_direct[pooled]
    lines += [
        "## Interpretation", "",
        f"- Gemma solo delta: **{solo_delta:+.4f}**.",
        f"- Qwen+Gemma reservoir-oracle delta: **{oracle_delta:+.4f}**.",
        "- The union oracle is diagnostic and non-deployable; it measures whether the "
        "prompt exposes additional correct facts, not whether the current selector finds them.",
        "- Prefer the synthetic prompt only if gains are not confined to formatting or a "
        "single relation and it preserves or improves complementary Qwen+Gemma coverage.", "",
    ]
    (output / "RESULT.md").write_text("\n".join(lines))
    print(f"comparison complete: {output / 'RESULT.md'}")
    print(f"Gemma solo delta: {solo_delta:+.6f}")
    print(f"Qwen+Gemma union-oracle delta: {oracle_delta:+.6f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "compare"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--base-run", default=str(DEFAULT_BASE))
        sub.add_argument("--output-dir", required=True)
        sub.add_argument("--agents", default=str(DEFAULT_CONFIG))
    prep = subparsers.choices["prepare"]
    prep.add_argument("--synthetic-cot", default=str(DEFAULT_SYNTHETIC))
    prep.add_argument("--seed", type=int, default=20260720)
    prep.add_argument("--n-proposals", type=int, default=3)
    prep.add_argument("--shot-policy", choices=["same", "disjoint"], default="same")
    comp = subparsers.choices["compare"]
    comp.add_argument("--base-agents", default=str(Path(__file__).with_name("agents.json")))
    comp.add_argument("--gold", default=str(ROOT / "data/train.jsonl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return prepare(args) if args.command == "prepare" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
