#!/usr/bin/env python3
"""Prepare and evaluate the legal Qwen-9B + Gemma-12B validation baseline.

This module deliberately stops before the proposed heterogeneous-memory router.
It runs only Gemma, reuses the frozen all-9B v0495 prediction artifact, and
reports simple output-level union/intersection baselines plus gold-aware
diagnostic ceilings.  Validation labels are never written into model tasks.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate import RELATION_TYPE, evaluate_per_sr_pair, macro_average_per_relation, true_positives
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    NUMERIC_RELATIONS,
    SINGLE_RELATIONS,
    build_agent_tasks,
    canonical_key,
    load_agent_config,
    load_synthetic_by_relation,
    proposal_prompt,
    proposal_candidates,
    proposal_parse_status,
    read_jsonl,
    select_synthetic_shots,
    sha256,
    validate_inputs,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.gemma_prompt_ablation import _minimum_overlap_shots
from run_inference import extract_after_think, parse_answer_items


ROOT = Path(__file__).resolve().parents[2]
GEMMA = "gemma_independent"
QWEN = "qwen_recall"
PROMPT_POLICY_ARTIFACT = (
    ROOT / "experiments/heterogeneous_agents/runs/gemma_prompt_oof_20260721_v1/"
    "FROZEN_PROMPT_POLICY.json"
)

# Frozen after a five-fold OOF assessment on the 130-row labeled-train prompt pilot.
# The OOF verdict was mixed overall, but every full-fit relation choice was
# selected in at least four of five folds. City changed from disjoint to shared
# after aligning confirmation with the proposal-only operational decoder.
PROMPT_POLICY = {
    "countryLandBordersCountry": "shared_cot5",
    "companyTradesAtStockExchange": "disjoint_cot5",
    "personHasCityOfDeath": "shared_cot5",
    "hasArea": "direct",
    "hasCapacity": "shared_cot5",
    "awardWonBy": "disjoint_cot5",
}

EXPECTED_QWEN_RAW_HASHES = {
    "borders": "f745c81754a4a2372d6cb7926066a75fa5f67601aea6b2a0c44f873c42de4bbf",
    "fp16_recovered": "60552db771ce0daaa2363b4e454afcec9c86d5fc0bea3b94152bc3d35858b8c6",
}


def _agent(config: Mapping[str, Any], agent_id: str) -> dict:
    matches = [dict(agent) for agent in config["agents"] if agent["id"] == agent_id]
    if len(matches) != 1:
        raise ContractError(f"expected exactly one {agent_id} config")
    return matches[0]


def apply_prompt_policy(tasks: Sequence[dict], *, gemma_agent: Mapping[str, Any],
                        qwen_agent: Mapping[str, Any],
                        synthetic: Mapping[str, Sequence[dict]], seed: int) -> dict:
    """Apply the train-frozen per-relation Gemma proposal policy in-place."""
    diagnostics: Counter = Counter()
    for task in tasks:
        if task["phase"] != "propose":
            continue
        relation, subject = task["relation"], task["subject"]
        policy = PROMPT_POLICY[relation]
        pool = synthetic.get(relation, [])
        reference = select_synthetic_shots(
            pool, subject=subject, relation=relation,
            count=int(qwen_agent["synthetic_shots"]), seed=seed)
        reference_subjects = [row["SubjectEntity"] for row in reference]
        prompt_agent = dict(gemma_agent)
        if policy == "direct":
            prompt_agent["role"] = "independent_direct_recall"
            shots = []
        elif policy == "shared_cot5":
            prompt_agent["role"] = "synthetic_cot_recall"
            shots = reference
        elif policy == "disjoint_cot5":
            prompt_agent["role"] = "synthetic_cot_recall"
            shots = _minimum_overlap_shots(
                pool, subject=subject, relation=relation,
                count=int(gemma_agent["synthetic_shots"]), seed=seed,
                reference_subjects=reference_subjects)
        else:  # pragma: no cover - PROMPT_POLICY is a source constant.
            raise ContractError(f"unknown prompt policy {policy!r}")
        expected = 0 if policy == "direct" else int(gemma_agent["synthetic_shots"])
        if len(shots) != expected:
            raise ContractError(
                f"{task['task_id']}: expected {expected} shots, got {len(shots)}")
        subjects = [row["SubjectEntity"] for row in shots]
        if subject in subjects or len(subjects) != len(set(subjects)):
            raise ContractError(f"{task['task_id']}: target leakage or duplicate shot subject")
        overlap = len(set(subjects) & set(reference_subjects))
        if policy == "disjoint_cot5":
            eligible = {
                row.get("SubjectEntity") for row in pool
                if isinstance(row.get("SubjectEntity"), str)
                and row.get("SubjectEntity") != subject
            }
            minimum = max(0, expected - len(eligible - set(reference_subjects)))
            if overlap != minimum:
                raise ContractError(
                    f"{task['task_id']}: non-minimal shot overlap {overlap} != {minimum}")
        task["prompt"] = proposal_prompt(prompt_agent, subject, relation, shots)
        task["shot_subjects"] = subjects
        task["prompt_policy"] = policy
        task["reference_qwen_shot_subjects"] = reference_subjects
        diagnostics[f"{relation}::{policy}::shots={len(shots)}::overlap={overlap}"] += 1
    return dict(sorted(diagnostics.items()))


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_dir = output / "plan"
    task_dir = plan_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input).resolve()
    agents_path = Path(args.agents).resolve()
    synthetic_path = Path(args.synthetic_cot).resolve()
    qwen_predictions = Path(args.qwen_predictions).resolve()
    record_predictions = Path(args.record_predictions).resolve()
    qwen_border_raw = Path(args.qwen_border_raw).resolve()
    qwen_fp16_raw = Path(args.qwen_fp16_raw).resolve()
    prompt_policy_artifact = Path(args.prompt_policy_artifact).resolve()

    try:
        prompt_policy_record = json.loads(prompt_policy_artifact.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"cannot read frozen prompt-policy artifact {prompt_policy_artifact}: {exc}") from exc
    if (not isinstance(prompt_policy_record, dict)
            or prompt_policy_record.get("schema") != "gemma-prompt-policy-v1"
            or prompt_policy_record.get("policy") != PROMPT_POLICY):
        raise ContractError("frozen prompt-policy artifact does not match production policy")

    config = load_agent_config(agents_path)
    qwen_agent, gemma_agent = _agent(config, QWEN), _agent(config, GEMMA)
    rows_with_labels = read_jsonl(input_path)
    validate_inputs(rows_with_labels)
    rows = [{"SubjectEntity": row["SubjectEntity"], "Relation": row["Relation"]}
            for row in rows_with_labels]
    synthetic = load_synthetic_by_relation(synthetic_path)
    tasks = build_agent_tasks(
        rows, gemma_agent, synthetic, seed=args.seed, n_proposals=args.n_proposals)
    policy_diagnostics = apply_prompt_policy(
        tasks, gemma_agent=gemma_agent, qwen_agent=qwen_agent,
        synthetic=synthetic, seed=args.seed)
    # This is intentionally the pre-architecture baseline. Blind existence and
    # cardinality commitments belong to the proposed router and would both
    # contaminate this control and add avoidable model forwards.
    tasks = [task for task in tasks if task["phase"] == "propose"]

    input_subset = plan_dir / "input_subset.jsonl"
    tasks_path = task_dir / f"{GEMMA}.jsonl"
    write_jsonl_atomic(input_subset, rows)
    write_jsonl_atomic(tasks_path, tasks)
    for required in (qwen_predictions, record_predictions, qwen_border_raw, qwen_fp16_raw):
        if not required.is_file():
            raise ContractError(f"missing frozen comparison artifact: {required}")
    if sha256(qwen_border_raw) != EXPECTED_QWEN_RAW_HASHES["borders"]:
        raise ContractError("frozen Qwen border raw hash mismatch")
    if sha256(qwen_fp16_raw) != EXPECTED_QWEN_RAW_HASHES["fp16_recovered"]:
        raise ContractError("frozen Qwen recovered-fp16 raw hash mismatch")
    manifest = {
        "schema": "qwen-gemma-simple-validation-plan-v1",
        "purpose": "simple dual-model baseline before heterogeneous-memory routing",
        "input": str(input_path), "input_sha256": sha256(input_path),
        "input_subset": str(input_subset), "input_subset_sha256": sha256(input_subset),
        "agents": str(agents_path), "agents_sha256": sha256(agents_path),
        "synthetic_cot": str(synthetic_path), "synthetic_cot_sha256": sha256(synthetic_path),
        "qwen_all9b_predictions": str(qwen_predictions),
        "qwen_all9b_predictions_sha256": sha256(qwen_predictions),
        "record_v0501_predictions": str(record_predictions),
        "record_v0501_predictions_sha256": sha256(record_predictions),
        "qwen_border_raw": str(qwen_border_raw),
        "qwen_border_raw_sha256": sha256(qwen_border_raw),
        "qwen_fp16_recovered_raw": str(qwen_fp16_raw),
        "qwen_fp16_recovered_raw_sha256": sha256(qwen_fp16_raw),
        "task_file": str(tasks_path), "task_sha256": sha256(tasks_path),
        "rows": len(rows), "tasks": len(tasks), "seed": args.seed,
        "n_proposals": args.n_proposals,
        "prompt_policy": PROMPT_POLICY,
        "prompt_policy_artifact": str(prompt_policy_artifact),
        "prompt_policy_artifact_sha256": sha256(prompt_policy_artifact),
        "prompt_policy_oof_conclusion": prompt_policy_record.get(
            "cross_fitted_conclusion"),
        "prompt_policy_diagnostics": policy_diagnostics,
        "legally_counted_parameter_total": config["verified_parameter_total"],
        "parameter_cap": config["parameter_cap"],
        "validation_labels_in_tasks": False,
        "target_excluded_labeled_icl_examples": True,
        "qwen_gpu_rerun_required": False,
        "gemma_gpu_run_required": True,
        "blind_commitments_included": False,
        "proposal_only_self_consistency": True,
    }
    (plan_dir / "PLAN.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"prepared {len(rows)} validation rows; Gemma tasks={len(tasks)}")
    print(f"Qwen reused exactly from {qwen_predictions}")
    print(f"legal Qwen+Gemma count={config['verified_parameter_total']/1e9:.6f}B")
    print(f"plan: {plan_dir / 'PLAN.json'}")
    return 0


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return row["SubjectEntity"], row["Relation"]


def _indexed(rows: Sequence[Mapping[str, Any]], label: str) -> dict:
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ContractError(f"{label}: duplicate key {key}")
        result[key] = dict(row)
    return result


def _number(objects: Sequence[str]) -> float | None:
    if len(objects) != 1:
        return None
    try:
        value = float(str(objects[0]).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def fuse_objects(qwen: Sequence[str], gemma: Sequence[str], relation: str,
                 policy: str) -> list[str]:
    if relation in NUMERIC_RELATIONS:
        values = [value for value in (_number(qwen), _number(gemma)) if value is not None]
        if not values:
            return []
        if policy == "intersection" and len(values) != 2:
            return []
        return [format(statistics.median(values), ".12g")]
    qmap = {canonical_key(item, relation): item for item in qwen
            if canonical_key(item, relation)}
    gmap = {canonical_key(item, relation): item for item in gemma
            if canonical_key(item, relation)}
    keys = ((list(qmap) + [key for key in gmap if key not in qmap])
            if policy == "union" else [key for key in qmap if key in gmap])
    return [qmap[key] if key in qmap else gmap[key] for key in keys]


def _gold_aliases(row: Mapping[str, Any]) -> list:
    gold = row.get("ObjectEntities", [])
    return [[item] for item in gold] if gold and isinstance(gold[0], str) else gold


def candidate_oracle(qwen_rows: Sequence[Mapping[str, Any]],
                     gemma_rows: Sequence[Mapping[str, Any]],
                     gold_rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    qidx, gidx = _indexed(qwen_rows, "qwen"), _indexed(gemma_rows, "gemma")
    output = []
    for gold in gold_rows:
        key, relation = _key(gold), gold["Relation"]
        # Preserve each numeric estimate as a separate candidate. Averaging is
        # a deployable fusion rule, not a candidate-reservoir operation.
        candidates = []
        seen = set()
        for item in (list(qidx[key].get("ObjectEntities", []))
                     + list(gidx[key].get("ObjectEntities", []))):
            candidate_key = canonical_key(item, relation)
            if candidate_key and candidate_key not in seen:
                candidates.append(item)
                seen.add(candidate_key)
        aliases = _gold_aliases(gold)
        selected, matched = [], 0
        for item in candidates:
            new_matched = true_positives(
                selected + [item], aliases, RELATION_TYPE[relation], 0.05)
            if new_matched > matched:
                selected.append(item)
                matched = new_matched
        output.append({"SubjectEntity": key[0], "Relation": relation,
                       "ObjectEntities": selected})
    return output


def _candidate_map_from_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict:
    result = {}
    for row in rows:
        key = _key(row)
        values, seen = [], set()
        for item in row.get("ObjectEntities", []):
            candidate_key = canonical_key(item, row["Relation"])
            if candidate_key and candidate_key not in seen:
                values.append(item)
                seen.add(candidate_key)
        result[key] = values
    return result


def qwen_raw_candidate_map(plan: Mapping[str, Any], qwen_rows: Sequence[Mapping[str, Any]]) -> dict:
    """Recover exact v0495 Qwen candidates with explicit per-relation coverage."""
    result = _candidate_map_from_prediction_rows(qwen_rows)
    raw_paths = [Path(plan["qwen_border_raw"]), Path(plan["qwen_fp16_recovered_raw"])]
    expected = {
        str(raw_paths[0]): EXPECTED_QWEN_RAW_HASHES["borders"],
        str(raw_paths[1]): EXPECTED_QWEN_RAW_HASHES["fp16_recovered"],
    }
    raw_keys = set()
    for path in raw_paths:
        if sha256(path) != expected[str(path)]:
            raise ContractError(f"Qwen raw provenance changed: {path}")
        for row in read_jsonl(path):
            key, relation = _key(row), row["Relation"]
            raw_keys.add(key)
            values = result.setdefault(key, [])
            seen = {canonical_key(item, relation) for item in values}
            for raw in row.get("raw_samples", []):
                answer = extract_after_think(str(raw))
                for item in parse_answer_items(answer, relation):
                    candidate_key = canonical_key(item, relation)
                    if candidate_key and candidate_key not in seen:
                        values.append(item)
                        seen.add(candidate_key)
    counts = Counter(relation for _, relation in raw_keys)
    required = {"countryLandBordersCountry": 68, "companyTradesAtStockExchange": 100,
                "personHasCityOfDeath": 100, "hasArea": 100, "hasCapacity": 100}
    if dict(counts) != required:
        raise ContractError(f"unexpected v0495 raw relation coverage: {dict(counts)}")
    return result


def gemma_raw_candidate_map(response_map: Mapping[str, Mapping[str, Any]]) -> dict:
    result = {}
    for response in response_map.values():
        key = response["subject"], response["relation"]
        result[key] = [candidate["item"] for candidate in proposal_candidates(response)]
    return result


def oracle_from_candidate_maps(candidate_maps: Sequence[Mapping[tuple, Sequence[str]]],
                               gold_rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    output = []
    for gold in gold_rows:
        key, relation, seen = _key(gold), gold["Relation"], set()
        candidates = []
        for candidate_map in candidate_maps:
            for item in candidate_map.get(key, []):
                candidate_key = canonical_key(item, relation)
                if candidate_key and candidate_key not in seen:
                    candidates.append(item)
                    seen.add(candidate_key)
        aliases = _gold_aliases(gold)
        selected, matched = [], 0
        for item in candidates:
            new_matched = true_positives(
                selected + [item], aliases, RELATION_TYPE[relation], 0.05)
            if new_matched > matched:
                selected.append(item)
                matched = new_matched
        output.append({"SubjectEntity": key[0], "Relation": relation,
                       "ObjectEntities": selected})
    return output


def complementarity_diagnostics(qwen_candidates: Mapping[tuple, Sequence[str]],
                                gemma_candidates: Mapping[tuple, Sequence[str]],
                                gold_rows: Sequence[Mapping[str, Any]]) -> dict:
    overall, by_relation = Counter(), {}
    relation_counts: dict[str, Counter] = {}
    for gold in gold_rows:
        key, relation, aliases = _key(gold), gold["Relation"], _gold_aliases(gold)
        if not aliases:
            continue
        qhit = true_positives(list(qwen_candidates.get(key, [])), aliases,
                              RELATION_TYPE[relation], 0.05) > 0
        ghit = true_positives(list(gemma_candidates.get(key, [])), aliases,
                              RELATION_TYPE[relation], 0.05) > 0
        bucket = "both" if qhit and ghit else "qwen_only" if qhit else "gemma_only" if ghit else "neither"
        overall[bucket] += 1
        relation_counts.setdefault(relation, Counter())[bucket] += 1
    for relation, counts in sorted(relation_counts.items()):
        by_relation[relation] = dict(counts)
    return {"nonempty_gold_rows": sum(overall.values()),
            "overall": dict(overall), "by_relation": by_relation}


def complete_answer_oracle(qwen_rows: Sequence[Mapping[str, Any]],
                           gemma_rows: Sequence[Mapping[str, Any]],
                           gold_rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    qidx, gidx = _indexed(qwen_rows, "qwen"), _indexed(gemma_rows, "gemma")
    qordered = [qidx[_key(row)] for row in gold_rows]
    gordered = [gidx[_key(row)] for row in gold_rows]
    qs = evaluate_per_sr_pair(qordered, list(gold_rows), RELATION_TYPE, tolerance=0.05)
    gs = evaluate_per_sr_pair(gordered, list(gold_rows), RELATION_TYPE, tolerance=0.05)
    qscore = {(row["SubjectEntity"], row["Relation"]): row["f1"] for row in qs}
    gscore = {(row["SubjectEntity"], row["Relation"]): row["f1"] for row in gs}
    return [dict(gidx[key] if gscore[key] > qscore[key] else qidx[key])
            for key in [_key(row) for row in gold_rows]]


def _load_validated_responses(plan_dir: Path, response_path: Path) -> tuple[list[dict], dict]:
    tasks = read_jsonl(plan_dir / "tasks" / f"{GEMMA}.jsonl")
    responses = read_jsonl(response_path)
    task_map = {row["task_id"]: row for row in tasks}
    response_map = {row.get("task_id"): row for row in responses}
    if len(task_map) != len(tasks) or len(response_map) != len(responses):
        raise ContractError("duplicate task or response id")
    if set(task_map) != set(response_map):
        missing = sorted(set(task_map) - set(response_map))[:5]
        extra = sorted(set(response_map) - set(task_map))[:5]
        raise ContractError(f"response coverage mismatch: missing={missing}, extra={extra}")
    for task_id, task in task_map.items():
        validate_task_response(task, response_map[task_id])
    manifest_path = response_path.with_suffix(response_path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ContractError(f"missing response manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("output_sha256") != sha256(response_path):
        raise ContractError("Gemma response manifest hash mismatch")
    return tasks, response_map


def proposal_only_prediction(response: Mapping[str, Any]) -> list[str]:
    """Plain self-consistency control with no heterogeneous-router signals."""
    relation = response["relation"]
    candidates = proposal_candidates(response)
    generations = response.get("generations", [])
    if relation in NUMERIC_RELATIONS:
        values = []
        for candidate in candidates:
            try:
                value = float(str(candidate["item"]).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                values.extend([value] * int(candidate["support"]))
        return [format(statistics.median(values), ".12g")] if values else []
    if relation in SINGLE_RELATIONS:
        none_support = sum(
            proposal_parse_status(str(generation), relation)[0] == "explicit_none"
            for generation in generations)
        if not candidates or none_support >= int(candidates[0]["support"]):
            return []
        return [candidates[0]["item"]]
    threshold = max(1, math.ceil(len(generations) / 2))
    return [candidate["item"] for candidate in candidates
            if int(candidate["support"]) >= threshold]


def evaluate(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_dir = output / "plan"
    result_dir = output / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((plan_dir / "PLAN.json").read_text())
    input_rows = read_jsonl(plan_dir / "input_subset.jsonl")
    gold = read_jsonl(Path(args.gold).resolve())
    if [_key(row) for row in input_rows] != [_key(row) for row in gold]:
        raise ContractError("gold rows do not exactly match the frozen task order")
    config = load_agent_config(Path(plan["agents"]))
    gemma_agent = _agent(config, GEMMA)
    response_path = output / "responses" / f"{GEMMA}.jsonl"
    _, response_map = _load_validated_responses(plan_dir, response_path)
    gemma_rows = []
    for source in input_rows:
        matches = [response for response in response_map.values()
                   if response["subject"] == source["SubjectEntity"]
                   and response["relation"] == source["Relation"]]
        if len(matches) != 1:
            raise ContractError(f"expected one Gemma proposal for {_key(source)}, got {len(matches)}")
        gemma_rows.append({"SubjectEntity": source["SubjectEntity"],
                           "Relation": source["Relation"],
                           "ObjectEntities": proposal_only_prediction(matches[0])})
    qwen_rows = read_jsonl(Path(plan["qwen_all9b_predictions"]))
    record_rows = read_jsonl(Path(plan["record_v0501_predictions"]))
    qidx, gidx = _indexed(qwen_rows, "qwen"), _indexed(gemma_rows, "gemma")
    expected_keys = [_key(row) for row in gold]
    if set(qidx) != set(expected_keys) or set(gidx) != set(expected_keys):
        raise ContractError("Qwen/Gemma predictions do not cover validation exactly")
    qordered, gordered = [qidx[key] for key in expected_keys], [gidx[key] for key in expected_keys]
    policies = {"qwen_all9b_v0495": qordered, "gemma": gordered,
                "record_v0501_reference": record_rows}
    for policy in ("union", "intersection"):
        policies[f"simple_dual_{policy}"] = [
            {"SubjectEntity": key[0], "Relation": key[1],
             "ObjectEntities": fuse_objects(
                 qidx[key].get("ObjectEntities", []), gidx[key].get("ObjectEntities", []),
                 key[1], policy)}
            for key in expected_keys
        ]
    policies["complete_answer_oracle"] = complete_answer_oracle(qordered, gordered, gold)
    policies["candidate_union_oracle"] = candidate_oracle(qordered, gordered, gold)
    qwen_raw_candidates = qwen_raw_candidate_map(plan, qordered)
    gemma_raw_candidates = gemma_raw_candidate_map(response_map)
    policies["raw_proposal_union_oracle"] = oracle_from_candidate_maps(
        [qwen_raw_candidates, gemma_raw_candidates], gold)
    scores = {name: score(rows, gold) for name, rows in policies.items()}
    for oracle in ("complete_answer_oracle", "candidate_union_oracle",
                   "raw_proposal_union_oracle"):
        for source in ("qwen_all9b_v0495", "gemma"):
            if scores[oracle]["*** All Relations ***"] + 1e-12 < scores[source]["*** All Relations ***"]:
                raise ContractError(f"{oracle} fell below source policy {source}")
    for name, rows in policies.items():
        write_jsonl_atomic(result_dir / f"{name}.jsonl", rows)
    payload = {
        "schema": "qwen-gemma-simple-validation-result-v1",
        "development_validation_not_blind_test": True,
        "plan_sha256": sha256(plan_dir / "PLAN.json"),
        "gemma_response_sha256": sha256(response_path),
        "scores": scores,
        "diagnostic_non_deployable": ["complete_answer_oracle", "candidate_union_oracle",
                                       "raw_proposal_union_oracle"],
        "final_output_complementarity": complementarity_diagnostics(
            _candidate_map_from_prediction_rows(qordered),
            _candidate_map_from_prediction_rows(gordered), gold),
        "raw_proposal_complementarity": complementarity_diagnostics(
            qwen_raw_candidates, gemma_raw_candidates, gold),
        "qwen_raw_coverage": {
            "countryLandBordersCountry": "10-sample raw",
            "companyTradesAtStockExchange": "10-sample raw",
            "personHasCityOfDeath": "10-sample raw",
            "hasArea": "10-sample raw",
            "hasCapacity": "10-sample recovered raw",
            "awardWonBy": "final iterative System-2 output only",
        },
        "novel_router_included": False,
        "blind_commitments_included": False,
    }
    (result_dir / "RESULT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    relations = ["*** All Relations ***", "countryLandBordersCountry",
                 "companyTradesAtStockExchange", "personHasCityOfDeath",
                 "hasArea", "hasCapacity", "awardWonBy"]
    lines = ["# Simple Qwen + Gemma validation baseline", "",
             "This is a labeled-validation development comparison, not a blind-test result.",
             "The heterogeneous-memory router is intentionally absent.", "",
             "Union and intersection are deliberately weak structural controls; poor scores do not imply that Gemma adds no knowledge.",
             "Shared-CoT borders/capacity are model-heterogeneous but prompt-coupled.",
             "Numeric union/intersection use a naive two-point median (arithmetic mean when both exist), not a ceiling.",
             "Prompt routes were cross-fitted in five folds on the 130-row train pilot. The OOF verdict was mixed: routing beat direct prompting but not the best static prompt.", "",
             "| policy | " + " | ".join(relations) + " |",
             "|---|" + "|".join(["---:"] * len(relations)) + "|"]
    for name, values in scores.items():
        lines.append("| " + name + " | " + " | ".join(
            f"{values.get(relation, float('nan')):.4f}" for relation in relations) + " |")
    lines += ["", "All `*oracle` policies use gold labels and are non-deployable.",
              "`raw_proposal_union_oracle` uses exact v0495 raw caches for five relations and the final iterative Qwen award output.", "",
              "## Unique correct-candidate coverage", "", "```json",
              json.dumps(payload["raw_proposal_complementarity"], indent=2, sort_keys=True),
              "```", ""]
    (result_dir / "RESULT.md").write_text("\n".join(lines))
    print(f"evaluation complete: {result_dir / 'RESULT.md'}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--input", default=str(ROOT / "data/val.jsonl"))
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--agents", default=str(Path(__file__).with_name("agents_qwen_gemma_synthetic.json")))
    prep.add_argument("--synthetic-cot", default=str(ROOT / "data/synthetic_cot_faithful.jsonl"))
    prep.add_argument("--qwen-predictions", default=str(ROOT / "baselines/v0495_2026-07-14/pred_hybrid_0495.jsonl"))
    prep.add_argument("--record-predictions", default=str(ROOT / "baselines/v0501_2026-07-15/pred_hybrid_0501.jsonl"))
    prep.add_argument("--qwen-border-raw", default=str(ROOT / "archive/experiments/architecture_validation_20260713/border_4bit/raw.jsonl"))
    prep.add_argument("--qwen-fp16-raw", default=str(ROOT / "baselines/v0491_2026-07-14/system1_fp16_raw_capacity_recovered.jsonl"))
    prep.add_argument("--prompt-policy-artifact", default=str(PROMPT_POLICY_ARTIFACT))
    prep.add_argument("--seed", type=int, default=20260720)
    prep.add_argument("--n-proposals", type=int, default=3)
    prep.set_defaults(func=prepare)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--output-dir", required=True)
    ev.add_argument("--gold", default=str(ROOT / "data/val.jsonl"))
    ev.set_defaults(func=evaluate)
    return ap


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
