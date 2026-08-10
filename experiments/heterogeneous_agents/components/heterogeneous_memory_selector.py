#!/usr/bin/env python3
"""Inference-legal heterogeneous parametric-memory candidate routing.

This module is intentionally separate from candidate generation.  It builds a
typed Qwen/Gemma candidate graph, fits transparent probability calibrators on
labeled *training* rows, and decodes validation/test graphs without labels.

The CLI keeps the leakage boundary explicit:

* ``build-train-graph`` and ``build-validation-graph`` never open gold labels.
* ``fit`` is the only command that opens training labels.
* ``decode`` refuses labels and consumes only a frozen calibration artifact.
* ``evaluate`` is post-hoc reporting and never changes the artifact.

The current validation proposal run deliberately omitted blind commitments.
``prepare-commitments`` creates resumable choice-scoring tasks for those
signals.  Missing commitments are represented explicitly; they are never
silently inferred from a proposal or candidate identity.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Permit both ``python -m experiments...`` and direct review/debug execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluate import RELATION_TYPE, true_positives
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    NULLABLE_RELATIONS,
    NUMERIC_RELATIONS,
    SINGLE_RELATIONS,
    build_agent_tasks,
    canonical_key,
    load_agent_config,
    load_synthetic_by_relation,
    normalize_string,
    proposal_candidates,
    proposal_parse_status,
    read_jsonl,
    sha256,
    validate_task_response,
    validate_inputs,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import (
    GEMMA,
    PROMPT_POLICY,
    QWEN,
    proposal_only_prediction,
)
from architecture_candidate_v3 import company_prediction, city_prediction
from numeric_aggregation import aggregate_quantile
from run_inference import (aggregate, aggregate_numeric_cluster,
                           extract_after_think, parse_answer_items)


ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "experiments/heterogeneous_agents/runs"
PILOT_DIRECT = RUN_ROOT / "portfolio_pilot_20260720_v1"
PILOT_SHARED = RUN_ROOT / "gemma_synthetic_ablation_20260721_v1"
PILOT_DISJOINT = RUN_ROOT / "gemma_synthetic_disjoint_ablation_20260721_v1"
DUAL_VALIDATION = RUN_ROOT / "dual_model_validation_fast_oofroute_20260721_v1"
FOLD_MANIFEST = RUN_ROOT / "gemma_prompt_oof_20260721_v1/FOLDS.jsonl"
OOF_CONFIRMATION = RUN_ROOT / "gemma_prompt_oof_20260721_v1/OOF_CONFIRMATION.json"
QWEN_TRAIN_OOF = ROOT / "archive/experiments/overnight_robust_20260711/a0_legacy"

RELATIONS = tuple(sorted(RELATION_TYPE))
PROMPT_POLICIES = ("direct", "shared_cot5", "disjoint_cot5")
AGENTS = (QWEN, GEMMA)
DECODER_UTILITY_SCHEMA = "symmetric-output-utility-v2"


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _indexed(rows: Sequence[Mapping[str, Any]], label: str) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ContractError(f"{label}: duplicate key {key}")
        result[key] = dict(row)
    return result


def _load_graph(path: Path, *, expected_split: str | None = None) -> list[dict]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = _json(manifest_path)
    if manifest.get("schema") != "heterogeneous-memory-graph-manifest-v1":
        raise ContractError(f"unsupported graph manifest: {manifest_path}")
    if manifest.get("gold_aware") is True:
        raise ContractError(f"gold-aware graph is not inference-legal: {manifest_path}")
    if manifest.get("contains_labels") is not False:
        raise ContractError(f"graph does not certify label exclusion: {manifest_path}")
    if manifest.get("output_sha256") != sha256(path):
        raise ContractError(f"graph hash mismatch: {path}")
    if expected_split is not None and manifest.get("split") != expected_split:
        raise ContractError(
            f"graph split mismatch: {manifest.get('split')!r} != {expected_split!r}")
    rows = read_jsonl(path)
    if int(manifest.get("rows", -1)) != len(rows):
        raise ContractError(f"graph row-count mismatch: {path}")
    if len({_key(row) for row in rows}) != len(rows):
        raise ContractError(f"duplicate graph key: {path}")
    if any(row.get("schema") != "heterogeneous-memory-graph-row-v1" for row in rows):
        raise ContractError(f"graph row schema mismatch: {path}")
    return rows


def _response_map(path: Path, *, expected_agent: str | None = None) -> dict[str, dict]:
    if not path.is_file():
        raise ContractError(f"missing response artifact: {path}")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ContractError(f"missing response manifest: {manifest_path}")
    manifest = _json(manifest_path)
    if manifest.get("output_sha256") != sha256(path):
        raise ContractError(f"response hash mismatch: {path}")
    if expected_agent is not None and manifest.get("agent_id") != expected_agent:
        raise ContractError(f"response agent mismatch: {path}")
    result = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id in result:
            raise ContractError(f"invalid or duplicate task id in {path}: {task_id!r}")
        if expected_agent is not None and row.get("agent_id") != expected_agent:
            raise ContractError(f"foreign response in {path}: {task_id}")
        result[task_id] = row
    return result


def _phase(response_map: Mapping[str, Mapping[str, Any]], subject: str,
           relation: str, phase: str) -> dict:
    rows = [dict(row) for row in response_map.values()
            if row.get("subject") == subject and row.get("relation") == relation
            and row.get("phase") == phase]
    if len(rows) != 1:
        raise ContractError(f"expected one {phase} response for {(subject, relation)}, got {len(rows)}")
    return rows[0]


def _commitment(response_map: Mapping[str, Mapping[str, Any]] | None,
                subject: str, relation: str, phase: str) -> dict:
    if response_map is None:
        return {"available": False, "selected": None, "probabilities": {}}
    try:
        row = _phase(response_map, subject, relation, phase)
    except ContractError:
        return {"available": False, "selected": None, "probabilities": {}}
    probabilities = row.get("choice_probabilities", {})
    if not isinstance(probabilities, dict):
        raise ContractError(f"invalid commitment probabilities for {(subject, relation, phase)}")
    return {
        "available": True,
        "selected": row.get("selected_choice"),
        "probabilities": {str(key): float(value) for key, value in probabilities.items()},
    }


def _proposal_summary(response: Mapping[str, Any]) -> dict:
    generations = list(response.get("generations", []))
    relation = str(response["relation"])
    statuses = [proposal_parse_status(str(text), relation)[0] for text in generations]
    candidates = proposal_candidates(response)
    return {
        "n_samples": len(generations),
        "none_count": statuses.count("explicit_none"),
        "parse_failures": sum(status in {
            "missing_answer_field", "conflicting_answer_fields", "unparseable_answer_field"
        } for status in statuses),
        "candidates": candidates,
    }


def _qwen_raw_response(row: Mapping[str, Any]) -> dict:
    relation = str(row["Relation"])
    converted = []
    for raw in row.get("raw_samples", []):
        answer = extract_after_think(str(raw)).strip()
        items = parse_answer_items(answer, relation) if answer else []
        normalized = normalize_string(answer)
        if normalized in {"none", "null", "no answer"}:
            converted.append("ANSWER: None")
        elif items:
            converted.append("ANSWER: " + ", ".join(items))
        else:
            # Keep failures explicit so support denominators cannot silently shrink.
            converted.append("")
    return {
        "agent_id": QWEN,
        "subject": row["SubjectEntity"],
        "relation": relation,
        "phase": "propose",
        "mode": "generate",
        "generations": converted,
    }


def _synthetic_response(subject: str, relation: str, objects: Sequence[str]) -> dict:
    answer = "None" if not objects else ", ".join(str(item) for item in objects)
    return {
        "agent_id": QWEN,
        "subject": subject,
        "relation": relation,
        "phase": "propose",
        "mode": "generate",
        "generations": [f"ANSWER: {answer}"],
    }


def _numeric_dispersion(candidates: Sequence[Mapping[str, Any]]) -> float | None:
    values = []
    for candidate in candidates:
        try:
            value = float(str(candidate["item"]).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            values.extend([value] * max(1, int(candidate.get("support", 1))))
    if len(values) < 2:
        return 0.0 if values else None
    logs = [math.log(value) for value in values]
    center = statistics.median(logs)
    return statistics.median(abs(value - center) for value in logs)


def _make_graph(index: int, subject: str, relation: str,
                proposals: Mapping[str, Mapping[str, Any]],
                final_objects: Mapping[str, Sequence[str]],
                commitments: Mapping[str, Mapping[str, Mapping[str, Any]]],
                prompt_policy: str) -> dict:
    summaries = {agent: _proposal_summary(proposals[agent]) for agent in AGENTS}
    nodes: dict[str, dict] = {}
    for agent in AGENTS:
        n_samples = int(summaries[agent]["n_samples"])
        for candidate in summaries[agent]["candidates"]:
            key = str(candidate["key"])
            node = nodes.setdefault(key, {
                "key": key, "item": candidate["item"], "type": (
                    "numeric" if relation in NUMERIC_RELATIONS else "string"),
                "sources": {}, "selected_by": {},
            })
            count = int(candidate["support"])
            node["sources"][agent] = {
                "support": count,
                "samples": n_samples,
                "support_rate": count / n_samples if n_samples else 0.0,
            }
    for agent in AGENTS:
        selected_keys = {
            canonical_key(str(item), relation) for item in final_objects.get(agent, [])
            if canonical_key(str(item), relation)
        }
        for node in nodes.values():
            node["selected_by"][agent] = node["key"] in selected_keys
    ordered = sorted(nodes.values(), key=lambda node: (
        -len(node["sources"]),
        -sum(source["support_rate"] for source in node["sources"].values()),
        node["key"],
    ))
    return {
        "schema": "heterogeneous-memory-graph-row-v1",
        "SubjectEntity": subject,
        "Relation": relation,
        "input_index": index,
        "prompt_policy": prompt_policy,
        "baseline_agent": QWEN,
        "baseline_objects": list(final_objects.get(QWEN, [])),
        "agent_outputs": {agent: list(final_objects.get(agent, [])) for agent in AGENTS},
        "agents": {
            agent: {
                "n_samples": int(summaries[agent]["n_samples"]),
                "none_count": int(summaries[agent]["none_count"]),
                "none_rate": (summaries[agent]["none_count"] /
                              summaries[agent]["n_samples"]
                              if summaries[agent]["n_samples"] else 0.0),
                "parse_failures": int(summaries[agent]["parse_failures"]),
                "numeric_log_mad": _numeric_dispersion(summaries[agent]["candidates"]),
                "existence": commitments[agent]["existence"],
                "cardinality": commitments[agent]["cardinality"],
            } for agent in AGENTS
        },
        "candidates": ordered,
    }


def _load_pilot_variant(run: Path, agent: str) -> dict[str, dict]:
    return _response_map(run / f"responses/{agent}.jsonl", expected_agent=agent)


def _load_qwen_train_oof(root: Path) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict], dict]:
    raw_by_key, prediction_by_key, provenance = {}, {}, {}
    for fold in range(5):
        directory = root / f"fold_{fold}"
        raw_path, prediction_path = directory / "raw.jsonl", directory / "predictions.jsonl"
        manifest_path = raw_path.with_suffix(raw_path.suffix + ".manifest.json")
        manifest = _json(manifest_path)
        if manifest.get("raw_cache_sha256") != sha256(raw_path):
            raise ContractError(f"Qwen train OOF raw hash mismatch: {raw_path}")
        if manifest.get("output_sha256") != sha256(prediction_path):
            raise ContractError(f"Qwen train OOF prediction hash mismatch: {prediction_path}")
        if int(manifest.get("n_consistency", -1)) != 10:
            raise ContractError(f"Qwen train OOF fold is not N=10: {raw_path}")
        raw_rows, prediction_rows = read_jsonl(raw_path), read_jsonl(prediction_path)
        for row in raw_rows:
            key = _key(row)
            if key in raw_by_key:
                raise ContractError(f"duplicate Qwen train OOF raw key: {key}")
            raw_by_key[key] = _qwen_raw_response(row)
        for row in prediction_rows:
            key = _key(row)
            if key in prediction_by_key:
                raise ContractError(f"duplicate Qwen train OOF prediction key: {key}")
            prediction_by_key[key] = dict(row)
        provenance[str(fold)] = {
            "raw": str(raw_path), "raw_sha256": sha256(raw_path),
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
        }
    if len(raw_by_key) != 477 or set(raw_by_key) != set(prediction_by_key):
        raise ContractError("Qwen train OOF artifacts do not cover 477 unique rows")
    # Recompose the portions of the frozen v0495 policy that need only the
    # available Qwen samples. System-2 corroboration is unavailable on train,
    # so company/city use their predeclared high-support paths only. Award
    # retains the fold prediction because the production route is iterative.
    original_raw = {}
    for fold in range(5):
        for row in read_jsonl(root / f"fold_{fold}/raw.jsonl"):
            original_raw[_key(row)] = row
    for key, row in original_raw.items():
        subject, relation = key
        samples = list(row.get("raw_samples", []))
        if relation == "countryLandBordersCountry":
            objects = aggregate(
                relation, subject, samples,
                response_protocol="legacy-cot", aggregation_profile="relation-v1")
        elif relation == "companyTradesAtStockExchange":
            objects = company_prediction(subject, samples, [])
        elif relation == "personHasCityOfDeath":
            objects = city_prediction(subject, samples, [], corroborate=False)
        elif relation == "hasArea":
            objects = aggregate_quantile([extract_after_think(sample) for sample in samples], .55)
        elif relation == "hasCapacity":
            objects = aggregate_numeric_cluster(
                [extract_after_think(sample) for sample in samples], .30)
        else:
            continue
        prediction_by_key[key] = {
            "SubjectEntity": subject, "Relation": relation, "ObjectEntities": objects}
    provenance["production_like_reaggregation"] = {
        "borders": "relation-v1",
        "company": "K4 Qwen-only; no train System-2 corroborator",
        "city": "K6 Qwen-only; no train System-2 corroborator",
        "area": "strict quantile 0.55",
        "capacity": "relative cluster 0.30",
        "award": "legacy fold prediction; iterative train artifact unavailable",
    }
    return raw_by_key, prediction_by_key, provenance


def _oof_prompt_routes(path: Path) -> dict[tuple[str, str], str]:
    record = _json(path)
    if record.get("selection_uses_holdout_labels") is not False:
        raise ContractError("prompt OOF artifact does not certify holdout-label isolation")
    routes = {}
    for selection in record.get("selections", []):
        policy = str(selection.get("selected_policy"))
        if policy not in PROMPT_POLICIES:
            raise ContractError(f"invalid OOF prompt policy: {policy}")
        for raw_key in selection.get("holdout_keys", []):
            key = tuple(raw_key)
            if len(key) != 2 or key in routes:
                raise ContractError(f"invalid/duplicate OOF prompt key: {raw_key}")
            routes[(str(key[0]), str(key[1]))] = policy
    return routes


def build_train_graph(args: argparse.Namespace) -> int:
    direct, shared, disjoint = map(Path, (args.direct_run, args.shared_run, args.disjoint_run))
    input_path = direct / "plan/input_subset.jsonl"
    inputs = read_jsonl(input_path)
    validate_inputs(inputs)
    qwen = _load_pilot_variant(direct, QWEN)
    qwen_oof_raw, qwen_oof_predictions, qwen_oof_provenance = ({}, {}, {})
    if args.qwen_proposals == "oof-n10":
        qwen_oof_raw, qwen_oof_predictions, qwen_oof_provenance = _load_qwen_train_oof(
            Path(args.qwen_oof_run).resolve())
    variants = {
        "direct": _load_pilot_variant(direct, GEMMA),
        "shared_cot5": _load_pilot_variant(shared, GEMMA),
        "disjoint_cot5": _load_pilot_variant(disjoint, GEMMA),
    }
    oof_routes = (_oof_prompt_routes(Path(args.oof_confirmation).resolve())
                  if args.training_prompt_routing == "oof" else {})
    if oof_routes and set(oof_routes) != {_key(row) for row in inputs}:
        raise ContractError("OOF prompt routes do not exactly cover the pilot inputs")
    graphs = []
    for index, source in enumerate(inputs):
        subject, relation = _key(source)
        qproposal = (qwen_oof_raw[(subject, relation)] if qwen_oof_raw
                     else _phase(qwen, subject, relation, "propose"))
        qsummary = (qwen_oof_predictions[(subject, relation)].get("ObjectEntities", [])
                    if qwen_oof_predictions else proposal_only_prediction(qproposal))
        qcommit = {
            "existence": _commitment(qwen, subject, relation, "commit_existence"),
            "cardinality": _commitment(qwen, subject, relation, "commit_cardinality"),
        }
        prompt_variants = {}
        for policy in PROMPT_POLICIES:
            gproposal = _phase(variants[policy], subject, relation, "propose")
            gsummary = proposal_only_prediction(gproposal)
            commits = {
                QWEN: qcommit,
                GEMMA: {
                    "existence": _commitment(
                        variants[policy], subject, relation, "commit_existence"),
                    "cardinality": _commitment(
                        variants[policy], subject, relation, "commit_cardinality"),
                },
            }
            prompt_variants[policy] = _make_graph(
                index, subject, relation,
                {QWEN: qproposal, GEMMA: gproposal},
                {QWEN: qsummary, GEMMA: gsummary}, commits, policy)
        # Preserve the prior row-level OOF view for inspection. ``fit`` does
        # not trust this view for its outer-fold estimate: it selects one
        # relation policy using only each outer fold's training partition.
        display_policy = (oof_routes[(subject, relation)] if oof_routes
                          else PROMPT_POLICY[relation])
        graph = copy.deepcopy(prompt_variants[display_policy])
        graph["prompt_variants"] = prompt_variants
        graphs.append(graph)
    output = Path(args.output).resolve()
    write_jsonl_atomic(output, graphs)
    legal_config_path = ROOT / "experiments/heterogeneous_agents/agents_qwen_gemma_synthetic.json"
    legal_config = load_agent_config(legal_config_path)
    manifest = {
        "schema": "heterogeneous-memory-graph-manifest-v1",
        "split": "train-pilot", "rows": len(graphs),
        "contains_labels": False, "output_sha256": sha256(output),
        "input": str(input_path.resolve()), "input_sha256": sha256(input_path),
        "prompt_policy": PROMPT_POLICY,
        "training_prompt_routing": args.training_prompt_routing,
        "oof_confirmation": (str(Path(args.oof_confirmation).resolve())
                             if oof_routes else None),
        "oof_confirmation_sha256": (sha256(Path(args.oof_confirmation).resolve())
                                    if oof_routes else None),
        "sources": {name: str(path.resolve()) for name, path in {
            "direct": direct, "shared_cot5": shared, "disjoint_cot5": disjoint}.items()},
        "qwen_samples": sorted({g["agents"][QWEN]["n_samples"] for g in graphs}),
        "gemma_samples": sorted({g["agents"][GEMMA]["n_samples"] for g in graphs}),
        "legal_agent_config": str(legal_config_path),
        "legal_agent_config_sha256": sha256(legal_config_path),
        "verified_parameter_total": legal_config["verified_parameter_total"],
        "parameter_cap": legal_config["parameter_cap"],
        "qwen_proposals": args.qwen_proposals,
        "qwen_oof_run": (str(Path(args.qwen_oof_run).resolve()) if qwen_oof_raw else None),
        "qwen_oof_provenance": qwen_oof_provenance,
        "prompt_variant_storage": list(PROMPT_POLICIES),
        "selector_cross_validation": "nested fold-local relation prompt selection",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"train graph complete: {output} ({len(graphs)} rows)")
    return 0


def _load_commitment_maps(directory: Path | None) -> dict[str, dict[str, dict] | None]:
    if directory is None:
        return {agent: None for agent in AGENTS}
    plan_path = directory.parent / "PLAN.json"
    plan = _json(plan_path)
    if plan.get("schema") != "heterogeneous-memory-commitment-plan-v1":
        raise ContractError(f"unsupported commitment plan: {plan_path}")
    if plan.get("contains_labels") is not False:
        raise ContractError(f"commitment plan does not certify label exclusion: {plan_path}")
    task_files = plan.get("task_files")
    if not isinstance(task_files, dict) or set(task_files) != set(AGENTS):
        raise ContractError(f"commitment plan agent coverage mismatch: {plan_path}")
    loaded: dict[str, dict[str, dict]] = {}
    for agent in AGENTS:
        task_record = task_files[agent]
        task_path = Path(task_record["path"]).resolve()
        if task_record.get("sha256") != sha256(task_path):
            raise ContractError(f"commitment task hash mismatch: {task_path}")
        tasks = read_jsonl(task_path)
        task_by_id = {str(task["task_id"]): task for task in tasks}
        if len(task_by_id) != len(tasks):
            raise ContractError(f"duplicate commitment task ids: {task_path}")
        if int(task_record.get("tasks", -1)) != len(tasks):
            raise ContractError(f"commitment task count mismatch: {task_path}")
        response_path = directory / f"{agent}.jsonl"
        responses = _response_map(response_path, expected_agent=agent)
        response_manifest = _json(
            response_path.with_suffix(response_path.suffix + ".manifest.json"))
        if response_manifest.get("task_sha256") != sha256(task_path):
            raise ContractError(
                f"response was produced from a different task artifact: {response_path}")
        if set(responses) != set(task_by_id):
            missing = sorted(set(task_by_id) - set(responses))[:3]
            extra = sorted(set(responses) - set(task_by_id))[:3]
            raise ContractError(
                f"commitment response coverage mismatch for {agent}: "
                f"missing={missing}, extra={extra}")
        for task_id, task in task_by_id.items():
            validate_task_response(task, responses[task_id])
        loaded[agent] = responses
    return loaded


def build_validation_graph(args: argparse.Namespace) -> int:
    run = Path(args.run).resolve()
    plan = _json(run / "plan/PLAN.json")
    inputs = read_jsonl(run / "plan/input_subset.jsonl")
    validate_inputs(inputs)
    tasks = read_jsonl(Path(plan["task_file"]))
    task_by_key = {(str(row["subject"]), str(row["relation"])): row for row in tasks}
    gemma_map = _response_map(run / f"responses/{GEMMA}.jsonl", expected_agent=GEMMA)
    gemma_by_key = {
        (str(row["subject"]), str(row["relation"])): row
        for row in gemma_map.values() if row.get("phase") == "propose"
    }
    qwen_raw: dict[tuple[str, str], dict] = {}
    for raw_path in (Path(plan["qwen_border_raw"]), Path(plan["qwen_fp16_recovered_raw"])):
        for row in read_jsonl(raw_path):
            key = _key(row)
            if key in qwen_raw:
                raise ContractError(f"duplicate Qwen raw row {key}")
            qwen_raw[key] = _qwen_raw_response(row)
    # The heterogeneous portfolio is Qwen-9B + Gemma-12B. v0501 is only an
    # evaluation reference because its Qwen-2.5-14B area route would make the
    # three-checkpoint portfolio exceed the 32B cap.
    qwen_fallback = _indexed(read_jsonl(Path(plan["qwen_all9b_predictions"])), "qwen-v0495")
    commitment_dir = Path(args.commitment_dir).resolve() if args.commitment_dir else None
    commitment_maps = _load_commitment_maps(commitment_dir)
    graphs = []
    for index, source in enumerate(inputs):
        subject, relation = _key(source)
        key = subject, relation
        qobjects = qwen_fallback[key].get("ObjectEntities", [])
        qproposal = qwen_raw.get(key)
        if qproposal is None:
            # Award has no exact Qwen raw cache; preserve the iterative final output.
            qproposal = _synthetic_response(subject, relation, qobjects)
        if key not in gemma_by_key or key not in task_by_key:
            raise ContractError(f"missing Gemma validation proposal/task for {key}")
        gproposal = gemma_by_key[key]
        commits = {
            agent: {
                "existence": _commitment(commitment_maps[agent], subject, relation,
                                         "commit_existence"),
                "cardinality": _commitment(commitment_maps[agent], subject, relation,
                                           "commit_cardinality"),
            } for agent in AGENTS
        }
        graphs.append(_make_graph(
            index, subject, relation,
            {QWEN: qproposal, GEMMA: gproposal},
            {QWEN: qobjects, GEMMA: proposal_only_prediction(gproposal)},
            commits, str(task_by_key[key].get("prompt_policy", PROMPT_POLICY[relation]))))
    output = Path(args.output).resolve()
    write_jsonl_atomic(output, graphs)
    available = {
        agent: sum(graph["agents"][agent]["existence"]["available"]
                   or graph["agents"][agent]["cardinality"]["available"]
                   for graph in graphs) for agent in AGENTS
    }
    manifest = {
        "schema": "heterogeneous-memory-graph-manifest-v1",
        "split": "validation", "rows": len(graphs), "contains_labels": False,
        "output_sha256": sha256(output), "source_plan": str((run / 'plan/PLAN.json')),
        "source_plan_sha256": sha256(run / "plan/PLAN.json"),
        "commitment_dir": str(commitment_dir) if commitment_dir else None,
        "rows_with_any_commitment": available,
        "prompt_policy": PROMPT_POLICY,
        "qwen_raw_award_limitation": "iterative final output only",
        "legal_baseline": "all-9B v0495",
        "v0501_usage": "comparison only; excluded because adding its 14B area model would exceed 32B",
        "verified_parameter_total": plan["legally_counted_parameter_total"],
        "parameter_cap": plan["parameter_cap"],
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"validation graph complete: {output} ({len(graphs)} rows)")
    if not commitment_dir:
        print("WARNING: blind commitments are explicitly missing; prepare/run them before production decode")
    return 0


def prepare_commitments(args: argparse.Namespace) -> int:
    input_path, agents_path = Path(args.input).resolve(), Path(args.agents).resolve()
    inputs = read_jsonl(input_path)
    validate_inputs(inputs)
    config = load_agent_config(agents_path)
    selected = [agent for agent in config["agents"] if agent["id"] in AGENTS]
    if {agent["id"] for agent in selected} != set(AGENTS):
        raise ContractError("commitment config must contain Qwen and Gemma")
    synthetic = load_synthetic_by_relation(Path(args.synthetic_cot).resolve())
    output = Path(args.output_dir).resolve()
    task_dir = output / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    label_free_input = output / "input_subset.jsonl"
    write_jsonl_atomic(label_free_input, [{
        "SubjectEntity": row["SubjectEntity"], "Relation": row["Relation"]}
        for row in inputs])
    task_records = {}
    for agent in selected:
        all_tasks = build_agent_tasks(inputs, agent, synthetic, seed=args.seed, n_proposals=1)
        tasks = [row for row in all_tasks if row["phase"] != "propose"]
        path = task_dir / f"{agent['id']}.jsonl"
        write_jsonl_atomic(path, tasks)
        task_records[agent["id"]] = {"path": str(path), "sha256": sha256(path),
                                     "tasks": len(tasks)}
    manifest = {
        "schema": "heterogeneous-memory-commitment-plan-v1",
        "source_input": str(input_path), "source_input_sha256": sha256(input_path),
        "input_subset": str(label_free_input),
        "input_subset_sha256": sha256(label_free_input),
        "contains_labels": False, "rows": len(inputs), "seed": args.seed,
        "agents": str(agents_path), "agents_sha256": sha256(agents_path),
        "task_files": task_records,
    }
    (output / "PLAN.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"commitment plan complete: {output / 'PLAN.json'}")
    return 0


def _prob(commitment: Mapping[str, Any], label: str) -> tuple[float, float]:
    if not commitment.get("available"):
        return 0.0, 1.0
    probs = commitment.get("probabilities", {})
    return float(probs.get(label, 0.0)), 0.0


def feature_names() -> list[str]:
    base = [
        "intercept", "qwen_source", "gemma_source", "cross_model",
        "qwen_support", "gemma_support", "max_support", "mean_support",
        "qwen_selected", "gemma_selected", "candidate_count",
        "qwen_exist_yes", "gemma_exist_yes", "qwen_exist_missing",
        "gemma_exist_missing", "qwen_card_one", "gemma_card_one",
        "qwen_card_many", "gemma_card_many", "numeric_cross_tolerance",
        "numeric_qwen_distance", "numeric_gemma_distance",
    ]
    base += [f"relation={relation}" for relation in RELATIONS]
    base += [f"qwen_source*{relation}" for relation in RELATIONS]
    base += [f"gemma_source*{relation}" for relation in RELATIONS]
    base += [f"support*{relation}" for relation in RELATIONS]
    base += [f"prompt={policy}" for policy in PROMPT_POLICIES]
    return base


def _numeric_value(item: Any) -> float | None:
    try:
        value = float(str(item).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _agent_numeric_center(graph: Mapping[str, Any], agent: str) -> float | None:
    values = []
    for node in graph["candidates"]:
        source = node["sources"].get(agent)
        value = _numeric_value(node["item"])
        if source and value is not None:
            values.extend([value] * max(1, int(source["support"])))
    return statistics.median(values) if values else None


def candidate_features(graph: Mapping[str, Any], node: Mapping[str, Any]) -> list[float]:
    relation = str(graph["Relation"])
    qsource, gsource = node["sources"].get(QWEN), node["sources"].get(GEMMA)
    qs, gs = (float(qsource["support_rate"]) if qsource else 0.0,
              float(gsource["support_rate"]) if gsource else 0.0)
    qyes, qmiss = _prob(graph["agents"][QWEN]["existence"], "YES")
    gyes, gmiss = _prob(graph["agents"][GEMMA]["existence"], "YES")
    qone, _ = _prob(graph["agents"][QWEN]["cardinality"], "ONE")
    gone, _ = _prob(graph["agents"][GEMMA]["cardinality"], "ONE")
    qmany, _ = _prob(graph["agents"][QWEN]["cardinality"], "MANY")
    gmany, _ = _prob(graph["agents"][GEMMA]["cardinality"], "MANY")
    value = _numeric_value(node["item"])
    qcenter, gcenter = _agent_numeric_center(graph, QWEN), _agent_numeric_center(graph, GEMMA)
    def distance(center: float | None) -> float:
        if value is None or center is None:
            return 0.0
        return min(1.0, abs(math.log(value / center)) / 5.0)
    numeric_agree = 0.0
    if value is not None and (qsource or gsource):
        other_agent = GEMMA if qsource and not gsource else QWEN if gsource and not qsource else None
        if qsource and gsource:
            numeric_agree = 1.0
        elif other_agent is not None:
            for other in graph["candidates"]:
                other_value = _numeric_value(other["item"])
                if (other_agent in other["sources"] and other_value is not None
                        and abs(other_value - value) / max(abs(value), 1e-12) <= 0.05):
                    numeric_agree = 1.0
                    break
    values = [
        1.0, float(qsource is not None), float(gsource is not None),
        float(qsource is not None and gsource is not None), qs, gs, max(qs, gs),
        (qs + gs) / max(1, int(qsource is not None) + int(gsource is not None)),
        float(node["selected_by"].get(QWEN, False)),
        float(node["selected_by"].get(GEMMA, False)),
        min(1.0, len(graph["candidates"]) / 20.0),
        qyes, gyes, qmiss, gmiss, qone, gone, qmany, gmany,
        numeric_agree, distance(qcenter), distance(gcenter),
    ]
    values += [float(relation == candidate) for candidate in RELATIONS]
    values += [float(qsource is not None and relation == candidate) for candidate in RELATIONS]
    values += [float(gsource is not None and relation == candidate) for candidate in RELATIONS]
    values += [max(qs, gs) if relation == candidate else 0.0 for candidate in RELATIONS]
    values += [float(graph.get("prompt_policy") == policy) for policy in PROMPT_POLICIES]
    if len(values) != len(feature_names()):
        raise AssertionError("candidate feature schema drift")
    return values


def null_feature_names() -> list[str]:
    names = [
        "intercept", "qwen_none_rate", "gemma_none_rate", "candidate_count",
        "max_qwen_support", "max_gemma_support", "cross_model_candidate",
        "qwen_exist_no", "gemma_exist_no", "qwen_exist_missing",
        "gemma_exist_missing", "qwen_card_zero", "gemma_card_zero",
    ]
    names += [f"relation={relation}" for relation in RELATIONS]
    names += [f"qwen_none*{relation}" for relation in RELATIONS]
    names += [f"gemma_none*{relation}" for relation in RELATIONS]
    return names


def null_features(graph: Mapping[str, Any]) -> list[float]:
    relation = str(graph["Relation"])
    qno, qmiss = _prob(graph["agents"][QWEN]["existence"], "NO")
    gno, gmiss = _prob(graph["agents"][GEMMA]["existence"], "NO")
    qzero, _ = _prob(graph["agents"][QWEN]["cardinality"], "ZERO")
    gzero, _ = _prob(graph["agents"][GEMMA]["cardinality"], "ZERO")
    qnone, gnone = (float(graph["agents"][QWEN]["none_rate"]),
                    float(graph["agents"][GEMMA]["none_rate"]))
    max_q = max((node["sources"].get(QWEN, {}).get("support_rate", 0.0)
                 for node in graph["candidates"]), default=0.0)
    max_g = max((node["sources"].get(GEMMA, {}).get("support_rate", 0.0)
                 for node in graph["candidates"]), default=0.0)
    cross = any(len(node["sources"]) == 2 for node in graph["candidates"])
    values = [1.0, qnone, gnone, min(1.0, len(graph["candidates"]) / 20.0),
              max_q, max_g, float(cross), qno, gno, qmiss, gmiss, qzero, gzero]
    values += [float(relation == candidate) for candidate in RELATIONS]
    values += [qnone if relation == candidate else 0.0 for candidate in RELATIONS]
    values += [gnone if relation == candidate else 0.0 for candidate in RELATIONS]
    if len(values) != len(null_feature_names()):
        raise AssertionError("null feature schema drift")
    return values


class LogisticCalibrator:
    """Small deterministic L2-regularized logistic calibrator (IRLS)."""

    def __init__(self, names: Sequence[str], l2: float = 2.0):
        self.names = list(names)
        self.l2 = float(l2)
        self.coefficients = np.zeros(len(self.names), dtype=np.float64)

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float],
            weights: Sequence[float] | None = None) -> "LogisticCalibrator":
        matrix = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.names) or len(target) != len(matrix):
            raise ContractError("invalid calibration matrix")
        if not len(target) or set(np.unique(target)) - {0.0, 1.0}:
            raise ContractError("logistic targets must be nonempty binary values")
        sample_weight = (np.ones(len(target), dtype=np.float64) if weights is None
                         else np.asarray(weights, dtype=np.float64))
        if sample_weight.shape != target.shape or np.any(sample_weight <= 0):
            raise ContractError("invalid calibration sample weights")
        beta = np.zeros(matrix.shape[1], dtype=np.float64)
        penalty = np.eye(matrix.shape[1], dtype=np.float64) * self.l2
        penalty[0, 0] = 0.0
        for _ in range(60):
            logits = np.clip(matrix @ beta, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            variance = np.maximum(probs * (1.0 - probs), 1e-6)
            effective = sample_weight * variance
            working = logits + (target - probs) / variance
            lhs = matrix.T @ (matrix * effective[:, None]) + penalty
            rhs = matrix.T @ (effective * working)
            try:
                updated = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            if np.max(np.abs(updated - beta)) < 1e-8:
                beta = updated
                break
            beta = updated
        if not np.all(np.isfinite(beta)):
            raise ContractError("non-finite calibration coefficients")
        self.coefficients = beta
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        logits = np.clip(matrix @ self.coefficients, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict:
        return {"feature_names": self.names, "l2": self.l2,
                "coefficients": self.coefficients.tolist()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogisticCalibrator":
        model = cls(value["feature_names"], float(value["l2"]))
        model.coefficients = np.asarray(value["coefficients"], dtype=np.float64)
        if model.coefficients.shape != (len(model.names),):
            raise ContractError("calibration coefficient shape mismatch")
        return model


def _gold_aliases(row: Mapping[str, Any]) -> list:
    gold = row.get("ObjectEntities", [])
    return [[item] for item in gold] if gold and isinstance(gold[0], str) else gold


def _candidate_label(graph: Mapping[str, Any], node: Mapping[str, Any],
                     gold: Mapping[str, Any]) -> float:
    return float(true_positives(
        [node["item"]], _gold_aliases(gold), RELATION_TYPE[graph["Relation"]], 0.05) > 0)


def _candidate_source_group(node: Mapping[str, Any]) -> str:
    sources = set(node.get("sources", {}))
    if sources == {QWEN}:
        return "qwen_only"
    if sources == {GEMMA}:
        return "gemma_only"
    if sources == {QWEN, GEMMA}:
        return "shared"
    raise ContractError(f"candidate has invalid source set: {sorted(sources)}")


def _candidate_training_weights(
        graph: Mapping[str, Any], weighting: str) -> list[float]:
    """Return candidate weights whose total is one for every nonempty row.

    ``row`` reproduces the original implementation. ``row-agent-balanced``
    allocates one half of the row mass to every agent with at least one
    proposal, then spreads that mass uniformly across that agent's candidates.
    A shared candidate receives contributions from both agents. This prevents
    Qwen N=10 from receiving more calibration weight merely because it emits
    more distinct nodes than Gemma N=1.
    """
    nodes = list(graph.get("candidates", []))
    if not nodes:
        return []
    if weighting == "row":
        return [1.0 / len(nodes)] * len(nodes)
    if weighting != "row-agent-balanced":
        raise ContractError(f"unsupported candidate weighting: {weighting!r}")
    counts = {
        agent: sum(agent in node.get("sources", {}) for node in nodes)
        for agent in AGENTS
    }
    active = [agent for agent, count in counts.items() if count]
    if not active:
        raise ContractError(f"candidate row has no active agent sources: {_key(graph)}")
    per_agent_mass = 1.0 / len(active)
    weights = [
        sum(per_agent_mass / counts[agent] for agent in active
            if agent in node.get("sources", {}))
        for node in nodes
    ]
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        raise ContractError(f"invalid candidate weights for {_key(graph)}")
    # Floating-point roundoff is removed so every row has exactly unit mass.
    return [weight / total for weight in weights]


def _training_matrices(graphs: Sequence[Mapping[str, Any]],
                       gold_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
                       candidate_weighting: str = "row") -> tuple:
    cx, cy, cw, nx, ny = [], [], [], [], []
    for graph in graphs:
        key = _key(graph)
        if key not in gold_by_key:
            raise ContractError(f"missing training gold for {key}")
        gold = gold_by_key[key]
        weights = _candidate_training_weights(graph, candidate_weighting)
        for node, weight in zip(graph["candidates"], weights):
            cx.append(candidate_features(graph, node))
            cy.append(_candidate_label(graph, node, gold))
            cw.append(weight)
        nx.append(null_features(graph))
        ny.append(float(not gold.get("ObjectEntities")))
    if not cx:
        raise ContractError("training graph contains no candidates")
    return cx, cy, cw, nx, ny


def _calibration_summary(records: Sequence[Mapping[str, Any]]) -> dict:
    """Weighted reliability summary for held-out calibration predictions."""
    if not records:
        return {"count": 0, "effective_weight": 0.0}
    total = sum(float(row["weight"]) for row in records)
    if total <= 0:
        raise ContractError("calibration diagnostics have nonpositive weight")
    clipped = [
        min(1.0 - 1e-12, max(1e-12, float(row["probability"])))
        for row in records
    ]
    labels = [float(row["label"]) for row in records]
    weights = [float(row["weight"]) for row in records]
    mean_probability = sum(w * p for w, p in zip(weights, clipped)) / total
    positive_rate = sum(w * y for w, y in zip(weights, labels)) / total
    brier = sum(w * (p - y) ** 2
                for w, p, y in zip(weights, clipped, labels)) / total
    log_loss = -sum(
        w * (y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
        for w, p, y in zip(weights, clipped, labels)) / total
    bins = []
    expected_calibration_error = 0.0
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        selected = [
            position for position, probability in enumerate(clipped)
            if lower <= probability < upper or (index == 9 and probability == 1.0)
        ]
        if not selected:
            continue
        bin_weight = sum(weights[position] for position in selected)
        confidence = sum(
            weights[position] * clipped[position] for position in selected) / bin_weight
        accuracy = sum(
            weights[position] * labels[position] for position in selected) / bin_weight
        expected_calibration_error += bin_weight / total * abs(confidence - accuracy)
        bins.append({
            "lower": lower, "upper": upper, "count": len(selected),
            "effective_weight": bin_weight, "mean_probability": confidence,
            "positive_rate": accuracy,
        })
    return {
        "count": len(records), "effective_weight": total,
        "mean_probability": mean_probability, "positive_rate": positive_rate,
        "brier": brier, "log_loss": log_loss,
        "expected_calibration_error_10bin": expected_calibration_error,
        "reliability_bins": bins,
    }


def _grouped_calibration_diagnostics(
        records: Sequence[Mapping[str, Any]], dimensions: Sequence[str]) -> dict:
    result = {"overall": _calibration_summary(records)}
    for dimension in dimensions:
        values = sorted({str(row[dimension]) for row in records})
        result[f"by_{dimension}"] = {
            value: _calibration_summary(
                [row for row in records if str(row[dimension]) == value])
            for value in values
        }
    return result


def _training_source_diagnostics(
        graphs: Sequence[Mapping[str, Any]],
        gold_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
        candidate_weighting: str) -> dict:
    records = []
    coverage: dict[str, dict[str, dict[str, float]]] = {
        agent: defaultdict(lambda: {"matched_gold": 0.0, "gold_objects": 0.0})
        for agent in AGENTS
    }
    for graph in graphs:
        gold = gold_by_key[_key(graph)]
        relation = str(graph["Relation"])
        weights = _candidate_training_weights(graph, candidate_weighting)
        for node, weight in zip(graph["candidates"], weights):
            records.append({
                "relation": relation,
                "source_group": _candidate_source_group(node),
                "label": _candidate_label(graph, node, gold),
                "weight": weight,
            })
        aliases = _gold_aliases(gold)
        gold_count = len(gold.get("ObjectEntities", []))
        for agent in AGENTS:
            items = [
                node["item"] for node in graph["candidates"]
                if agent in node.get("sources", {})
            ]
            matched = true_positives(
                items, aliases, RELATION_TYPE[relation], 0.05)
            coverage[agent][relation]["matched_gold"] += float(matched)
            coverage[agent][relation]["gold_objects"] += float(gold_count)
    grouped = {}
    for dimension in ("source_group", "relation"):
        grouped[f"by_{dimension}"] = {}
        for value in sorted({str(row[dimension]) for row in records}):
            subset = [row for row in records if str(row[dimension]) == value]
            total_weight = sum(float(row["weight"]) for row in subset)
            grouped[f"by_{dimension}"][value] = {
                "candidate_nodes": len(subset),
                "effective_weight": total_weight,
                "positive_nodes": int(sum(row["label"] for row in subset)),
                "weighted_precision": (
                    sum(float(row["weight"]) * float(row["label"]) for row in subset)
                    / total_weight if total_weight else 0.0),
            }
    coverage_rows = {}
    for agent in AGENTS:
        coverage_rows[agent] = {}
        for relation in RELATIONS:
            values = coverage[agent][relation]
            coverage_rows[agent][relation] = {
                **values,
                "micro_gold_coverage": (
                    values["matched_gold"] / values["gold_objects"]
                    if values["gold_objects"] else 0.0),
            }
    return {
        "candidate_weighting": candidate_weighting,
        "rows": len(graphs),
        "candidate_nodes": len(records),
        **grouped,
        "gold_coverage_by_agent_relation": coverage_rows,
    }


def _expected_cardinality(graph: Mapping[str, Any], candidate_sum: float) -> float:
    estimates = []
    for agent in AGENTS:
        commitment = graph["agents"][agent]["cardinality"]
        if not commitment.get("available"):
            continue
        probs = commitment.get("probabilities", {})
        estimates.append(float(probs.get("ONE", 0.0)) +
                         2.0 * float(probs.get("MANY", 0.0)))
    return max(candidate_sum, statistics.mean(estimates) if estimates else candidate_sum)


def _weighted_median(items: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(items)
    total = sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= total / 2:
            return value
    return ordered[-1][0]


def _decode_numeric(graph: Mapping[str, Any], candidates: Sequence[tuple[dict, float]]) -> tuple[list[str], float]:
    usable = [(node, probability, _numeric_value(node["item"]))
              for node, probability in candidates]
    usable = [(node, probability, value) for node, probability, value in usable
              if value is not None]
    if not usable:
        return [], 0.0
    clusters = []
    for anchor_node, anchor_probability, anchor in usable:
        members = [(node, probability, value) for node, probability, value in usable
                   if abs(value - anchor) / max(abs(anchor), 1e-12) <= 0.05]
        score_value = sum(probability * (0.5 + 0.5 * max(
            source["support_rate"] for source in node["sources"].values()))
                          for node, probability, _ in members)
        clusters.append((score_value, members))
    cluster_score, best = max(clusters, key=lambda row: (row[0], -len(row[1])))
    value = _weighted_median([
        (number, max(1e-6, probability) * sum(
            source["support_rate"] for source in node["sources"].values()))
        for node, probability, number in best
    ])
    return [format(value, ".12g")], min(1.0, cluster_score)


def _set_utility(selected: Sequence[tuple[dict, float]], all_probabilities: Sequence[float],
                 expected_cardinality: float) -> float:
    if not selected:
        return 0.0
    expected_tp = sum(probability for _, probability in selected)
    denominator = len(selected) + max(expected_cardinality, sum(all_probabilities))
    return 2.0 * expected_tp / denominator if denominator > 0 else 0.0


def _objects_utility(graph: Mapping[str, Any], objects: Sequence[str],
                     scored: Sequence[tuple[dict, float]], null_probability: float) -> float:
    """Score any decoded object set on one relation-consistent utility scale.

    This function is deliberately shared by the proposed and Qwen-baseline
    sides of the guard. Decoder-internal ranking scores must never be compared
    directly with this output utility.
    """
    if not objects:
        return null_probability if graph["Relation"] in NULLABLE_RELATIONS else 0.0
    relation = graph["Relation"]
    if relation in NUMERIC_RELATIONS:
        references = [_numeric_value(item) for item in objects]
        references = [value for value in references if value is not None]
        return max((probability for node, probability in scored
                    if any(_numeric_value(node["item"]) is not None
                           and abs(_numeric_value(node["item"]) - reference) /
                           max(abs(reference), 1e-12) <= 0.05
                           for reference in references)), default=0.0)
    keys = {canonical_key(str(item), relation) for item in objects}
    chosen = [(node, probability) for node, probability in scored if node["key"] in keys]
    if relation in SINGLE_RELATIONS:
        # A legal single-valued prediction earns F1=1 exactly when that one
        # candidate is correct, so its expected F1 is its calibrated
        # correctness probability. This also handles malformed multi-object
        # baselines conservatively by taking their strongest candidate.
        return max((probability for _, probability in chosen), default=0.0)
    expected = _expected_cardinality(graph, sum(probability for _, probability in scored))
    return _set_utility(chosen, [probability for _, probability in scored], expected)


def decode_graph(graph: Mapping[str, Any], candidate_model: LogisticCalibrator,
                 null_model: LogisticCalibrator, *, guard_margin: float,
                 require_commitments: bool) -> tuple[list[str], dict]:
    if require_commitments and any(
            not graph["agents"][agent][phase]["available"]
            for agent in AGENTS for phase in ("existence", "cardinality")):
        raise ContractError(f"missing blind commitments for {_key(graph)}")
    nodes = list(graph["candidates"])
    probabilities = (candidate_model.predict([candidate_features(graph, node) for node in nodes])
                     if nodes else np.asarray([], dtype=np.float64))
    scored = list(zip(nodes, [float(value) for value in probabilities]))
    null_probability = float(null_model.predict([null_features(graph)])[0])
    relation = graph["Relation"]
    if relation in NUMERIC_RELATIONS:
        proposed, decoder_selection_utility = _decode_numeric(graph, scored)
    elif relation in SINGLE_RELATIONS:
        best = max(scored, key=lambda pair: pair[1], default=None)
        if best is None or null_probability >= best[1]:
            proposed, decoder_selection_utility = [], null_probability
        else:
            proposed, decoder_selection_utility = [best[0]["item"]], best[1]
    else:
        ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0]["key"]))
        expected = _expected_cardinality(graph, sum(probability for _, probability in ranked))
        choices = []
        for count in range(1, len(ranked) + 1):
            selected = ranked[:count]
            choices.append((_set_utility(
                selected, [probability for _, probability in ranked], expected), selected))
        if choices:
            decoder_selection_utility, selected = max(
                choices, key=lambda row: (row[0], -len(row[1])))
            proposed = [node["item"] for node, _ in selected]
        else:
            decoder_selection_utility, proposed = 0.0, []
        if relation in NULLABLE_RELATIONS and null_probability >= decoder_selection_utility:
            decoder_selection_utility, proposed = null_probability, []
    # Critical invariant: both sides of the guard are evaluated by the exact
    # same function. Numeric cluster mass and list-prefix scores are selection
    # heuristics, not commensurate output utilities.
    proposed_utility = _objects_utility(graph, proposed, scored, null_probability)
    baseline = list(graph.get("baseline_objects", []))
    baseline_utility = _objects_utility(graph, baseline, scored, null_probability)
    use_baseline = baseline_utility + guard_margin >= proposed_utility
    guarded = baseline if use_baseline else proposed
    diagnostics = {
        "null_probability": null_probability,
        "proposed_objects": proposed,
        "proposed_utility": proposed_utility,
        "decoder_selection_utility": decoder_selection_utility,
        "baseline_objects": baseline,
        "baseline_utility": baseline_utility,
        "guard_margin": guard_margin,
        "used_baseline": use_baseline,
        "candidate_probabilities": [
            {"key": node["key"], "item": node["item"], "probability": probability}
            for node, probability in sorted(scored, key=lambda pair: -pair[1])
        ],
    }
    return guarded, diagnostics


def _fit_models(graphs: Sequence[Mapping[str, Any]], gold_by_key: Mapping,
                l2: float, candidate_weighting: str = "row"
                ) -> tuple[LogisticCalibrator, LogisticCalibrator]:
    cx, cy, cw, nx, ny = _training_matrices(
        graphs, gold_by_key, candidate_weighting)
    candidate = LogisticCalibrator(feature_names(), l2=l2).fit(cx, cy, cw)
    null = LogisticCalibrator(null_feature_names(), l2=l2).fit(nx, ny)
    return candidate, null


def _materialize_prompt_routes(
        graphs: Sequence[Mapping[str, Any]], routes: Mapping[str, str]) -> list[dict]:
    """Select one pre-generated Gemma prompt variant per relation."""
    materialized = []
    for graph in graphs:
        relation = str(graph["Relation"])
        policy = routes.get(relation)
        if policy not in PROMPT_POLICIES:
            raise ContractError(f"missing/invalid prompt route for {relation}: {policy!r}")
        variants = graph.get("prompt_variants")
        if not isinstance(variants, dict) or set(variants) != set(PROMPT_POLICIES):
            raise ContractError(f"training graph lacks complete prompt variants: {_key(graph)}")
        selected = copy.deepcopy(variants[policy])
        if selected.get("prompt_policy") != policy or _key(selected) != _key(graph):
            raise ContractError(f"prompt variant provenance mismatch: {_key(graph)}")
        materialized.append(selected)
    return materialized


def _select_prompt_routes(
        graphs: Sequence[Mapping[str, Any]],
        gold_by_key: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[dict[str, str], dict]:
    """Choose relation policies using only the supplied (outer-train) rows."""
    routes, diagnostics = {}, {}
    for relation in RELATIONS:
        subset = [graph for graph in graphs if graph["Relation"] == relation]
        if not subset:
            raise ContractError(f"prompt-route training has no rows for {relation}")
        gold = [gold_by_key[_key(graph)] for graph in subset]
        policy_scores = {}
        for policy in PROMPT_POLICIES:
            selected = _materialize_prompt_routes(subset, {relation: policy})
            predictions = [{
                "SubjectEntity": graph["SubjectEntity"], "Relation": relation,
                "ObjectEntities": graph["agent_outputs"][GEMMA],
            } for graph in selected]
            policy_scores[policy] = float(score(predictions, gold)[relation])
        # Prefer the simpler direct prompt on an exact tie, followed by shared
        # and disjoint CoT. The order is fixed before seeing any labels.
        policy = max(PROMPT_POLICIES, key=lambda name: (
            policy_scores[name], -PROMPT_POLICIES.index(name)))
        routes[relation] = policy
        diagnostics[relation] = {
            "selected": policy, "training_rows": len(subset), "scores": policy_scores}
    return routes, diagnostics


def _fixed_prompt_routes(graphs: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Read one already-frozen prompt policy per relation from materialized rows."""
    routes = {}
    for relation in RELATIONS:
        policies = {str(graph.get("prompt_policy")) for graph in graphs
                    if graph["Relation"] == relation}
        if len(policies) != 1 or next(iter(policies), None) not in PROMPT_POLICIES:
            raise ContractError(
                f"fixed prompt graph must contain exactly one valid policy for "
                f"{relation}, got {sorted(policies)}")
        routes[relation] = next(iter(policies))
    return routes


def _prediction_rows(graphs: Sequence[Mapping[str, Any]], candidate: LogisticCalibrator,
                     null: LogisticCalibrator, *, margin: float,
                     require_commitments: bool) -> tuple[list[dict], list[dict]]:
    rows, diagnostics = [], []
    for graph in graphs:
        objects, detail = decode_graph(
            graph, candidate, null, guard_margin=margin,
            require_commitments=require_commitments)
        rows.append({"SubjectEntity": graph["SubjectEntity"], "Relation": graph["Relation"],
                     "ObjectEntities": objects})
        diagnostics.append({"SubjectEntity": graph["SubjectEntity"],
                            "Relation": graph["Relation"], **detail})
    return rows, diagnostics


def _select_guard_margin_one_se(
        fold_scores_by_margin: Mapping[float, Mapping[int, float]],
        baseline_fold_scores: Mapping[int, float]) -> tuple[float, dict]:
    """Choose a conservative guard using paired OOF fold improvements.

    Raw fold F1 varies substantially because the small pilot folds contain
    different relation mixtures.  Pairing every candidate policy with the
    unchanged Qwen baseline on the same fold removes most of that nuisance
    variation.  We then apply the conventional one-standard-error rule: keep
    the largest (most conservative) predeclared margin whose mean paired
    improvement is within one standard error of the best mean improvement.

    This selection consumes train-fold scores only.  Validation labels are not
    accepted by this function or by ``fit_selector``.
    """
    if not fold_scores_by_margin:
        raise ContractError("guard selection requires at least one margin")
    fold_ids = sorted(baseline_fold_scores)
    if len(fold_ids) < 2:
        raise ContractError("one-standard-error guard selection needs at least two folds")
    diagnostics = {}
    for margin, fold_scores in sorted(fold_scores_by_margin.items()):
        if sorted(fold_scores) != fold_ids:
            raise ContractError(f"guard margin {margin} does not cover every fold")
        deltas = [float(fold_scores[fold] - baseline_fold_scores[fold])
                  for fold in fold_ids]
        mean_delta = statistics.mean(deltas)
        standard_error = statistics.stdev(deltas) / math.sqrt(len(deltas))
        diagnostics[str(margin)] = {
            "fold_scores": {str(fold): float(fold_scores[fold]) for fold in fold_ids},
            "baseline_fold_scores": {
                str(fold): float(baseline_fold_scores[fold]) for fold in fold_ids},
            "paired_fold_deltas": {
                str(fold): float(delta) for fold, delta in zip(fold_ids, deltas)},
            "mean_paired_delta": float(mean_delta),
            "standard_error_paired_delta": float(standard_error),
        }
    best_margin = max(
        fold_scores_by_margin,
        key=lambda margin: (diagnostics[str(margin)]["mean_paired_delta"], margin))
    best_mean = diagnostics[str(best_margin)]["mean_paired_delta"]
    best_se = diagnostics[str(best_margin)]["standard_error_paired_delta"]
    threshold = best_mean - best_se
    eligible = [margin for margin in fold_scores_by_margin
                if diagnostics[str(margin)]["mean_paired_delta"] >= threshold - 1e-12]
    selected = max(eligible)
    for margin in fold_scores_by_margin:
        diagnostics[str(margin)]["within_one_se_of_best"] = margin in eligible
    return selected, {
        "rule": (
            "largest predeclared margin whose mean paired OOF improvement over "
            "the same-fold Qwen baseline is within one standard error of the "
            "best mean paired improvement"),
        "best_mean_margin": float(best_margin),
        "best_mean_paired_delta": float(best_mean),
        "best_margin_standard_error": float(best_se),
        "eligibility_threshold": float(threshold),
        "selected_margin": float(selected),
        "margins": diagnostics,
    }


def fit_selector(args: argparse.Namespace) -> int:
    graph_path, gold_path = Path(args.graph).resolve(), Path(args.gold).resolve()
    graph_manifest = _json(
        graph_path.with_suffix(graph_path.suffix + ".manifest.json"))
    training_split = str(graph_manifest.get("split"))
    if training_split not in {"train-pilot", "train"}:
        raise ContractError(
            f"fit accepts train/train-pilot graphs, got {training_split!r}")
    graphs = _load_graph(graph_path, expected_split=training_split)
    gold_rows = read_jsonl(gold_path)
    gold_by_key = _indexed(gold_rows, "training-gold")
    fold_rows = read_jsonl(Path(args.folds).resolve())
    folds = {_key(row): int(row["fold"]) for row in fold_rows}
    if set(folds) != {_key(graph) for graph in graphs}:
        raise ContractError("fold manifest does not exactly cover training graphs")
    margins = [float(value) for value in args.guard_margins.split(",")]
    if not margins or any(value < 0 or not math.isfinite(value) for value in margins):
        raise ContractError("guard margins must be finite and nonnegative")
    candidate_weighting = getattr(args, "candidate_weighting", "row")
    if candidate_weighting not in {"row", "row-agent-balanced"}:
        raise ContractError(
            f"unsupported candidate weighting: {candidate_weighting!r}")
    oof_by_margin = {margin: [] for margin in margins}
    fold_scores_by_margin = {margin: {} for margin in margins}
    baseline_fold_scores = {}
    nested_prompt_routes = {}
    candidate_oof_records, null_oof_records = [], []
    fixed_routes = (_fixed_prompt_routes(graphs)
                    if getattr(args, "fixed_prompt_routes", False) else None)
    for fold in sorted(set(folds.values())):
        train_base = [graph for graph in graphs if folds[_key(graph)] != fold]
        holdout_base = [graph for graph in graphs if folds[_key(graph)] == fold]
        if fixed_routes is None:
            routes, route_diagnostics = _select_prompt_routes(train_base, gold_by_key)
            nested_prompt_routes[str(fold)] = route_diagnostics
            train = _materialize_prompt_routes(train_base, routes)
            holdout = _materialize_prompt_routes(holdout_base, routes)
        else:
            routes = fixed_routes
            nested_prompt_routes[str(fold)] = {
                "selection": "pre-frozen relation routes", "routes": routes}
            train, holdout = train_base, holdout_base
        candidate, null = _fit_models(
            train, gold_by_key, args.l2, candidate_weighting)
        holdout_gold = [gold_by_key[_key(graph)] for graph in holdout]
        for graph in holdout:
            weights = _candidate_training_weights(graph, candidate_weighting)
            probabilities = (
                candidate.predict([
                    candidate_features(graph, node)
                    for node in graph["candidates"]])
                if graph["candidates"] else np.asarray([], dtype=np.float64))
            for node, probability, weight in zip(
                    graph["candidates"], probabilities, weights):
                candidate_oof_records.append({
                    "fold": fold, "relation": graph["Relation"],
                    "source_group": _candidate_source_group(node),
                    "probability": float(probability),
                    "label": _candidate_label(
                        graph, node, gold_by_key[_key(graph)]),
                    "weight": float(weight),
                })
            null_probability = float(null.predict([null_features(graph)])[0])
            null_oof_records.append({
                "fold": fold, "relation": graph["Relation"],
                "probability": null_probability,
                "label": float(not gold_by_key[_key(graph)].get("ObjectEntities")),
                "weight": 1.0,
            })
        baseline_rows = [{
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": list(graph.get("baseline_objects", [])),
        } for graph in holdout]
        baseline_fold_scores[fold] = float(
            score(baseline_rows, holdout_gold)["*** All Relations ***"])
        for margin in margins:
            rows, _ = _prediction_rows(
                holdout, candidate, null, margin=margin, require_commitments=True)
            oof_by_margin[margin].extend(rows)
            fold_scores_by_margin[margin][fold] = float(
                score(rows, holdout_gold)["*** All Relations ***"])
    if fixed_routes is None:
        final_routes, final_route_diagnostics = _select_prompt_routes(graphs, gold_by_key)
        final_graphs = _materialize_prompt_routes(graphs, final_routes)
        prompt_route_selection = (
            "each OOF fold selects relation policies using only its outer-training rows; "
            "the production route uses all labeled training-pilot rows")
    else:
        final_routes, final_route_diagnostics = fixed_routes, {
            relation: {"selected": policy, "selection": "pre-frozen"}
            for relation, policy in fixed_routes.items()}
        final_graphs = list(graphs)
        prompt_route_selection = (
            "relation prompt routes were frozen before this fit and were not "
            "selected from calibrator-fold labels")
    gold_subset = [gold_by_key[_key(graph)] for graph in final_graphs]
    source_scores = {}
    for agent in AGENTS:
        source_rows = [{
            "SubjectEntity": graph["SubjectEntity"], "Relation": graph["Relation"],
            "ObjectEntities": graph["agent_outputs"][agent],
        } for graph in final_graphs]
        source_scores[agent] = score(source_rows, gold_subset)
    oof_scores = {}
    for margin, rows in oof_by_margin.items():
        by_key = _indexed(rows, f"oof-margin-{margin}")
        ordered = [by_key[_key(graph)] for graph in final_graphs]
        oof_scores[margin] = score(ordered, gold_subset)
    best_margin, guard_selection = _select_guard_margin_one_se(
        fold_scores_by_margin, baseline_fold_scores)
    best_score = oof_scores[best_margin]["*** All Relations ***"]
    chosen_by_key = _indexed(oof_by_margin[best_margin], "chosen-oof-predictions")
    chosen_oof = [chosen_by_key[_key(graph)] for graph in final_graphs]
    candidate, null = _fit_models(
        final_graphs, gold_by_key, args.l2, candidate_weighting)
    output = Path(args.output).resolve()
    oof_path = output.with_name(output.stem + ".oof_predictions.jsonl")
    write_jsonl_atomic(oof_path, chosen_oof)
    artifact = {
        "schema": "heterogeneous-memory-selector-v1",
        "decoder_utility_schema": DECODER_UTILITY_SCHEMA,
        "training_split": training_split,
        "training_graph": str(graph_path), "training_graph_sha256": sha256(graph_path),
        "training_gold": str(gold_path), "training_gold_sha256": sha256(gold_path),
        "fold_manifest": str(Path(args.folds).resolve()),
        "fold_manifest_sha256": sha256(Path(args.folds).resolve()),
        "rows": len(graphs), "folds": len(set(folds.values())), "l2": args.l2,
        "candidate_weighting": candidate_weighting,
        "candidate_model": candidate.to_dict(), "null_model": null.to_dict(),
        "prompt_route": final_routes,
        "prompt_route_full_train_diagnostics": final_route_diagnostics,
        "nested_oof_prompt_routes": nested_prompt_routes,
        "prompt_route_selection": prompt_route_selection,
        "guard_margin": best_margin,
        "guard_margin_selection": guard_selection["rule"],
        "guard_margin_selection_diagnostics": guard_selection,
        "oof_scores": {str(margin): values for margin, values in oof_scores.items()},
        "source_scores_on_same_train_pilot": source_scores,
        "training_source_diagnostics": _training_source_diagnostics(
            final_graphs, gold_by_key, candidate_weighting),
        "oof_candidate_calibration": _grouped_calibration_diagnostics(
            candidate_oof_records, ("relation", "source_group")),
        "oof_null_calibration": _grouped_calibration_diagnostics(
            null_oof_records, ("relation",)),
        "chosen_oof_predictions": str(oof_path),
        "chosen_oof_predictions_sha256": sha256(oof_path),
        "validation_labels_used": False,
        "known_domain_shift": (
            "train Qwen proposals use five OOF N=10 legacy-fold caches; validation "
            "Qwen proposals use the frozen v0495 mixed-source raw artifacts. Support "
            "is normalized, but train System-2 corroboration is unavailable"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"selector fit complete: {output}; OOF margin={best_margin} score={best_score:.6f}")
    return 0


def decode(args: argparse.Namespace) -> int:
    graph_path, selector_path = Path(args.graph).resolve(), Path(args.selector).resolve()
    graphs = _load_graph(graph_path)
    graph_manifest = _json(graph_path.with_suffix(graph_path.suffix + ".manifest.json"))
    if graph_manifest.get("split") not in {"validation", "test"}:
        raise ContractError("decode accepts only validation/test graphs")
    artifact = _json(selector_path)
    if artifact.get("schema") != "heterogeneous-memory-selector-v1":
        raise ContractError("unsupported selector artifact")
    if artifact.get("decoder_utility_schema") != DECODER_UTILITY_SCHEMA:
        raise ContractError(
            "selector predates the symmetric baseline-guard utility fix")
    candidate = LogisticCalibrator.from_dict(artifact["candidate_model"])
    null = LogisticCalibrator.from_dict(artifact["null_model"])
    if candidate.names != feature_names():
        raise ContractError("selector candidate feature schema is stale or incompatible")
    if null.names != null_feature_names():
        raise ContractError("selector null feature schema is stale or incompatible")
    prompt_route = artifact.get("prompt_route")
    if not isinstance(prompt_route, dict) or set(prompt_route) != set(RELATIONS):
        raise ContractError("selector prompt route is missing or incomplete")
    for graph in graphs:
        expected = prompt_route.get(graph["Relation"])
        if graph.get("prompt_policy") != expected:
            raise ContractError(
                f"validation prompt route mismatch for {_key(graph)}: "
                f"{graph.get('prompt_policy')!r} != {expected!r}")
    margin = float(artifact["guard_margin"] if args.guard_margin is None else args.guard_margin)
    rows, diagnostics = _prediction_rows(
        graphs, candidate, null, margin=margin,
        require_commitments=not args.allow_missing_commitments)
    output = Path(args.output).resolve()
    write_jsonl_atomic(output, rows)
    diagnostic_path = output.with_suffix(output.suffix + ".diagnostics.jsonl")
    write_jsonl_atomic(diagnostic_path, diagnostics)
    manifest = {
        "schema": "heterogeneous-memory-predictions-v1",
        "graph": str(graph_path), "graph_sha256": sha256(graph_path),
        "selector": str(selector_path), "selector_sha256": sha256(selector_path),
        "output_sha256": sha256(output), "rows": len(rows),
        "guard_margin": margin,
        "decoder_utility_schema": DECODER_UTILITY_SCHEMA,
        "missing_commitments_allowed": bool(args.allow_missing_commitments),
        "gold_labels_used": False,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"decode complete: {output}")
    return 0


def evaluate_predictions(args: argparse.Namespace) -> int:
    predictions, gold = read_jsonl(Path(args.predictions)), read_jsonl(Path(args.gold))
    if [_key(row) for row in predictions] != [_key(row) for row in gold]:
        raise ContractError("prediction and gold order/coverage differ")
    result = score(predictions, gold)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps({
            "schema": "heterogeneous-memory-evaluation-v1",
            "development_only": True,
            "predictions": str(Path(args.predictions).resolve()),
            "predictions_sha256": sha256(Path(args.predictions)),
            "gold": str(Path(args.gold).resolve()), "scores": result,
        }, indent=2, sort_keys=True) + "\n")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    train = sub.add_parser("build-train-graph")
    train.add_argument("--output", required=True)
    train.add_argument("--direct-run", default=str(PILOT_DIRECT))
    train.add_argument("--shared-run", default=str(PILOT_SHARED))
    train.add_argument("--disjoint-run", default=str(PILOT_DISJOINT))
    train.add_argument("--training-prompt-routing", choices=("oof", "full-fit"),
                       default="oof")
    train.add_argument("--oof-confirmation", default=str(OOF_CONFIRMATION))
    train.add_argument("--qwen-proposals", choices=("oof-n10", "pilot-n3"),
                       default="oof-n10")
    train.add_argument("--qwen-oof-run", default=str(QWEN_TRAIN_OOF))
    train.set_defaults(func=build_train_graph)
    val = sub.add_parser("build-validation-graph")
    val.add_argument("--run", default=str(DUAL_VALIDATION))
    val.add_argument("--commitment-dir")
    val.add_argument("--output", required=True)
    val.set_defaults(func=build_validation_graph)
    commitments = sub.add_parser("prepare-commitments")
    commitments.add_argument("--input", default=str(ROOT / "data/val.jsonl"))
    commitments.add_argument("--agents", default=str(
        ROOT / "experiments/heterogeneous_agents/agents_qwen_gemma_synthetic.json"))
    commitments.add_argument("--synthetic-cot", default=str(
        ROOT / "data/synthetic_cot_faithful.jsonl"))
    commitments.add_argument("--output-dir", required=True)
    commitments.add_argument("--seed", type=int, default=20260720)
    commitments.set_defaults(func=prepare_commitments)
    fit = sub.add_parser("fit")
    fit.add_argument("--graph", required=True)
    fit.add_argument("--gold", default=str(ROOT / "data/train.jsonl"))
    fit.add_argument("--folds", default=str(FOLD_MANIFEST))
    fit.add_argument("--output", required=True)
    fit.add_argument("--l2", type=float, default=2.0)
    fit.add_argument("--guard-margins", default="0,0.02,0.05,0.1")
    fit.add_argument(
        "--candidate-weighting", choices=("row", "row-agent-balanced"),
        default="row",
        help=(
            "row reproduces the pilot; row-agent-balanced gives each proposing "
            "agent equal mass within a subject-relation row"))
    fit.add_argument(
        "--fixed-prompt-routes", action="store_true",
        help="consume one pre-frozen prompt policy per relation from materialized graphs")
    fit.set_defaults(func=fit_selector)
    dec = sub.add_parser("decode")
    dec.add_argument("--graph", required=True)
    dec.add_argument("--selector", required=True)
    dec.add_argument("--output", required=True)
    dec.add_argument("--guard-margin", type=float)
    dec.add_argument("--allow-missing-commitments", action="store_true")
    dec.set_defaults(func=decode)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--gold", required=True)
    ev.add_argument("--output")
    ev.set_defaults(func=evaluate_predictions)
    return ap


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
