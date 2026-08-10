#!/usr/bin/env python3
"""Frozen validation confirmation for typed Ministral N=3 admission.

The experiment answers two separate questions without choosing on validation:

1. Does the typed Ministral reservoir add correct validation candidates?
2. Does the train-approved decoder rule convert any of that supply to F1?

``prepare`` never opens ``data/val.jsonl``.  It creates zero-shot N=3
Ministral tasks directly from the certified label-free validation graph and
freezes the exact Codabench-matched 0.511138728 baseline.  ``admit`` builds the
typed graph using the already audited complete-link component policy.
``decode`` applies one train-approved action only: for ``hasArea``, replace the
incumbent when exactly one *new* typed Ministral component has unanimous 3/3
support.  ``evaluate`` verifies all hashes before it opens validation labels.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.assemble_and_audit import (
    assemble_graphs,
    load_responses,
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
from experiments.heterogeneous_agents.components.ministral_candidate_supply import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    MINISTRAL,
    _agent,
    _json,
)
from experiments.heterogeneous_agents.components.ministral_consistency_admission import (
    N_PROPOSALS,
    ROUTE,
    _key,
)
from experiments.heterogeneous_agents.components.ministral_typed_component_admission import (
    _merge_typed_row,
    _proposal_response,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks
from experiments.heterogeneous_agents.components.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    compose_competition_train_oof,
    competition_validation_predictions,
)


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
DEFAULT_OUTPUT = RUNS / "ministral_typed_validation_confirmation_20260729_v3"
DEFAULT_CONFIG = ROOT / "configs/final/portfolio_supply.json"
DEFAULT_SOURCE_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/graphs/"
    "validation_graph.jsonl"
)
DEFAULT_SYNTHETIC = ROOT / "data/synthetic_cot_faithful.jsonl"
DEFAULT_VALIDATION_GOLD = ROOT / "data/val.jsonl"
DEFAULT_TRAIN_RESULT = (
    RUNS / "ministral_typed_component_admission_20260729_v3/"
    "analysis/RESULT.json"
)
DEFAULT_TRAIN_TYPED_GRAPH = (
    RUNS / "ministral_typed_component_admission_20260729_v2/"
    "graph/TYPED_ADMITTED_GRAPH.jsonl"
)
DEFAULT_TRAIN_GOLD = ROOT / "data/train.jsonl"
PLAN_SCHEMA = "ministral-typed-validation-confirmation-plan-v1"
GRAPH_SCHEMA = "ministral-typed-validation-graph-manifest-v1"
PREDICTION_SCHEMA = "ministral-typed-validation-predictions-manifest-v1"
RESULT_SCHEMA = "ministral-typed-validation-confirmation-result-v1"
SEED = 20260730
TRAIN_APPROVED_POLICY = "area_unanimous_new_component_replace"
TRAIN_BASE_SCORE = 0.48245401578759645
TRAIN_AREA_DELTA = 0.020000000000000018
TRAIN_POOLED_DELTA = 0.004192872117400381
TRAIN_SWITCHES = 6


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ContractError(f"missing artifact or manifest: {path}")
    value = _json(manifest_path)
    if value.get("output_sha256") != sha256(path):
        raise ContractError(f"artifact hash mismatch: {path}")
    return value


def _validate_source_graph(path: Path) -> dict[str, Any]:
    manifest = _manifest(path)
    if (
        manifest.get("contains_labels") is not False
        or manifest.get("gold_aware") is not False
        or manifest.get("split") != "validation"
        or manifest.get("rows") != 478
    ):
        raise ContractError("source is not a certified label-free validation graph")
    return manifest


def _validate_train_gate(path: Path) -> dict[str, Any]:
    result = _json(path)
    if (
        result.get("schema") != "ministral-typed-component-admission-result-v1"
        or result.get("validation_opened") is not False
        or result.get("validation_labels_used") is not False
        or float(result["typed_oracle_delta_over_base"]) < 0.010
        or float(result["typed_oracle_gain_over_exact"]) < 0.007
    ):
        raise ContractError("typed admission did not pass its frozen train gate")
    return result


def _validate_plan(output: Path, require_response: bool = True) -> dict[str, Any]:
    path = output / "plan/PLAN.json"
    plan = _json(path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or plan.get("n_proposals") != N_PROPOSALS
        or plan.get("selected_decoder_policy") != TRAIN_APPROVED_POLICY
    ):
        raise ContractError("invalid validation confirmation plan")
    for name in (
        "input_rows", "source_graph", "synthetic_cot", "agents",
        "baseline_predictions", "train_result", "train_typed_graph",
        "train_gold",
    ):
        if sha256(Path(plan[name])) != plan[f"{name}_sha256"]:
            raise ContractError(f"frozen plan artifact changed: {name}")
    if sha256(Path(plan["task_path"])) != plan["task_sha256"]:
        raise ContractError("frozen validation tasks changed")
    _validate_source_graph(Path(plan["source_graph"]))
    _validate_train_gate(Path(plan["train_result"]))
    if require_response:
        manifest = _manifest(Path(plan["response_path"]))
        if (
            manifest.get("task_sha256") != plan["task_sha256"]
            or manifest.get("tasks") != plan["task_count"]
            or manifest.get("agent_id") != MINISTRAL
            or manifest.get("model") != EXPECTED_MODEL
            or manifest.get("revision") != EXPECTED_REVISION
        ):
            raise ContractError("stale or foreign validation responses")
    return plan


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    source_graph = Path(args.source_graph).resolve()
    config_path = Path(args.agents).resolve()
    synthetic_path = Path(args.synthetic_cot).resolve()
    train_result_path = Path(args.train_result).resolve()
    train_typed_graph = Path(args.train_typed_graph).resolve()
    train_gold = Path(args.train_gold).resolve()
    _validate_source_graph(source_graph)
    _validate_train_gate(train_result_path)
    train_graph_manifest = _manifest(train_typed_graph)
    if (
        train_graph_manifest.get("contains_labels") is not False
        or train_graph_manifest.get("validation_opened") is not False
        or train_graph_manifest.get("split") != "train"
    ):
        raise ContractError("train typed graph is not certified label-free")
    config = load_agent_config(config_path)
    agent = _agent(config, MINISTRAL)
    if (
        agent["model"] != EXPECTED_MODEL
        or agent.get("revision") != EXPECTED_REVISION
        or int(agent.get("synthetic_shots", -1)) != 0
    ):
        raise ContractError("unexpected Ministral zero-shot checkpoint")

    source_rows = read_jsonl(source_graph)
    rows = [
        {
            "SubjectEntity": str(row["SubjectEntity"]),
            "Relation": str(row["Relation"]),
        }
        for row in source_rows
    ]
    if len(rows) != 478 or len({_key(row) for row in rows}) != 478:
        raise ContractError("expected 478 unique validation rows")

    baseline, baseline_detail = competition_validation_predictions()
    if {_key(row) for row in baseline} != {_key(row) for row in rows}:
        raise ContractError("competition baseline does not cover validation graph")
    baseline_path = Path(baseline_detail["prediction_path"]).resolve()
    train_baseline, train_baseline_detail = compose_competition_train_oof()
    train_graphs = read_jsonl(train_typed_graph)
    train_gold_rows = read_jsonl(train_gold)
    train_predictions, train_decisions = apply_frozen_policy(
        train_baseline, train_graphs)
    train_base_scores = score(train_baseline, train_gold_rows)
    train_policy_scores = score(train_predictions, train_gold_rows)
    train_policy_gate = {
        "baseline": train_base_scores["*** All Relations ***"],
        "policy_score": train_policy_scores["*** All Relations ***"],
        "area_delta":
            train_policy_scores["hasArea"] - train_base_scores["hasArea"],
        "pooled_delta":
            train_policy_scores["*** All Relations ***"]
            - train_base_scores["*** All Relations ***"],
        "switches": sum(bool(row["changed"]) for row in train_decisions),
        "support_required": 3,
        "relation": "hasArea",
        "capacity_route_rejected": True,
        "two_of_three_route_rejected": True,
        "subject_grouped_oof": train_baseline_detail["subject_grouped_oof"],
        "oof_model_excludes_subject":
            train_baseline_detail["oof_model_excludes_subject"],
    }
    if (
        train_policy_gate["baseline"] != TRAIN_BASE_SCORE
        or train_policy_gate["area_delta"] != TRAIN_AREA_DELTA
        or train_policy_gate["pooled_delta"] != TRAIN_POOLED_DELTA
        or train_policy_gate["switches"] != TRAIN_SWITCHES
    ):
        raise ContractError(
            f"train policy audit changed: {train_policy_gate}")

    synthetic = load_synthetic_by_relation(synthetic_path)
    tasks = build_agent_tasks(
        rows, agent, synthetic, seed=SEED, n_proposals=N_PROPOSALS)
    proposals = [task for task in tasks if task["phase"] == "propose"]
    if (
        len(proposals) != 478
        or any(task.get("shot_subjects") for task in proposals)
        or any(task.get("n_samples") != N_PROPOSALS for task in proposals)
    ):
        raise ContractError("validation route must remain zero-shot N=3")
    validate_tasks(tasks, MINISTRAL)

    plan_dir = output / "plan"
    row_path = plan_dir / "INPUT_ROWS.jsonl"
    task_path = plan_dir / f"tasks/{MINISTRAL}.jsonl"
    smoke_path = plan_dir / f"smoke/{MINISTRAL}.jsonl"
    response_path = output / f"responses/{MINISTRAL}.jsonl"
    write_jsonl_atomic(row_path, rows)
    write_jsonl_atomic(task_path, tasks)

    smoke_keys: set[tuple[str, str]] = set()
    seen_relations: set[str] = set()
    for row in rows:
        if row["Relation"] not in seen_relations:
            seen_relations.add(row["Relation"])
            smoke_keys.add(_key(row))
    smoke = [
        task for task in tasks
        if (str(task["subject"]), str(task["relation"])) in smoke_keys
    ]
    if len(smoke_keys) != 6 or len(smoke) != 18:
        raise ContractError("expected one complete N=3 smoke row per relation")
    write_jsonl_atomic(smoke_path, smoke)

    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": True,
        "gold_awareness_scope": "train_policy_selection_only",
        "validation_opened": False,
        "validation_labels_used": False,
        "rows": len(rows),
        "n_proposals": N_PROPOSALS,
        "seed": SEED,
        "selected_decoder_policy": TRAIN_APPROVED_POLICY,
        "train_policy_gate": train_policy_gate,
        "input_rows": str(row_path),
        "input_rows_sha256": sha256(row_path),
        "source_graph": str(source_graph),
        "source_graph_sha256": sha256(source_graph),
        "source_graph_manifest_sha256": sha256(
            source_graph.with_suffix(source_graph.suffix + ".manifest.json")),
        "synthetic_cot": str(synthetic_path),
        "synthetic_cot_sha256": sha256(synthetic_path),
        "agents": str(config_path),
        "agents_sha256": sha256(config_path),
        "baseline_predictions": str(baseline_path),
        "baseline_predictions_sha256": sha256(baseline_path),
        "baseline_pipeline_id": COMPETITION_PIPELINE_ID,
        "baseline_reported_score": baseline_detail["reported_score"],
        "baseline_validation_selected_lineage": True,
        "train_result": str(train_result_path),
        "train_result_sha256": sha256(train_result_path),
        "train_typed_graph": str(train_typed_graph),
        "train_typed_graph_sha256": sha256(train_typed_graph),
        "train_gold": str(train_gold),
        "train_gold_sha256": sha256(train_gold),
        "task_path": str(task_path),
        "task_sha256": sha256(task_path),
        "task_count": len(tasks),
        "smoke_path": str(smoke_path),
        "smoke_sha256": sha256(smoke_path),
        "response_path": str(response_path),
        "typed_graph": str(output / "graph/TYPED_VALIDATION_GRAPH.jsonl"),
        "verification_queue": str(output / "graph/VERIFICATION_QUEUE.jsonl"),
        "predictions": str(output / "VALIDATION_PREDICTIONS.jsonl"),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(plan_dir / "PLAN.json", plan)
    print(json.dumps({
        "plan": str(plan_dir / "PLAN.json"),
        "rows": len(rows),
        "tasks": len(tasks),
        "proposal_generations": len(rows) * N_PROPOSALS,
        "smoke_tasks": len(smoke),
        "selected_decoder_policy": TRAIN_APPROVED_POLICY,
        "validation_opened": False,
    }, indent=2, sort_keys=True))
    return 0


def admit(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    config = load_agent_config(Path(plan["agents"]))
    agent = _agent(config, MINISTRAL)
    rows = read_jsonl(Path(plan["input_rows"]))
    responses = load_responses(Path(plan["response_path"]).parent, [agent])
    third_graphs = assemble_graphs(rows, [agent], responses)
    base_by = {_key(row): row for row in read_jsonl(Path(plan["source_graph"]))}
    third_by = {_key(row): row for row in third_graphs}
    response_map = responses[MINISTRAL]
    expected = {_key(row) for row in rows}
    if set(base_by) != expected or set(third_by) != expected:
        raise ContractError("validation graph coverage mismatch")

    graphs, queue, audit = [], [], []
    for source in rows:
        key = _key(source)
        proposal = _proposal_response(response_map, *key)
        graph, pending, detail = _merge_typed_row(
            base_by[key], third_by[key], proposal)
        graphs.append(graph)
        queue.extend(pending)
        audit.append(detail)

    graph_path = Path(plan["typed_graph"])
    queue_path = Path(plan["verification_queue"])
    audit_path = graph_path.parent / "ADMISSION_AUDIT.jsonl"
    write_jsonl_atomic(graph_path, graphs)
    write_jsonl_atomic(queue_path, queue)
    write_jsonl_atomic(audit_path, audit)
    common = {
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "split": "validation",
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "source_graph": plan["source_graph"],
        "source_graph_sha256": plan["source_graph_sha256"],
        "response": plan["response_path"],
        "response_sha256": sha256(Path(plan["response_path"])),
        "policy": "typed-component-v1",
        "ministral_commitments_preserved": True,
        "verification_queue_graph_connected": True,
        "dormant_candidates_directly_outputtable": False,
        "route_selection_normalization":
            "canonical-agent-output-selection-v2",
        "route_selection_requires_agent_outputs": True,
        "legacy_selected_by_fallback_allowed": False,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(graph_path.with_suffix(graph_path.suffix + ".manifest.json"), {
        **common,
        "schema": GRAPH_SCHEMA,
        "rows": len(graphs),
        "output": str(graph_path),
        "output_sha256": sha256(graph_path),
        "verification_queue": str(queue_path),
        "verification_queue_sha256": sha256(queue_path),
        "admission_audit": str(audit_path),
        "admission_audit_sha256": sha256(audit_path),
        "relational_graph_rebuilt": True,
        "dormant_candidates": sum(
            len(row.get("dormant_candidates", [])) for row in graphs),
    })
    _write_json(queue_path.with_suffix(queue_path.suffix + ".manifest.json"), {
        **common,
        "schema": "ministral-typed-validation-verification-queue-v1",
        "rows": len(queue),
        "output": str(queue_path),
        "output_sha256": sha256(queue_path),
        "directly_consumable_as_predictions": False,
    })
    print(json.dumps({
        "typed_graph": str(graph_path),
        "rows": len(graphs),
        "admitted_new_candidates": sum(
            row["admitted_new_candidates"] for row in audit),
        "verification_queue_candidates": len(queue),
    }, indent=2, sort_keys=True))
    return 0


def _unanimous_new_area(graph: Mapping[str, Any]) -> list[str]:
    """Return the sole legal unanimous new area, otherwise fail closed."""
    if graph.get("Relation") != "hasArea":
        return []
    candidates = []
    for node in graph.get("candidates", []):
        evidence = node.get("routes", {}).get(ROUTE, {})
        if (
            evidence.get("admission_reason")
            == "numeric_complete_link_self_consistent_new"
            and int(evidence.get("support", 0)) == N_PROPOSALS
            and int(evidence.get("samples", 0)) == N_PROPOSALS
        ):
            candidates.append(str(node["item"]))
    return candidates if len(candidates) == 1 else []


def apply_frozen_policy(
    baseline: Sequence[Mapping[str, Any]],
    graphs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph_by = {_key(row): row for row in graphs}
    if len(graph_by) != len(graphs):
        raise ContractError("duplicate typed validation graph key")
    output, decisions = [], []
    for source in baseline:
        key = _key(source)
        if key not in graph_by:
            raise ContractError(f"typed graph lacks baseline row {key}")
        row = copy.deepcopy(source)
        proposed = _unanimous_new_area(graph_by[key])
        before = [str(item) for item in row.get("ObjectEntities", [])]
        changed = bool(proposed and proposed != before)
        if changed:
            row["ObjectEntities"] = proposed
        output.append(row)
        decisions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "policy": TRAIN_APPROVED_POLICY,
            "incumbent": before,
            "proposal": proposed,
            "changed": changed,
            "reason": (
                "single_new_unanimous_typed_area_component"
                if changed else "keep_incumbent"
            ),
        })
    return output, decisions


def decode(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    graph_path = Path(plan["typed_graph"])
    graph_manifest = _manifest(graph_path)
    if (
        graph_manifest.get("schema") != GRAPH_SCHEMA
        or graph_manifest.get("contains_labels") is not False
        or graph_manifest.get("validation_opened") is not False
    ):
        raise ContractError("typed validation graph is not label-free")
    baseline = read_jsonl(Path(plan["baseline_predictions"]))
    graphs = read_jsonl(graph_path)
    predictions, decisions = apply_frozen_policy(baseline, graphs)
    prediction_path = Path(plan["predictions"])
    decision_path = output / "DECISIONS.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(decision_path, decisions)
    manifest = {
        "schema": PREDICTION_SCHEMA,
        "contains_labels": False,
        "gold_aware": True,
        "gold_awareness_scope": "train_policy_selection_only",
        "validation_opened": False,
        "validation_labels_used": False,
        "development_only": True,
        "deployable": False,
        "baseline_validation_selected_lineage": True,
        "policy": TRAIN_APPROVED_POLICY,
        "rows": len(predictions),
        "switches": sum(bool(row["changed"]) for row in decisions),
        "output": str(prediction_path),
        "output_sha256": sha256(prediction_path),
        "decisions": str(decision_path),
        "decisions_sha256": sha256(decision_path),
        "baseline": plan["baseline_predictions"],
        "baseline_sha256": plan["baseline_predictions_sha256"],
        "typed_graph": str(graph_path),
        "typed_graph_sha256": sha256(graph_path),
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
    }
    _write_json(
        prediction_path.with_suffix(prediction_path.suffix + ".manifest.json"),
        manifest)
    print(json.dumps({
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "switches": manifest["switches"],
        "validation_opened": False,
    }, indent=2, sort_keys=True))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    graph_path = Path(plan["typed_graph"])
    prediction_path = Path(plan["predictions"])
    graph_manifest = _manifest(graph_path)
    prediction_manifest = _manifest(prediction_path)
    if (
        graph_manifest.get("contains_labels") is not False
        or prediction_manifest.get("contains_labels") is not False
        or prediction_manifest.get("validation_opened") is not False
        or prediction_manifest.get("validation_labels_used") is not False
        or prediction_manifest.get("policy") != TRAIN_APPROVED_POLICY
    ):
        raise ContractError("cannot evaluate contaminated or foreign artifact")

    rows = read_jsonl(Path(plan["input_rows"]))
    base_graphs = read_jsonl(Path(plan["source_graph"]))
    typed_graphs = read_jsonl(graph_path)
    config = load_agent_config(Path(plan["agents"]))
    agent = _agent(config, MINISTRAL)
    responses = load_responses(Path(plan["response_path"]).parent, [agent])
    raw_graphs = assemble_graphs(rows, [agent], responses)
    base_by = {_key(row): row for row in base_graphs}
    raw_by = {_key(row): row for row in raw_graphs}
    combined_raw = []
    for row in rows:
        key = _key(row)
        merged = copy.deepcopy(base_by[key])
        merged["candidates"].extend(copy.deepcopy(raw_by[key]["candidates"]))
        combined_raw.append(merged)

    gold_path = Path(args.validation_gold).resolve()
    gold = read_jsonl(gold_path)
    expected = {_key(row) for row in rows}
    gold_by = {_key(row): row for row in gold}
    if len(gold) != 478 or set(gold_by) != expected:
        raise ContractError("validation gold does not match frozen plan")
    ordered_gold = [gold_by[_key(row)] for row in rows]
    baseline = read_jsonl(Path(plan["baseline_predictions"]))
    predictions = read_jsonl(prediction_path)
    scores = {
        "baseline_predictions": score(baseline, ordered_gold),
        "typed_policy_predictions": score(predictions, ordered_gold),
        "base_candidate_oracle": score(
            oracle_rows(base_graphs, ordered_gold), ordered_gold),
        "raw_n3_candidate_oracle": score(
            oracle_rows(combined_raw, ordered_gold), ordered_gold),
        "typed_candidate_oracle": score(
            oracle_rows(typed_graphs, ordered_gold), ordered_gold),
    }
    pooled = "*** All Relations ***"
    result = {
        "schema": RESULT_SCHEMA,
        "development_only": True,
        "deployable": False,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": True,
        "validation_labels_used_for_selection": False,
        "selected_policy_frozen_on_train": True,
        "policy": TRAIN_APPROVED_POLICY,
        "scores": scores,
        "typed_policy_pooled_delta":
            scores["typed_policy_predictions"][pooled]
            - scores["baseline_predictions"][pooled],
        "typed_policy_area_delta":
            scores["typed_policy_predictions"]["hasArea"]
            - scores["baseline_predictions"]["hasArea"],
        "raw_n3_oracle_delta":
            scores["raw_n3_candidate_oracle"][pooled]
            - scores["base_candidate_oracle"][pooled],
        "typed_oracle_delta":
            scores["typed_candidate_oracle"][pooled]
            - scores["base_candidate_oracle"][pooled],
        "switches": prediction_manifest["switches"],
        "baseline_predictions": plan["baseline_predictions"],
        "baseline_predictions_sha256": plan["baseline_predictions_sha256"],
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "typed_graph": str(graph_path),
        "typed_graph_sha256": sha256(graph_path),
        "validation_gold": str(gold_path),
        "validation_gold_sha256": sha256(gold_path),
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
    }
    analysis = output / "analysis"
    _write_json(analysis / "RESULT.json", result)
    lines = [
        "# Typed Ministral validation confirmation",
        "",
        "Development validation confirmation. The candidate graph and final "
        "predictions were frozen before validation labels were opened.",
        "",
        f"- Frozen baseline F1: "
        f"**{scores['baseline_predictions'][pooled]:.6f}**",
        f"- Typed-policy F1: "
        f"**{scores['typed_policy_predictions'][pooled]:.6f}**",
        f"- End-to-end delta: **{result['typed_policy_pooled_delta']:+.6f}**",
        f"- hasArea delta: **{result['typed_policy_area_delta']:+.4f}**",
        f"- Base/raw-N3/typed oracle: "
        f"**{scores['base_candidate_oracle'][pooled]:.6f} / "
        f"{scores['raw_n3_candidate_oracle'][pooled]:.6f} / "
        f"{scores['typed_candidate_oracle'][pooled]:.6f}**",
        f"- Raw N3 oracle delta: **{result['raw_n3_oracle_delta']:+.6f}**",
        f"- Typed oracle delta: **{result['typed_oracle_delta']:+.6f}**",
        f"- Frozen decoder switches: **{result['switches']}**",
        "",
        "| relation | baseline | typed policy | base oracle | typed oracle |",
        "|---|---:|---:|---:|---:|",
    ]
    for relation in sorted(scores["baseline_predictions"]):
        lines.append(
            f"| {relation} | "
            f"{scores['baseline_predictions'][relation]:.4f} | "
            f"{scores['typed_policy_predictions'][relation]:.4f} | "
            f"{scores['base_candidate_oracle'][relation]:.4f} | "
            f"{scores['typed_candidate_oracle'][relation]:.4f} |")
    lines.extend([
        "",
        "The baseline is the Codabench-matched validation-selected lineage, "
        "so this is not a blind-test claim. No rule or threshold was selected "
        "from these validation scores.",
        "",
    ])
    (analysis / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps({
        "baseline": scores["baseline_predictions"][pooled],
        "typed_policy": scores["typed_policy_predictions"][pooled],
        "typed_policy_delta": result["typed_policy_pooled_delta"],
        "base_oracle": scores["base_candidate_oracle"][pooled],
        "raw_n3_oracle": scores["raw_n3_candidate_oracle"][pooled],
        "typed_oracle": scores["typed_candidate_oracle"][pooled],
        "typed_oracle_delta": result["typed_oracle_delta"],
        "switches": result["switches"],
        "result": str(analysis / "RESULT.md"),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subs = result.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--agents", default=str(DEFAULT_CONFIG))
    p.add_argument("--source-graph", default=str(DEFAULT_SOURCE_GRAPH))
    p.add_argument("--synthetic-cot", default=str(DEFAULT_SYNTHETIC))
    p.add_argument("--train-result", default=str(DEFAULT_TRAIN_RESULT))
    p.add_argument(
        "--train-typed-graph", default=str(DEFAULT_TRAIN_TYPED_GRAPH))
    p.add_argument("--train-gold", default=str(DEFAULT_TRAIN_GOLD))
    p.set_defaults(function=prepare)
    p = subs.add_parser("admit")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.set_defaults(function=admit)
    p = subs.add_parser("decode")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.set_defaults(function=decode)
    p = subs.add_parser("evaluate")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--validation-gold", default=str(DEFAULT_VALIDATION_GOLD))
    p.set_defaults(function=evaluate)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
