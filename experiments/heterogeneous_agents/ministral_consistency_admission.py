#!/usr/bin/env python3
"""Nested Ministral N=1/N=3 candidate supply and safe graph admission.

The experiment deliberately separates three concerns:

* ``prepare`` is label-free and creates one three-generation proposal task for
  every frozen training row, plus the existing candidate-blind commitments.
* ``admit`` is label-free.  It preserves every source-graph candidate, attaches
  Ministral provenance when a source candidate is independently corroborated,
  and admits a new Ministral-only candidate only when at least two of the three
  generations produce the same canonical value.
* ``analyze`` validates the immutable admitted-graph manifest before opening
  training labels.  It reports nested N=1/N=3 raw supply, admitted supply,
  candidate precision/noise, and unique-answer retention.  Validation is
  structurally absent.

The admitted graph is not a new final decoder.  It is a label-free evidence
artifact suitable for a later frozen selector experiment if the predeclared
candidate-supply gate passes.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
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
    canonical_key,
    load_agent_config,
    load_synthetic_by_relation,
    proposal_parse_status,
    read_jsonl,
    sha256,
    validate_inputs,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.ministral_candidate_supply import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    MINISTRAL,
    _agent,
    _has_correct,
    _items,
    _json,
    _source_manifest,
)
from experiments.heterogeneous_agents.relational_candidate_graph import (
    augment_relational_graph,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_OUTPUT = RUNS / "ministral_consistency_admission_20260728_v1"
DEFAULT_CONFIG = ROOT / "configs/final/portfolio_supply.json"
DEFAULT_INPUT = ROOT / "data/train.jsonl"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_SYNTHETIC = ROOT / "data/synthetic_cot_faithful.jsonl"
DEFAULT_SOURCE_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/graphs/train_graph.jsonl"
)
PLAN_SCHEMA = "ministral-consistency-admission-plan-v1"
GRAPH_SCHEMA = "ministral-consistency-admitted-graph-manifest-v1"
RESULT_SCHEMA = "ministral-consistency-admission-result-v1"
ROUTE = "ministral:self_consistency"
SEED = 20260729
N_PROPOSALS = 3
ADMISSION_SUPPORT = 2

# These gates are frozen before inference.  They require N=3 to improve the
# raw reservoir, while the admitted reservoir must retain useful discoveries
# and materially reduce unsupported new candidates.
MIN_RAW_N3_INCREMENT_OVER_N1 = 0.003
MIN_ADMITTED_ORACLE_DELTA = 0.010
MIN_ADMITTED_UNIQUE_ROWS = 8
MIN_ADMITTED_UNIQUE_RELATIONS = 3
MIN_UNIQUE_RETENTION = 0.70
MIN_NEW_CANDIDATE_REJECTION_RATE = 0.40
MAX_PARSE_FAILURE_RATE = 0.10


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _response_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ContractError(f"missing response or manifest: {path}")
    manifest = _json(manifest_path)
    if manifest.get("output_sha256") != sha256(path):
        raise ContractError(f"response hash mismatch: {path}")
    return manifest


def _validate_plan(output: Path) -> dict[str, Any]:
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or plan.get("n_proposals") != N_PROPOSALS
        or plan.get("admission_support") != ADMISSION_SUPPORT
    ):
        raise ContractError("invalid or label-contaminated Ministral N=3 plan")
    for field in ("input_rows", "source_graph", "synthetic_cot", "agents"):
        path = Path(plan[field])
        if sha256(path) != plan[f"{field}_sha256"]:
            raise ContractError(f"frozen plan artifact changed: {field}")
    _source_manifest(Path(plan["source_graph"]))
    task_path = Path(plan["task_path"])
    response_path = Path(plan["response_path"])
    manifest = _response_manifest(response_path)
    if (
        sha256(task_path) != plan["task_sha256"]
        or manifest.get("task_sha256") != plan["task_sha256"]
        or manifest.get("tasks") != plan["task_count"]
        or manifest.get("agent_id") != MINISTRAL
        or manifest.get("model") != EXPECTED_MODEL
        or manifest.get("revision") != EXPECTED_REVISION
    ):
        raise ContractError("stale or foreign Ministral N=3 responses")
    tasks = read_jsonl(task_path)
    validate_tasks(tasks, MINISTRAL)
    proposal_tasks = [task for task in tasks if task["phase"] == "propose"]
    if (
        len(proposal_tasks) != plan["rows"]
        or any(task.get("n_samples") != N_PROPOSALS for task in proposal_tasks)
    ):
        raise ContractError("Ministral proposal tasks are not complete N=3 tasks")
    return plan


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    config_path = Path(args.agents).resolve()
    input_path = Path(args.input).resolve()
    synthetic_path = Path(args.synthetic_cot).resolve()
    source_graph = Path(args.source_graph).resolve()
    config = load_agent_config(config_path)
    agent = _agent(config, MINISTRAL)
    if (
        agent["model"] != EXPECTED_MODEL
        or agent.get("revision") != EXPECTED_REVISION
        or int(agent.get("synthetic_shots", -1)) != 0
    ):
        raise ContractError("Ministral config is not the audited zero-shot checkpoint")
    _source_manifest(source_graph)

    labeled_rows = read_jsonl(input_path)
    validate_inputs(labeled_rows)
    rows = [
        {
            "SubjectEntity": str(row["SubjectEntity"]),
            "Relation": str(row["Relation"]),
        }
        for row in labeled_rows
    ]
    if len(rows) != 477 or len({_key(row) for row in rows}) != len(rows):
        raise ContractError(f"expected 477 unique train rows, found {len(rows)}")
    if {_key(row) for row in read_jsonl(source_graph)} != {
            _key(row) for row in rows}:
        raise ContractError("source graph does not cover frozen train rows")

    synthetic = load_synthetic_by_relation(synthetic_path)
    tasks = build_agent_tasks(
        rows, agent, synthetic, seed=SEED, n_proposals=N_PROPOSALS)
    proposal_tasks = [task for task in tasks if task["phase"] == "propose"]
    if (
        len(proposal_tasks) != len(rows)
        or any(task.get("shot_subjects") for task in proposal_tasks)
        or any(task.get("n_samples") != N_PROPOSALS for task in proposal_tasks)
    ):
        raise ContractError("Ministral route must remain zero-shot nested N=3")
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
            smoke_keys.add(_key(row))
            seen_relations.add(row["Relation"])
    smoke = [
        task for task in tasks
        if (str(task["subject"]), str(task["relation"])) in smoke_keys
    ]
    if len(smoke_keys) != 6 or len(smoke) != 18:
        raise ContractError("expected one complete three-task smoke row per relation")
    write_jsonl_atomic(smoke_path, smoke)

    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "rows": len(rows),
        "n_proposals": N_PROPOSALS,
        "nested_prefixes": [1, 3],
        "seed": SEED,
        "admission_support": ADMISSION_SUPPORT,
        "admission_policy": {
            "preserve_all_source_candidates": True,
            "attach_ministral_to_source_match": True,
            "new_ministral_candidate_minimum_support": ADMISSION_SUPPORT,
            "use_ministral_commitments_in_decoder": False,
            "labels_used": False,
        },
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
        "model": agent["model"],
        "revision": agent["revision"],
        "task_path": str(task_path),
        "task_sha256": sha256(task_path),
        "task_count": len(tasks),
        "smoke_path": str(smoke_path),
        "smoke_sha256": sha256(smoke_path),
        "response_path": str(response_path),
        "admitted_graph": str(output / "graph/ADMITTED_GRAPH.jsonl"),
        "admission_audit": str(output / "graph/ADMISSION_AUDIT.jsonl"),
        "declared_parameter_total": config["declared_parameter_total"],
        "verified_parameter_total": config["verified_parameter_total"],
        "parameter_cap": config["parameter_cap"],
        "parameter_headroom": config["declared_parameter_headroom"],
        "promotion_gate": {
            "raw_n3_increment_over_n1_minimum":
                MIN_RAW_N3_INCREMENT_OVER_N1,
            "admitted_oracle_delta_minimum": MIN_ADMITTED_ORACLE_DELTA,
            "admitted_unique_rows_minimum": MIN_ADMITTED_UNIQUE_ROWS,
            "admitted_unique_relations_minimum": MIN_ADMITTED_UNIQUE_RELATIONS,
            "unique_correct_retention_minimum": MIN_UNIQUE_RETENTION,
            "new_candidate_rejection_rate_minimum":
                MIN_NEW_CANDIDATE_REJECTION_RATE,
            "parse_failure_rate_maximum": MAX_PARSE_FAILURE_RATE,
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    plan_path = plan_dir / "PLAN.json"
    _write_json(plan_path, plan)
    print(json.dumps({
        "plan": str(plan_path),
        "rows": len(rows),
        "tasks": len(tasks),
        "proposal_generations": len(rows) * N_PROPOSALS,
        "smoke_tasks": len(smoke),
        "verified_parameter_total": config["verified_parameter_total"],
        "parameter_headroom": config["declared_parameter_headroom"],
    }, indent=2, sort_keys=True))
    return 0


def _prefix_graphs(
    rows: Sequence[Mapping[str, Any]],
    agent: Mapping[str, Any],
    response_map: Mapping[str, Mapping[str, dict]],
    n: int,
) -> list[dict[str, Any]]:
    if n not in {1, 3}:
        raise ContractError(f"unsupported nested prefix N={n}")
    copied = copy.deepcopy(response_map)
    for response in copied[MINISTRAL].values():
        if response.get("phase") == "propose":
            generations = response.get("generations", [])
            if len(generations) != N_PROPOSALS:
                raise ContractError(
                    f"{response.get('task_id')}: expected three generations")
            response["generations"] = generations[:n]
    return assemble_graphs(rows, [agent], copied)


def _source_index(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Canonicalize and coalesce legacy duplicate source surfaces in-place.

    The frozen production graph predates typed numeric keys on every route.
    Consequently, the same numeric surface can occur once as ``numeric:923768``
    and once as ``923768``.  They are not competing facts and must not become
    separate components.  Coalescing takes the maximum within-route support
    (rather than summing duplicated evidence) and unions independent routes.
    """
    relation = str(graph["Relation"])
    result: dict[str, dict[str, Any]] = {}
    coalesced: list[dict[str, Any]] = []
    for node in graph.get("candidates", []):
        key = canonical_key(str(node["item"]), relation)
        if not key:
            continue
        if key not in result:
            node["key"] = key
            result[key] = node
            coalesced.append(node)
            continue
        target = result[key]
        for agent_id, evidence in node.get("sources", {}).items():
            current = target.setdefault("sources", {}).get(agent_id)
            if (
                current is None
                or float(evidence.get("support_rate", 0.0))
                > float(current.get("support_rate", 0.0))
            ):
                target["sources"][agent_id] = copy.deepcopy(evidence)
        for route, evidence in node.get("routes", {}).items():
            current = target.setdefault("routes", {}).get(route)
            if current is None:
                target["routes"][route] = copy.deepcopy(evidence)
            else:
                if (
                    float(evidence.get("support_rate", 0.0))
                    > float(current.get("support_rate", 0.0))
                ):
                    preserved_selected = bool(current.get("selected", False))
                    target["routes"][route] = copy.deepcopy(evidence)
                    target["routes"][route]["selected"] = (
                        preserved_selected
                        or bool(evidence.get("selected", False))
                    )
                else:
                    current["selected"] = (
                        bool(current.get("selected", False))
                        or bool(evidence.get("selected", False))
                    )
        for agent_id, selected in node.get("selected_by", {}).items():
            target.setdefault("selected_by", {})[agent_id] = (
                bool(target["selected_by"].get(agent_id, False))
                or bool(selected)
            )
        _refresh_route_summary(target)
    if isinstance(graph, dict):
        graph["candidates"] = coalesced
    return result


def _refresh_route_summary(node: dict[str, Any]) -> None:
    sources = set(str(value) for value in node.get("sources", {}))
    routes = set(str(value) for value in node.get("routes", {}))
    node.setdefault("route_summary", {})
    node["route_summary"]["model_family_count"] = len(sources)
    node["route_summary"]["route_count"] = len(routes)
    node["route_summary"]["cross_model_agreement"] = len(sources) >= 2
    node["route_summary"]["qwen_sc_only"] = routes == {"qwen:self_consistency"}
    node["route_summary"]["gemma_only"] = routes == {"gemma:independent"}
    node["route_summary"]["system2_only"] = routes == {"qwen:system2"}
    node["route_summary"]["system2_supported"] = "qwen:system2" in routes
    node["route_summary"]["within_qwen_route_agreement"] = {
        "qwen:self_consistency", "qwen:system2"}.issubset(routes)
    node["route_summary"]["ministral_supported"] = MINISTRAL in sources
    node["route_summary"]["ministral_self_consistent"] = (
        int(node.get("sources", {}).get(
            MINISTRAL, {}).get("support", 0)) >= ADMISSION_SUPPORT
    )


def _attach_ministral(
    node: dict[str, Any], *, support: int, samples: int,
    selected: bool, reason: str,
) -> None:
    node.setdefault("sources", {})[MINISTRAL] = {
        "samples": samples,
        "support": support,
        "support_rate": support / samples,
    }
    node.setdefault("routes", {})[ROUTE] = {
        "model_family": MINISTRAL,
        "route_type": "sampled-self-consistency",
        "samples": samples,
        "support": support,
        "support_rate": support / samples,
        "selected": selected,
        "admission_reason": reason,
    }
    node.setdefault("selected_by", {})[MINISTRAL] = selected
    _refresh_route_summary(node)


def _new_node(
    candidate: Mapping[str, Any], relation: str,
) -> dict[str, Any]:
    support = int(candidate["proposal_support"][MINISTRAL])
    item = str(candidate["item"])
    key = canonical_key(item, relation)
    node = {
        "key": key,
        "item": item,
        "type": "numeric" if relation in {"hasArea", "hasCapacity"} else "string",
        "sources": {},
        "routes": {},
        "selected_by": {},
        "route_summary": {
            "qwen_sc_only": False,
            "gemma_only": False,
            "system2_only": False,
            "system2_supported": False,
            "within_qwen_route_agreement": False,
        },
    }
    _attach_ministral(
        node,
        support=support,
        samples=N_PROPOSALS,
        selected=True,
        reason="ministral_self_consistent_new",
    )
    return node


def _merge_row(
    base: Mapping[str, Any], third: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _key(base) != _key(third):
        raise ContractError("cannot merge mismatched graph rows")
    row = copy.deepcopy(base)
    relation = str(row["Relation"])
    source_nodes = _source_index(row)
    raw_new = 0
    admitted_new = 0
    rejected_new = 0
    corroborated = 0
    candidate_audit = []
    for candidate in third.get("candidates", []):
        item = str(candidate["item"])
        key = canonical_key(item, relation)
        support = int(candidate["proposal_support"][MINISTRAL])
        if not key:
            continue
        if key in source_nodes:
            corroborated += 1
            _attach_ministral(
                source_nodes[key],
                support=support,
                samples=N_PROPOSALS,
                selected=support >= ADMISSION_SUPPORT,
                reason="cross_model_corroborated",
            )
            decision = "corroborate_source"
        else:
            raw_new += 1
            if support >= ADMISSION_SUPPORT:
                node = _new_node(candidate, relation)
                row.setdefault("candidates", []).append(node)
                source_nodes[key] = node
                admitted_new += 1
                decision = "admit_self_consistent_new"
            else:
                rejected_new += 1
                decision = "reject_single_sample_new"
        candidate_audit.append({
            "key": key,
            "item": item,
            "support": support,
            "support_rate": support / N_PROPOSALS,
            "decision": decision,
        })

    commitment = third.get("commitments", {}).get(MINISTRAL, {})
    row.setdefault("agents", {})[MINISTRAL] = {
        "candidate_supply_only": True,
        "n_samples": N_PROPOSALS,
        "existence": {
            "available": bool(commitment),
            "selected": commitment.get("existence", "UNKNOWN"),
            "probabilities": commitment.get("existence_probabilities", {}),
        },
        "cardinality": {
            "available": bool(commitment),
            "selected": commitment.get("cardinality", "UNKNOWN"),
            "probabilities": commitment.get("cardinality_probabilities", {}),
        },
        "decoder_commitments_enabled": False,
    }
    row.setdefault("agent_outputs", {})[MINISTRAL] = [
        str(node["item"])
        for node in third.get("candidates", [])
        if int(node["proposal_support"][MINISTRAL]) >= ADMISSION_SUPPORT
    ]
    row.setdefault("proposal_routes", {})[ROUTE] = {
        "available": True,
        "model_family": MINISTRAL,
        "n_samples": N_PROPOSALS,
        "admission_support": ADMISSION_SUPPORT,
    }
    row.setdefault("production_match", {})[
        "ministral_candidate_admission"
    ] = "corroborated-or-two-of-three-v1"
    row["candidates"].sort(key=lambda node: (
        -len(node.get("sources", {})),
        -sum(
            float(value.get("support_rate", 0.0))
            for value in node.get("sources", {}).values()
        ),
        canonical_key(str(node["item"]), relation),
    ))
    # The source relational graph cannot be retained after candidate mutation.
    # Rebuilding it is essential: otherwise downstream component decoders would
    # silently ignore every newly admitted candidate.
    row.pop("relational_graph", None)
    row.pop("relational_graph_schema", None)
    row = augment_relational_graph(row)
    return row, {
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": relation,
        "raw_ministral_candidates": len(third.get("candidates", [])),
        "raw_new_candidates": raw_new,
        "admitted_new_candidates": admitted_new,
        "rejected_new_candidates": rejected_new,
        "corroborated_source_candidates": corroborated,
        "candidate_decisions": candidate_audit,
    }


def admit(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    config = load_agent_config(Path(plan["agents"]))
    agent = _agent(config, MINISTRAL)
    rows = read_jsonl(Path(plan["input_rows"]))
    responses = load_responses(Path(plan["response_path"]).parent, [agent])
    third_graphs = _prefix_graphs(rows, agent, responses, N_PROPOSALS)
    base_graphs = read_jsonl(Path(plan["source_graph"]))
    base_by = {_key(row): row for row in base_graphs}
    third_by = {_key(row): row for row in third_graphs}
    expected = {_key(row) for row in rows}
    if set(base_by) != expected or set(third_by) != expected:
        raise ContractError("candidate graph coverage mismatch")

    admitted_rows, audit_rows = [], []
    for row in rows:
        admitted_row, audit_row = _merge_row(
            base_by[_key(row)], third_by[_key(row)])
        admitted_rows.append(admitted_row)
        audit_rows.append(audit_row)
    graph_path = Path(plan["admitted_graph"])
    audit_path = Path(plan["admission_audit"])
    write_jsonl_atomic(graph_path, admitted_rows)
    write_jsonl_atomic(audit_path, audit_rows)
    manifest = {
        "schema": GRAPH_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "split": "train",
        "rows": len(admitted_rows),
        "output": str(graph_path),
        "output_sha256": sha256(graph_path),
        "admission_audit": str(audit_path),
        "admission_audit_sha256": sha256(audit_path),
        "source_graph": plan["source_graph"],
        "source_graph_sha256": plan["source_graph_sha256"],
        "responses": plan["response_path"],
        "responses_sha256": sha256(Path(plan["response_path"])),
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "admission_policy": plan["admission_policy"],
        "relational_graph_rebuilt": True,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(
        graph_path.with_suffix(graph_path.suffix + ".manifest.json"), manifest)
    print(json.dumps({
        "graph": str(graph_path),
        "rows": len(admitted_rows),
        "raw_new_candidates": sum(
            row["raw_new_candidates"] for row in audit_rows),
        "admitted_new_candidates": sum(
            row["admitted_new_candidates"] for row in audit_rows),
        "rejected_new_candidates": sum(
            row["rejected_new_candidates"] for row in audit_rows),
        "corroborated_source_candidates": sum(
            row["corroborated_source_candidates"] for row in audit_rows),
    }, indent=2, sort_keys=True))
    return 0


def _combined_raw(
    base: Mapping[str, Any], third: Mapping[str, Any],
) -> dict[str, Any]:
    relation = str(base["Relation"])
    candidates: dict[str, dict[str, str]] = {}
    for item in _items(base) + _items(third):
        key = canonical_key(item, relation)
        if key:
            candidates.setdefault(key, {"item": item})
    return {
        "SubjectEntity": str(base["SubjectEntity"]),
        "Relation": relation,
        "candidates": list(candidates.values()),
    }


def _graph_manifest(plan: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(plan["admitted_graph"])
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("contains_labels") is not False
        or manifest.get("gold_aware") is not False
        or manifest.get("validation_opened") is not False
        or manifest.get("validation_labels_used") is not False
        or manifest.get("output_sha256") != sha256(path)
        or manifest.get("plan_sha256") != sha256(
            Path(plan["admitted_graph"]).parents[1] / "plan/PLAN.json")
    ):
        raise ContractError("admitted graph failed label-free manifest validation")
    return manifest


def _candidate_counts(
    base: Mapping[str, Any],
    third: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> dict[str, int]:
    relation = str(base["Relation"])
    base_keys = {
        canonical_key(item, relation) for item in _items(base)
        if canonical_key(item, relation)
    }
    result = Counter()
    for node in third.get("candidates", []):
        key = canonical_key(str(node["item"]), relation)
        if not key or key in base_keys:
            continue
        support = int(node["proposal_support"][MINISTRAL])
        correct = _has_correct([str(node["item"])], gold, relation)
        result["raw_new"] += 1
        result["raw_new_correct"] += int(correct)
        if support >= ADMISSION_SUPPORT:
            result["admitted_new"] += 1
            result["admitted_new_correct"] += int(correct)
    return dict(result)


def _unique_rows(
    rows: Sequence[Mapping[str, Any]],
    base_by: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate_by: Mapping[tuple[str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[list[str]], Counter[str]]:
    unique, relations = [], Counter()
    for row in rows:
        key = _key(row)
        relation = key[1]
        base_correct = _has_correct(
            _items(base_by[key]), gold_by[key], relation)
        candidate_correct = _has_correct(
            _items(candidate_by[key]), gold_by[key], relation)
        if candidate_correct and not base_correct:
            unique.append([key[0], relation])
            relations[relation] += 1
    return unique, relations


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    _graph_manifest(plan)
    config = load_agent_config(Path(plan["agents"]))
    agent = _agent(config, MINISTRAL)
    rows = read_jsonl(Path(plan["input_rows"]))
    responses = load_responses(Path(plan["response_path"]).parent, [agent])
    prefix_graphs = {
        n: _prefix_graphs(rows, agent, responses, n) for n in (1, 3)
    }
    base_graphs = read_jsonl(Path(plan["source_graph"]))
    admitted_graphs = read_jsonl(Path(plan["admitted_graph"]))
    base_by = {_key(row): row for row in base_graphs}
    admitted_by = {_key(row): row for row in admitted_graphs}
    prefix_by = {
        n: {_key(row): row for row in graphs}
        for n, graphs in prefix_graphs.items()
    }
    expected = {_key(row) for row in rows}
    if (
        set(base_by) != expected
        or set(admitted_by) != expected
        or any(set(value) != expected for value in prefix_by.values())
    ):
        raise ContractError("analysis graph coverage mismatch")

    gold_path = Path(args.train_gold).resolve()
    gold_by = {_key(row): row for row in read_jsonl(gold_path)}
    if set(gold_by) != expected:
        raise ContractError("train label coverage mismatch")
    ordered_gold = [gold_by[_key(row)] for row in rows]
    ordered_base = [base_by[_key(row)] for row in rows]
    base_scores = score(oracle_rows(ordered_base, ordered_gold), ordered_gold)

    raw_scores, raw_deltas = {}, {}
    for n in (1, 3):
        raw_graphs = [
            _combined_raw(base_by[_key(row)], prefix_by[n][_key(row)])
            for row in rows
        ]
        scores = score(oracle_rows(raw_graphs, ordered_gold), ordered_gold)
        raw_scores[str(n)] = scores
        raw_deltas[str(n)] = {
            relation: scores[relation] - base_scores[relation]
            for relation in base_scores
        }
    admitted_scores = score(
        oracle_rows(admitted_graphs, ordered_gold), ordered_gold)
    admitted_deltas = {
        relation: admitted_scores[relation] - base_scores[relation]
        for relation in base_scores
    }

    raw_unique, raw_unique_relations = _unique_rows(
        rows, base_by, prefix_by[3], gold_by)
    admitted_unique, admitted_unique_relations = _unique_rows(
        rows, base_by, admitted_by, gold_by)
    candidate_counts = Counter()
    parse_counts = Counter()
    row_truth_audit = []
    for row in rows:
        key = _key(row)
        counts = _candidate_counts(
            base_by[key], prefix_by[3][key], gold_by[key])
        candidate_counts.update(counts)
        relation = key[1]
        response_graph = prefix_by[3][key]
        parse_counts.update(
            response_graph.get(
                "proposal_parse_diagnostics", {}).get(MINISTRAL, {}))
        row_truth_audit.append({
            "SubjectEntity": key[0],
            "Relation": relation,
            **counts,
            "raw_unique_correct": [key[0], relation] in raw_unique,
            "admitted_unique_correct": [key[0], relation] in admitted_unique,
        })
    parse_total = sum(parse_counts.values())
    parse_failures = (
        parse_total
        - parse_counts.get("parsed_nonempty", 0)
        - parse_counts.get("explicit_none", 0)
    )
    parse_failure_rate = (
        parse_failures / parse_total if parse_total else 1.0)
    raw_new = candidate_counts["raw_new"]
    admitted_new = candidate_counts["admitted_new"]
    rejection_rate = 1.0 - admitted_new / raw_new if raw_new else 0.0
    unique_retention = (
        len(admitted_unique) / len(raw_unique) if raw_unique else 0.0)
    raw_precision = (
        candidate_counts["raw_new_correct"] / raw_new if raw_new else 0.0)
    admitted_precision = (
        candidate_counts["admitted_new_correct"] / admitted_new
        if admitted_new else 0.0)
    pooled = "*** All Relations ***"
    n3_increment = (
        raw_deltas["3"][pooled] - raw_deltas["1"][pooled])
    gate_checks = {
        "raw_n3_increment_over_n1":
            n3_increment >= MIN_RAW_N3_INCREMENT_OVER_N1,
        "admitted_oracle_delta":
            admitted_deltas[pooled] >= MIN_ADMITTED_ORACLE_DELTA,
        "admitted_unique_rows":
            len(admitted_unique) >= MIN_ADMITTED_UNIQUE_ROWS,
        "admitted_unique_relations":
            len(admitted_unique_relations) >= MIN_ADMITTED_UNIQUE_RELATIONS,
        "unique_correct_retention":
            unique_retention >= MIN_UNIQUE_RETENTION,
        "new_candidate_rejection_rate":
            rejection_rate >= MIN_NEW_CANDIDATE_REJECTION_RATE,
        "parse_failure_rate":
            parse_failure_rate <= MAX_PARSE_FAILURE_RATE,
    }

    analysis_dir = output / "analysis"
    truth_audit_path = analysis_dir / "ROW_TRUTH_AUDIT.jsonl"
    write_jsonl_atomic(truth_audit_path, row_truth_audit)
    result = {
        "schema": RESULT_SCHEMA,
        "development_only": True,
        "deployable": False,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "model": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "n_proposals": N_PROPOSALS,
        "admission_support": ADMISSION_SUPPORT,
        "verified_parameter_total": config["verified_parameter_total"],
        "parameter_headroom": config["declared_parameter_headroom"],
        "base_candidate_union_oracle_scores": base_scores,
        "raw_candidate_union_oracle_scores": raw_scores,
        "raw_candidate_union_oracle_deltas": raw_deltas,
        "raw_n3_increment_over_n1": n3_increment,
        "admitted_candidate_union_oracle_scores": admitted_scores,
        "admitted_candidate_union_oracle_deltas": admitted_deltas,
        "raw_unique_correct_rows": raw_unique,
        "raw_unique_correct_rows_count": len(raw_unique),
        "raw_unique_correct_by_relation": dict(raw_unique_relations),
        "admitted_unique_correct_rows": admitted_unique,
        "admitted_unique_correct_rows_count": len(admitted_unique),
        "admitted_unique_correct_by_relation": dict(
            admitted_unique_relations),
        "unique_correct_retention": unique_retention,
        "candidate_filter": {
            **dict(candidate_counts),
            "raw_new_candidate_precision": raw_precision,
            "admitted_new_candidate_precision": admitted_precision,
            "new_candidate_rejection_rate": rejection_rate,
        },
        "parse": {
            "counts": dict(parse_counts),
            "failures": parse_failures,
            "samples": parse_total,
            "failure_rate": parse_failure_rate,
        },
        "promotion_gate_checks": gate_checks,
        "promotion_gate_passed": all(gate_checks.values()),
        "next_stage": (
            "freeze_and_validate_admitted_graph"
            if all(gate_checks.values())
            else "do_not_promote_admitted_graph"
        ),
        "admitted_graph": plan["admitted_graph"],
        "admitted_graph_sha256": sha256(Path(plan["admitted_graph"])),
        "row_truth_audit": str(truth_audit_path),
        "row_truth_audit_sha256": sha256(truth_audit_path),
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
    }
    result_path = analysis_dir / "RESULT.json"
    _write_json(result_path, result)

    lines = [
        "# Ministral N=3 consistency and graph-admission audit",
        "",
        "Development-only train audit. Validation was not opened.",
        "",
        f"- Model: `{EXPECTED_MODEL}`",
        f"- Legal portfolio: **{config['verified_parameter_total']:,} / "
        f"{config['parameter_cap']:,}**",
        f"- Base two-model oracle: **{base_scores[pooled]:.6f}**",
        f"- Raw nested N=1 oracle: **{raw_scores['1'][pooled]:.6f}** "
        f"({raw_deltas['1'][pooled]:+.6f})",
        f"- Raw N=3 oracle: **{raw_scores['3'][pooled]:.6f}** "
        f"({raw_deltas['3'][pooled]:+.6f})",
        f"- N=3 incremental gain over nested N=1: "
        f"**{n3_increment:+.6f}**",
        f"- Corroboration-aware admitted oracle: "
        f"**{admitted_scores[pooled]:.6f}** "
        f"({admitted_deltas[pooled]:+.6f})",
        f"- Raw/admitted unique correct rows: "
        f"**{len(raw_unique)} / {len(admitted_unique)}**",
        f"- Raw/admitted unique relation coverage: "
        f"**{len(raw_unique_relations)} / "
        f"{len(admitted_unique_relations)}**",
        f"- Unique-correct retention: **{unique_retention:.2%}**",
        f"- Unsupported-new-candidate rejection: **{rejection_rate:.2%}**",
        f"- Raw/admitted new-candidate precision: "
        f"**{raw_precision:.2%} / {admitted_precision:.2%}**",
        f"- Parse failure rate: **{parse_failure_rate:.2%}**",
        f"- Promotion gate: "
        f"**{'PASS' if all(gate_checks.values()) else 'FAIL'}**",
        "",
        "| relation | base oracle | raw N=1 | raw N=3 | admitted N=3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for relation in sorted(base_scores):
        lines.append(
            f"| {relation} | {base_scores[relation]:.4f} | "
            f"{raw_scores['1'][relation]:.4f} | "
            f"{raw_scores['3'][relation]:.4f} | "
            f"{admitted_scores[relation]:.4f} |")
    lines.extend([
        "",
        "## Promotion checks",
        "",
    ])
    for name, passed in gate_checks.items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend([
        "",
        "The raw and admitted oracles are gold-aware and nondeployable. "
        "The admitted graph itself was constructed before labels were opened; "
        "its manifest certifies `contains_labels=false` and `gold_aware=false`.",
        "",
    ])
    (analysis_dir / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps({
        "base_oracle": base_scores[pooled],
        "raw_n1_oracle": raw_scores["1"][pooled],
        "raw_n3_oracle": raw_scores["3"][pooled],
        "raw_n3_increment_over_n1": n3_increment,
        "admitted_oracle": admitted_scores[pooled],
        "admitted_oracle_delta": admitted_deltas[pooled],
        "raw_unique_rows": len(raw_unique),
        "admitted_unique_rows": len(admitted_unique),
        "promotion_gate_passed": all(gate_checks.values()),
        "result": str(analysis_dir / "RESULT.md"),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    prepare_parser.add_argument("--agents", default=str(DEFAULT_CONFIG))
    prepare_parser.add_argument("--input", default=str(DEFAULT_INPUT))
    prepare_parser.add_argument("--synthetic-cot", default=str(DEFAULT_SYNTHETIC))
    prepare_parser.add_argument("--source-graph", default=str(DEFAULT_SOURCE_GRAPH))
    prepare_parser.set_defaults(function=prepare)
    admit_parser = subparsers.add_parser("admit")
    admit_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    admit_parser.set_defaults(function=admit)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    analyze_parser.add_argument("--train-gold", default=str(DEFAULT_GOLD))
    analyze_parser.set_defaults(function=analyze)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
