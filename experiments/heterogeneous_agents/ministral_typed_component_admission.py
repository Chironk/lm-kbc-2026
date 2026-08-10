#!/usr/bin/env python3
"""Typed component admission over the frozen Ministral N=3 responses.

This experiment fixes the failure exposed by exact 2-of-3 voting:
semantically equivalent numeric answers such as 590, 592.55, and 603.67 did
not share a canonical string even though all are within the official numeric
tolerance.  ``admit`` clusters numeric values using complete-link 5% distance
and counts support by distinct generation.  It also:

* attaches any Ministral evidence that corroborates an existing source node;
* admits a new numeric component only with support from two generations;
* admits repeated new award names because that relation is open-ended;
* sends unsupported numeric and all other novel string candidates to a
  label-free verification queue instead of silently deleting them;
* rebuilds the typed relational graph after candidate mutation.

``prepare`` and ``admit`` never open labels.  ``analyze`` validates both
label-free manifests before opening training labels and compares raw N=3,
legacy exact admission, and typed admission.  Validation is absent.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
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
    canonical_key,
    load_agent_config,
    proposal_parse_status,
    read_jsonl,
    sha256,
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
from experiments.heterogeneous_agents.ministral_consistency_admission import (
    ADMISSION_SUPPORT,
    N_PROPOSALS,
    ROUTE,
    _attach_ministral,
    _combined_raw,
    _key,
    _new_node,
    _source_index,
)
from experiments.heterogeneous_agents.relational_candidate_graph import (
    augment_relational_graph,
)
from experiments.heterogeneous_agents.route_aware_candidate_graph import (
    normalize_route_selection,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_OUTPUT = RUNS / "ministral_typed_component_admission_20260729_v3"
DEFAULT_CONFIG = ROOT / "configs/final/portfolio_supply.json"
DEFAULT_SOURCE_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/graphs/train_graph.jsonl"
)
DEFAULT_N3_RUN = RUNS / "ministral_consistency_admission_20260728_v1"
DEFAULT_EXACT_GRAPH = DEFAULT_N3_RUN / "graph/ADMITTED_GRAPH.jsonl"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
PLAN_SCHEMA = "ministral-typed-component-admission-plan-v1"
GRAPH_SCHEMA = "ministral-typed-component-graph-manifest-v1"
QUEUE_SCHEMA = "ministral-verification-queue-manifest-v1"
RESULT_SCHEMA = "ministral-typed-component-admission-result-v1"
NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}
OPEN_LIST_RELATIONS = {"awardWonBy"}
NUMERIC_TOLERANCE = 0.05

# Frozen train-only promotion criteria.  The experiment must improve materially
# over exact admission while retaining useful raw discoveries and increasing
# candidate precision rather than merely enlarging the reservoir.
MIN_ORACLE_DELTA_OVER_BASE = 0.010
MIN_ORACLE_GAIN_OVER_EXACT = 0.007
MIN_UNIQUE_ROWS = 5
MIN_UNIQUE_RELATIONS = 2
MIN_RAW_UNIQUE_RETENTION = 0.40
MIN_PRECISION_MULTIPLIER_OVER_RAW = 1.25


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


def _validate_n3_source(path: Path) -> dict[str, Any]:
    plan = _json(path / "plan/PLAN.json")
    response = path / f"responses/{MINISTRAL}.jsonl"
    manifest = _manifest(response)
    if (
        plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or plan.get("n_proposals") != N_PROPOSALS
        or manifest.get("agent_id") != MINISTRAL
        or manifest.get("model") != EXPECTED_MODEL
        or manifest.get("revision") != EXPECTED_REVISION
        or manifest.get("task_sha256") != plan.get("task_sha256")
    ):
        raise ContractError("invalid frozen Ministral N=3 source")
    return plan


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    source_graph = Path(args.source_graph).resolve()
    n3_run = Path(args.n3_run).resolve()
    exact_graph = Path(args.exact_graph).resolve()
    config_path = Path(args.agents).resolve()
    _source_manifest(source_graph)
    n3_plan = _validate_n3_source(n3_run)
    exact_manifest = _manifest(exact_graph)
    if (
        exact_manifest.get("contains_labels") is not False
        or exact_manifest.get("gold_aware") is not False
        or exact_manifest.get("validation_opened") is not False
        or exact_manifest.get("validation_labels_used") is not False
    ):
        raise ContractError("legacy exact admitted graph is not label-free")
    config = load_agent_config(config_path)
    agent = _agent(config, MINISTRAL)
    if (
        agent["model"] != EXPECTED_MODEL
        or agent.get("revision") != EXPECTED_REVISION
    ):
        raise ContractError("unexpected Ministral checkpoint")

    rows = read_jsonl(Path(n3_plan["input_rows"]))
    if len(rows) != 477:
        raise ContractError("expected 477 frozen training rows")
    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "rows": len(rows),
        "n_proposals": N_PROPOSALS,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "minimum_distinct_generation_support": ADMISSION_SUPPORT,
        "typed_policy": {
            "numeric": "complete_link_5pct_two_generations",
            "open_list": "exact_two_generations",
            "other_string": "source_corroboration_or_verification_queue",
            "source_candidates_preserved": True,
            "ministral_commitments_enabled": True,
            "verification_queue_graph_connected": True,
        },
        "agents": str(config_path),
        "agents_sha256": sha256(config_path),
        "source_graph": str(source_graph),
        "source_graph_sha256": sha256(source_graph),
        "n3_run": str(n3_run),
        "n3_plan": str(n3_run / "plan/PLAN.json"),
        "n3_plan_sha256": sha256(n3_run / "plan/PLAN.json"),
        "n3_response": n3_plan["response_path"],
        "n3_response_sha256": sha256(Path(n3_plan["response_path"])),
        "input_rows": n3_plan["input_rows"],
        "input_rows_sha256": sha256(Path(n3_plan["input_rows"])),
        "legacy_exact_graph": str(exact_graph),
        "legacy_exact_graph_sha256": sha256(exact_graph),
        "typed_graph": str(output / "graph/TYPED_ADMITTED_GRAPH.jsonl"),
        "verification_queue": str(output / "graph/VERIFICATION_QUEUE.jsonl"),
        "verified_parameter_total": config["verified_parameter_total"],
        "parameter_cap": config["parameter_cap"],
        "parameter_headroom": config["declared_parameter_headroom"],
        "promotion_gate": {
            "oracle_delta_over_base_minimum": MIN_ORACLE_DELTA_OVER_BASE,
            "oracle_gain_over_exact_minimum": MIN_ORACLE_GAIN_OVER_EXACT,
            "unique_rows_minimum": MIN_UNIQUE_ROWS,
            "unique_relations_minimum": MIN_UNIQUE_RELATIONS,
            "raw_unique_retention_minimum": MIN_RAW_UNIQUE_RETENTION,
            "precision_multiplier_over_raw_minimum":
                MIN_PRECISION_MULTIPLIER_OVER_RAW,
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    plan_path = output / "plan/PLAN.json"
    _write_json(plan_path, plan)
    print(json.dumps({
        "plan": str(plan_path),
        "rows": len(rows),
        "reuses_frozen_n3_responses": True,
        "additional_gpu_inference": False,
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
    ):
        raise ContractError("invalid typed-admission plan")
    for field in (
        "agents", "source_graph", "n3_plan", "n3_response", "input_rows",
        "legacy_exact_graph",
    ):
        if sha256(Path(plan[field])) != plan[f"{field}_sha256"]:
            raise ContractError(f"frozen typed-admission input changed: {field}")
    _source_manifest(Path(plan["source_graph"]))
    _validate_n3_source(Path(plan["n3_run"]))
    return plan


def _distance(left: float, right: float) -> float:
    scale = max(abs(left), abs(right))
    return abs(left - right) / scale if scale else 0.0


def _numeric_occurrences(
    generations: Sequence[str], relation: str,
) -> list[dict[str, Any]]:
    occurrences = []
    for generation_index, generation in enumerate(generations):
        _, items = proposal_parse_status(str(generation), relation)
        seen = set()
        for item in items:
            key = canonical_key(item, relation)
            if not key or key in seen:
                continue
            seen.add(key)
            value = float(key.split(":", 1)[1])
            occurrences.append({
                "generation": generation_index,
                "item": str(item),
                "key": key,
                "value": value,
            })
    return sorted(occurrences, key=lambda item: (
        item["value"], item["generation"], item["key"]))


def _complete_link_numeric_clusters(
    occurrences: Sequence[Mapping[str, Any]],
    tolerance: float = NUMERIC_TOLERANCE,
) -> list[dict[str, Any]]:
    """Deterministic non-transitive numeric clustering.

    Every new member must be within tolerance of every existing member.  This
    prevents the classic 100~104~108 bridge from merging 100 and 108 even
    though their endpoints are not officially equivalent.
    """
    clusters: list[list[Mapping[str, Any]]] = []
    for occurrence in sorted(occurrences, key=lambda item: (
            float(item["value"]), int(item["generation"]), str(item["key"]))):
        compatible = [
            cluster for cluster in clusters
            if all(
                _distance(float(occurrence["value"]), float(member["value"]))
                <= tolerance + 1e-12
                for member in cluster
            )
        ]
        if compatible:
            # Prefer the nearest current median, then the earliest cluster.
            cluster = min(compatible, key=lambda value: abs(
                float(occurrence["value"])
                - statistics.median(float(item["value"]) for item in value)))
            cluster.append(occurrence)
        else:
            clusters.append([occurrence])

    result = []
    for members in clusters:
        generations = sorted({int(item["generation"]) for item in members})
        values = [float(item["value"]) for item in members]
        median = statistics.median(values)
        representative = min(members, key=lambda item: (
            abs(float(item["value"]) - median),
            len(str(item["item"])),
            str(item["item"]),
        ))
        result.append({
            "item": str(representative["item"]),
            "key": canonical_key(str(representative["item"]), "hasArea"),
            "value": float(representative["value"]),
            "generation_support": len(generations),
            "generations": generations,
            "member_items": [str(item["item"]) for item in members],
            "member_values": values,
        })
    return result


def _proposal_response(
    response_map: Mapping[str, dict], subject: str, relation: str,
) -> dict[str, Any]:
    matches = [
        row for row in response_map.values()
        if row.get("phase") == "propose"
        and row.get("subject") == subject
        and row.get("relation") == relation
    ]
    if len(matches) != 1:
        raise ContractError(
            f"{(subject, relation)}: expected one proposal response")
    generations = matches[0].get("generations", [])
    if len(generations) != N_PROPOSALS:
        raise ContractError(
            f"{(subject, relation)}: expected three proposal generations")
    return matches[0]


def _numeric_source_match(
    source_nodes: Mapping[str, Mapping[str, Any]],
    value: float,
) -> dict[str, Any] | None:
    matches = []
    for node in source_nodes.values():
        key = canonical_key(str(node["item"]), "hasArea")
        if not key:
            continue
        source_value = float(key.split(":", 1)[1])
        distance = _distance(value, source_value)
        if distance <= NUMERIC_TOLERANCE + 1e-12:
            matches.append((distance, -len(node.get("sources", {})), node))
    return min(matches, key=lambda item: (item[0], item[1]))[2] if matches else None


def _typed_new_node(
    item: str, relation: str, support: int, reason: str,
) -> dict[str, Any]:
    candidate = {
        "item": item,
        "proposal_support": {MINISTRAL: support},
    }
    node = _new_node(candidate, relation)
    node["routes"][ROUTE]["admission_reason"] = reason
    return node


def _ministral_agent(third: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the already-parsed Ministral decision commitments."""
    commitment = third.get("commitments", {}).get(MINISTRAL, {})
    return {
        "candidate_supply_only": False,
        "n_samples": N_PROPOSALS,
        "typed_component_admission": True,
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
        "decoder_commitments_enabled": bool(commitment),
    }


def _dormant_node(
    item: str,
    relation: str,
    support: int,
    reason: str,
    *,
    member_items: Sequence[str] = (),
    member_values: Sequence[float] = (),
) -> dict[str, Any]:
    """Represent queued evidence in-row without making it output-eligible."""
    key = canonical_key(item, relation)
    route = {
        "model_family": MINISTRAL,
        "route_type": "independent-direct-recall",
        "support": int(support),
        "samples": N_PROPOSALS,
        "support_rate": support / N_PROPOSALS,
        "selected": False,
        "admission_reason": reason,
    }
    if member_items:
        route["cluster_members"] = [str(value) for value in member_items]
    if member_values:
        route["cluster_member_values"] = [
            float(value) for value in member_values]
    return {
        "item": str(item),
        "key": key,
        "type": "numeric" if relation in NUMERIC_RELATIONS else "string",
        "sources": {},
        "selected_by": {MINISTRAL: False},
        "routes": {ROUTE: route},
        "output_eligible": False,
        "dormant": True,
        "dormant_reason": reason,
    }


def _merge_typed_row(
    base: Mapping[str, Any],
    third: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if _key(base) != _key(third):
        raise ContractError("typed admission row mismatch")
    # Repair historical route flags before any new evidence is attached.
    # Selection is recomputed from agent output surfaces with current typed
    # canonicalization, not inherited from split-specific legacy keys.
    row = normalize_route_selection(base)
    subject, relation = _key(row)
    source_nodes = _source_index(row)
    queue = []
    dormant = list(row.get("dormant_candidates", []))
    admitted_new = 0
    corroborated = 0
    typed_groups = 0

    if relation in NUMERIC_RELATIONS:
        groups = _complete_link_numeric_clusters(
            _numeric_occurrences(proposal["generations"], relation))
        typed_groups = len(groups)
        for group in groups:
            match = _numeric_source_match(source_nodes, float(group["value"]))
            support = int(group["generation_support"])
            if match is not None:
                _attach_ministral(
                    match,
                    support=support,
                    samples=N_PROPOSALS,
                    selected=support >= ADMISSION_SUPPORT,
                    reason="numeric_component_corroborates_source",
                )
                match["routes"][ROUTE]["cluster_members"] = group["member_items"]
                corroborated += 1
            elif support >= ADMISSION_SUPPORT:
                node = _typed_new_node(
                    str(group["item"]),
                    relation,
                    support,
                    "numeric_complete_link_self_consistent_new",
                )
                node["routes"][ROUTE]["cluster_members"] = group["member_items"]
                row["candidates"].append(node)
                source_nodes[canonical_key(str(node["item"]), relation)] = node
                admitted_new += 1
            else:
                detail = {
                    "SubjectEntity": subject,
                    "Relation": relation,
                    "candidate": str(group["item"]),
                    "candidate_key": canonical_key(
                        str(group["item"]), relation),
                    "reason": "isolated_numeric_candidate",
                    "generation_support": support,
                    "member_items": group["member_items"],
                    "member_values": group["member_values"],
                }
                queue.append(detail)
                dormant.append(_dormant_node(
                    str(group["item"]),
                    relation,
                    support,
                    str(detail["reason"]),
                    member_items=group["member_items"],
                    member_values=group["member_values"],
                ))
    else:
        for candidate in third.get("candidates", []):
            item = str(candidate["item"])
            key = canonical_key(item, relation)
            support = int(candidate["proposal_support"][MINISTRAL])
            if not key:
                continue
            if key in source_nodes:
                _attach_ministral(
                    source_nodes[key],
                    support=support,
                    samples=N_PROPOSALS,
                    selected=support >= ADMISSION_SUPPORT,
                    reason="exact_string_corroborates_source",
                )
                corroborated += 1
            elif relation in OPEN_LIST_RELATIONS and support >= ADMISSION_SUPPORT:
                node = _typed_new_node(
                    item, relation, support,
                    "open_list_exact_self_consistent_new")
                row["candidates"].append(node)
                source_nodes[key] = node
                admitted_new += 1
            else:
                detail = {
                    "SubjectEntity": subject,
                    "Relation": relation,
                    "candidate": item,
                    "candidate_key": key,
                    "reason": (
                        "novel_string_requires_verification"
                        if support >= ADMISSION_SUPPORT
                        else "isolated_string_candidate"
                    ),
                    "generation_support": support,
                }
                queue.append(detail)
                dormant.append(_dormant_node(
                    item, relation, support, str(detail["reason"])))

    row.setdefault("agents", {})[MINISTRAL] = _ministral_agent(third)
    row.setdefault("agent_outputs", {})[MINISTRAL] = [
        str(node["item"])
        for node in third.get("candidates", [])
        if int(node["proposal_support"][MINISTRAL]) >= ADMISSION_SUPPORT
    ]
    row["dormant_candidates"] = dormant
    row.setdefault("proposal_routes", {})[ROUTE] = {
        "available": True,
        "model_family": MINISTRAL,
        "n_samples": N_PROPOSALS,
        "admission_policy": "typed-component-v1",
    }
    row.setdefault("production_match", {})[
        "ministral_candidate_admission"
    ] = "typed-component-v1"
    row["candidates"].sort(key=lambda node: (
        -len(node.get("sources", {})),
        -sum(
            float(value.get("support_rate", 0.0))
            for value in node.get("sources", {}).values()
        ),
        canonical_key(str(node["item"]), relation),
    ))
    row.pop("relational_graph", None)
    row.pop("relational_graph_schema", None)
    row = augment_relational_graph(row)
    return row, queue, {
        "SubjectEntity": subject,
        "Relation": relation,
        "typed_groups": typed_groups,
        "admitted_new_candidates": admitted_new,
        "corroborated_source_candidates": corroborated,
        "verification_queue_candidates": len(queue),
    }


def admit(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    config = load_agent_config(Path(plan["agents"]))
    agent = _agent(config, MINISTRAL)
    rows = read_jsonl(Path(plan["input_rows"]))
    responses = load_responses(Path(plan["n3_response"]).parent, [agent])
    third_graphs = assemble_graphs(rows, [agent], responses)
    base_by = {
        _key(row): row for row in read_jsonl(Path(plan["source_graph"]))}
    third_by = {_key(row): row for row in third_graphs}
    expected = {_key(row) for row in rows}
    if set(base_by) != expected or set(third_by) != expected:
        raise ContractError("typed admission graph coverage mismatch")

    graph_rows, queue_rows, audit_rows = [], [], []
    response_map = responses[MINISTRAL]
    for source in rows:
        key = _key(source)
        proposal = _proposal_response(response_map, *key)
        graph, queue, audit = _merge_typed_row(
            base_by[key], third_by[key], proposal)
        graph_rows.append(graph)
        queue_rows.extend(queue)
        audit_rows.append(audit)

    graph_path = Path(plan["typed_graph"])
    queue_path = Path(plan["verification_queue"])
    audit_path = graph_path.parent / "ADMISSION_AUDIT.jsonl"
    write_jsonl_atomic(graph_path, graph_rows)
    write_jsonl_atomic(queue_path, queue_rows)
    write_jsonl_atomic(audit_path, audit_rows)
    common = {
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "split": "train",
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "source_graph": plan["source_graph"],
        "source_graph_sha256": plan["source_graph_sha256"],
        "n3_response": plan["n3_response"],
        "n3_response_sha256": plan["n3_response_sha256"],
        "policy": plan["typed_policy"],
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
    graph_manifest = {
        **common,
        "schema": GRAPH_SCHEMA,
        "rows": len(graph_rows),
        "output": str(graph_path),
        "output_sha256": sha256(graph_path),
        "verification_queue": str(queue_path),
        "verification_queue_sha256": sha256(queue_path),
        "admission_audit": str(audit_path),
        "admission_audit_sha256": sha256(audit_path),
        "relational_graph_rebuilt": True,
        "dormant_candidates": sum(
            len(row.get("dormant_candidates", [])) for row in graph_rows),
    }
    queue_manifest = {
        **common,
        "schema": QUEUE_SCHEMA,
        "rows": len(queue_rows),
        "output": str(queue_path),
        "output_sha256": sha256(queue_path),
        "directly_consumable_as_predictions": False,
    }
    _write_json(
        graph_path.with_suffix(graph_path.suffix + ".manifest.json"),
        graph_manifest)
    _write_json(
        queue_path.with_suffix(queue_path.suffix + ".manifest.json"),
        queue_manifest)
    print(json.dumps({
        "typed_graph": str(graph_path),
        "rows": len(graph_rows),
        "admitted_new_candidates": sum(
            row["admitted_new_candidates"] for row in audit_rows),
        "corroborated_source_candidates": sum(
            row["corroborated_source_candidates"] for row in audit_rows),
        "verification_queue_candidates": len(queue_rows),
    }, indent=2, sort_keys=True))
    return 0


def _validate_output_manifest(path: Path, schema: str) -> dict[str, Any]:
    manifest = _manifest(path)
    if (
        manifest.get("schema") != schema
        or manifest.get("contains_labels") is not False
        or manifest.get("gold_aware") is not False
        or manifest.get("validation_opened") is not False
        or manifest.get("validation_labels_used") is not False
    ):
        raise ContractError(f"{path}: output manifest is not label-free")
    return manifest


def _unique_rows(
    rows: Sequence[Mapping[str, Any]],
    base_by: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate_by: Mapping[tuple[str, str], Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[list[str]], Counter[str]]:
    unique, relations = [], Counter()
    for source in rows:
        key = _key(source)
        relation = key[1]
        if (
            _has_correct(_items(candidate_by[key]), gold_by[key], relation)
            and not _has_correct(_items(base_by[key]), gold_by[key], relation)
        ):
            unique.append([key[0], relation])
            relations[relation] += 1
    return unique, relations


def _new_candidate_truth(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> tuple[int, int]:
    relation = str(base["Relation"])
    base_keys = {
        canonical_key(item, relation) for item in _items(base)
        if canonical_key(item, relation)
    }
    total = correct = 0
    for item in _items(candidate):
        key = canonical_key(item, relation)
        if not key or key in base_keys:
            continue
        total += 1
        correct += int(_has_correct([item], gold, relation))
    return total, correct


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _validate_plan(output)
    typed_path = Path(plan["typed_graph"])
    queue_path = Path(plan["verification_queue"])
    _validate_output_manifest(typed_path, GRAPH_SCHEMA)
    _validate_output_manifest(queue_path, QUEUE_SCHEMA)
    rows = read_jsonl(Path(plan["input_rows"]))
    base_graphs = read_jsonl(Path(plan["source_graph"]))
    exact_graphs = read_jsonl(Path(plan["legacy_exact_graph"]))
    typed_graphs = read_jsonl(typed_path)
    config = load_agent_config(Path(plan["agents"]))
    agent = _agent(config, MINISTRAL)
    responses = load_responses(Path(plan["n3_response"]).parent, [agent])
    raw_graphs = assemble_graphs(rows, [agent], responses)
    base_by = {_key(row): row for row in base_graphs}
    exact_by = {_key(row): row for row in exact_graphs}
    typed_by = {_key(row): row for row in typed_graphs}
    raw_by = {_key(row): row for row in raw_graphs}
    expected = {_key(row) for row in rows}
    if any(set(value) != expected for value in (
            base_by, exact_by, typed_by, raw_by)):
        raise ContractError("typed analysis graph coverage mismatch")

    gold_path = Path(args.train_gold).resolve()
    gold_by = {_key(row): row for row in read_jsonl(gold_path)}
    if set(gold_by) != expected:
        raise ContractError("typed analysis gold coverage mismatch")
    ordered_gold = [gold_by[_key(row)] for row in rows]
    ordered_base = [base_by[_key(row)] for row in rows]
    combined_raw = [
        _combined_raw(base_by[_key(row)], raw_by[_key(row)])
        for row in rows
    ]
    scores = {
        "base": score(oracle_rows(ordered_base, ordered_gold), ordered_gold),
        "raw_n3": score(oracle_rows(combined_raw, ordered_gold), ordered_gold),
        "exact": score(oracle_rows(exact_graphs, ordered_gold), ordered_gold),
        "typed": score(oracle_rows(typed_graphs, ordered_gold), ordered_gold),
    }
    raw_unique, raw_relations = _unique_rows(
        rows, base_by, raw_by, gold_by)
    exact_unique, exact_relations = _unique_rows(
        rows, base_by, exact_by, gold_by)
    typed_unique, typed_relations = _unique_rows(
        rows, base_by, typed_by, gold_by)

    raw_total = raw_correct = typed_total = typed_correct = 0
    for source in rows:
        key = _key(source)
        total, correct = _new_candidate_truth(
            base_by[key], raw_by[key], gold_by[key])
        raw_total += total
        raw_correct += correct
        total, correct = _new_candidate_truth(
            base_by[key], typed_by[key], gold_by[key])
        typed_total += total
        typed_correct += correct
    raw_precision = raw_correct / raw_total if raw_total else 0.0
    typed_precision = typed_correct / typed_total if typed_total else 0.0
    precision_multiplier = (
        typed_precision / raw_precision if raw_precision else 0.0)
    retention = len(typed_unique) / len(raw_unique) if raw_unique else 0.0
    pooled = "*** All Relations ***"
    delta_base = scores["typed"][pooled] - scores["base"][pooled]
    gain_exact = scores["typed"][pooled] - scores["exact"][pooled]
    gate_checks = {
        "oracle_delta_over_base":
            delta_base >= MIN_ORACLE_DELTA_OVER_BASE,
        "oracle_gain_over_exact":
            gain_exact >= MIN_ORACLE_GAIN_OVER_EXACT,
        "unique_rows": len(typed_unique) >= MIN_UNIQUE_ROWS,
        "unique_relations": len(typed_relations) >= MIN_UNIQUE_RELATIONS,
        "raw_unique_retention": retention >= MIN_RAW_UNIQUE_RETENTION,
        "precision_multiplier_over_raw":
            precision_multiplier >= MIN_PRECISION_MULTIPLIER_OVER_RAW,
    }

    queue_rows = read_jsonl(queue_path)
    queue_correct = 0
    queue_correct_rows = set()
    for item in queue_rows:
        key = _key(item)
        if _has_correct(
            [str(item["candidate"])], gold_by[key], key[1]):
            queue_correct += 1
            queue_correct_rows.add(key)
    result = {
        "schema": RESULT_SCHEMA,
        "development_only": True,
        "deployable": False,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "scores": scores,
        "typed_oracle_delta_over_base": delta_base,
        "typed_oracle_gain_over_exact": gain_exact,
        "raw_unique_correct_rows": raw_unique,
        "raw_unique_correct_by_relation": dict(raw_relations),
        "exact_unique_correct_rows": exact_unique,
        "exact_unique_correct_by_relation": dict(exact_relations),
        "typed_unique_correct_rows": typed_unique,
        "typed_unique_correct_by_relation": dict(typed_relations),
        "typed_raw_unique_retention": retention,
        "candidate_precision": {
            "raw_new_total": raw_total,
            "raw_new_correct": raw_correct,
            "raw_new_precision": raw_precision,
            "typed_new_total": typed_total,
            "typed_new_correct": typed_correct,
            "typed_new_precision": typed_precision,
            "typed_precision_multiplier_over_raw": precision_multiplier,
        },
        "verification_queue": {
            "candidates": len(queue_rows),
            "correct_candidates": queue_correct,
            "rows_with_correct_candidate": len(queue_correct_rows),
            "correct_rows_by_relation": dict(Counter(
                relation for _, relation in queue_correct_rows)),
        },
        "promotion_gate_checks": gate_checks,
        "promotion_gate_passed": all(gate_checks.values()),
        "next_stage": (
            "freeze_and_validate_typed_admission"
            if all(gate_checks.values())
            else "retain_n3_supply_and_improve_verification"
        ),
        "typed_graph": str(typed_path),
        "typed_graph_sha256": sha256(typed_path),
        "queue": str(queue_path),
        "queue_sha256": sha256(queue_path),
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
    }
    analysis = output / "analysis"
    _write_json(analysis / "RESULT.json", result)
    lines = [
        "# Ministral typed-component admission audit",
        "",
        "Development-only train audit. Validation was not opened.",
        "",
        f"- Base two-model oracle: **{scores['base'][pooled]:.6f}**",
        f"- Raw Ministral N=3 oracle: **{scores['raw_n3'][pooled]:.6f}**",
        f"- Legacy exact-admission oracle: **{scores['exact'][pooled]:.6f}**",
        f"- Typed-component admission oracle: "
        f"**{scores['typed'][pooled]:.6f}**",
        f"- Typed delta over base: **{delta_base:+.6f}**",
        f"- Typed gain over exact admission: **{gain_exact:+.6f}**",
        f"- Raw/exact/typed unique rows: "
        f"**{len(raw_unique)} / {len(exact_unique)} / "
        f"{len(typed_unique)}**",
        f"- Typed unique relation coverage: **{len(typed_relations)}/6**",
        f"- Typed retention of raw unique rows: **{retention:.2%}**",
        f"- Raw/typed new-candidate precision: "
        f"**{raw_precision:.2%} / {typed_precision:.2%}**",
        f"- Verification queue: **{len(queue_rows)} candidates; "
        f"{queue_correct} correct across {len(queue_correct_rows)} rows**",
        f"- Promotion gate: "
        f"**{'PASS' if all(gate_checks.values()) else 'FAIL'}**",
        "",
        "| relation | base | raw N=3 | exact admit | typed admit |",
        "|---|---:|---:|---:|---:|",
    ]
    for relation in sorted(scores["base"]):
        lines.append(
            f"| {relation} | {scores['base'][relation]:.4f} | "
            f"{scores['raw_n3'][relation]:.4f} | "
            f"{scores['exact'][relation]:.4f} | "
            f"{scores['typed'][relation]:.4f} |")
    lines.extend(["", "## Promotion checks", ""])
    for name, passed in gate_checks.items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend([
        "",
        "All oracle and precision measurements are gold-aware and "
        "nondeployable. The typed graph and verification queue were built "
        "before labels were opened and have fail-closed label-free manifests.",
        "",
    ])
    (analysis / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps({
        "base_oracle": scores["base"][pooled],
        "raw_n3_oracle": scores["raw_n3"][pooled],
        "exact_oracle": scores["exact"][pooled],
        "typed_oracle": scores["typed"][pooled],
        "typed_delta_over_base": delta_base,
        "typed_gain_over_exact": gain_exact,
        "typed_unique_rows": len(typed_unique),
        "typed_unique_relations": len(typed_relations),
        "promotion_gate_passed": all(gate_checks.values()),
        "result": str(analysis / "RESULT.md"),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    prepare_parser.add_argument("--agents", default=str(DEFAULT_CONFIG))
    prepare_parser.add_argument(
        "--source-graph", default=str(DEFAULT_SOURCE_GRAPH))
    prepare_parser.add_argument("--n3-run", default=str(DEFAULT_N3_RUN))
    prepare_parser.add_argument("--exact-graph", default=str(DEFAULT_EXACT_GRAPH))
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
