#!/usr/bin/env python3
"""One graph-native decoder for the full heterogeneous system.

The deployed 0.518450 prediction file is produced by a chain of separate
programs: a component/surface decoder over a Qwen+Gemma graph, then two
post-processing scripts that re-open that output and edit rows using
Ministral evidence held *outside* the graph.  The architecture claim of the
system ("a decoder reasons over a typed evidence graph carrying all three
model families") is therefore not literally true of the deployed code.

This module closes that gap without changing any decision:

``build``
    Materialises ONE unified typed graph in which every model family is a
    first-class evidence route: ``qwen:self_consistency``, ``qwen:system2``,
    ``gemma:independent``, ``ministral:self_consistency`` (N=3 zero-shot,
    carrying typed admission reasons) and ``ministral:cot5_cap40_n10``
    (N=10 synthetic-CoT, carrying per-generation provenance).  The certified
    chain incumbent and the frozen capacity-veto policy ledger are attached
    as typed nodes, so nothing the decoder consults lives outside the graph.

``decode``
    Runs ONE pass over that graph and applies every frozen policy in order.
    Each policy declares the evidence routes it consumes; a policy fitted on
    Qwen/Gemma features is served a route-scoped projection of the graph so
    that consolidation cannot silently change its decisions.  Per-row
    provenance for every layer is written to ``DECISIONS.jsonl``.

``verify``
    Parity gate.  The consolidated decoder must reproduce the deployed
    prediction artifact exactly (identical sha256 and identical per-row
    objects).  Any divergence is reported row-by-row rather than absorbed.

The module never opens validation labels to make a decision; ``verify``
scores the two artifacts with the official scorer purely to report that the
consolidation is score-neutral.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.heterogeneous_agents.baseline_relative_route_decoder import (
    ResidualRidge,
)
from experiments.heterogeneous_agents.component_aware_decoder import (
    decode as component_decode,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    answer_field_status,
    canonical_key,
    normalize_string,
    proposal_parse_status,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.heterogeneous_memory_selector import _key


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CANONICAL = ROOT / "results/heterogeneous/canonical_runtime/sota_pipeline_20260726"

DEFAULT_OUTPUT = RUNS / "unified_graph_decoder_20260730_v1"

# ---------------------------------------------------------------- inputs ---
# Typed graph already carrying qwen/gemma routes plus the Ministral N=3
# self-consistency route with its typed admission reasons.
TYPED_GRAPH = (
    RUNS / "ministral_typed_validation_confirmation_20260729_v3"
    / "graph/TYPED_VALIDATION_GRAPH.jsonl")
# Raw CoT40 generations; the fifth route is materialised from these.
COT40_RESPONSES = (
    RUNS / "ministral_cot40_validation_confirmation_20260729_v1"
    / "responses/ministral_cot5_cap40_n10.jsonl")
# The certified chain incumbent the frozen decoders were applied to.
CHAIN_INCUMBENT = (
    RUNS / "route_residual_decoder_20260723_v1/validation_train_selected.jsonl")
# Frozen learned policy: component/surface residual ridge.
COMPONENT_MODELS = CANONICAL / "component/models.json"
# Frozen policy ledger: Qwen precision-vetoed capacity switches.
CAPACITY_LEDGER = (
    RUNS / "july_legacy_capacity_veto_20260726_v1/VALIDATION_DECISIONS.jsonl")
# The artifact this decoder must reproduce.
DEPLOYED_PREDICTIONS = (
    RUNS / "ministral_cot40_validation_confirmation_20260729_v1"
    / "VALIDATION_PREDICTIONS.jsonl")
# Support counts the deployed residual layer actually used, for parity forensics.
DEPLOYED_DECISIONS = (
    RUNS / "ministral_cot40_validation_confirmation_20260729_v1"
    / "DECISIONS.jsonl")
DEFAULT_GOLD = ROOT / "data/val.jsonl"

# ---------------------------------------------------------------- routes ---
QWEN_SC = "qwen:self_consistency"
QWEN_S2 = "qwen:system2"
GEMMA = "gemma:independent"
MINISTRAL_N3 = "ministral:self_consistency"
MINISTRAL_COT40 = "ministral:cot5_cap40_n10"

CHAIN_ROUTES = frozenset({QWEN_SC, QWEN_S2, GEMMA})
ALL_ROUTES = frozenset(
    {QWEN_SC, QWEN_S2, GEMMA, MINISTRAL_N3, MINISTRAL_COT40})

NUMERIC_RELATIONS = ("hasArea", "hasCapacity")
ROWS = 478

# ------------------------------------------------------------- constants ---
COT40_SAMPLES = 10
COT40_SUPPORT_REQUIRED = 7
COT40_POLICY = "two_thirds"
PARSER_MODES = ("legacy-20260729", "corrected")
AREA_POLICY = "area_unanimous_new_component_replace"
AREA_PROPOSALS = 3
AREA_ADMISSION_REASON = "numeric_complete_link_self_consistent_new"

GRAPH_SCHEMA = "unified-heterogeneous-graph-v1"
GRAPH_MANIFEST_SCHEMA = "unified-heterogeneous-graph-manifest-v1"
PLAN_SCHEMA = "unified-graph-decoder-plan-v1"
DECISION_SCHEMA = "unified-graph-decoder-decisions-v1"
PARITY_SCHEMA = "unified-graph-decoder-parity-v1"

# Ordered policy stack.  ``routes`` is the evidence scope each policy is
# permitted to consult; a frozen model may only see the routes it was fitted
# against, otherwise consolidation would silently alter its decisions.
POLICIES = (
    {
        "id": "component_surface_residual_ridge",
        "kind": "frozen_learned_model",
        "routes": sorted(CHAIN_ROUTES),
        "relations": ("awardWonBy", "companyTradesAtStockExchange"),
    },
    {
        "id": "capacity_qwen_precision_veto",
        "kind": "frozen_policy_ledger",
        "routes": sorted(CHAIN_ROUTES),
        "relations": ("hasCapacity",),
    },
    {
        "id": AREA_POLICY,
        "kind": "graph_rule",
        "routes": [MINISTRAL_N3],
        "relations": ("hasArea",),
    },
    {
        "id": f"cot40_{COT40_POLICY}_support_{COT40_SUPPORT_REQUIRED}",
        "kind": "graph_rule",
        "routes": [MINISTRAL_COT40],
        "relations": (),  # all relations
    },
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _rows_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[
    tuple[str, str], Mapping[str, Any]
]:
    out = {_key(row): row for row in rows}
    if len(out) != len(rows):
        raise ContractError("duplicate (subject, relation) key")
    return out


# ------------------------------------------------------------------ build --
def cot40_route_evidence(
    generations: Sequence[str], relation: str, *,
    parser_mode: str = "corrected",
) -> tuple[dict[str, dict[str, Any]], list[int]]:
    """Per-candidate distinct-generation support for the CoT40 route.

    Mirrors the frozen residual policy's evidence construction exactly: a
    candidate is supported once per generation that mentions it, regardless
    of how many times it appears inside that generation.
    """
    if parser_mode not in PARSER_MODES:
        raise ContractError(f"unsupported CoT40 parser mode: {parser_mode}")
    values: dict[str, dict[str, Any]] = {}
    none_generations: list[int] = []
    for index, generation in enumerate(generations):
        if parser_mode == "legacy-20260729":
            status, items = legacy_proposal_parse_status(
                str(generation), relation)
        else:
            status, items = proposal_parse_status(str(generation), relation)
        if status == "explicit_none":
            none_generations.append(index)
        seen: set[str] = set()
        for item in items:
            canonical = canonical_key(str(item), relation)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            current = values.setdefault(canonical, {
                "key": canonical,
                "item": str(item),
                "generation_indices": [],
            })
            current["generation_indices"].append(index)
    return values, none_generations


def legacy_proposal_parse_status(
    text: str, relation: str,
) -> tuple[str, list[str]]:
    """Parse proposals exactly as the frozen 2026-07-29 SOTA run did.

    The historical parser did not remove a model-echoed second ``ANSWER:``
    prefix.  That behavior is incorrect, but it is part of the byte-level
    development artifact.  Keeping it behind an explicit compatibility mode
    lets the reproduction runner prove exact parity while the default
    ``corrected`` mode continues to exercise the repaired parser.
    """
    status, answer = answer_field_status(text)
    if answer is None:
        return status, []
    if normalize_string(answer) in {"none", "null", "no answer"}:
        return "explicit_none", []
    # Lazy import preserves the CUDA worker-isolation contract in core.py.
    from run_inference import parse_answer_items
    items = parse_answer_items(
        answer, relation, response_protocol="legacy-cot")
    if not items:
        return "unparseable_answer_field", []
    return "parsed_nonempty", [str(item) for item in items]


def _attach_cot40_route(
    row: dict[str, Any],
    response: Mapping[str, Any],
    *,
    parser_mode: str,
) -> dict[str, Any]:
    """Insert the CoT40 generations as a first-class typed evidence route."""
    relation = str(row["Relation"])
    generations = list(response["generations"])
    if len(generations) != COT40_SAMPLES:
        raise ContractError(
            f"{_key(row)}: expected {COT40_SAMPLES} CoT40 generations")
    occurrences, none_generations = cot40_route_evidence(
        generations, relation, parser_mode=parser_mode)

    candidates = list(row.get("candidates", []))
    by_canonical: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        canonical = canonical_key(str(candidate["item"]), relation)
        if canonical and canonical not in by_canonical:
            by_canonical[canonical] = candidate

    for canonical, value in sorted(occurrences.items()):
        candidate = by_canonical.get(canonical)
        if candidate is None:
            candidate = {
                "key": canonical,
                "item": str(value["item"]),
                "type": (
                    "numeric" if relation in NUMERIC_RELATIONS else "string"),
                "routes": {},
                "sources": {},
                "selected_by": {},
                "output_eligible": True,
                "admitted_by": MINISTRAL_COT40,
            }
            candidates.append(candidate)
            by_canonical[canonical] = candidate
        support = sorted({int(i) for i in value["generation_indices"]})
        candidate.setdefault("routes", {})[MINISTRAL_COT40] = {
            "model_family": "ministral_independent",
            "route_type": "independent-synthetic-cot-recall",
            "support": len(support),
            "samples": COT40_SAMPLES,
            "support_rate": len(support) / COT40_SAMPLES,
            "selected": len(support) >= COT40_SUPPORT_REQUIRED,
            "generation_indices": support,
            "display_item": str(value["item"]),
        }
    row["candidates"] = candidates
    row.setdefault("proposal_routes", {})[MINISTRAL_COT40] = {
        "available": True,
        "model_family": "ministral_independent",
        "n_samples": COT40_SAMPLES,
        "route_type": "independent-synthetic-cot-recall",
        "generation_provenance_available": True,
        "none_generation_indices": none_generations,
        "parsed_candidate_count": len(occurrences),
        "parser_mode": parser_mode,
    }
    return row


def build(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    typed_path = Path(args.typed_graph).resolve()
    cot40_path = Path(args.cot40_responses).resolve()
    incumbent_path = Path(args.chain_incumbent).resolve()
    ledger_path = Path(args.capacity_ledger).resolve()
    parser_mode = str(getattr(args, "parser_mode", "corrected"))
    if parser_mode not in PARSER_MODES:
        raise ContractError(f"unsupported CoT40 parser mode: {parser_mode}")

    graphs = read_jsonl(typed_path)
    if len(graphs) != ROWS:
        raise ContractError(f"typed graph must contain {ROWS} validation rows")
    graph_by = _rows_by_key(graphs)

    responses = {}
    for response in read_jsonl(cot40_path):
        key = (str(response["subject"]), str(response["relation"]))
        if key in responses:
            raise ContractError(f"duplicate CoT40 response for {key}")
        responses[key] = response
    if set(responses) != set(graph_by):
        raise ContractError("CoT40 responses do not cover the typed graph")

    incumbents = _rows_by_key(read_jsonl(incumbent_path))
    if set(incumbents) != set(graph_by):
        raise ContractError("chain incumbent does not cover the typed graph")

    ledger = {}
    for decision in read_jsonl(ledger_path):
        ledger[_key(decision)] = decision

    unified = []
    admitted_by_cot40 = 0
    # Preserve the certified source order.  Sorting by subject was harmless
    # for scoring, but made byte-level reproduction impossible even when all
    # 478 prediction rows were semantically identical to the deployed file.
    for source_row in graphs:
        key = _key(source_row)
        row = copy.deepcopy(dict(graph_by[key]))
        before = len(row.get("candidates", []))
        row = _attach_cot40_route(
            row, responses[key], parser_mode=parser_mode)
        admitted_by_cot40 += len(row["candidates"]) - before

        objects = [str(item) for item in incumbents[key].get(
            "ObjectEntities", [])]
        row["incumbent"] = {
            "node_type": "incumbent_hypothesis",
            "objects": objects,
            "provenance": {
                # Logical identifiers keep graph bytes independent of the
                # clone/output directory.  The hash is the identity check.
                "artifact": "chain_incumbent",
                "sha256": sha256(incumbent_path),
                "description":
                    "certified heterogeneous-chain incumbent (pre-decoder)",
            },
        }
        decision = ledger.get(key)
        if decision is not None:
            row["frozen_policy_ledger"] = {
                "policy": "capacity_qwen_precision_veto",
                "artifact": "capacity_ledger",
                "sha256": sha256(ledger_path),
                "action_id": decision.get("action_id"),
                "proposal": [
                    str(item) for item in decision.get("proposal", []) or []],
                "incumbent": [
                    str(item) for item in decision.get("incumbent", []) or []],
                "estimated_improvement": decision.get("estimated_improvement"),
                "margin": decision.get("margin"),
                "qwen_precision_vetoed": decision.get("qwen_precision_vetoed"),
                "switched": decision.get("switched"),
            }
        row["schema"] = GRAPH_SCHEMA
        row["evidence_routes"] = sorted(ALL_ROUTES)
        row["contains_labels"] = False
        row["gold_aware"] = False
        unified.append(row)

    graph_path = output / "graph/UNIFIED_VALIDATION_GRAPH.jsonl"
    write_jsonl_atomic(graph_path, unified)

    route_counts: Counter[str] = Counter()
    for row in unified:
        for candidate in row.get("candidates", []):
            for route in candidate.get("routes", {}):
                route_counts[route] += 1
    missing = ALL_ROUTES - set(route_counts)
    if missing:
        raise ContractError(f"unified graph is missing routes: {sorted(missing)}")

    _write_json(graph_path.with_suffix(graph_path.suffix + ".manifest.json"), {
        "schema": GRAPH_MANIFEST_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "split": "validation",
        "rows": len(unified),
        "output": str(graph_path),
        "output_sha256": sha256(graph_path),
        "typed_graph": str(typed_path),
        "typed_graph_sha256": sha256(typed_path),
        "cot40_responses": str(cot40_path),
        "cot40_responses_sha256": sha256(cot40_path),
        "chain_incumbent": str(incumbent_path),
        "chain_incumbent_sha256": sha256(incumbent_path),
        "capacity_ledger": str(ledger_path),
        "capacity_ledger_sha256": sha256(ledger_path),
        "evidence_routes": sorted(ALL_ROUTES),
        "route_candidate_counts": dict(sorted(route_counts.items())),
        "candidates_admitted_by_cot40": admitted_by_cot40,
        "cot40_parser_mode": parser_mode,
    })

    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "split": "validation",
        "graph": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "component_models": str(Path(args.component_models).resolve()),
        "component_models_sha256": sha256(Path(args.component_models)),
        "deployed_predictions": str(Path(args.deployed).resolve()),
        "deployed_predictions_sha256": sha256(Path(args.deployed)),
        "policies": [dict(policy) for policy in POLICIES],
        "cot40_samples": COT40_SAMPLES,
        "cot40_support_required": COT40_SUPPORT_REQUIRED,
        "cot40_parser_mode": parser_mode,
        "area_proposals": AREA_PROPOSALS,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "parity_target": "byte_identical_to_deployed_predictions",
    }
    plan_path = output / "plan/PLAN.json"
    _write_json(plan_path, plan)
    print(json.dumps({
        "graph": str(graph_path),
        "graph_sha256": plan["graph_sha256"],
        "rows": len(unified),
        "evidence_routes": sorted(ALL_ROUTES),
        "route_candidate_counts": dict(sorted(route_counts.items())),
        "candidates_admitted_by_cot40": admitted_by_cot40,
        "cot40_parser_mode": parser_mode,
        "plan": str(plan_path),
    }, indent=2, sort_keys=True))
    return 0


# ----------------------------------------------------------------- decode --
def scope_graph(
    graph: Mapping[str, Any], routes: frozenset[str],
) -> dict[str, Any]:
    """Project the graph onto one policy's declared evidence routes.

    A candidate surviving the projection must retain at least one in-scope
    route; a candidate admitted only by an out-of-scope model family is not
    visible to that policy.  This is what makes consolidation decision
    preserving: a frozen model keeps seeing exactly the evidence it was fitted
    against, while the graph as a whole carries every model family.
    """
    scoped = copy.deepcopy(dict(graph))

    def retain(node: dict[str, Any]) -> bool:
        node_routes = node.get("routes")
        if not isinstance(node_routes, dict):
            return True
        kept = {
            name: value for name, value in node_routes.items()
            if name in routes
        }
        node["routes"] = kept
        return bool(kept)

    if isinstance(scoped.get("candidates"), list):
        scoped["candidates"] = [
            node for node in scoped["candidates"] if retain(node)]
    relational = scoped.get("relational_graph")
    if isinstance(relational, dict):
        for field in ("components", "nodes"):
            if isinstance(relational.get(field), list):
                relational[field] = [
                    node for node in relational[field] if retain(node)]
    return scoped


def _revive(blob: Mapping[str, Any]) -> ResidualRidge:
    model = ResidualRidge(list(blob["feature_names"]), float(blob["l2"]))
    model.mean = np.asarray(blob["mean"], dtype=np.float64)
    model.scale = np.asarray(blob["scale"], dtype=np.float64)
    model.coefficients = np.asarray(blob["coefficients"], dtype=np.float64)
    if (
        model.mean.shape != (len(model.names),)
        or model.scale.shape != (len(model.names),)
        or model.coefficients.shape != (len(model.names) + 1,)
    ):
        raise ContractError("frozen residual ridge has inconsistent shape")
    return model


def apply_component_residual(
    graph: Mapping[str, Any],
    objects: Sequence[str],
    models: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    relation = str(graph["Relation"])
    arm = models["chosen_arm"].get(relation)
    enabled = models["enabled"]
    if arm is None or not enabled.get(arm, {}).get(relation, False):
        return list(objects), {"applied": False, "reason": "arm_not_enabled"}
    margin = float(models["selected_margins"][arm][relation])
    scoped = scope_graph(graph, CHAIN_ROUTES)
    selected, detail = component_decode(
        _revive(models["models"][arm][relation]), scoped, list(objects),
        arm, margin)
    return list(selected), {
        "applied": not bool(detail["used_control"]),
        "arm": arm,
        "guard_margin": margin,
        "estimated_f1_delta": float(detail["estimated_f1_delta"]),
        "action_count": int(detail["action_count"]),
        "evidence_routes": sorted(CHAIN_ROUTES),
    }


def apply_capacity_veto(
    graph: Mapping[str, Any],
    objects: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    if str(graph["Relation"]) != "hasCapacity":
        return list(objects), {"applied": False, "reason": "relation_out_of_scope"}
    ledger = graph.get("frozen_policy_ledger")
    if not isinstance(ledger, Mapping) or ledger.get("action_id") is None:
        return list(objects), {"applied": False, "reason": "no_switch_recorded"}
    proposal = [str(item) for item in ledger.get("proposal", [])]
    if not proposal or not ledger.get("switched"):
        return list(objects), {"applied": False, "reason": "not_switched"}
    return proposal, {
        "applied": True,
        "action_id": ledger.get("action_id"),
        "estimated_improvement": ledger.get("estimated_improvement"),
        "margin": ledger.get("margin"),
        "qwen_precision_vetoed": ledger.get("qwen_precision_vetoed"),
        "evidence": "frozen_policy_ledger",
    }


def apply_area_unanimity(
    graph: Mapping[str, Any],
    objects: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Replace an area incumbent with a unanimous, typed-new N=3 component."""
    if str(graph["Relation"]) != "hasArea":
        return list(objects), {"applied": False, "reason": "relation_out_of_scope"}
    proposals = []
    for candidate in graph.get("candidates", []):
        route = candidate.get("routes", {}).get(MINISTRAL_N3, {})
        if (
            route.get("admission_reason") == AREA_ADMISSION_REASON
            and int(route.get("support", 0)) == AREA_PROPOSALS
            and int(route.get("samples", 0)) == AREA_PROPOSALS
        ):
            proposals.append(str(candidate["item"]))
    if len(proposals) != 1:
        return list(objects), {
            "applied": False,
            "reason": "no_unique_unanimous_new_component",
            "candidate_count": len(proposals),
            "evidence_routes": [MINISTRAL_N3],
        }
    selected = [proposals[0]]
    return selected, {
        "applied": selected != list(objects),
        "policy": AREA_POLICY,
        "proposal": selected,
        "evidence_routes": [MINISTRAL_N3],
    }


def apply_cot40_support(
    graph: Mapping[str, Any],
    objects: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Frozen two-thirds CoT40 residual, read from typed graph routes."""
    relation = str(graph["Relation"])
    counts: dict[str, int] = {}
    displays: dict[str, str] = {}
    for candidate in graph.get("candidates", []):
        route = candidate.get("routes", {}).get(MINISTRAL_COT40)
        if not isinstance(route, Mapping):
            continue
        canonical = canonical_key(str(candidate["item"]), relation)
        if not canonical:
            continue
        counts[canonical] = int(route["support"])
        displays[canonical] = str(route.get(
            "display_item", candidate["item"]))
    base = [str(item) for item in objects]
    if relation in NUMERIC_RELATIONS:
        if counts:
            highest = max(counts.values())
            winners = [
                canonical for canonical, count in counts.items()
                if count == highest and count >= COT40_SUPPORT_REQUIRED
            ]
            selected = (
                [displays[winners[0]]] if len(winners) == 1 else list(base))
        else:
            selected = list(base)
    else:
        by_canonical: dict[str, str] = {}
        for item in base:
            canonical = canonical_key(item, relation)
            if canonical:
                by_canonical.setdefault(canonical, item)
        for canonical, count in counts.items():
            if count >= COT40_SUPPORT_REQUIRED:
                by_canonical.setdefault(canonical, displays[canonical])
        selected = list(by_canonical.values())
    return selected, {
        "applied": selected != base,
        "policy": COT40_POLICY,
        "support_required": COT40_SUPPORT_REQUIRED,
        "candidate_support": {
            displays[canonical]: count
            for canonical, count in sorted(counts.items())
        },
        "evidence_routes": [MINISTRAL_COT40],
    }


def decode_row(
    graph: Mapping[str, Any],
    models: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Apply the full frozen policy stack in one pass over one graph row."""
    incumbent = graph.get("incumbent")
    if not isinstance(incumbent, Mapping):
        raise ContractError(f"{_key(graph)}: unified graph lacks an incumbent")
    objects = [str(item) for item in incumbent["objects"]]
    trace: list[dict[str, Any]] = []
    stack = (
        ("component_surface_residual_ridge",
         lambda g, o: apply_component_residual(g, o, models)),
        ("capacity_qwen_precision_veto", apply_capacity_veto),
        (AREA_POLICY, apply_area_unanimity),
        (f"cot40_{COT40_POLICY}_support_{COT40_SUPPORT_REQUIRED}",
         apply_cot40_support),
    )
    for policy_id, function in stack:
        before = list(objects)
        objects, detail = function(graph, objects)
        objects = [str(item) for item in objects]
        trace.append({
            "policy": policy_id,
            "before": before,
            "after": list(objects),
            "changed": objects != before,
            **detail,
        })
    return objects, trace


def decode(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = _json(output / "plan/PLAN.json")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ContractError("unified decoder plan schema mismatch")
    graph_path = Path(plan["graph"])
    if sha256(graph_path) != plan["graph_sha256"]:
        raise ContractError("unified graph changed since the plan was frozen")
    models_path = Path(plan["component_models"])
    if sha256(models_path) != plan["component_models_sha256"]:
        raise ContractError("frozen component models changed")
    models = _json(models_path)
    if models.get("schema") != "component-aware-decoder-models-v1":
        raise ContractError("unexpected component model schema")

    graphs = read_jsonl(graph_path)
    if len(graphs) != ROWS:
        raise ContractError(f"unified graph must contain {ROWS} rows")

    predictions, decisions = [], []
    layer_changes: Counter[str] = Counter()
    for graph in graphs:
        objects, trace = decode_row(graph, models)
        key = _key(graph)
        predictions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": objects,
        })
        for step in trace:
            if step["changed"]:
                layer_changes[str(step["policy"])] += 1
        decisions.append({
            "schema": DECISION_SCHEMA,
            "SubjectEntity": key[0],
            "Relation": key[1],
            "incumbent": [
                str(item) for item in graph["incumbent"]["objects"]],
            "prediction": objects,
            "changed": objects != [
                str(item) for item in graph["incumbent"]["objects"]],
            "layers": trace,
        })

    prediction_path = output / "VALIDATION_PREDICTIONS.jsonl"
    decision_path = output / "DECISIONS.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(decision_path, decisions)
    _write_json(
        prediction_path.with_suffix(prediction_path.suffix + ".manifest.json"),
        {
            "schema": "unified-graph-decoder-predictions-v1",
            "contains_labels": False,
            "gold_aware": False,
            "single_pass_graph_native": True,
            "rows": len(predictions),
            "output": str(prediction_path),
            "output_sha256": sha256(prediction_path),
            "graph": str(graph_path),
            "graph_sha256": plan["graph_sha256"],
            "policies": [dict(policy) for policy in POLICIES],
            "layer_changed_rows": dict(sorted(layer_changes.items())),
        })
    print(json.dumps({
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "decisions": str(decision_path),
        "rows": len(predictions),
        "layer_changed_rows": dict(sorted(layer_changes.items())),
    }, indent=2, sort_keys=True))
    return 0


# ----------------------------------------------------------------- verify --
def verify(args: argparse.Namespace) -> int:
    from experiments.heterogeneous_agents.assemble_and_audit import score

    output = Path(args.output_dir).resolve()
    plan = _json(output / "plan/PLAN.json")
    prediction_path = output / "VALIDATION_PREDICTIONS.jsonl"
    deployed_path = Path(plan["deployed_predictions"])
    if sha256(deployed_path) != plan["deployed_predictions_sha256"]:
        raise ContractError("deployed prediction artifact changed")

    produced = _rows_by_key(read_jsonl(prediction_path))
    deployed = _rows_by_key(read_jsonl(deployed_path))
    if set(produced) != set(deployed):
        raise ContractError("row coverage differs from the deployed artifact")

    # The deployed residual layer recorded the support counts it actually
    # used.  Carrying them into the parity artifact makes the cause of any
    # divergence checkable instead of asserted.
    deployed_support: dict[tuple[str, str], Mapping[str, Any]] = {}
    ledger_path = Path(args.deployed_decisions)
    if ledger_path.is_file():
        deployed_support = {
            _key(row): row for row in read_jsonl(ledger_path)}

    unified_decisions = {
        _key(row): row for row in read_jsonl(output / "DECISIONS.jsonl")}

    divergent = []
    for key in sorted(produced):
        left = [str(item) for item in produced[key]["ObjectEntities"]]
        right = [str(item) for item in deployed[key]["ObjectEntities"]]
        if left == right:
            continue
        record = {
            "SubjectEntity": key[0],
            "Relation": key[1],
            "unified": left,
            "deployed": right,
        }
        deployed_row = deployed_support.get(key)
        if deployed_row is not None:
            record["deployed_candidate_support"] = deployed_row.get(
                "candidate_support")
        unified_row = unified_decisions.get(key)
        if unified_row is not None:
            for layer in unified_row.get("layers", []):
                if "candidate_support" in layer:
                    record["unified_candidate_support"] = layer[
                        "candidate_support"]
        # A support key that only differs from another by a leading answer
        # marker means the deployed run split one candidate into two.
        fragmented = sorted({
            surface
            for surface in (record.get("deployed_candidate_support") or {})
            if str(surface).upper().startswith("ANSWER:")
        })
        if fragmented:
            record["deployed_parse_fragmentation"] = fragmented
        divergent.append(record)

    gold = read_jsonl(Path(args.gold))
    unified_scores = score(list(produced.values()), gold)
    deployed_scores = score(list(deployed.values()), gold)
    pooled = "*** All Relations ***"
    identical = sha256(prediction_path) == sha256(deployed_path)
    explained = [
        row for row in divergent if row.get("deployed_parse_fragmentation")]

    parity = {
        "schema": PARITY_SCHEMA,
        "divergences_explained_by_deployed_parse_fragmentation": len(explained),
        "divergences_unexplained": len(divergent) - len(explained),
        "contains_labels": True,
        "gold_aware": True,
        "gold_used_for": "reporting_score_neutrality_only",
        "byte_identical": identical,
        "row_identical": not divergent,
        "divergent_rows": len(divergent),
        "divergences": divergent[:50],
        "unified_sha256": sha256(prediction_path),
        "deployed_sha256": sha256(deployed_path),
        "unified_score": unified_scores[pooled],
        "deployed_score": deployed_scores[pooled],
        "score_delta": unified_scores[pooled] - deployed_scores[pooled],
        "unified_per_relation": unified_scores,
        "deployed_per_relation": deployed_scores,
        "policies": [dict(policy) for policy in POLICIES],
    }
    _write_json(output / "analysis/PARITY.json", parity)

    lines = [
        "# Unified graph-native decoder — consolidation parity",
        "",
        "One decoder, one typed evidence graph carrying all three model "
        "families as first-class routes, one pass. The deployed artifact was "
        "produced by a base decoder plus two post-processing scripts; this "
        "run must reproduce it exactly.",
        "",
        f"- Byte-identical to deployed artifact: **{identical}**",
        f"- Divergent rows: **{len(divergent)}/{len(produced)}** "
        f"({len(explained)} explained by deployed answer-marker parse "
        f"fragmentation, {len(divergent) - len(explained)} unexplained)",
        f"- Unified pooled macro-F1: **{unified_scores[pooled]:.6f}**",
        f"- Deployed pooled macro-F1: **{deployed_scores[pooled]:.6f}**",
        f"- Score delta: **{parity['score_delta']:+.6f}**",
        "",
        "| policy | kind | evidence routes |",
        "|---|---|---|",
    ]
    for policy in POLICIES:
        lines.append(
            f"| `{policy['id']}` | {policy['kind']} | "
            f"{', '.join('`' + r + '`' for r in policy['routes'])} |")
    if divergent:
        lines.extend([
            "",
            "## Divergences",
            "",
            "| subject | relation | unified | deployed |",
            "|---|---|---|---|",
        ])
        for row in divergent[:25]:
            lines.append(
                f"| {row['SubjectEntity']} | {row['Relation']} | "
                f"{row['unified']} | {row['deployed']} |")
    (output / "analysis").mkdir(parents=True, exist_ok=True)
    (output / "analysis/RESULT.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "byte_identical": identical,
        "divergent_rows": len(divergent),
        "unified_score": unified_scores[pooled],
        "deployed_score": deployed_scores[pooled],
        "score_delta": parity["score_delta"],
        "result": str(output / "analysis/RESULT.md"),
    }, indent=2, sort_keys=True))
    return 0 if identical else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--typed-graph", default=str(TYPED_GRAPH))
    build_parser.add_argument("--cot40-responses", default=str(COT40_RESPONSES))
    build_parser.add_argument("--chain-incumbent", default=str(CHAIN_INCUMBENT))
    build_parser.add_argument("--capacity-ledger", default=str(CAPACITY_LEDGER))
    build_parser.add_argument(
        "--parser-mode", choices=PARSER_MODES, default="corrected",
        help=(
            "Use corrected parsing for new systems or legacy-20260729 for "
            "byte-identical reproduction of the frozen development artifact."))
    build_parser.add_argument(
        "--component-models", default=str(COMPONENT_MODELS))
    build_parser.add_argument("--deployed", default=str(DEPLOYED_PREDICTIONS))
    build_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    build_parser.set_defaults(function=build)

    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    decode_parser.set_defaults(function=decode)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    verify_parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    verify_parser.add_argument(
        "--deployed-decisions", default=str(DEPLOYED_DECISIONS))
    verify_parser.set_defaults(function=verify)
    return value


def main() -> int:
    arguments = parser().parse_args()
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
