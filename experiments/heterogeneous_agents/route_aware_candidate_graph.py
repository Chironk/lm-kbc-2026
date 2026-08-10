#!/usr/bin/env python3
"""Add explicit proposal-route evidence to heterogeneous candidate graphs.

The existing graph merges proposals by model family and uses System-2 only
inside the frozen incumbent composer.  This transform preserves the existing
Qwen/Gemma evidence fields while adding three explicit proposal routes:

* ``qwen:self_consistency`` -- the N=10 System-1 proposal reservoir;
* ``qwen:system2`` -- the alternate System-2 prompt/judge output;
* ``gemma:independent`` -- the independent Gemma proposal.

System-2 uses the already-counted Qwen checkpoint, so this is an inference
route, not another parameter-counted model.  A System-2-only identity becomes
a candidate node instead of disappearing when the frozen corroboration rule
does not promote it.

``build`` is label-free and fail-closed. ``analyze`` is explicitly gold-aware
and measures graph coverage/oracle diagnostics only; it never emits deployable
predictions or selects a decoder.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate import RELATION_TYPE, true_positives
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.heterogeneous_memory_selector import (
    _key,
    _load_graph,
)
from experiments.heterogeneous_agents.production_matched_graph import (
    _validate_system2,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "production_matched_oof_20260723_v1")
DEFAULT_TRAIN_SYSTEM2 = (
    ROOT / "archive/experiments/architecture_overnight_20260713/"
    "system2_fp16/predictions.jsonl")
DEFAULT_VALIDATION_SYSTEM2 = (
    ROOT / "archive/experiments/architecture_validation_20260713/"
    "system2_fp16/predictions.jsonl")
TARGET_RELATIONS = {
    "companyTradesAtStockExchange",
    "personHasCityOfDeath",
}
ROUTE_QWEN_SC = "qwen:self_consistency"
ROUTE_QWEN_SYSTEM2 = "qwen:system2"
ROUTE_GEMMA = "gemma:independent"


def _route(
        *, model_family: str, support: int, samples: int, selected: bool,
        route_type: str,
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "route_type": route_type,
        "support": int(support),
        "samples": int(samples),
        "support_rate": support / samples if samples else 0.0,
        "selected": bool(selected),
    }


def _summarize_routes(routes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    families = {
        str(route["model_family"]) for route in routes.values()}
    qwen_routes = {
        name for name, route in routes.items()
        if route["model_family"] == QWEN}
    return {
        "route_count": len(routes),
        "model_family_count": len(families),
        "cross_model_agreement": len(families) > 1,
        "within_qwen_route_agreement": len(qwen_routes) > 1,
        "system2_supported": ROUTE_QWEN_SYSTEM2 in routes,
        "system2_only": (
            set(routes) == {ROUTE_QWEN_SYSTEM2}),
        "gemma_only": set(routes) == {ROUTE_GEMMA},
        "qwen_sc_only": set(routes) == {ROUTE_QWEN_SC},
    }


def _selected_keys(
    graph: Mapping[str, Any],
    agent: str,
    relation: str,
    *,
    required: bool,
) -> set[str]:
    """Return canonical selected outputs, failing closed when required.

    Historical numeric train graphs used untyped candidate keys (``"38"``)
    while current canonicalization produces ``"numeric:38"``.  Their
    ``selected_by`` flags were consequently all false even though
    ``agent_outputs`` contained a selected numeric answer.  Recompute route
    selection exclusively from the actual output surfaces.  A graph that
    claims an agent/route but lacks its output is invalid; silently inheriting
    ``selected_by`` would reintroduce split-dependent legacy semantics.
    """
    outputs = graph.get("agent_outputs", {})
    if agent not in outputs:
        if required:
            raise ContractError(
                f"{_key(graph)}: required agent_outputs missing {agent}")
        return set()
    if not isinstance(outputs[agent], list):
        raise ContractError(
            f"{_key(graph)}: agent_outputs[{agent!r}] must be a list")
    return {
        key for item in outputs.get(agent, [])
        if (key := canonical_key(str(item), relation))
    }


def _route_required(
    graph: Mapping[str, Any], agent: str, route: str,
) -> bool:
    """Whether a row claims evidence from an agent/route."""
    if agent in graph.get("agents", {}):
        return True
    return any(
        agent in node.get("sources", {})
        or route in node.get("routes", {})
        for node in graph.get("candidates", [])
    )


def normalize_route_selection(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize existing Qwen/Gemma route flags with split-invariant logic."""
    row = copy.deepcopy(graph)
    relation = str(row["Relation"])
    route_by_agent = {
        QWEN: ROUTE_QWEN_SC,
        GEMMA: ROUTE_GEMMA,
    }
    required = {
        agent: _route_required(row, agent, route)
        for agent, route in route_by_agent.items()
    }
    selected = {
        agent: _selected_keys(
            row, agent, relation, required=required[agent])
        for agent in (QWEN, GEMMA)
    }
    for node in row.get("candidates", []):
        item_key = canonical_key(str(node.get("item", "")), relation)
        for agent, route in route_by_agent.items():
            evidence = node.get("routes", {}).get(route)
            if evidence is None:
                continue
            evidence["selected"] = bool(
                item_key and item_key in selected[agent])
            node.setdefault("selected_by", {})[agent] = bool(
                evidence["selected"])
        if "routes" in node:
            node["route_summary"] = _summarize_routes(node["routes"])
    row.setdefault("route_selection_normalization", {}).update({
        "schema": "canonical-agent-output-selection-v2",
        "qwen_outputs_required": required[QWEN],
        "gemma_outputs_required": required[GEMMA],
        "qwen_outputs_available": (
            not required[QWEN] or QWEN in row.get("agent_outputs", {})),
        "gemma_outputs_available": (
            not required[GEMMA] or GEMMA in row.get("agent_outputs", {})),
        "legacy_selected_by_fallback_allowed": False,
    })
    return row


def augment_graph(
        graph: Mapping[str, Any], system2_objects: Sequence[str],
) -> dict[str, Any]:
    """Return one graph with explicit route evidence and merged identities."""
    row = copy.deepcopy(graph)
    relation = str(row["Relation"])
    selected_keys = {
        agent: _selected_keys(row, agent, relation, required=True)
        for agent in (QWEN, GEMMA)
    }
    nodes: dict[str, dict] = {}
    for original in row.get("candidates", []):
        node = copy.deepcopy(original)
        item_key = canonical_key(str(node.get("item", "")), relation)
        routes: dict[str, dict] = {}
        qwen = node.get("sources", {}).get(QWEN)
        gemma = node.get("sources", {}).get(GEMMA)
        if qwen is not None:
            qwen_selected = bool(
                item_key and item_key in selected_keys[QWEN])
            routes[ROUTE_QWEN_SC] = _route(
                model_family=QWEN,
                support=int(qwen.get("support", 0)),
                samples=int(qwen.get("samples", 0)),
                selected=qwen_selected,
                route_type="sampled-self-consistency",
            )
            node.setdefault("selected_by", {})[QWEN] = qwen_selected
        if gemma is not None:
            gemma_selected = bool(
                item_key and item_key in selected_keys[GEMMA])
            routes[ROUTE_GEMMA] = _route(
                model_family=GEMMA,
                support=int(gemma.get("support", 0)),
                samples=int(gemma.get("samples", 0)),
                selected=gemma_selected,
                route_type="independent-direct-recall",
            )
            node.setdefault("selected_by", {})[GEMMA] = gemma_selected
        node["routes"] = routes
        node["route_summary"] = _summarize_routes(routes)
        nodes[str(node["key"])] = node

    system2_keys = set()
    for item in system2_objects:
        key = canonical_key(str(item), relation)
        if not key or key in system2_keys:
            continue
        system2_keys.add(key)
        node = nodes.setdefault(key, {
            "key": key,
            "item": str(item),
            "type": (
                "numeric" if relation in {"hasArea", "hasCapacity"}
                else "string"),
            # ``sources`` retains its original model-proposal semantics for
            # backward compatibility. New consumers must use ``routes``.
            "sources": {},
            "selected_by": {QWEN: False, GEMMA: False},
            "routes": {},
            "route_summary": {},
        })
        node["routes"][ROUTE_QWEN_SYSTEM2] = _route(
            model_family=QWEN,
            support=1,
            samples=1,
            selected=True,
            route_type="alternate-prompt-judge",
        )
        node["route_summary"] = _summarize_routes(node["routes"])
    for node in nodes.values():
        # Recompute summaries for pre-existing nodes after System-2 merging.
        node["route_summary"] = _summarize_routes(node["routes"])

    row["candidates"] = sorted(nodes.values(), key=lambda node: (
        -int(node["route_summary"]["model_family_count"]),
        -int(node["route_summary"]["route_count"]),
        -sum(float(route["support_rate"])
             for route in node["routes"].values()),
        str(node["key"]),
    ))
    row["proposal_routes"] = {
        ROUTE_QWEN_SC: {
            "model_family": QWEN,
            "available": True,
            "n_samples": int(row["agents"][QWEN]["n_samples"]),
        },
        ROUTE_QWEN_SYSTEM2: {
            "model_family": QWEN,
            "available": relation in TARGET_RELATIONS,
            "n_samples": 1 if relation in TARGET_RELATIONS else 0,
            "objects": list(system2_objects),
        },
        ROUTE_GEMMA: {
            "model_family": GEMMA,
            "available": True,
            "n_samples": int(row["agents"][GEMMA]["n_samples"]),
        },
    }
    row["route_graph_schema"] = "explicit-proposal-routes-v1"
    row["route_selection_normalization"] = {
        "schema": "canonical-agent-output-selection-v2",
        "qwen_outputs_required": True,
        "gemma_outputs_required": True,
        "qwen_outputs_available": True,
        "gemma_outputs_available": True,
        "legacy_selected_by_fallback_allowed": False,
    }
    return row


def _augment_split(
        graphs: Sequence[Mapping[str, Any]],
        system2: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict]:
    output = []
    for graph in graphs:
        key = _key(graph)
        relation = key[1]
        if relation in TARGET_RELATIONS:
            if key not in system2:
                raise ContractError(f"System-2 misses graph row {key}")
            objects = list(system2[key].get("ObjectEntities", []))
        else:
            objects = []
        output.append(augment_graph(graph, objects))
    return output


def _write_graph(
        path: Path, rows: Sequence[Mapping[str, Any]], *, split: str,
        source_graph: Path, system2_provenance: Mapping[str, Any],
) -> None:
    write_jsonl_atomic(path, rows)
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({
        "schema": "heterogeneous-memory-graph-manifest-v1",
        "route_graph_schema": "explicit-proposal-routes-v1",
        "split": split,
        "rows": len(rows),
        "contains_labels": False,
        "gold_aware": False,
        "output_sha256": sha256(path),
        "source_graph": str(source_graph),
        "source_graph_sha256": sha256(source_graph),
        "system2": system2_provenance,
        "model_families": [QWEN, GEMMA],
        "proposal_routes": [
            ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2, ROUTE_GEMMA],
        "parameter_count_delta": 0,
    }, indent=2, sort_keys=True) + "\n")


def build(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    train_graph_path = source / "graphs/train_graph.jsonl"
    validation_graph_path = source / "graphs/validation_graph.jsonl"
    train = _load_graph(train_graph_path, expected_split="train")
    validation = _load_graph(validation_graph_path, expected_split="validation")
    train_system2_path = Path(args.train_system2).resolve()
    validation_system2_path = Path(args.validation_system2).resolve()
    train_system2, train_provenance = _validate_system2(train_system2_path)
    validation_system2, validation_provenance = _validate_system2(
        validation_system2_path)
    train_augmented = _augment_split(train, train_system2)
    validation_augmented = _augment_split(validation, validation_system2)

    graph_dir = output / "graphs"
    plan_dir = output / "plan"
    graph_dir.mkdir(parents=True, exist_ok=True)
    plan_dir.mkdir(parents=True, exist_ok=True)
    source_plan = source / "plan/PLAN.json"
    if not source_plan.is_file():
        raise ContractError(f"missing source plan {source_plan}")
    shutil.copy2(source_plan, plan_dir / "PLAN.json")
    train_output = graph_dir / "train_graph.jsonl"
    validation_output = graph_dir / "validation_graph.jsonl"
    _write_graph(
        train_output, train_augmented, split="train",
        source_graph=train_graph_path, system2_provenance=train_provenance)
    _write_graph(
        validation_output, validation_augmented, split="validation",
        source_graph=validation_graph_path,
        system2_provenance=validation_provenance)
    record = {
        "schema": "route-aware-candidate-graph-build-v1",
        "labels_opened": False,
        "validation_labels_opened": False,
        "train_rows": len(train_augmented),
        "validation_rows": len(validation_augmented),
        "model_families": [QWEN, GEMMA],
        "proposal_routes": [
            ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2, ROUTE_GEMMA],
        "parameter_count_delta": 0,
        "train_graph": str(train_output),
        "train_graph_sha256": sha256(train_output),
        "validation_graph": str(validation_output),
        "validation_graph_sha256": sha256(validation_output),
    }
    (output / "BUILD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"route-aware graphs ready: {len(train_augmented)} train, "
        f"{len(validation_augmented)} validation")
    print(f"build={output / 'BUILD.json'}")
    return 0


def _gold_aliases(row: Mapping[str, Any]) -> list[list[str]]:
    values = row.get("ObjectEntities", [])
    if values and isinstance(values[0], str):
        return [[str(item)] for item in values]
    return values


def _row_f1(
        objects: Sequence[str], gold: Mapping[str, Any], relation: str,
) -> float:
    aliases = _gold_aliases(gold)
    predictions = list(dict.fromkeys(str(item) for item in objects))
    tp = true_positives(
        predictions, aliases, RELATION_TYPE[relation], 0.05)
    precision = tp / len(predictions) if predictions else 1.0
    recall = tp / len(aliases) if aliases else 1.0
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0)


def _candidate_hit(
        graph: Mapping[str, Any], gold: Mapping[str, Any],
) -> bool:
    return true_positives(
        [str(node["item"]) for node in graph["candidates"]],
        _gold_aliases(gold), RELATION_TYPE[str(graph["Relation"])], 0.05) > 0


def _candidate_oracle(
        graph: Mapping[str, Any], gold: Mapping[str, Any],
) -> float:
    relation = str(graph["Relation"])
    if not gold.get("ObjectEntities"):
        return 1.0
    correct = [
        str(node["item"]) for node in graph["candidates"]
        if true_positives(
            [str(node["item"])], _gold_aliases(gold),
            RELATION_TYPE[relation], 0.05) > 0]
    if relation == "personHasCityOfDeath":
        actions = [[], *[[item] for item in correct]]
        return max(_row_f1(action, gold, relation) for action in actions)
    return _row_f1(correct, gold, relation)


def _metrics(
        graphs: Sequence[Mapping[str, Any]],
        gold: Mapping[tuple[str, str], Mapping[str, Any]],
        relation: str,
) -> dict[str, Any]:
    rows = [row for row in graphs if row["Relation"] == relation]
    nonnull = [row for row in rows if gold[_key(row)]["ObjectEntities"]]
    null = [row for row in rows if not gold[_key(row)]["ObjectEntities"]]
    system2_only_nodes = [
        node for row in rows for node in row["candidates"]
        if node.get("route_summary", {}).get("system2_only")]
    unique_correct_rows = sum(
        any(
            node.get("route_summary", {}).get("system2_only")
            and true_positives(
                [str(node["item"])], _gold_aliases(gold[_key(row)]),
                RELATION_TYPE[relation], 0.05) > 0
            for node in row["candidates"])
        for row in nonnull)
    return {
        "rows": len(rows),
        "nonnull_rows": len(nonnull),
        "null_rows": len(null),
        "nonnull_candidate_recall": (
            sum(_candidate_hit(row, gold[_key(row)]) for row in nonnull)
            / len(nonnull) if nonnull else 0.0),
        "null_rows_with_candidates": sum(
            bool(row["candidates"]) for row in null),
        "mean_candidate_count": (
            sum(len(row["candidates"]) for row in rows) / len(rows)
            if rows else 0.0),
        "candidate_oracle_mean_f1": (
            sum(_candidate_oracle(row, gold[_key(row)]) for row in rows)
            / len(rows) if rows else 0.0),
        "system2_only_nodes": len(system2_only_nodes),
        "system2_only_correct_nonnull_rows": unique_correct_rows,
        "system2_only_nodes_on_null_rows": sum(
            node.get("route_summary", {}).get("system2_only", False)
            for row in null for node in row["candidates"]),
    }


def _route_bucket_metrics(
        graphs: Sequence[Mapping[str, Any]],
        gold: Mapping[tuple[str, str], Mapping[str, Any]],
        relation: str,
) -> dict[str, dict[str, float | int]]:
    rows = [row for row in graphs if row["Relation"] == relation]
    predicates = {
        "cross_model_agreement": lambda summary, routes: bool(
            summary.get("cross_model_agreement")),
        "within_qwen_route_agreement": lambda summary, routes: bool(
            summary.get("within_qwen_route_agreement")),
        "system2_supported": lambda summary, routes: bool(
            summary.get("system2_supported")),
        "system2_only": lambda summary, routes: bool(
            summary.get("system2_only")),
        "all_three_routes": lambda summary, routes: set(routes) == {
            ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2, ROUTE_GEMMA},
        "qwen_sc_only": lambda summary, routes: bool(
            summary.get("qwen_sc_only")),
        "gemma_only": lambda summary, routes: bool(
            summary.get("gemma_only")),
    }
    counts = {
        name: {"nodes": 0, "correct_nodes": 0}
        for name in predicates}
    for row in rows:
        target = gold[_key(row)]
        for node in row["candidates"]:
            summary = node.get("route_summary", {})
            routes = node.get("routes", {})
            correct = true_positives(
                [str(node["item"])], _gold_aliases(target),
                RELATION_TYPE[relation], 0.05) > 0
            for name, predicate in predicates.items():
                if predicate(summary, routes):
                    counts[name]["nodes"] += 1
                    counts[name]["correct_nodes"] += int(correct)
    return {
        name: {
            **values,
            "precision": (
                values["correct_nodes"] / values["nodes"]
                if values["nodes"] else 0.0),
        }
        for name, values in counts.items()
    }


def analyze(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    base = Path(args.base_output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = _load_graph(
        source / "graphs/train_graph.jsonl", expected_split="train")
    validation = _load_graph(
        source / "graphs/validation_graph.jsonl", expected_split="validation")
    base_train = _load_graph(
        base / "graphs/train_graph.jsonl", expected_split="train")
    base_validation = _load_graph(
        base / "graphs/validation_graph.jsonl", expected_split="validation")
    train_gold_rows = read_jsonl(Path(args.train_gold).resolve())
    validation_gold_rows = read_jsonl(Path(args.validation_gold).resolve())
    train_gold = {_key(row): row for row in train_gold_rows}
    validation_gold = {_key(row): row for row in validation_gold_rows}
    if {_key(row) for row in train} != set(train_gold):
        raise ContractError("training graph/gold coverage mismatch")
    if {_key(row) for row in validation} != set(validation_gold):
        raise ContractError("validation graph/gold coverage mismatch")

    metrics = {}
    for split, old_rows, new_rows, gold in (
            ("train", base_train, train, train_gold),
            ("validation", base_validation, validation, validation_gold)):
        if split == "train":
            eligible = {
                _key(row) for row in new_rows
                if row.get("calibration_eligible", True) is not False}
            old_rows = [row for row in old_rows if _key(row) in eligible]
            new_rows = [row for row in new_rows if _key(row) in eligible]
        metrics[split] = {}
        for relation in sorted(TARGET_RELATIONS):
            old = _metrics(old_rows, gold, relation)
            new = _metrics(new_rows, gold, relation)
            metrics[split][relation] = {
                "base": old,
                "route_aware": new,
                "delta": {
                    key: new[key] - old[key]
                    for key in (
                        "nonnull_candidate_recall",
                        "null_rows_with_candidates",
                        "mean_candidate_count",
                        "candidate_oracle_mean_f1")
                },
                "route_bucket_precision": _route_bucket_metrics(
                    new_rows, gold, relation),
            }
    result = {
        "schema": "route-aware-candidate-graph-analysis-v1",
        "development_only": True,
        "gold_aware": True,
        "nondeployable": True,
        "purpose": (
            "measure graph reservoir coverage; does not select a decoder or "
            "emit predictions"),
        "train_metrics_exclude_calibration_ineligible_rows": True,
        "metrics": metrics,
        "route_graph": str(source),
        "base_graph": str(base),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Route-aware candidate graph analysis",
        "",
        "Gold-aware development diagnostic only. No decoder was selected and "
        "no deployable predictions were emitted.",
        "",
        "| split | relation | non-null recall base → route | "
        "oracle F1 base → route | mean candidates base → route | "
        "System-2-only correct rows | System-2-only nodes on null rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation"):
        for relation in sorted(TARGET_RELATIONS):
            record = metrics[split][relation]
            old, new = record["base"], record["route_aware"]
            lines.append(
                f"| {split} | {relation} | "
                f"{old['nonnull_candidate_recall']:.4f} → "
                f"{new['nonnull_candidate_recall']:.4f} | "
                f"{old['candidate_oracle_mean_f1']:.4f} → "
                f"{new['candidate_oracle_mean_f1']:.4f} | "
                f"{old['mean_candidate_count']:.2f} → "
                f"{new['mean_candidate_count']:.2f} | "
                f"{new['system2_only_correct_nonnull_rows']} | "
                f"{new['system2_only_nodes_on_null_rows']} |")
    lines += [
        "",
        "## Route identity precision",
        "",
        "| split | relation | route bucket | correct / nodes | precision |",
        "|---|---|---|---:|---:|",
    ]
    for split in ("train", "validation"):
        for relation in sorted(TARGET_RELATIONS):
            buckets = metrics[split][relation]["route_bucket_precision"]
            for name in (
                    "all_three_routes", "cross_model_agreement",
                    "within_qwen_route_agreement", "system2_supported",
                    "system2_only", "qwen_sc_only", "gemma_only"):
                bucket = buckets[name]
                lines.append(
                    f"| {split} | {relation} | {name} | "
                    f"{bucket['correct_nodes']} / {bucket['nodes']} | "
                    f"{bucket['precision']:.4f} |")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(f"route-aware graph analysis complete: {output / 'RESULT.md'}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--source-output-dir", default=str(DEFAULT_SOURCE))
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--train-system2", default=str(DEFAULT_TRAIN_SYSTEM2))
    build_parser.add_argument(
        "--validation-system2", default=str(DEFAULT_VALIDATION_SYSTEM2))
    build_parser.set_defaults(func=build)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--source-output-dir", required=True)
    analyze_parser.add_argument(
        "--base-output-dir", default=str(DEFAULT_SOURCE))
    analyze_parser.add_argument("--output-dir", required=True)
    analyze_parser.add_argument(
        "--train-gold", default=str(ROOT / "data/train.jsonl"))
    analyze_parser.add_argument(
        "--validation-gold", default=str(ROOT / "data/val.jsonl"))
    analyze_parser.set_defaults(func=analyze)
    return ap


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(parsed.func(parsed))
