#!/usr/bin/env python3
"""Insert frozen capacity-specific proposals into the typed candidate graph.

The numeric supply pilot measured new correct candidates, but its first
downstream selector used proposal values only as votes over *pre-existing*
components.  Values absent from the old graph therefore could not be selected.
This module repairs that missing architectural link:

    compact proposal -> typed candidate surface -> numeric component -> decoder

``build`` is label-free and train-only.  It validates the frozen pilot
responses, adds the Gemma direct-capacity route without collapsing provenance,
rebuilds relational components, and writes a manifest.  No validation graph or
gold label is accepted.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate import try_parse_number
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
from experiments.heterogeneous_agents.relational_candidate_graph import (
    augment_relational_graph,
)
from experiments.heterogeneous_agents.route_aware_candidate_graph import (
    _summarize_routes,
)
from experiments.heterogeneous_agents.numeric_supply_pilot import (
    ARM_AGENT,
    _sample_values,
    _validated_responses,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "targeted_company_gemma_n3_20260724_v1")
DEFAULT_PILOT = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "numeric_supply_pilot_20260724_v1")
RELATION = "hasCapacity"
ROUTE = "gemma:capacity_direct"
ARM_ROUTES = {
    "gemma_direct": ROUTE,
    "gemma_magnitude": "gemma:capacity_magnitude",
    "qwen_direct": "qwen:capacity_direct",
}
ROUTE_AGENTS = {
    ARM_ROUTES["gemma_direct"]: GEMMA,
    ARM_ROUTES["gemma_magnitude"]: GEMMA,
    ARM_ROUTES["qwen_direct"]: QWEN,
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _valid_number(value: Any) -> float | None:
    parsed = try_parse_number(str(value))
    if parsed is None or not math.isfinite(parsed) or parsed <= 0:
        return None
    return float(parsed)


def _same_number(left: Any, right: float) -> bool:
    parsed = _valid_number(left)
    if parsed is None:
        return False
    scale = max(abs(parsed), abs(right), 1.0)
    return abs(parsed - right) / scale <= 1e-12


def _display(value: float) -> str:
    return format(value, ".15g")


def augment_capacity_row(
        graph: Mapping[str, Any],
        route_values: Mapping[str, Sequence[float]] | Sequence[float],
) -> dict[str, Any]:
    """Add exact proposal surfaces, then rebuild 5%-equivalent components."""
    row = copy.deepcopy(graph)
    if str(row["Relation"]) != RELATION:
        return row
    # Backward-compatible convenience for focused unit tests and callers.
    if not isinstance(route_values, Mapping):
        route_values = {ROUTE: route_values}
    candidates = list(row.get("candidates", []))
    for route, values in sorted(route_values.items()):
        counts = Counter(float(value) for value in values)
        for value, support in sorted(counts.items()):
            matching = next((
                candidate for candidate in candidates
                if _same_number(candidate.get("item"), value)
            ), None)
            if matching is None:
                item = _display(value)
                matching = {
                    "key": canonical_key(item, RELATION),
                    "item": item,
                    "type": "numeric",
                    "sources": {},
                    "selected_by": {
                        "qwen_recall": False,
                        "gemma_independent": False,
                    },
                    "routes": {},
                    "route_summary": {},
                }
                candidates.append(matching)
            routes = matching.setdefault("routes", {})
            routes[route] = {
                "model_family": ROUTE_AGENTS[route],
                "route_type": "capacity-specific-compact-proposal",
                "support": int(support),
                "samples": len(values),
                "support_rate": support / len(values) if values else 0.0,
                "selected": support == max(counts.values()),
            }
            matching["route_summary"] = _summarize_routes(routes)
    row["candidates"] = sorted(candidates, key=lambda candidate: (
        -sum(float(route.get("support_rate", 0.0))
             for route in candidate.get("routes", {}).values()),
        str(candidate["key"]),
    ))
    for arm, route in ARM_ROUTES.items():
        if route not in route_values:
            continue
        values = route_values[route]
        row.setdefault("proposal_routes", {})[route] = {
            "model_family": ROUTE_AGENTS[route],
            "available": bool(values),
            "n_samples": len(values),
            "route_type": "capacity-specific-compact-proposal",
            "source_pilot_arm": arm,
        }
    return augment_relational_graph(row)


def load_capacity_routes(
        pilot: Path,
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Load all frozen capacity proposal arms without opening pilot labels."""
    plan = _json(pilot / "plan/PLAN.json")
    active = [arm for arm in ARM_ROUTES if arm in plan.get("arms", [])]
    if not active:
        raise ContractError("pilot contains no supported capacity arms")
    responses: dict[str, Mapping[str, Any]] = {}
    for agent in sorted({ARM_AGENT[arm] for arm in active}):
        responses.update(_validated_responses(plan, agent))
    result: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in read_jsonl(Path(plan["inputs"])):
        key = str(row["SubjectEntity"]), str(row["Relation"])
        if key[1] != RELATION:
            continue
        per_route = {}
        for arm in active:
            agent = ARM_AGENT[arm]
            task_id = f"{agent}::{arm}::{row['input_index']}::proposal"
            parsed = _sample_values(
                list(responses[task_id].get("generations", [])), RELATION)
            per_route[ARM_ROUTES[arm]] = [
                float(value) for value in parsed if value is not None]
        result[key] = per_route
    return result


def build(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    pilot = Path(args.pilot_dir).resolve()
    output = Path(args.output_dir).resolve()
    source_graph = source / "graphs/train_graph.jsonl"
    source_manifest = source_graph.with_suffix(
        source_graph.suffix + ".manifest.json")
    manifest = _json(source_manifest)
    if manifest.get("split") != "train":
        raise ContractError("capacity proposal graph accepts train only")
    if manifest.get("contains_labels") or manifest.get("gold_aware"):
        raise ContractError("source graph must be label-free")
    if manifest.get("output_sha256") != sha256(source_graph):
        raise ContractError("source graph hash mismatch")

    plan = _json(pilot / "plan/PLAN.json")
    if plan.get("split") != "train":
        raise ContractError("capacity proposal pilot must be train-only")
    if "gemma_direct" not in plan.get("arms", []):
        raise ContractError("pilot lacks gemma_direct arm")
    routes = load_capacity_routes(pilot)
    rows = _load_graph(source_graph, expected_split="train")
    expected = {
        _key(row) for row in rows if str(row["Relation"]) == RELATION}
    if set(routes) != expected:
        raise ContractError("capacity proposal response coverage mismatch")

    before_surfaces = before_components = 0
    after_surfaces = after_components = 0
    augmented = []
    for row in rows:
        if str(row["Relation"]) == RELATION:
            before_surfaces += len(row.get("candidates", []))
            before_components += len(
                row.get("relational_graph", {}).get("components", []))
            new_row = augment_capacity_row(row, routes[_key(row)])
            after_surfaces += len(new_row.get("candidates", []))
            after_components += len(
                new_row.get("relational_graph", {}).get("components", []))
        else:
            new_row = copy.deepcopy(row)
        augmented.append(new_row)

    graph_dir = output / "graphs"
    plan_dir = output / "plan"
    graph_dir.mkdir(parents=True, exist_ok=True)
    plan_dir.mkdir(parents=True, exist_ok=True)
    target = graph_dir / "train_graph.jsonl"
    write_jsonl_atomic(target, augmented)
    target.with_suffix(target.suffix + ".manifest.json").write_text(
        json.dumps({
            "schema": "heterogeneous-memory-graph-manifest-v1",
            "split": "train",
            "rows": len(augmented),
            "contains_labels": False,
            "gold_aware": False,
            "output_sha256": sha256(target),
            "source_graph": str(source_graph),
            "source_graph_sha256": sha256(source_graph),
            "proposal_pilot": str(pilot),
            "proposal_plan_sha256": sha256(pilot / "plan/PLAN.json"),
            "proposal_route": ROUTE,
            "proposal_routes": sorted({
                route for values in routes.values() for route in values}),
            "parameter_count_delta": 0,
            "validation_graph_created": False,
        }, indent=2, sort_keys=True) + "\n")
    source_plan = source / "plan/PLAN.json"
    shutil.copy2(source_plan, plan_dir / "PLAN.json")
    report = {
        "schema": "capacity-proposal-graph-build-v1",
        "labels_opened": False,
        "validation_opened": False,
        "relation": RELATION,
        "routes": sorted({
            route for values in routes.values() for route in values}),
        "rows": len(expected),
        "before_surfaces": before_surfaces,
        "after_surfaces": after_surfaces,
        "added_surfaces": after_surfaces - before_surfaces,
        "before_components": before_components,
        "after_components": after_components,
        "added_components": after_components - before_components,
        "train_graph": str(target),
        "train_graph_sha256": sha256(target),
    }
    (output / "BUILD.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--source-output-dir", default=str(DEFAULT_SOURCE))
    root.add_argument("--pilot-dir", default=str(DEFAULT_PILOT))
    return root


def main() -> int:
    return build(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
