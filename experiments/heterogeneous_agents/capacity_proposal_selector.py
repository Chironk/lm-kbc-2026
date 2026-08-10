#!/usr/bin/env python3
"""Train-only selector test for capacity-specific proposal components.

The selector tests a deliberately small, predeclared family:

* ``never``: keep the incumbent;
* ``new_unanimous``: switch only to a genuinely new component supported by
  all three capacity-direct samples;
* ``new_majority``: the same, with at least two of three samples;
* ``any_unanimous``: allow an old or new non-incumbent component with 3/3.

A component is "new" only when its sole evidence route is
``gemma:capacity_direct``.  This directly measures whether newly supplied
facts—not re-ranking old candidates—can improve the incumbent.

This module has no validation mode.  It opens train gold for an OOF stability
audit and uses the standing gate: positive mean delta, wins in at least three
of five frozen folds, and no harmful fold.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate import try_parse_number
from experiments.heterogeneous_agents.capacity_proposal_graph import (
    ARM_ROUTES,
    ROUTE,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.heterogeneous_memory_selector import (
    _key,
    _load_graph,
)
from experiments.heterogeneous_agents.relation_specific_structured_decoder import (
    _row_f1,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_SOURCE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "capacity_proposal_graph_20260725_v1")
RELATION = "hasCapacity"
GEMMA_DIRECT = ARM_ROUTES["gemma_direct"]
GEMMA_MAGNITUDE = ARM_ROUTES["gemma_magnitude"]
QWEN_DIRECT = ARM_ROUTES["qwen_direct"]
PROPOSAL_ROUTES = set(ARM_ROUTES.values())
RULES = (
    "never",
    "new_cross_model",
    "new_two_views",
    "new_gemma_views",
    "new_unanimous",
    "new_majority",
    "any_unanimous",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _number(value: Any) -> float | None:
    parsed = try_parse_number(str(value))
    if parsed is None or not math.isfinite(parsed) or parsed <= 0:
        return None
    return float(parsed)


def _within(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right))
    return scale > 0 and abs(left - right) / scale <= 0.05 + 1e-12


def _component_values(component: Mapping[str, Any]) -> list[float]:
    return [
        value for item in component.get("member_items", [])
        if (value := _number(item)) is not None]


def proposal_action(
        graph: Mapping[str, Any], rule: str,
) -> str | None:
    if rule == "never":
        return None
    incumbent = [
        value for item in graph.get("baseline_objects", [])
        if (value := _number(item)) is not None]
    required = 2 if rule == "new_majority" else 3
    new_only = rule.startswith("new_")
    choices = []
    for component in graph.get(
            "relational_graph", {}).get("components", []):
        component_routes = component.get("routes", {})
        route = component_routes.get(GEMMA_DIRECT)
        proposal_routes = set(component_routes) & PROPOSAL_ROUTES
        if rule == "new_cross_model":
            if not (
                    QWEN_DIRECT in proposal_routes
                    and proposal_routes & {GEMMA_DIRECT, GEMMA_MAGNITUDE}):
                continue
            support = len(proposal_routes)
        elif rule == "new_two_views":
            if len(proposal_routes) < 2:
                continue
            support = len(proposal_routes)
        elif rule == "new_gemma_views":
            if not {GEMMA_DIRECT, GEMMA_MAGNITUDE} <= proposal_routes:
                continue
            support = len(proposal_routes)
        else:
            if route is None:
                continue
            support = int(round(
                float(route.get("max_support_rate", 0.0)) * 3))
            if support < required:
                continue
        values = _component_values(component)
        if not values or any(
                _within(value, reference)
                for value in values for reference in incumbent):
            continue
        routes = set(component_routes)
        if new_only and routes - PROPOSAL_ROUTES:
            continue
        choices.append((
            support,
            -len(routes),
            str(component["representative"]),
        ))
    return max(choices)[2] if choices else None


def run(args: argparse.Namespace) -> int:
    source = Path(args.graph_source).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    graph_path = source / "graphs/train_graph.jsonl"
    manifest = _json(graph_path.with_suffix(
        graph_path.suffix + ".manifest.json"))
    if manifest.get("split") != "train":
        raise ContractError("selector accepts only a train graph")
    if manifest.get("contains_labels") or manifest.get("gold_aware"):
        raise ContractError("selector input graph must be label-free")
    if manifest.get("output_sha256") != sha256(graph_path):
        raise ContractError("input graph hash mismatch")
    if manifest.get("proposal_route") != ROUTE:
        raise ContractError("graph lacks the capacity proposal route")
    original_path = Path(manifest["source_graph"])
    if sha256(original_path) != manifest["source_graph_sha256"]:
        raise ContractError("original source graph hash mismatch")

    graphs = [
        graph for graph in _load_graph(graph_path, expected_split="train")
        if str(graph["Relation"]) == RELATION]
    original_graphs = {
        _key(graph): graph
        for graph in _load_graph(original_path, expected_split="train")
        if str(graph["Relation"]) == RELATION}
    plan = _json(source / "plan/PLAN.json")
    gold_path = Path(plan["train_gold"])
    if sha256(gold_path) != plan["train_gold_sha256"]:
        raise ContractError("train gold hash mismatch")
    gold = {_key(row): row for row in read_jsonl(gold_path)}
    folds_path = Path(plan["folds"])
    folds = {_key(row): int(row["fold"]) for row in read_jsonl(folds_path)}
    if len(graphs) != 100:
        raise ContractError(f"expected 100 capacity rows, got {len(graphs)}")

    current_supply = 0
    augmented_supply = 0
    for graph in graphs:
        key = _key(graph)
        gold_value = _number(gold[key]["ObjectEntities"][0][0])
        if gold_value is None:
            raise ContractError(f"{key}: invalid gold")
        # Read the pre-augmentation graph rather than removing every augmented
        # node carrying ROUTE: an exact proposal can legitimately merge into a
        # pre-existing candidate, and filtering that node would undercount the
        # original supply.
        old_values = [
            value for node in original_graphs[key].get("candidates", [])
            if (value := _number(node.get("item"))) is not None]
        all_values = [
            value for node in graph.get("candidates", [])
            if (value := _number(node.get("item"))) is not None]
        current_supply += any(_within(value, gold_value)
                              for value in old_values)
        augmented_supply += any(_within(value, gold_value)
                                for value in all_values)

    report = {}
    diagnostics = []
    for rule in RULES[1:]:
        fold_deltas = {}
        helpful = harmful = neutral = actions = 0
        for fold in range(5):
            deltas = []
            for graph in graphs:
                key = _key(graph)
                if folds[key] != fold:
                    continue
                replacement = proposal_action(graph, rule)
                if replacement is None:
                    deltas.append(0.0)
                    continue
                actions += 1
                delta = (
                    _row_f1([replacement], gold[key], RELATION)
                    - _row_f1(
                        list(graph.get("baseline_objects", [])),
                        gold[key], RELATION))
                deltas.append(delta)
                helpful += delta > 1e-12
                harmful += delta < -1e-12
                neutral += abs(delta) <= 1e-12
                diagnostics.append({
                    "SubjectEntity": key[0],
                    "Relation": RELATION,
                    "rule": rule,
                    "fold": fold,
                    "incumbent": list(graph.get("baseline_objects", [])),
                    "replacement": [replacement],
                    "delta": delta,
                })
            fold_deltas[str(fold)] = statistics.mean(deltas)
        values = list(fold_deltas.values())
        wins = sum(value > 1e-12 for value in values)
        harmful_folds = sum(value < -1e-12 for value in values)
        mean = statistics.mean(values)
        report[rule] = {
            "fold_deltas": fold_deltas,
            "mean_paired_delta": mean,
            "winning_folds": wins,
            "harmful_folds": harmful_folds,
            "actions": actions,
            "helpful_actions": helpful,
            "harmful_actions": harmful,
            "neutral_actions": neutral,
            "gate_passed": (
                mean > 1e-12 and wins >= 3 and harmful_folds == 0),
        }
    passing = [
        rule for rule in RULES[1:] if report[rule]["gate_passed"]]
    selected = (max(passing, key=lambda rule: (
        report[rule]["mean_paired_delta"],
        -report[rule]["actions"])) if passing else "never")
    result = {
        "schema": "capacity-proposal-selector-result-v1",
        "scope": "train-only-gold-aware",
        "contains_labels": True,
        "gold_aware": True,
        "deployable": False,
        "validation_opened": False,
        "graph_source": str(source),
        "graph_sha256": sha256(graph_path),
        "current_supply_oracle": current_supply / len(graphs),
        "augmented_supply_oracle": augmented_supply / len(graphs),
        "newly_reachable_rows": augmented_supply - current_supply,
        "gate": "mean>0, wins>=3/5, no harmful fold",
        "rules": report,
        "selected_rule": selected,
        "passed": selected != "never",
        "conclusion": (
            "proposal-to-component integration is correct, but no new-component "
            "selection rule passes the standing stability gate"
            if selected == "never" else
            f"freeze {selected} for a separately authorized confirmation"),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_jsonl_atomic(output / "action_diagnostics.jsonl", diagnostics)
    lines = [
        "# Capacity proposal-component selector",
        "",
        "Train-only gold-aware audit; validation remained sealed.",
        "",
        f"- supply oracle: {current_supply/len(graphs):.3f} -> "
        f"{augmented_supply/len(graphs):.3f} "
        f"(+{augmented_supply-current_supply} rows)",
        f"- selected rule: **{selected}**",
        f"- gate: **{'PASSED' if selected != 'never' else 'FAILED'}**",
        "",
        "| rule | mean delta | folds | actions | helpful | harmful | gate |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for rule in RULES[1:]:
        item = report[rule]
        lines.append(
            f"| {rule} | {item['mean_paired_delta']:+.3f} | "
            f"{list(item['fold_deltas'].values())} | {item['actions']} | "
            f"{item['helpful_actions']} | {item['harmful_actions']} | "
            f"{'PASS' if item['gate_passed'] else 'fail'} |")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--output-dir", required=True)
    root.add_argument("--graph-source", default=str(DEFAULT_GRAPH_SOURCE))
    return root


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
