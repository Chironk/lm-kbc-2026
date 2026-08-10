#!/usr/bin/env python3
"""State-expanded training and cycle-safe walking over the unified graph.

The first unified selector saw only the deployed incumbent state for each
training row.  This module expands the same label-free graph into bounded
counterfactual answer states, then learns local edit existence and transition
utility from all of them.  Every original row has total training weight one,
regardless of its number of candidates, states, or legal actions.

No validation labels, per-relation thresholds, or relation-specific models are
used.  Nested subject-grouped OOF evaluation remains the deployment gate.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.relation_specific_structured_decoder import (
    _row_f1,
)
from experiments.heterogeneous_agents.unified_memory_action_graph import (
    DEFAULT_AGENTS,
    DEFAULT_GOLD,
    DEFAULT_GRAPH,
    FEATURE_NAMES,
    INNER_FOLDS,
    L2_GRID,
    OUTER_FOLDS,
    PARAMETER_CAP,
    RELATIONS,
    ROW_GATE_FEATURE_NAMES,
    RUNS,
    UnifiedSelector,
    WeightedLogistic,
    WeightedRidge,
    _agent_parameter_total,
    _canonical_objects,
    _deployment_gate,
    _key,
    _legal_actions,
    _oracle_predictions,
    _prediction_rows,
    _relation_deltas,
    _subset_gold,
    action_features,
    build_hierarchical_row,
    decode_one,
    grouped_relation_folds,
    row_gate_features,
)


DEFAULT_OUTPUT = RUNS / "walking_memory_graph_selector_20260727_v1"
MAX_TRAIN_STATES = 24
TRAIN_STATE_DEPTH = 2
MAX_WALK_STEPS = 3


def state_key(graph: Mapping[str, Any], objects: Sequence[str]) -> tuple[str, ...]:
    return _canonical_objects(objects, str(graph["Relation"]))


def state_view(
    graph: Mapping[str, Any], objects: Sequence[str],
) -> dict[str, Any]:
    """Return an inference-legal view rooted at a counterfactual answer state."""
    cache_key = state_key(graph, objects)
    cache = (
        graph.setdefault("_state_view_cache", {})
        if isinstance(graph, dict) else {})
    if cache_key in cache:
        return cache[cache_key]
    view = {
        key: value for key, value in graph.items()
        if key not in {"actions", "incumbent_objects", "_state_view_cache"}
    }
    view["incumbent_objects"] = [str(item) for item in objects]
    view["actions"] = _legal_actions(
        graph["_source"], view["incumbent_objects"])
    cache[cache_key] = view
    return view


def _action_priority(
    graph: Mapping[str, Any], action: Mapping[str, Any],
) -> tuple[float, int, tuple[str, ...]]:
    features = action_features(graph, action)
    index = {name: offset for offset, name in enumerate(FEATURE_NAMES)}
    evidence = (
        features[index["added_qwen_support"]]
        + features[index["added_gemma_support"]]
        + features[index["removed_qwen_support"]]
        + features[index["removed_gemma_support"]]
    )
    action_order = {
        "COLLAPSE": 5, "EMPTY": 4, "REPLACE": 3,
        "ADD": 2, "DROP": 1, "KEEP": 0,
    }
    return (
        evidence,
        action_order[str(action["action_type"])],
        state_key(graph, action["objects"]),
    )


def expand_states(
    graph: Mapping[str, Any], *, depth: int = TRAIN_STATE_DEPTH,
    max_states: int = MAX_TRAIN_STATES,
) -> list[dict[str, Any]]:
    """Bounded label-free BFS over unique answer states."""
    if depth < 0 or max_states < 1:
        raise ValueError("invalid state expansion bounds")
    root = state_view(graph, graph["incumbent_objects"])
    output = [root]
    seen = {state_key(root, root["incumbent_objects"])}
    frontier = [(root, 0)]
    while frontier and len(output) < max_states:
        current, current_depth = frontier.pop(0)
        if current_depth >= depth:
            continue
        alternatives = [
            action for action in current["actions"]
            if action["action_type"] != "KEEP"]
        alternatives.sort(
            key=lambda action: _action_priority(current, action),
            reverse=True)
        for action in alternatives:
            key = state_key(current, action["objects"])
            if key in seen:
                continue
            seen.add(key)
            child = state_view(graph, action["objects"])
            output.append(child)
            frontier.append((child, current_depth + 1))
            if len(output) >= max_states:
                break
    return output


def state_training_arrays(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[
    list[list[float]], list[float], list[float],
    list[list[float]], list[float], list[float],
    dict[str, Any],
]:
    """Build row-balanced action and edit-gate supervision."""
    action_x, action_y, action_weights = [], [], []
    gate_x, gate_y, gate_weights = [], [], []
    state_counts, action_counts = [], []
    for graph in graphs:
        states = expand_states(graph)
        state_counts.append(len(states))
        state_weight = 1.0 / len(states)
        gold = gold_by[_key(graph)]
        relation = str(graph["Relation"])
        for state in states:
            baseline = _row_f1(
                state["incumbent_objects"], gold, relation)
            deltas = [
                _row_f1(action["objects"], gold, relation) - baseline
                for action in state["actions"]
            ]
            action_counts.append(len(deltas))
            action_weight = state_weight / len(deltas)
            for action, delta in zip(state["actions"], deltas):
                action_x.append(action_features(state, action))
                action_y.append(delta)
                action_weights.append(action_weight)
            gate_x.append(row_gate_features(state))
            gate_y.append(float(max(deltas) > 1e-12))
            gate_weights.append(state_weight)
    diagnostics = {
        "rows": len(graphs),
        "states": sum(state_counts),
        "actions": sum(action_counts),
        "mean_states_per_row": statistics.mean(state_counts),
        "max_states_per_row": max(state_counts, default=0),
        "mean_actions_per_state": statistics.mean(action_counts),
        "row_action_weight_range": _row_weight_range(
            graphs, action_weights, state_counts, action_counts),
    }
    return (
        action_x, action_y, action_weights,
        gate_x, gate_y, gate_weights,
        diagnostics,
    )


def prepare_row_examples(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Prepare each row once so nested fits only concatenate cached arrays."""
    prepared, totals = {}, Counter()
    for graph in graphs:
        arrays = state_training_arrays([graph], gold_by)
        prepared[_key(graph)] = {
            "action_x": arrays[0],
            "action_y": arrays[1],
            "action_weights": arrays[2],
            "gate_x": arrays[3],
            "gate_y": arrays[4],
            "gate_weights": arrays[5],
            "diagnostics": arrays[6],
        }
        totals.update({
            "rows": 1,
            "states": arrays[6]["states"],
            "actions": arrays[6]["actions"],
        })
    diagnostics = {
        "rows": totals["rows"],
        "states": totals["states"],
        "actions": totals["actions"],
        "mean_states_per_row": totals["states"] / max(totals["rows"], 1),
        "mean_actions_per_state": (
            totals["actions"] / max(totals["states"], 1)),
        "max_states_per_row": max(
            value["diagnostics"]["states"] for value in prepared.values()),
        "row_action_weight_range": [
            min(sum(value["action_weights"]) for value in prepared.values()),
            max(sum(value["action_weights"]) for value in prepared.values()),
        ],
    }
    return prepared, diagnostics


def _row_weight_range(
    graphs: Sequence[Mapping[str, Any]], weights: Sequence[float],
    state_counts: Sequence[int], action_counts: Sequence[int],
) -> list[float]:
    totals, weight_offset, action_offset = [], 0, 0
    for _, state_count in zip(graphs, state_counts):
        total = 0.0
        for count in action_counts[action_offset:action_offset + state_count]:
            total += sum(weights[weight_offset:weight_offset + count])
            weight_offset += count
        action_offset += state_count
        totals.append(total)
    if weight_offset != len(weights) or action_offset != len(action_counts):
        raise AssertionError("row action weight accounting drift")
    return [min(totals, default=0.0), max(totals, default=0.0)]


def fit_state_expanded(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]], l2: float,
    prepared: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[UnifiedSelector, dict[str, Any]]:
    if prepared is None:
        prepared, diagnostics = prepare_row_examples(graphs, gold_by)
    else:
        subset = [prepared[_key(graph)] for graph in graphs]
        diagnostics = {
            "rows": len(subset),
            "states": sum(item["diagnostics"]["states"] for item in subset),
            "actions": sum(item["diagnostics"]["actions"] for item in subset),
            "mean_states_per_row": statistics.mean(
                item["diagnostics"]["states"] for item in subset),
            "max_states_per_row": max(
                item["diagnostics"]["states"] for item in subset),
            "mean_actions_per_state": (
                sum(item["diagnostics"]["actions"] for item in subset)
                / sum(item["diagnostics"]["states"] for item in subset)),
            "row_action_weight_range": [
                min(sum(item["action_weights"]) for item in subset),
                max(sum(item["action_weights"]) for item in subset),
            ],
        }
    subset = [prepared[_key(graph)] for graph in graphs]
    action_x = [
        row for item in subset for row in item["action_x"]]
    action_y = [
        value for item in subset for value in item["action_y"]]
    action_weights = [
        value for item in subset for value in item["action_weights"]]
    gate_x = [row for item in subset for row in item["gate_x"]]
    gate_y = [value for item in subset for value in item["gate_y"]]
    gate_weights = [
        value for item in subset for value in item["gate_weights"]]
    action = WeightedRidge(l2).fit(
        action_x, action_y, action_weights)
    gate = WeightedLogistic(l2).fit(gate_x, gate_y, gate_weights)
    return UnifiedSelector(action, gate), diagnostics


def walk_with_chooser(
    graph: Mapping[str, Any],
    chooser: Callable[
        [Mapping[str, Any]], tuple[list[str], Mapping[str, Any]]],
    *, max_steps: int = MAX_WALK_STEPS,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Cycle-safe graph walk, exposed separately for deterministic testing."""
    current = list(graph["incumbent_objects"])
    seen = {state_key(graph, current)}
    trace: list[dict[str, Any]] = []
    for step in range(max_steps):
        view = state_view(graph, current)
        objects, raw_detail = chooser(view)
        detail = dict(raw_detail)
        detail["walk_step"] = step
        trace.append(detail)
        next_key = state_key(graph, objects)
        if (
            detail.get("selected_action") == "KEEP"
            or next_key == state_key(graph, current)
            or next_key in seen
        ):
            detail["stop_reason"] = (
                "keep" if detail.get("selected_action") == "KEEP"
                else "cycle_or_no_change")
            break
        current = list(objects)
        seen.add(next_key)
    else:
        if trace:
            trace[-1]["stop_reason"] = "max_steps"
    return current, trace


def walk_decode_one(
    model: UnifiedSelector, graph: Mapping[str, Any],
    *, max_steps: int = MAX_WALK_STEPS,
) -> tuple[list[str], dict[str, Any]]:
    objects, trace = walk_with_chooser(
        graph, lambda view: decode_one(model, view),
        max_steps=max_steps)
    return objects, {
        "SubjectEntity": graph["SubjectEntity"],
        "Relation": graph["Relation"],
        "initial_objects": graph["incumbent_objects"],
        "selected_objects": objects,
        "steps": len(trace),
        "changed": state_key(graph, objects) != state_key(
            graph, graph["incumbent_objects"]),
        "trace": trace,
    }


def walk_prediction_rows(
    model: UnifiedSelector, graphs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics = [], []
    for graph in graphs:
        objects, detail = walk_decode_one(model, graph)
        predictions.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": objects,
        })
        diagnostics.append(detail)
    return predictions, diagnostics


def _choose_l2(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]], *, seed: int,
    prepared: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[float, dict[str, Any]]:
    folds = grouped_relation_folds(graphs, INNER_FOLDS, seed=seed)
    summaries = {}
    for l2 in L2_GRID:
        deltas = []
        for fold in range(INNER_FOLDS):
            fit_rows = [row for row in graphs if folds[_key(row)] != fold]
            hold_rows = [row for row in graphs if folds[_key(row)] == fold]
            model, _ = fit_state_expanded(
                fit_rows, gold_by, l2, prepared)
            predictions, _ = walk_prediction_rows(model, hold_rows)
            control = [{
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "ObjectEntities": row["incumbent_objects"],
            } for row in hold_rows]
            gold = _subset_gold(hold_rows, gold_by)
            deltas.append(
                score(predictions, gold)["*** All Relations ***"]
                - score(control, gold)["*** All Relations ***"])
        summaries[str(l2)] = {
            "fold_deltas": deltas,
            "mean_delta": statistics.mean(deltas),
        }
    best = max(
        L2_GRID,
        key=lambda value: (summaries[str(value)]["mean_delta"], value))
    return float(best), summaries


def _reachable_state_oracle(
    graphs: Sequence[Mapping[str, Any]],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    predictions = []
    for graph in graphs:
        relation = str(graph["Relation"])
        gold = gold_by[_key(graph)]
        states = expand_states(graph)
        best = max(
            states,
            key=lambda state: (
                _row_f1(state["incumbent_objects"], gold, relation),
                state_key(graph, state["incumbent_objects"])
                == state_key(graph, graph["incumbent_objects"]),
            ))
        predictions.append({
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": relation,
            "ObjectEntities": best["incumbent_objects"],
        })
    return predictions


def run_audit(args: argparse.Namespace) -> int:
    graph_path = Path(args.train_graph).resolve()
    gold_path = Path(args.train_gold).resolve()
    agents_path = Path(args.agents).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = read_jsonl(graph_path)
    gold_rows = read_jsonl(gold_path)
    gold_by = {_key(row): row for row in gold_rows}
    if len(raw) != len(gold_rows) or {_key(row) for row in raw} != set(gold_by):
        raise ContractError("walking selector graph/gold mismatch")
    graphs = [build_hierarchical_row(row) for row in raw]
    prepared, prepared_diagnostics = prepare_row_examples(graphs, gold_by)
    folds = grouped_relation_folds(graphs, OUTER_FOLDS, seed=args.seed)
    oof_by, diagnostics, fold_records, selected_l2 = {}, [], [], []
    for fold in range(OUTER_FOLDS):
        fit_rows = [row for row in graphs if folds[_key(row)] != fold]
        hold_rows = [row for row in graphs if folds[_key(row)] == fold]
        l2, inner = _choose_l2(
            fit_rows, gold_by, seed=args.seed + 1009 * (fold + 1),
            prepared=prepared)
        selected_l2.append(l2)
        model, expansion = fit_state_expanded(
            fit_rows, gold_by, l2, prepared)
        predictions, detail = walk_prediction_rows(model, hold_rows)
        for row in predictions:
            oof_by[_key(row)] = row
        diagnostics.extend([
            {**item, "outer_fold": fold, "l2": l2} for item in detail])
        control = [{
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": row["incumbent_objects"],
        } for row in hold_rows]
        hold_gold = _subset_gold(hold_rows, gold_by)
        selected_score = score(predictions, hold_gold)
        control_score = score(control, hold_gold)
        fold_records.append({
            "fold": fold,
            "rows": len(hold_rows),
            "selected_l2": l2,
            "inner_cv": inner,
            "training_expansion": expansion,
            "control_score": control_score["*** All Relations ***"],
            "selected_score": selected_score["*** All Relations ***"],
            "delta": selected_score["*** All Relations ***"]
            - control_score["*** All Relations ***"],
        })
    if set(oof_by) != {_key(row) for row in graphs}:
        raise ContractError("walking selector OOF coverage failure")
    predictions = [oof_by[_key(row)] for row in graphs]
    controls = [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": row["incumbent_objects"],
    } for row in graphs]
    oracle = _reachable_state_oracle(graphs, gold_by)
    selected_scores = score(predictions, gold_rows)
    control_scores = score(controls, gold_rows)
    oracle_scores = score(oracle, gold_rows)
    pooled_delta = (
        selected_scores["*** All Relations ***"]
        - control_scores["*** All Relations ***"])
    relation_deltas = _relation_deltas(selected_scores, control_scores)
    gate = _deployment_gate(
        pooled_delta, [item["delta"] for item in fold_records],
        relation_deltas)
    final_l2 = max(
        sorted(set(selected_l2)),
        key=lambda value: (selected_l2.count(value), value))
    final_model, final_expansion = fit_state_expanded(
        graphs, gold_by, final_l2, prepared)
    agent_parameters = _agent_parameter_total(agents_path)
    total_parameters = agent_parameters + final_model.parameter_count
    if total_parameters > PARAMETER_CAP:
        raise ContractError("walking selector exceeds parameter cap")
    write_jsonl_atomic(output / "FOLDS.jsonl", [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "fold": folds[_key(row)],
    } for row in graphs])
    write_jsonl_atomic(output / "TRAIN_OOF_PREDICTIONS.jsonl", predictions)
    (output / "TRAIN_OOF_PREDICTIONS.jsonl.manifest.json").write_text(
        json.dumps({
            "schema": "walking-memory-graph-oof-predictions-manifest-v1",
            "split": "train",
            "rows": len(predictions),
            "contains_labels": False,
            "gold_aware": True,
            "deployable": False,
            "oof_model_excludes_subject": True,
            "selection_uses_train_labels": True,
            "validation_labels_used": False,
            "output_sha256": sha256(
                output / "TRAIN_OOF_PREDICTIONS.jsonl"),
        }, indent=2, sort_keys=True) + "\n")
    write_jsonl_atomic(output / "TRAIN_OOF_DIAGNOSTICS.jsonl", diagnostics)
    (output / "MODEL.json").write_text(json.dumps({
        "schema": "state-expanded-walking-memory-selector-model-v1",
        "development_only": True,
        "train_labels_used": True,
        "validation_labels_used": False,
        "relation_specific_models": False,
        "relation_specific_thresholds": False,
        "max_train_states": MAX_TRAIN_STATES,
        "train_state_depth": TRAIN_STATE_DEPTH,
        "max_walk_steps": MAX_WALK_STEPS,
        "selected_outer_l2": selected_l2,
        "final_l2": final_l2,
        "training_expansion": final_expansion,
        "prepared_expansion": prepared_diagnostics,
        "model": final_model.to_dict(),
        "selector_parameter_count": final_model.parameter_count,
        "agent_parameter_upper_bound": agent_parameters,
        "combined_parameter_upper_bound": total_parameters,
        "parameter_cap": PARAMETER_CAP,
    }, indent=2, sort_keys=True) + "\n")
    helped = harmed = 0
    for graph in graphs:
        gold = gold_by[_key(graph)]
        relation = str(graph["Relation"])
        delta = (
            _row_f1(oof_by[_key(graph)]["ObjectEntities"], gold, relation)
            - _row_f1(graph["incumbent_objects"], gold, relation))
        helped += delta > 1e-12
        harmed += delta < -1e-12
    result = {
        "schema": "state-expanded-walking-memory-train-audit-v1",
        "development_only": True,
        "validation_labels_used": False,
        "rows": len(graphs),
        "control_scores": control_scores,
        "selected_scores": selected_scores,
        "reachable_state_oracle_scores": oracle_scores,
        "pooled_delta": pooled_delta,
        "relation_deltas": relation_deltas,
        "folds": fold_records,
        "deployment_gate": gate,
        "changed_rows": sum(item["changed"] for item in diagnostics),
        "helped_rows": helped,
        "harmed_rows": harmed,
        "walk_step_counts": dict(Counter(
            item["steps"] for item in diagnostics)),
        "selector_parameter_count": final_model.parameter_count,
        "combined_parameter_upper_bound": total_parameters,
        "artifacts": {
            "model": str(output / "MODEL.json"),
            "model_sha256": sha256(output / "MODEL.json"),
            "oof_predictions": str(output / "TRAIN_OOF_PREDICTIONS.jsonl"),
            "oof_predictions_sha256": sha256(
                output / "TRAIN_OOF_PREDICTIONS.jsonl"),
        },
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# State-expanded walking heterogeneous-memory selector", "",
        "Train-only nested subject-grouped OOF audit. Validation was not read.",
        "",
        f"- Control pooled F1: "
        f"**{control_scores['*** All Relations ***']:.9f}**",
        f"- Walking selector pooled F1: "
        f"**{selected_scores['*** All Relations ***']:.9f}**",
        f"- Delta: **{pooled_delta:+.9f}**",
        f"- Reachable bounded-state oracle: "
        f"**{oracle_scores['*** All Relations ***']:.9f}**",
        f"- Changed/helped/harmed: **{result['changed_rows']} / "
        f"{helped} / {harmed}**",
        f"- Broad deployment gate: **{gate['passed']}**",
        f"- Training states/actions: **{final_expansion['states']} / "
        f"{final_expansion['actions']}**",
        f"- Learned parameters: **{final_model.parameter_count}**", "",
        "## Relation deltas", "",
        "| relation | control | selector | delta |",
        "|---|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        lines.append(
            f"| {relation} | {control_scores[relation]:.6f} | "
            f"{selected_scores[relation]:.6f} | "
            f"{relation_deltas[relation]:+.6f} |")
    lines.extend(["", "## Outer folds", "",
                  "| fold | rows | L2 | delta |",
                  "|---:|---:|---:|---:|"])
    for item in fold_records:
        lines.append(
            f"| {item['fold']} | {item['rows']} | "
            f"{item['selected_l2']:.1f} | {item['delta']:+.6f} |")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "pooled_delta": pooled_delta,
        "gate_passed": gate["passed"],
        "helped": helped,
        "harmed": harmed,
        "training_states": final_expansion["states"],
        "training_actions": final_expansion["actions"],
        "output": str(output),
    }, indent=2))
    return 0


def run_decode(args: argparse.Namespace) -> int:
    """Decode a label-free graph only when the train-only gate passed."""
    model_dir = Path(args.model_dir).resolve()
    result_path = model_dir / "RESULT.json"
    model_path = model_dir / "MODEL.json"
    if not result_path.is_file() or not model_path.is_file():
        raise ContractError("missing walking selector train audit")
    result = json.loads(result_path.read_text())
    if not result.get("deployment_gate", {}).get("passed"):
        raise ContractError(
            "walking selector failed the broad train-only deployment gate")
    artifact = json.loads(model_path.read_text())
    if (
        artifact.get("schema")
        != "state-expanded-walking-memory-selector-model-v1"
        or artifact.get("validation_labels_used") is not False
        or artifact.get("relation_specific_models") is not False
        or artifact.get("relation_specific_thresholds") is not False
    ):
        raise ContractError("invalid walking selector model provenance")
    model = UnifiedSelector.from_dict(artifact["model"])
    graph_path = Path(args.graph).resolve()
    output_path = Path(args.output).resolve()
    graphs = [
        build_hierarchical_row(row) for row in read_jsonl(graph_path)]
    predictions, diagnostics = walk_prediction_rows(model, graphs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_path, predictions)
    diagnostics_path = output_path.with_name(
        output_path.stem + ".diagnostics.jsonl")
    write_jsonl_atomic(diagnostics_path, diagnostics)
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps({
            "schema": "walking-memory-graph-predictions-manifest-v1",
            "rows": len(predictions),
            "contains_labels": False,
            "gold_aware": False,
            "validation_labels_used": False,
            "train_gate_passed": True,
            "source_graph": str(graph_path),
            "source_graph_sha256": sha256(graph_path),
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "output_sha256": sha256(output_path),
        }, indent=2, sort_keys=True) + "\n")
    print(f"walking predictions frozen: {output_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("train-audit")
    audit.add_argument("--train-graph", default=str(DEFAULT_GRAPH))
    audit.add_argument("--train-gold", default=str(DEFAULT_GOLD))
    audit.add_argument("--agents", default=str(DEFAULT_AGENTS))
    audit.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    audit.add_argument("--seed", type=int, default=20260727)
    audit.set_defaults(func=run_audit)
    decode = sub.add_parser("decode")
    decode.add_argument("--model-dir", default=str(DEFAULT_OUTPUT))
    decode.add_argument(
        "--graph",
        default=str(
            RUNS / "targeted_company_gemma_n3_20260724_v1/"
            "graphs/validation_graph.jsonl"))
    decode.add_argument(
        "--output", default=str(DEFAULT_OUTPUT / "PREDICTIONS.jsonl"))
    decode.set_defaults(func=run_decode)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
