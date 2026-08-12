#!/usr/bin/env python3
"""Replay the final decoder after removing Ministral's N=3 zero-shot route.

This is a paired, CPU-only development ablation.  It reuses the exact frozen
Qwen, Gemma, and Ministral CoT N=10 generations from a completed run, rebuilds
the candidate graph without ``ministral:self_consistency``, and applies the
same frozen decoder stack.  Removing the route at graph construction time
also disables the route's special unanimous-area replacement without changing
any other decoder stage.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents import end_to_end_pipeline as e2e
from experiments.heterogeneous_agents import final_submission_pipeline as final
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    augment_relational_graph,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT
    / "experiments/heterogeneous_agents/runs/stable_key_validation_20260810_v1"
)


def build_graph_without_route1(
    base: Mapping[str, Any],
    *,
    qwen_texts: Sequence[str],
    gemma_texts: Sequence[str],
    ministral_cot40: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct the production evidence graph with only Ministral CoT N=10."""
    graph = copy.deepcopy(dict(base))
    e2e._attach_supply_route(
        graph,
        ministral_cot40,
        route_name=e2e.MINISTRAL_COT40,
        samples=10,
    )
    graph.pop("relational_graph", None)
    graph.pop("relational_graph_schema", None)
    graph = augment_relational_graph(graph)

    final._replace_route_events(
        graph,
        route="qwen:self_consistency",
        family=e2e.QWEN,
        records=final._qwen_records(graph, qwen_texts),
        raw_texts=qwen_texts,
        provenance="frozen_split_generation",
    )
    final._replace_route_events(
        graph,
        route="gemma:independent",
        family=e2e.GEMMA,
        records=[final._generic_record(graph, text) for text in gemma_texts],
        raw_texts=gemma_texts,
        provenance="frozen_split_generation",
    )
    cot40_texts = [str(value) for value in ministral_cot40["generations"]]
    final._replace_route_events(
        graph,
        route=e2e.MINISTRAL_COT40,
        family=e2e.MINISTRAL,
        records=[final._generic_record(graph, text) for text in cot40_texts],
        raw_texts=cot40_texts,
        provenance="frozen_split_generation",
    )
    final._state_and_relation_edges(graph)
    graph["schema"] = final.GRAPH_SCHEMA
    graph["contains_labels"] = False
    graph["gold_aware"] = False
    return graph


def _objects(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[str]]:
    return {
        final._key(row): [str(value) for value in row["ObjectEntities"]]
        for row in rows
    }


def component_cot40_area(
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Apply the frozen 7/10 rule to typed numeric components, not surfaces."""
    if str(graph["Relation"]) != "hasArea":
        return list(incumbent), {"applied": False, "reason": "out_of_scope"}
    components: list[tuple[int, str]] = []
    for node in graph.get("relational_graph", {}).get("nodes", []):
        if node.get("node_type") != "candidate_component":
            continue
        route = node.get("routes", {}).get(e2e.MINISTRAL_COT40)
        if not isinstance(route, Mapping):
            continue
        components.append((
            int(route.get("distinct_generation_support", 0)),
            str(node["representative"]),
        ))
    if not components:
        return list(incumbent), {"applied": False, "reason": "no_components"}
    highest = max(support for support, _ in components)
    winners = [
        item for support, item in components
        if support == highest and support >= 7
    ]
    if len(winners) != 1:
        return list(incumbent), {
            "applied": False,
            "reason": "no_unique_7_of_10_component",
            "highest_support": highest,
            "winner_count": len(winners),
        }
    selected = [winners[0]]
    return selected, {
        "applied": selected != list(incumbent),
        "reason": "unique_7_of_10_numeric_component",
        "highest_support": highest,
        "selected": selected,
    }


def run(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    destination = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else source / "analysis/ministral_route1_ablation"
    )
    policy, model_paths = final._validate_policy(source)
    split = str(policy.get("split"))
    if split not in {"train", "validation"} or policy.get("blind"):
        raise ContractError(
            "route ablation requires a non-blind train or validation run")
    source_plan = e2e._validate_plan(source)

    primary, qwen_raw, system2 = final._primary_inputs(
        source, source_plan, policy)
    base_rows = final._assemble_from_primary(
        source, source_plan, primary, qwen_raw)
    gemma = final._response_map(source_plan, "gemma:independent")
    ministral_cot40 = final._response_map(
        source_plan, e2e.MINISTRAL_COT40)

    decision_graphs: list[dict[str, Any]] = []
    full_graphs: list[dict[str, Any]] = []
    for source_row in base_rows:
        key = final._key(source_row)
        base, qwen_texts, gemma_texts = final._prepare_base_row(
            source_row,
            {"generations": list(qwen_raw[key])},
            gemma[key],
            primary_objects=primary[key],
            system2_objects=system2.get(key, ()),
        )
        full = build_graph_without_route1(
            base,
            qwen_texts=qwen_texts,
            gemma_texts=gemma_texts,
            ministral_cot40=ministral_cot40[key],
        )
        if e2e.MINISTRAL_N3 in full.get("proposal_routes", {}):
            raise ContractError(f"route 1 survived graph construction: {key}")
        if any(
            e2e.MINISTRAL_N3 in node.get("routes", {})
            for node in full.get("candidates", [])
        ):
            raise ContractError(f"route 1 candidate evidence survived: {key}")
        decision_graphs.append(base)
        full_graphs.append(full)

    predictions, decisions = final._apply_frozen_stack(
        decision_graphs, full_graphs, model_paths)
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "PREDICTIONS_NO_MINISTRAL_N3.jsonl"
    decision_path = destination / "DECISIONS_NO_MINISTRAL_N3.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(decision_path, decisions)

    # The production route counts exact strings.  This second arm retains the
    # predeclared 7/10 threshold but reads the typed graph's complete-link 5%
    # numeric components, which is the representation used everywhere else in
    # the numeric graph.  Only hasArea is changed in this diagnostic arm.
    area_graphs = [
        graph for graph in full_graphs if graph["Relation"] == "hasArea"]
    area_incumbents: dict[tuple[str, str], list[str]] = {}
    component_details: dict[tuple[str, str], dict[str, Any]] = {}
    for graph, decision in zip(full_graphs, decisions, strict=True):
        if graph["Relation"] != "hasArea":
            continue
        key = final._key(graph)
        # The final layer's ``before`` value is the incumbent immediately
        # before exact-surface CoT40 admission.
        before = [str(value) for value in decision["layers"][-1]["before"]]
        objects, detail = component_cot40_area(graph, before)
        area_incumbents[key] = objects
        component_details[key] = detail
    area_predictions, area_proof = final._apply_relation_typed_graph_correction(
        area_graphs, area_incumbents)
    component_predictions = [dict(row) for row in predictions]
    index_by_key = {
        final._key(row): index
        for index, row in enumerate(component_predictions)
    }
    for row in area_predictions:
        component_predictions[index_by_key[final._key(row)]] = row
    component_path = destination / "PREDICTIONS_COMPONENT_COT40_AREA.jsonl"
    write_jsonl_atomic(component_path, component_predictions)

    control_path = source / "FINAL_PREDICTIONS.jsonl"
    control = read_jsonl(control_path)
    if len(control) != len(predictions):
        raise ContractError("control and ablation row counts differ")
    control_by = _objects(control)
    ablation_by = _objects(predictions)
    changed = [key for key in control_by if control_by[key] != ablation_by[key]]

    gold = Path(args.gold).resolve()
    control_scores = final._score(control, gold)
    ablation_scores = final._score(predictions, gold)
    component_scores = final._score(component_predictions, gold)
    gold_by = {
        final._key(row): row for row in read_jsonl(gold)
    }
    component_by = _objects(component_predictions)
    changed_component_area = [
        key for key in control_by
        if key[1] == "hasArea" and control_by[key] != component_by[key]
    ]
    component_effects: list[float] = []
    for key in changed_component_area:
        control_row = {
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": control_by[key],
        }
        component_row = {
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": component_by[key],
        }
        before = final.official.evaluate_per_sr_pair(
            [control_row], [gold_by[key]], final.official.RELATION_TYPE,
        )[0]["f1"]
        after = final.official.evaluate_per_sr_pair(
            [component_row], [gold_by[key]], final.official.RELATION_TYPE,
        )[0]["f1"]
        component_effects.append(float(after) - float(before))
    relation_deltas = {
        relation: (
            ablation_scores[relation]["macro-f1"]
            - control_scores[relation]["macro-f1"]
        )
        for relation in control_scores
    }
    area_changes = [key for key in changed if key[1] == "hasArea"]
    result = {
        "schema": "ministral-route1-removal-ablation-v1",
        "evaluation_split": split,
        "train_only": split == "train",
        "development_only": True,
        "gold_aware_evaluation": True,
        "policy_changed": "remove ministral:self_consistency (zero-shot N=3)",
        "policy_held_fixed": [
            "production Qwen incumbent",
            "Gemma independent route",
            "Ministral SyntheticCoT N=10 route",
            "all frozen decoder models and thresholds",
            "final relation-typed graph correction",
        ],
        "source": str(source),
        "source_predictions_sha256": sha256(control_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "decisions_sha256": sha256(decision_path),
        "rows": len(predictions),
        "changed_rows": len(changed),
        "changed_area_rows": len(area_changes),
        "control_pooled_macro_f1": control_scores[final.POOLED]["macro-f1"],
        "ablation_pooled_macro_f1": ablation_scores[final.POOLED]["macro-f1"],
        "pooled_delta": relation_deltas[final.POOLED],
        "control_hasArea_macro_f1": control_scores["hasArea"]["macro-f1"],
        "ablation_hasArea_macro_f1": ablation_scores["hasArea"]["macro-f1"],
        "hasArea_delta": relation_deltas["hasArea"],
        "component_cot40_area": {
            "description": (
                "remove N=3; apply the frozen 7/10 CoT rule to complete-link "
                "5% numeric graph components instead of exact surfaces"
            ),
            "predictions": str(component_path),
            "predictions_sha256": sha256(component_path),
            "pooled_macro_f1": component_scores[final.POOLED]["macro-f1"],
            "pooled_delta_vs_control": (
                component_scores[final.POOLED]["macro-f1"]
                - control_scores[final.POOLED]["macro-f1"]
            ),
            "hasArea_macro_f1": component_scores["hasArea"]["macro-f1"],
            "hasArea_delta_vs_control": (
                component_scores["hasArea"]["macro-f1"]
                - control_scores["hasArea"]["macro-f1"]
            ),
            "component_replacements": sum(
                bool(detail.get("applied"))
                for detail in component_details.values()
            ),
            "changed_area_rows_vs_control": len(changed_component_area),
            "helpful_area_rows": sum(value > 0 for value in component_effects),
            "harmful_area_rows": sum(value < 0 for value in component_effects),
            "neutral_area_rows": sum(value == 0 for value in component_effects),
            "proof_changed_rows": sum(
                bool(detail.get("changed")) for detail in area_proof
            ),
        },
        "relation_deltas": relation_deltas,
        "changed_keys": [list(key) for key in changed],
    }
    final._write_json(destination / "RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", default=str(DEFAULT_SOURCE))
    value.add_argument("--output-dir")
    value.add_argument("--gold", default=str(ROOT / "data/val.jsonl"))
    value.set_defaults(function=run)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.function(arguments))
