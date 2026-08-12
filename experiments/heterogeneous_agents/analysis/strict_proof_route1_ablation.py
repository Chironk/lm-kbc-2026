#!/usr/bin/env python3
"""Remove Ministral N=3 from the frozen 0.520729 development lineage.

The archived 0.520729 prediction consists of two independently scoped parts:

* the staged 0.518450 incumbent, whose N=3 Ministral rule is restricted to
  ``hasArea``; and
* the strict graph correction, whose validation graph was constructed with
  Qwen, Gemma, and only the N=10 SyntheticCoT Ministral route.  That final
  correction changed only stock-exchange and land-border rows.

This paired CPU-only ablation replaces the N=3 area stage with the already
frozen 7/10 rule over N=10 complete-link numeric components.  It then retains
the archived strict graph decisions on every non-area row.  The contracts
below prevent the composition from silently changing any other policy.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate as official

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
DEFAULT_REPRODUCTION = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "historical_sota_reproduction_20260810_v1"
)
DEFAULT_SAFE = (
    ROOT / "results/heterogeneous/candidates/frozen_20260803/"
    "safe_0_518450_validation.jsonl"
)
DEFAULT_STRICT = (
    ROOT / "results/heterogeneous/candidates/frozen_20260803/"
    "strict_proof_0_520729_validation.jsonl"
)
DEFAULT_GOLD = ROOT / "data/archive/validation_478_20260729.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "strict_proof_route1_ablation_20260810_v1"
)

POOLED = "*** All Relations ***"
AREA_RELATION = "hasArea"
N3_POLICY = "area_unanimous_new_component_replace"
COT40_ROUTE = "ministral:cot5_cap40_n10"
N3_ROUTE = "ministral:self_consistency"
COT40_SUPPORT = 7
ROWS = 478

# Pinned historical facts.  The strict validation graph was built by
# ``cot40_cardinality_validation_confirmation.py`` at research checkpoint
# 1a25798.  Its event parity contract required exactly ten events for
# ROUTE_MINISTRAL, whose value was ``ministral:cot5_cap40_n10``.  No N=3
# Ministral events entered that graph.
STRICT_GRAPH_SOURCE_COMMIT = "1a25798a5ae873fad8066ae1369f558bf6e1b73d"
STRICT_GRAPH_MINISTRAL_ROUTES = (COT40_ROUTE,)
STRICT_ALLOWED_CHANGED_RELATIONS = frozenset({
    "companyTradesAtStockExchange",
    "countryLandBordersCountry",
})


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def _by_key(
    rows: Sequence[Mapping[str, Any]], label: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {_key(row): row for row in rows}
    if len(result) != len(rows):
        raise ContractError(f"{label}: duplicate subject-relation key")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _score(
    predictions: Sequence[Mapping[str, Any]], gold_path: Path,
) -> dict[str, dict[str, float]]:
    return official.macro_average_per_relation(
        official.evaluate_per_sr_pair(
            list(predictions), read_jsonl(gold_path), official.RELATION_TYPE,
        )
    )


def _component_cot40_area(
    graph: Mapping[str, Any], incumbent: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Apply the frozen 7/10 threshold to 5%-linked numeric components."""
    components: list[tuple[int, str]] = []
    for node in graph.get("relational_graph", {}).get("nodes", []):
        if node.get("node_type") != "candidate_component":
            continue
        route = node.get("routes", {}).get(COT40_ROUTE)
        if not isinstance(route, Mapping):
            continue
        components.append((
            int(route.get("distinct_generation_support", 0)),
            str(node["representative"]),
        ))
    if not components:
        return list(incumbent), {
            "applied": False,
            "reason": "no_cot40_numeric_component",
        }
    highest = max(support for support, _ in components)
    winners = [
        item for support, item in components
        if support == highest and support >= COT40_SUPPORT
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


def _remove_n3_and_rebuild(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Delete N=3 proposal provenance before rebuilding graph components."""
    result = copy.deepcopy(dict(graph))
    result.get("proposal_routes", {}).pop(N3_ROUTE, None)
    evidence_routes = result.get("evidence_routes")
    if isinstance(evidence_routes, list):
        result["evidence_routes"] = [
            route for route in evidence_routes if str(route) != N3_ROUTE
        ]
    for field in ("candidates", "dormant_candidates"):
        retained = []
        for node in result.get(field, []):
            node.get("routes", {}).pop(N3_ROUTE, None)
            # A surface supplied solely by the removed route is not evidence
            # in the counterfactual graph.
            if node.get("routes"):
                retained.append(node)
        result[field] = retained
    result.pop("relational_graph", None)
    result.pop("relational_graph_schema", None)
    rebuilt = augment_relational_graph(result)
    if any(
        N3_ROUTE in node.get("routes", {})
        for node in rebuilt.get("relational_graph", {}).get("nodes", [])
    ):
        raise ContractError("N=3 route survived graph reconstruction")
    return rebuilt


def run(args: argparse.Namespace) -> int:
    reproduction = Path(args.reproduction).resolve()
    safe_path = Path(args.safe).resolve()
    strict_path = Path(args.strict).resolve()
    gold_path = Path(args.gold).resolve()
    output = Path(args.output_dir).resolve()

    safe = read_jsonl(safe_path)
    strict = read_jsonl(strict_path)
    graphs = read_jsonl(reproduction / "graph/UNIFIED_VALIDATION_GRAPH.jsonl")
    decisions = read_jsonl(reproduction / "DECISIONS.jsonl")
    reproduced = read_jsonl(reproduction / "VALIDATION_PREDICTIONS.jsonl")
    if not all(len(rows) == ROWS for rows in (
        safe, strict, graphs, decisions, reproduced,
    )):
        raise ContractError("expected 478 rows in every frozen artifact")
    if safe != reproduced:
        raise ContractError("historical reproduction is not byte-equivalent")

    safe_by = _by_key(safe, "safe")
    strict_by = _by_key(strict, "strict")
    graph_by = _by_key(graphs, "graph")
    decision_by = _by_key(decisions, "decisions")
    if not (set(safe_by) == set(strict_by) == set(graph_by) == set(decision_by)):
        raise ContractError("frozen artifact coverage differs")

    strict_changed = [
        key for key in safe_by
        if safe_by[key]["ObjectEntities"] != strict_by[key]["ObjectEntities"]
    ]
    if not strict_changed:
        raise ContractError("strict artifact contains no graph corrections")
    unexpected = {
        relation for _, relation in strict_changed
        if relation not in STRICT_ALLOWED_CHANGED_RELATIONS
    }
    if unexpected:
        raise ContractError(
            f"strict graph changed a route-1-sensitive relation: {unexpected}")

    route_free_area: dict[tuple[str, str], list[str]] = {}
    area_details: dict[tuple[str, str], dict[str, Any]] = {}
    for key, graph in graph_by.items():
        if key[1] != AREA_RELATION:
            continue
        layers = list(decision_by[key]["layers"])
        n3_layers = [layer for layer in layers if layer["policy"] == N3_POLICY]
        if len(n3_layers) != 1:
            raise ContractError(f"{key}: expected one historical N=3 layer")
        # The input to the removed N=3 rule is the retained staged incumbent.
        incumbent = [str(value) for value in n3_layers[0]["before"]]
        route_free_graph = _remove_n3_and_rebuild(graph)
        selected, detail = _component_cot40_area(route_free_graph, incumbent)
        route_free_area[key] = selected
        area_details[key] = detail

    predictions: list[dict[str, Any]] = []
    for row in strict:
        key = _key(row)
        objects = (
            route_free_area[key]
            if key[1] == AREA_RELATION
            else [str(value) for value in row["ObjectEntities"]]
        )
        predictions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": objects,
        })

    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "VALIDATION_PREDICTIONS_NO_MINISTRAL_N3.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    scores = _score(predictions, gold_path)
    safe_scores = _score(safe, gold_path)
    strict_scores = _score(strict, gold_path)

    prediction_by = _by_key(predictions, "route-free predictions")
    changed_vs_strict = [
        key for key in strict_by
        if strict_by[key]["ObjectEntities"]
        != prediction_by[key]["ObjectEntities"]
    ]
    if any(key[1] != AREA_RELATION for key in changed_vs_strict):
        raise ContractError("route removal changed a non-area strict output")
    gold_by = _by_key(read_jsonl(gold_path), "gold")
    changed_effects: list[float] = []
    for key in changed_vs_strict:
        before = official.evaluate_per_sr_pair(
            [strict_by[key]], [gold_by[key]], official.RELATION_TYPE,
        )[0]["f1"]
        after = official.evaluate_per_sr_pair(
            [prediction_by[key]], [gold_by[key]], official.RELATION_TYPE,
        )[0]["f1"]
        changed_effects.append(float(after) - float(before))

    result = {
        "schema": "strict-proof-ministral-route1-ablation-v1",
        "development_only": True,
        "contains_labels": True,
        "gold_aware_evaluation": True,
        "rows": len(predictions),
        "removed_route": "ministral:self_consistency",
        "retained_ministral_routes": [COT40_ROUTE],
        "strict_graph_source_commit": STRICT_GRAPH_SOURCE_COMMIT,
        "strict_graph_ministral_routes": list(STRICT_GRAPH_MINISTRAL_ROUTES),
        "safe_prediction": str(safe_path),
        "safe_prediction_sha256": sha256(safe_path),
        "strict_prediction": str(strict_path),
        "strict_prediction_sha256": sha256(strict_path),
        "prediction": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "gold": str(gold_path),
        "gold_sha256": sha256(gold_path),
        "strict_graph_changed_rows": len(strict_changed),
        "strict_graph_changed_relations": sorted({
            relation for _, relation in strict_changed
        }),
        "route_free_area_component_replacements": sum(
            bool(detail.get("applied")) for detail in area_details.values()
        ),
        "changed_rows_vs_archived_strict": len(changed_vs_strict),
        "changed_relations_vs_archived_strict": sorted({
            relation for _, relation in changed_vs_strict
        }),
        "changed_row_outcomes_vs_archived_strict": {
            "helpful": sum(value > 0 for value in changed_effects),
            "harmful": sum(value < 0 for value in changed_effects),
            "neutral": sum(value == 0 for value in changed_effects),
        },
        "safe_pooled_macro_f1": safe_scores[POOLED]["macro-f1"],
        "archived_strict_pooled_macro_f1": strict_scores[POOLED]["macro-f1"],
        "route_free_pooled_macro_f1": scores[POOLED]["macro-f1"],
        "delta_vs_archived_strict": (
            scores[POOLED]["macro-f1"]
            - strict_scores[POOLED]["macro-f1"]
        ),
        "performance_unchanged": (
            scores[POOLED]["macro-f1"]
            == strict_scores[POOLED]["macro-f1"]
        ),
        "archived_strict_hasArea_macro_f1": (
            strict_scores[AREA_RELATION]["macro-f1"]
        ),
        "route_free_hasArea_macro_f1": scores[AREA_RELATION]["macro-f1"],
        "hasArea_delta_vs_archived_strict": (
            scores[AREA_RELATION]["macro-f1"]
            - strict_scores[AREA_RELATION]["macro-f1"]
        ),
        "per_relation": scores,
    }
    result_path = output / "RESULT.json"
    _write_json(result_path, result)
    (output / "RESULT.md").write_text(
        "# Frozen strict-graph Ministral route ablation\n\n"
        "The zero-shot N=3 Ministral route is removed. The N=10 "
        "SyntheticCoT route supplies area components and was already the "
        "only Ministral route used by the strict graph correction.\n\n"
        f"- Archived strict macro-F1: **{strict_scores[POOLED]['macro-f1']:.6f}**\n"
        f"- Route-free macro-F1: **{scores[POOLED]['macro-f1']:.6f}**\n"
        f"- Delta: **{result['delta_vs_archived_strict']:+.6f}**\n"
        f"- Archived strict hasArea: **{strict_scores[AREA_RELATION]['macro-f1']:.6f}**\n"
        f"- Route-free hasArea: **{scores[AREA_RELATION]['macro-f1']:.6f}**\n"
        f"- Rows changed versus archived strict: **{len(changed_vs_strict)}**\n"
        f"- N=10 component replacements: **{result['route_free_area_component_replacements']}**\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--reproduction", default=str(DEFAULT_REPRODUCTION))
    value.add_argument("--safe", default=str(DEFAULT_SAFE))
    value.add_argument("--strict", default=str(DEFAULT_STRICT))
    value.add_argument("--gold", default=str(DEFAULT_GOLD))
    value.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    value.set_defaults(function=run)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.function(arguments))
