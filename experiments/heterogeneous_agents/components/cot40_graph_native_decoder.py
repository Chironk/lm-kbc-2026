#!/usr/bin/env python3
"""Train-only graph-native decoding of the frozen Ministral CoT40 N=10 run.

This experiment closes a specific architectural gap: the promoted CoT40
residual policy counted parsed strings outside the typed graph.  Here every
parsed generation is written back as route-provenanced candidate evidence,
components are rebuilt, and complete answer-set actions are scored.

The decisive ablation is matched:

* ``component_local`` sees the same components, local route support, actions,
  folds, targets, and ridge family, but no inter-component topology.
* ``typed_edges`` adds only features obtained by traversing typed graph edges:
  co-support, same-generation co-occurrence, contradiction, equivalence, and
  route-dependence structure.

Preparation is label-free.  Analysis opens training labels only and performs
nested, subject-grouped cross-validation.  There is intentionally no
validation command; a separate confirmation may be created only if the
predeclared train gate passes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluate import try_parse_number
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.components.baseline_relative_route_decoder import (
    ResidualRidge,
)
from experiments.heterogeneous_agents.components.component_aware_decoder import (
    _action_tokens,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    proposal_parse_status,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import _key
from experiments.heterogeneous_agents.components.ministral_cot40_training import (
    ARM_NAME,
    N_MAX,
    PLAN_SCHEMA as COT40_PLAN_SCHEMA,
)
from experiments.heterogeneous_agents.components.ministral_candidate_supply import (
    MINISTRAL,
)
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    _row_f1,
)
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    LIST_RELATIONS,
    NUMERIC_RELATIONS,
    SINGLE_RELATIONS,
    augment_relational_graph,
    collapse_prediction,
)
from experiments.heterogeneous_agents.components.sota_pipeline import (
    compose_competition_train_oof,
)
from experiments.heterogeneous_agents.components.three_model_component_decoder import (
    subject_grouped_folds,
)


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
DEFAULT_OUTPUT = RUNS / "cot40_graph_native_decoder_20260729_v1"
DEFAULT_COT40 = RUNS / "ministral_cot40_training_20260729_v1"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
ROUTE = "ministral:cot5_cap40_n10"
ARMS = ("component_local", "typed_edges")
RELATIONS = tuple(sorted(LIST_RELATIONS | SINGLE_RELATIONS | NUMERIC_RELATIONS))
ACTION_TYPES = ("KEEP", "EMPTY", "REPLACE", "ADD", "DROP", "GRAPH_SET")
# Fixed before analysis.  A one-value regularization contract removes an
# unnecessary inner hyperparameter search and makes the edge ablation harder
# to overfit; inner folds select only the global edit guard.
L2_VALUES = (10.0,)
GUARDS = (0.0, 0.02, 0.05, 0.10, 0.20)
SUPPORT_THRESHOLDS = (0.2, 0.3, 0.5, 0.7)
POOLED = "*** All Relations ***"
PLAN_SCHEMA = "cot40-graph-native-decoder-plan-v1"
GRAPH_SCHEMA = "cot40-enriched-typed-graph-manifest-v1"
RESULT_SCHEMA = "cot40-graph-native-decoder-result-v1"

# A graph result is promotable only when topology helps beyond an otherwise
# identical component-local decoder.  This prevents relabeling an ordinary
# candidate ranker as a graph contribution.
MIN_EDGE_INCREMENT = 0.003
MIN_EDGE_FOLD_WINS = 3
MAX_RELATION_REGRESSION = -0.01


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _response_map(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if (
            row.get("agent_id") != MINISTRAL
            or row.get("phase") != "propose"
            or not isinstance(row.get("generations"), list)
            or len(row["generations"]) != N_MAX
        ):
            raise ContractError("invalid frozen CoT40 proposal response")
        key = (str(row["subject"]), str(row["relation"]))
        if key in result:
            raise ContractError(f"duplicate CoT40 response: {key}")
        result[key] = row
    if len(result) != 477:
        raise ContractError(f"expected 477 CoT40 responses, got {len(result)}")
    return result


def _validate_cot40_inputs(cot40: Path) -> dict[str, Any]:
    """Validate only artifacts consumed here, while preserving hash history.

    The frozen CoT40 plan also pins ``sota_pipeline.py``.  That file later
    received a repository-path canonicalization with unchanged registered
    prediction hashes.  Requiring the historical implementation byte hash
    would make the immutable responses unusable even though this experiment
    consumes neither that implementation nor its path strings.  We continue
    to fail closed on the source graph, tasks, config, and response bytes.
    """
    path = cot40 / "plan/PLAN.json"
    plan = _json(path)
    if (
        plan.get("schema") != COT40_PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or plan.get("n_samples") != N_MAX
        or plan.get("responses") is None
        or plan.get("source_graph") is None
    ):
        raise ContractError("invalid frozen CoT40 input plan")
    for field in (
        "source_graph", "tasks", "config", "input_rows",
        "synthetic_cot",
    ):
        expected = plan.get(f"{field}_sha256")
        if (
            not isinstance(expected, str)
            or sha256(Path(plan[field])) != expected
        ):
            raise ContractError(f"CoT40 artifact changed: {field}")
    response_path = Path(plan["responses"])
    manifest_path = response_path.with_suffix(
        response_path.suffix + ".manifest.json")
    manifest = _json(manifest_path)
    if (
        manifest.get("output_sha256") != sha256(response_path)
        or manifest.get("agent_id") != MINISTRAL
        or manifest.get("task_sha256") != plan["tasks_sha256"]
    ):
        raise ContractError("CoT40 response manifest mismatch")
    return plan


def _surface_occurrences(
    generations: Sequence[str], relation: str,
) -> tuple[dict[str, dict[str, Any]], list[int]]:
    """Return exact typed surfaces with distinct-generation provenance."""
    values: dict[str, dict[str, Any]] = {}
    none_generations: list[int] = []
    for generation_index, generation in enumerate(generations):
        status, items = proposal_parse_status(str(generation), relation)
        if status == "explicit_none":
            none_generations.append(generation_index)
        seen: set[str] = set()
        for item in items:
            key = canonical_key(str(item), relation)
            if not key or key in seen:
                continue
            seen.add(key)
            current = values.setdefault(key, {
                "key": key,
                "item": str(item),
                "generation_indices": [],
            })
            current["generation_indices"].append(generation_index)
            # Prefer a concise surface while retaining typed identity.
            if len(str(item)) < len(str(current["item"])):
                current["item"] = str(item)
    return values, none_generations


def _attach_route(
    candidate: dict[str, Any], *, generations: Sequence[int],
) -> None:
    generation_indices = sorted({int(value) for value in generations})
    route = {
        "model_family": MINISTRAL,
        "route_type": "independent-cot-recall",
        "support": len(generation_indices),
        "samples": N_MAX,
        "support_rate": len(generation_indices) / N_MAX,
        "selected": len(generation_indices) >= 7,
        "generation_indices": generation_indices,
        "admission_reason": "complete_cot40_candidate_reservoir",
    }
    candidate.setdefault("routes", {})[ROUTE] = route
    candidate.setdefault("selected_by", {})[MINISTRAL] = route["selected"]
    candidate.setdefault("sources", {})[MINISTRAL] = {
        "support": route["support"],
        "samples": N_MAX,
        "support_rate": route["support_rate"],
    }
    candidate["output_eligible"] = True


def enrich_row(
    source: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Insert all CoT40 evidence into the typed graph without labels."""
    row = copy.deepcopy(source)
    key = _key(row)
    if key != (str(response["subject"]), str(response["relation"])):
        raise ContractError("source/CoT40 row mismatch")
    relation = key[1]
    occurrences, none_generations = _surface_occurrences(
        response["generations"], relation)
    candidates = list(row.get("candidates", []))
    exact: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_key = canonical_key(str(candidate["item"]), relation)
        if candidate_key and candidate_key not in exact:
            exact[candidate_key] = candidate
    for candidate_key, value in occurrences.items():
        candidate = exact.get(candidate_key)
        if candidate is None:
            candidate = {
                "key": candidate_key,
                "item": str(value["item"]),
                "type": (
                    "numeric" if relation in NUMERIC_RELATIONS else "string"
                ),
                "routes": {},
                "sources": {},
                "selected_by": {},
                "output_eligible": True,
            }
            candidates.append(candidate)
            exact[candidate_key] = candidate
        _attach_route(
            candidate, generations=value["generation_indices"])
    row["candidates"] = candidates
    row.setdefault("proposal_routes", {})[ROUTE] = {
        "available": True,
        "model_family": MINISTRAL,
        "n_samples": N_MAX,
        "route_type": "independent-cot-recall",
        "generation_provenance_available": True,
    }
    row.setdefault("agents", {})[MINISTRAL] = {
        "candidate_supply_only": True,
        "n_samples": N_MAX,
        "none_count": len(none_generations),
        "none_rate": len(none_generations) / N_MAX,
        "existence": {"available": False},
        "cardinality": {"available": False},
    }
    row.setdefault("agent_outputs", {})[MINISTRAL] = [
        str(value["item"])
        for value in occurrences.values()
        if len(value["generation_indices"]) >= 7
    ]
    row["cot40_generation_provenance"] = {
        "route": ROUTE,
        "samples": N_MAX,
        "none_generation_indices": none_generations,
        "parsed_candidate_count": len(occurrences),
    }
    row.pop("relational_graph", None)
    row.pop("relational_graph_schema", None)
    return augment_relational_graph(row)


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    cot40 = Path(args.cot40_run).resolve()
    plan = _validate_cot40_inputs(cot40)
    source_path = Path(plan["source_graph"]).resolve()
    response_path = Path(plan["responses"]).resolve()
    source_rows = read_jsonl(source_path)
    responses = _response_map(response_path)
    if len(source_rows) != 477:
        raise ContractError("expected complete 477-row source graph")
    graph_rows = []
    for source in source_rows:
        key = _key(source)
        if key not in responses:
            raise ContractError(f"missing CoT40 response: {key}")
        graph_rows.append(enrich_row(source, responses[key]))
    if len({_key(row) for row in graph_rows}) != 477:
        raise ContractError("enriched graph has duplicate rows")

    graph_path = output / "graph/COT40_TYPED_GRAPH.jsonl"
    write_jsonl_atomic(graph_path, graph_rows)
    graph_manifest = {
        "schema": GRAPH_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "split": "train",
        "rows": len(graph_rows),
        "route": ROUTE,
        "n_samples": N_MAX,
        "complete_candidate_reservoir": True,
        "generation_provenance_available": True,
        "output": str(graph_path),
        "output_sha256": sha256(graph_path),
        "source_graph": str(source_path),
        "source_graph_sha256": sha256(source_path),
        "responses": str(response_path),
        "responses_sha256": sha256(response_path),
    }
    _write_json(
        graph_path.with_suffix(graph_path.suffix + ".manifest.json"),
        graph_manifest,
    )
    experiment_plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "split": "train",
        "arms": list(ARMS),
        "matched_ablation":
            "identical_components_actions_targets_model; typed_edges adds "
            "only traversed inter-component edge features",
        "nested_subject_grouped_cross_validation": True,
        "l2_values": list(L2_VALUES),
        "guards": list(GUARDS),
        "promotion_gate": {
            "minimum_edge_increment": MIN_EDGE_INCREMENT,
            "minimum_edge_fold_wins": MIN_EDGE_FOLD_WINS,
            "maximum_relation_regression": MAX_RELATION_REGRESSION,
            "typed_edge_score_must_exceed_incumbent": True,
            "helpful_edits_must_exceed_harmful": True,
        },
        "cot40_run": str(cot40),
        "cot40_plan": str(cot40 / "plan/PLAN.json"),
        "cot40_plan_sha256": sha256(cot40 / "plan/PLAN.json"),
        "historical_incumbent_implementation_sha256":
            plan.get("incumbent_implementation_sha256"),
        "current_incumbent_implementation_sha256": sha256(Path(
            plan["incumbent_implementation"])),
        "historical_incumbent_hash_mismatch_is_path_canonicalization":
            plan.get("incumbent_implementation_sha256") != sha256(Path(
                plan["incumbent_implementation"])),
        "source_graph": str(source_path),
        "source_graph_sha256": sha256(source_path),
        "responses": str(response_path),
        "responses_sha256": sha256(response_path),
        "graph": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    plan_path = output / "plan/PLAN.json"
    _write_json(plan_path, experiment_plan)
    print(json.dumps({
        "plan": str(plan_path),
        "graph": str(graph_path),
        "rows": len(graph_rows),
        "validation_opened": False,
        "edge_types": sorted({
            edge["edge_type"]
            for row in graph_rows
            for edge in row["relational_graph"]["edges"]
        }),
    }, indent=2, sort_keys=True))
    return 0


def _validate_prepared(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or tuple(plan.get("arms", [])) != ARMS
    ):
        raise ContractError("invalid graph-native plan")
    for field in ("cot40_plan", "source_graph", "responses", "graph"):
        if sha256(Path(plan[field])) != plan[f"{field}_sha256"]:
            raise ContractError(f"frozen graph-native input changed: {field}")
    manifest = _json(
        Path(plan["graph"]).with_suffix(
            Path(plan["graph"]).suffix + ".manifest.json")
    )
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("contains_labels") is not False
        or manifest.get("validation_opened") is not False
        or manifest.get("output_sha256") != plan["graph_sha256"]
        or manifest.get("generation_provenance_available") is not True
    ):
        raise ContractError("invalid CoT40 typed graph manifest")
    graphs = read_jsonl(Path(plan["graph"]))
    if len(graphs) != 477 or len({_key(row) for row in graphs}) != 477:
        raise ContractError("typed graph coverage mismatch")
    return plan, graphs


def _component_id_map(graph: Mapping[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for component in graph["relational_graph"]["components"]:
        component_id = str(component["id"])
        for member in component.get("member_items", []):
            mapping[canonical_key(str(member), str(graph["Relation"]))] = (
                component_id
            )
        mapping[canonical_key(
            str(component["representative"]), str(graph["Relation"])
        )] = component_id
    return mapping


def _component_ids(
    graph: Mapping[str, Any], objects: Sequence[str],
) -> set[str]:
    mapping = _component_id_map(graph)
    relation = str(graph["Relation"])
    result: set[str] = set()
    for item in collapse_prediction(graph, objects):
        key = canonical_key(str(item), relation)
        if key in mapping:
            result.add(mapping[key])
    return result


def _objects_for_ids(
    graph: Mapping[str, Any], component_ids: Iterable[str],
) -> list[str]:
    wanted = set(component_ids)
    return [
        str(component["representative"])
        for component in graph["relational_graph"]["components"]
        if str(component["id"]) in wanted
    ]


def legal_actions(
    graph: Mapping[str, Any], incumbent: Sequence[str],
) -> list[dict[str, Any]]:
    """Enumerate bounded complete answer-set actions from graph structure."""
    cache_key = tuple(str(value) for value in incumbent)
    cached = graph.get("_graph_native_action_cache")
    if (
        isinstance(cached, Mapping)
        and tuple(cached.get("incumbent", ())) == cache_key
    ):
        return list(cached["actions"])
    relation = str(graph["Relation"])
    components = list(graph["relational_graph"]["components"])
    component_ids = [str(value["id"]) for value in components]
    incumbent_collapsed = collapse_prediction(graph, incumbent)
    incumbent_ids = _component_ids(graph, incumbent_collapsed)
    raw: list[tuple[str, list[str]]] = [("KEEP", list(incumbent))]
    if _action_tokens(
        graph, incumbent_collapsed, "component"
    ) != _action_tokens(graph, incumbent, "component"):
        raw.append(("GRAPH_SET", incumbent_collapsed))
    if relation not in NUMERIC_RELATIONS:
        raw.append(("EMPTY", []))
    if relation in SINGLE_RELATIONS | NUMERIC_RELATIONS:
        raw.extend(
            ("REPLACE", [str(component["representative"])])
            for component in components
        )
    else:
        for component in components:
            component_id = str(component["id"])
            representative = str(component["representative"])
            if component_id not in incumbent_ids:
                raw.append((
                    "ADD",
                    [*incumbent_collapsed, representative],
                ))
        for component_id in sorted(incumbent_ids):
            raw.append((
                "DROP",
                _objects_for_ids(graph, incumbent_ids - {component_id}),
            ))

        # Set-level graph actions: support contours and same-generation
        # Ministral subgraphs.  These allow multi-add/multi-drop decisions
        # without enumerating the exponential power set.
        for threshold in SUPPORT_THRESHOLDS:
            selected = {
                str(component["id"])
                for component in components
                if max(
                    (
                        float(route.get(
                            "component_support_rate",
                            route.get("max_support_rate", 0.0),
                        ))
                        for route in component.get("routes", {}).values()
                    ),
                    default=0.0,
                ) >= threshold
            }
            if selected:
                raw.append(("GRAPH_SET", _objects_for_ids(graph, selected)))
                raw.append((
                    "GRAPH_SET",
                    _objects_for_ids(graph, selected | incumbent_ids),
                ))
        for generation in range(N_MAX):
            selected = {
                str(component["id"])
                for component in components
                if generation in component.get("routes", {}).get(
                    ROUTE, {}).get("generation_indices", [])
            }
            if selected:
                raw.append(("GRAPH_SET", _objects_for_ids(graph, selected)))
                raw.append((
                    "GRAPH_SET",
                    _objects_for_ids(graph, selected | incumbent_ids),
                ))

    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for action_type, objects in raw:
        collapsed = collapse_prediction(
            graph, list(dict.fromkeys(str(value) for value in objects)))
        key = tuple(_action_tokens(graph, collapsed, "component"))
        existing = unique.get(key)
        if existing is None or action_type == "KEEP":
            unique[key] = {
                "action_type": action_type,
                "objects": list(objects) if action_type == "KEEP" else collapsed,
                "component_ids": sorted(_component_ids(graph, collapsed)),
            }
    actions = list(unique.values())
    if sum(value["action_type"] == "KEEP" for value in actions) != 1:
        raise ContractError("action inventory lacks exactly one KEEP")
    # Analysis-only in-memory cache.  It is never serialized into the
    # label-free graph artifact.
    graph["_graph_native_action_cache"] = {
        "incumbent": cache_key,
        "actions": actions,
    }
    return actions


def _route_family_map(graph: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(node["route"]): str(node.get("model_family", ""))
        for node in graph["relational_graph"]["nodes"]
        if node.get("node_type") == "evidence_route"
    }


def _support(
    component: Mapping[str, Any], family: str,
    route_families: Mapping[str, str],
) -> float:
    return max(
        (
            float(value.get(
                "component_support_rate",
                value.get("max_support_rate", 0.0),
            ))
            for route, value in component.get("routes", {}).items()
            if route_families.get(str(route)) == family
        ),
        default=0.0,
    )


def _summary(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    # ``values`` is often assembled from a set of component IDs.  Built-in
    # ``sum`` follows iteration order, so Python hash randomization can change
    # the final few bits, alter a tied inner-fold guard choice, and ultimately
    # change OOF edits.  ``math.fsum`` makes this aggregation stable.
    total = math.fsum(float(value) for value in values)
    return (
        max(values),
        total / len(values),
        min(values),
        min(1.0, total / max(1, len(values))),
    )


LOCAL_NAMES = (
    *[f"relation_{value}" for value in RELATIONS],
    *[f"action_{value.lower()}" for value in ACTION_TYPES],
    "incumbent_empty", "action_empty", "incumbent_size", "action_size",
    "size_delta", "edit_component_count", "component_count",
    *[
        f"{scope}_{family}_{stat}"
        for scope in ("selected", "added", "removed", "omitted")
        for family in ("qwen_recall", "gemma_independent", MINISTRAL)
        for stat in ("max", "mean", "min", "mass")
    ],
    "selected_independent_family_mean",
    "selected_independent_family_min",
    "ministral_none_rate",
    "qwen_none_rate",
    "gemma_none_rate",
    "numeric_log_distance",
)

EDGE_NAMES = (
    "internal_co_support_density",
    "internal_co_support_weight",
    "internal_same_generation_density",
    "internal_same_generation_rate",
    "co_support_boundary_density",
    "co_support_boundary_rate",
    "internal_contradiction_density",
    "contradiction_boundary_density",
    "null_selected_contradictions",
    "equivalence_edges_selected",
    "selected_alias_component_fraction",
    "selected_surface_per_component",
    "dependent_route_redundancy",
    "independence_discounted_support",
    "ministral_generation_coherence_mean",
    "ministral_generation_coherence_max",
    "ministral_generation_exact_set_rate",
    *[
        f"relation_{relation}_x_{feature}"
        for relation in RELATIONS
        for feature in (
            "same_generation_density",
            "co_support_weight",
            "contradiction_boundary",
        )
    ],
)


def local_features(
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    action: Mapping[str, Any],
) -> list[float]:
    cached = action.get("_component_local_features")
    if cached is not None:
        if len(cached) != len(LOCAL_NAMES):
            raise ContractError("cached local feature schema drift")
        return list(cached)
    relation = str(graph["Relation"])
    selected = set(action["component_ids"])
    incumbent_ids = _component_ids(graph, incumbent)
    added = selected - incumbent_ids
    removed = incumbent_ids - selected
    all_ids = {
        str(value["id"])
        for value in graph["relational_graph"]["components"]
    }
    omitted = all_ids - selected
    component_by = {
        str(value["id"]): value
        for value in graph["relational_graph"]["components"]
    }
    families = _route_family_map(graph)
    values: list[float] = [
        *[float(relation == value) for value in RELATIONS],
        *[
            float(action["action_type"] == value)
            for value in ACTION_TYPES
        ],
        float(not incumbent),
        float(not action["objects"]),
        min(1.0, len(incumbent_ids) / 10.0),
        min(1.0, len(selected) / 10.0),
        max(-1.0, min(1.0, (len(selected) - len(incumbent_ids)) / 10.0)),
        min(1.0, len(selected ^ incumbent_ids) / 10.0),
        min(1.0, len(all_ids) / 20.0),
    ]
    for ids in (selected, added, removed, omitted):
        for family in ("qwen_recall", "gemma_independent", MINISTRAL):
            values.extend(_summary([
                _support(component_by[item], family, families)
                for item in sorted(ids)
            ]))
    family_counts = [
        sum(
            _support(component_by[item], family, families) > 0.0
            for family in ("qwen_recall", "gemma_independent", MINISTRAL)
        )
        for item in selected
    ]
    values.extend([
        (
            sum(family_counts) / (3.0 * len(family_counts))
            if family_counts else 0.0
        ),
        (
            min(family_counts) / 3.0
            if family_counts else 0.0
        ),
        float(graph.get("agents", {}).get(
            MINISTRAL, {}).get("none_rate", 0.0)),
        float(graph.get("agents", {}).get(
            "qwen_recall", {}).get("none_rate", 0.0)),
        float(graph.get("agents", {}).get(
            "gemma_independent", {}).get("none_rate", 0.0)),
    ])
    numeric_distance = 0.0
    if relation in NUMERIC_RELATIONS and incumbent and action["objects"]:
        before = try_parse_number(str(incumbent[0]))
        after = try_parse_number(str(action["objects"][0]))
        if before and after and before > 0 and after > 0:
            numeric_distance = min(1.0, abs(math.log(after / before)) / 3.0)
    values.append(numeric_distance)
    if len(values) != len(LOCAL_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise ContractError("invalid component-local feature vector")
    action["_component_local_features"] = list(values)
    return values


def edge_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
) -> list[float]:
    """Traverse relational edges and summarize the proposed answer subgraph."""
    cached = action.get("_typed_edge_features")
    if cached is not None:
        if len(cached) != len(EDGE_NAMES):
            raise ContractError("cached edge feature schema drift")
        return list(cached)
    relation = str(graph["Relation"])
    # Inter-component co-support is set evidence.  For single-valued and
    # numeric relations the graph deliberately creates complete
    # contradiction cliques to encode action legality; their degree cannot
    # identify which alternative is true and must not be treated as a truth
    # message.  Those relations therefore use the matched component-local
    # decoder, while list relations receive typed edge messages.
    if relation not in LIST_RELATIONS:
        values = [0.0] * len(EDGE_NAMES)
        action["_typed_edge_features"] = list(values)
        return values
    selected = set(action["component_ids"])
    all_components = {
        str(value["id"]): value
        for value in graph["relational_graph"]["components"]
    }
    component_count = max(len(all_components), 1)
    internal_pairs = max(len(selected) * (len(selected) - 1) / 2, 1.0)
    boundary_pairs = max(
        len(selected) * (len(all_components) - len(selected)), 1)
    co_internal = co_boundary = 0
    co_weight = co_same_count = co_same_rate = boundary_rate = 0.0
    contradiction_internal = contradiction_boundary = null_selected = 0
    equivalent_selected = 0
    edges = graph["relational_graph"]["edges"]
    for edge in edges:
        edge_type = str(edge["edge_type"])
        source, target = str(edge["source"]), str(edge["target"])
        source_selected = source in selected
        target_selected = target in selected
        if edge_type == "co_supported_with":
            if source_selected and target_selected:
                co_internal += 1
                co_weight += float(edge.get("weight", 0.0))
                if int(edge.get("cooccurrence_count", 0)) > 0:
                    co_same_count += 1
                co_same_rate += float(edge.get("cooccurrence_rate", 0.0))
            elif source_selected != target_selected:
                co_boundary += 1
                boundary_rate += float(edge.get(
                    "cooccurrence_rate", 0.0))
        elif edge_type == "contradicts":
            if source == "hypothesis:null" or target == "hypothesis:null":
                other = target if source == "hypothesis:null" else source
                null_selected += int(other in selected)
            elif source_selected and target_selected:
                contradiction_internal += 1
            elif source_selected != target_selected:
                contradiction_boundary += 1
        elif edge_type == "equivalent_to":
            # Equivalence edges connect surface nodes; map their candidate
            # endpoints through member_of before attributing them.
            candidate_to_component = {
                str(value["source"]): str(value["target"])
                for value in edges
                if value["edge_type"] == "member_of"
            }
            if (
                candidate_to_component.get(source) in selected
                and candidate_to_component.get(target) in selected
            ):
                equivalent_selected += 1

    selected_components = [
        all_components[value] for value in selected
        if value in all_components
    ]
    alias_fraction = (
        sum(bool(value.get("alias_collapsed")) for value in selected_components)
        / max(1, len(selected_components))
    )
    surface_ratio = (
        sum(len(value.get("member_candidate_ids", []))
            for value in selected_components)
        / max(1, len(selected_components))
    )

    # Route-dependence is explicit in the graph.  Qwen's two routes count as
    # one memory family; independent-family support is the sum of per-family
    # maxima, not the sum of every route edge.
    route_families = _route_family_map(graph)
    dependent_routes = {
        frozenset((str(value["source"]), str(value["target"])))
        for value in edges if value["edge_type"] == "dependent_with"
    }
    raw_route_mass = discounted_mass = 0.0
    for component in selected_components:
        supports: dict[str, float] = {}
        route_supports: dict[str, float] = {}
        for route, metadata in component.get("routes", {}).items():
            support = float(metadata.get(
                "component_support_rate",
                metadata.get("max_support_rate", 0.0),
            ))
            route_supports[f"route:{route}"] = support
            family = route_families.get(str(route), "")
            supports[family] = max(supports.get(family, 0.0), support)
        raw_route_mass += sum(route_supports.values())
        discounted_mass += sum(supports.values())
    dependent_redundancy = max(0.0, raw_route_mass - discounted_mass)
    # Touch the dependency edges explicitly; absence would make the discount
    # an undocumented family heuristic rather than graph reasoning.
    if not dependent_routes:
        dependent_redundancy = 0.0

    generation_coverages = []
    exact_generation_sets = 0
    if selected:
        for generation in range(N_MAX):
            generation_set = {
                str(component["id"])
                for component in all_components.values()
                if generation in component.get("routes", {}).get(
                    ROUTE, {}).get("generation_indices", [])
            }
            overlap = len(selected & generation_set) / len(selected)
            generation_coverages.append(overlap)
            exact_generation_sets += int(generation_set == selected)
    coherence_mean = (
        sum(generation_coverages) / len(generation_coverages)
        if generation_coverages else 0.0
    )
    coherence_max = max(generation_coverages, default=0.0)
    exact_rate = exact_generation_sets / N_MAX if selected else 0.0

    base = [
        co_internal / internal_pairs,
        co_weight / internal_pairs,
        co_same_count / internal_pairs,
        co_same_rate / internal_pairs,
        co_boundary / boundary_pairs,
        boundary_rate / boundary_pairs,
        contradiction_internal / internal_pairs,
        contradiction_boundary / boundary_pairs,
        null_selected / component_count,
        min(1.0, equivalent_selected / max(1, len(selected))),
        alias_fraction,
        min(1.0, surface_ratio / 3.0),
        min(1.0, dependent_redundancy / max(1, len(selected))),
        min(1.0, discounted_mass / max(1, 3 * len(selected))),
        coherence_mean,
        coherence_max,
        exact_rate,
    ]
    interactions = []
    for value in RELATIONS:
        active = float(relation == value)
        interactions.extend([
            active * base[2],
            active * base[1],
            active * base[7],
        ])
    values = [*base, *interactions]
    if len(values) != len(EDGE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise ContractError("invalid typed-edge feature vector")
    action["_typed_edge_features"] = list(values)
    return values


def feature_names(arm: str) -> tuple[str, ...]:
    if arm == "component_local":
        return tuple(LOCAL_NAMES)
    if arm == "typed_edges":
        return tuple((*LOCAL_NAMES, *EDGE_NAMES))
    raise ContractError(f"unknown decoder arm: {arm}")


def action_features(
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    action: Mapping[str, Any],
    arm: str,
) -> list[float]:
    values = local_features(graph, incumbent, action)
    if arm == "typed_edges":
        values.extend(edge_features(graph, action))
    if len(values) != len(feature_names(arm)):
        raise AssertionError("graph-native feature schema drift")
    return values


def fit_model(
    graphs: Sequence[Mapping[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    arm: str,
    l2: float,
) -> ResidualRidge:
    x: list[list[float]] = []
    y: list[float] = []
    weights: list[float] = []
    for graph in graphs:
        key = _key(graph)
        incumbent = list(controls[key])
        actions = legal_actions(graph, incumbent)
        baseline = _row_f1(incumbent, gold[key], key[1])
        row_weight = 1.0 / len(actions)
        for action in actions:
            x.append(action_features(graph, incumbent, action, arm))
            y.append(
                _row_f1(action["objects"], gold[key], key[1]) - baseline)
            weights.append(row_weight)
    return ResidualRidge(feature_names(arm), l2).fit(x, y, weights)


def propose(
    model: ResidualRidge,
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    arm: str,
) -> tuple[list[str], float, str, int]:
    actions = legal_actions(graph, incumbent)
    estimates = model.predict([
        action_features(graph, incumbent, action, arm)
        for action in actions
    ])
    keep_index = next(
        index for index, action in enumerate(actions)
        if action["action_type"] == "KEEP"
    )
    best = max(range(len(actions)), key=lambda index: (
        float(estimates[index]),
        actions[index]["action_type"] == "KEEP",
        -len(actions[index]["objects"]),
        -index,
    ))
    advantage = float(estimates[best] - estimates[keep_index])
    return (
        list(actions[best]["objects"]),
        advantage,
        str(actions[best]["action_type"]),
        len(actions),
    )


def _prediction_rows(
    controls: Sequence[Mapping[str, Any]],
    replacements: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict[str, Any]]:
    return [{
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": str(row["Relation"]),
        "ObjectEntities": list(replacements.get(
            _key(row), row.get("ObjectEntities", []))),
    } for row in controls]


def cot40_count_anchor(
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    support_required: int = 7,
) -> list[str]:
    """Apply the frozen CoT40 support rule through typed components.

    This is the correct incumbent for graph decoding: edge reasoning must add
    value after the already-selected CoT40 policy, not replace it with an
    older two-model pipeline.
    """
    relation = str(graph["Relation"])
    components = list(graph["relational_graph"]["components"])
    supported = [
        (
            int(component.get("routes", {}).get(
                ROUTE, {}).get("distinct_generation_support", 0)),
            str(component["representative"]),
        )
        for component in components
    ]
    supported = [
        value for value in supported if value[0] >= support_required
    ]
    if relation in NUMERIC_RELATIONS:
        if not supported:
            return list(incumbent)
        maximum = max(value[0] for value in supported)
        winners = [value[1] for value in supported if value[0] == maximum]
        return winners if len(winners) == 1 else list(incumbent)
    return collapse_prediction(
        graph, [*incumbent, *(value[1] for value in supported)])


def _score_predictions(
    controls: Sequence[Mapping[str, Any]],
    proposals: Mapping[tuple[str, str], tuple[list[str], float, str, int]],
    guard: float,
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    replacements: dict[tuple[str, str], list[str]] = {}
    changed = helped = harmed = neutral = 0
    action_counts: Counter[str] = Counter()
    decisions = []
    for row in controls:
        key = _key(row)
        incumbent = list(row.get("ObjectEntities", []))
        proposal, advantage, action_type, action_count = proposals[key]
        selected = proposal if advantage > guard else incumbent
        replacements[key] = selected
        is_changed = tuple(selected) != tuple(incumbent)
        if is_changed:
            changed += 1
            action_counts[action_type] += 1
            before = _row_f1(incumbent, gold[key], key[1])
            after = _row_f1(selected, gold[key], key[1])
            helped += int(after > before + 1e-12)
            harmed += int(after < before - 1e-12)
            neutral += int(abs(after - before) <= 1e-12)
        decisions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "incumbent": incumbent,
            "proposal": proposal,
            "selected": selected,
            "predicted_advantage": advantage,
            "guard": guard,
            "action_type": action_type,
            "action_count": action_count,
            "changed": is_changed,
        })
    predictions = _prediction_rows(controls, replacements)
    scores = score(predictions, [gold[_key(row)] for row in predictions])
    return scores, {
        "changed": changed,
        "helped": helped,
        "harmed": harmed,
        "neutral": neutral,
        "action_counts": dict(action_counts),
    }, decisions


def _nested_oof(
    graphs: Sequence[Mapping[str, Any]],
    controls_rows: Sequence[Mapping[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    folds: Mapping[tuple[str, str], int],
    arm: str,
) -> dict[str, Any]:
    graph_by = {_key(row): row for row in graphs}
    control_by = {_key(row): row for row in controls_rows}
    fold_ids = sorted(set(folds.values()))
    oof: dict[tuple[str, str], tuple[list[str], float, str, int]] = {}
    diagnostics = []
    for outer in fold_ids:
        print(
            f"{arm}: outer fold {outer + 1}/{len(fold_ids)}",
            flush=True,
        )
        outer_fit = [row for row in graphs if folds[_key(row)] != outer]
        outer_hold = [row for row in graphs if folds[_key(row)] == outer]
        fit_keys = {_key(row) for row in outer_fit}
        # ``fit_keys`` is a set.  Preserve a deterministic scoring order so
        # floating-point ties between guards cannot depend on PYTHONHASHSEED.
        inner_controls = [
            control_by[key] for key in sorted(fit_keys)
        ]
        candidates = []
        for l2 in L2_VALUES:
            inner_proposals: dict[
                tuple[str, str], tuple[list[str], float, str, int]
            ] = {}
            for inner in fold_ids:
                if inner == outer:
                    continue
                inner_train = [
                    row for row in outer_fit if folds[_key(row)] != inner
                ]
                inner_hold = [
                    row for row in outer_fit if folds[_key(row)] == inner
                ]
                model = fit_model(
                    inner_train, controls, gold, arm, l2)
                for graph in inner_hold:
                    key = _key(graph)
                    inner_proposals[key] = propose(
                        model, graph, controls[key], arm)
            if set(inner_proposals) != fit_keys:
                raise ContractError("inner OOF proposal coverage mismatch")
            for guard in GUARDS:
                scores, audit, _ = _score_predictions(
                    inner_controls, inner_proposals, guard, gold)
                candidates.append((scores[POOLED], guard, l2, scores, audit))
        chosen = max(candidates, key=lambda value: (
            value[0], value[1], value[2]))
        _, guard, l2, inner_scores, inner_audit = chosen
        model = fit_model(outer_fit, controls, gold, arm, l2)
        for graph in outer_hold:
            key = _key(graph)
            oof[key] = propose(model, graph, controls[key], arm)
        hold_controls = [control_by[_key(row)] for row in outer_hold]
        hold_proposals = {_key(row): oof[_key(row)] for row in outer_hold}
        hold_scores, hold_audit, _ = _score_predictions(
            hold_controls, hold_proposals, guard, gold)
        for graph in outer_hold:
            proposal, advantage, action_type, count = oof[_key(graph)]
            oof[_key(graph)] = (
                proposal if advantage > guard else list(controls[_key(graph)]),
                1.0 if advantage > guard else 0.0,
                action_type,
                count,
            )
        diagnostics.append({
            "outer_fold": outer,
            "selected_l2": l2,
            "selected_guard": guard,
            "inner_scores": inner_scores,
            "inner_audit": inner_audit,
            "hold_scores": hold_scores,
            "hold_audit": hold_audit,
        })
    if set(oof) != set(graph_by):
        raise ContractError("outer OOF proposal coverage mismatch")
    # Guard decisions were encoded above as advantage 1/0; a 0.5 boundary
    # reconstructs the exact nested outer prediction without reusing labels.
    scores, audit, decisions = _score_predictions(
        controls_rows, oof, 0.5, gold)
    return {
        "scores": scores,
        "audit": audit,
        "decisions": decisions,
        "fold_diagnostics": diagnostics,
        "fold_scores": [
            value["hold_scores"][POOLED] for value in diagnostics
        ],
    }


def boundary_actions(
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    cot40_target: Sequence[str],
) -> list[dict[str, Any]]:
    """Return the coherent two-state decision at the CoT40 admission boundary."""
    keep = {
        "action_type": "KEEP",
        "objects": list(incumbent),
        "component_ids": sorted(_component_ids(graph, incumbent)),
    }
    target = {
        "action_type": "GRAPH_SET",
        "objects": list(cot40_target),
        "component_ids": sorted(_component_ids(graph, cot40_target)),
    }
    if _action_tokens(
        graph, keep["objects"], "component"
    ) == _action_tokens(graph, target["objects"], "component"):
        return [keep]
    return [keep, target]


def fit_boundary_model(
    graphs: Sequence[Mapping[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
    targets: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    arm: str,
    l2: float,
) -> ResidualRidge:
    x: list[list[float]] = []
    y: list[float] = []
    weights: list[float] = []
    for graph in graphs:
        key = _key(graph)
        incumbent = list(controls[key])
        actions = boundary_actions(graph, incumbent, targets[key])
        baseline = _row_f1(incumbent, gold[key], key[1])
        row_weight = 1.0 / len(actions)
        for action in actions:
            x.append(action_features(graph, incumbent, action, arm))
            y.append(
                _row_f1(action["objects"], gold[key], key[1]) - baseline)
            weights.append(row_weight)
    return ResidualRidge(feature_names(arm), l2).fit(x, y, weights)


def propose_boundary(
    model: ResidualRidge,
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    target: Sequence[str],
    arm: str,
) -> tuple[list[str], float, str, int]:
    actions = boundary_actions(graph, incumbent, target)
    estimates = model.predict([
        action_features(graph, incumbent, action, arm)
        for action in actions
    ])
    keep_index = 0
    best = max(range(len(actions)), key=lambda index: (
        float(estimates[index]),
        actions[index]["action_type"] == "KEEP",
    ))
    return (
        list(actions[best]["objects"]),
        float(estimates[best] - estimates[keep_index]),
        str(actions[best]["action_type"]),
        len(actions),
    )


def _nested_boundary_oof(
    graphs: Sequence[Mapping[str, Any]],
    controls_rows: Sequence[Mapping[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
    targets: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    folds: Mapping[tuple[str, str], int],
    arm: str,
) -> dict[str, Any]:
    control_by = {_key(row): row for row in controls_rows}
    fold_ids = sorted(set(folds.values()))
    oof: dict[tuple[str, str], tuple[list[str], float, str, int]] = {}
    diagnostics = []
    for outer in fold_ids:
        print(
            f"boundary_{arm}: outer fold {outer + 1}/{len(fold_ids)}",
            flush=True,
        )
        outer_fit = [row for row in graphs if folds[_key(row)] != outer]
        outer_hold = [row for row in graphs if folds[_key(row)] == outer]
        fit_keys = {_key(row) for row in outer_fit}
        # See the full-action decoder above: inner policy selection must not
        # depend on randomized set iteration.
        inner_controls = [control_by[key] for key in sorted(fit_keys)]
        candidates = []
        for l2 in L2_VALUES:
            inner_proposals: dict[
                tuple[str, str], tuple[list[str], float, str, int]
            ] = {}
            for inner in fold_ids:
                if inner == outer:
                    continue
                inner_train = [
                    row for row in outer_fit if folds[_key(row)] != inner
                ]
                inner_hold = [
                    row for row in outer_fit if folds[_key(row)] == inner
                ]
                model = fit_boundary_model(
                    inner_train, controls, targets, gold, arm, l2)
                for graph in inner_hold:
                    key = _key(graph)
                    inner_proposals[key] = propose_boundary(
                        model, graph, controls[key], targets[key], arm)
            if set(inner_proposals) != fit_keys:
                raise ContractError("boundary inner OOF coverage mismatch")
            for guard in GUARDS:
                scores, audit, _ = _score_predictions(
                    inner_controls, inner_proposals, guard, gold)
                candidates.append((scores[POOLED], guard, l2, scores, audit))
        _, guard, l2, inner_scores, inner_audit = max(
            candidates, key=lambda value: (value[0], value[1], value[2]))
        model = fit_boundary_model(
            outer_fit, controls, targets, gold, arm, l2)
        hold_proposals = {}
        for graph in outer_hold:
            key = _key(graph)
            proposal = propose_boundary(
                model, graph, controls[key], targets[key], arm)
            hold_proposals[key] = proposal
            oof[key] = proposal
        hold_controls = [control_by[_key(row)] for row in outer_hold]
        hold_scores, hold_audit, _ = _score_predictions(
            hold_controls, hold_proposals, guard, gold)
        for graph in outer_hold:
            key = _key(graph)
            proposal, advantage, action_type, count = oof[key]
            oof[key] = (
                proposal if advantage > guard else list(controls[key]),
                1.0 if advantage > guard else 0.0,
                action_type,
                count,
            )
        diagnostics.append({
            "outer_fold": outer,
            "selected_l2": l2,
            "selected_guard": guard,
            "inner_scores": inner_scores,
            "inner_audit": inner_audit,
            "hold_scores": hold_scores,
            "hold_audit": hold_audit,
        })
    scores, audit, decisions = _score_predictions(
        controls_rows, oof, 0.5, gold)
    return {
        "scores": scores,
        "audit": audit,
        "decisions": decisions,
        "fold_diagnostics": diagnostics,
        "fold_scores": [
            value["hold_scores"][POOLED] for value in diagnostics
        ],
    }


def boundary_analyze(args: argparse.Namespace) -> int:
    """Run the coherent KEEP-vs-complete-CoT40-set graph decision."""
    output = Path(args.output_dir).resolve()
    plan, graphs = _validate_prepared(output)
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    gold = {_key(row): row for row in gold_rows}
    base_rows, base_detail = compose_competition_train_oof()
    controls = {
        _key(row): list(row.get("ObjectEntities", [])) for row in base_rows
    }
    graph_by = {_key(row): row for row in graphs}
    if set(gold) != set(controls) or set(gold) != set(graph_by):
        raise ContractError("boundary graph/control/gold coverage mismatch")
    targets = {
        key: cot40_count_anchor(graph_by[key], controls[key])
        for key in controls
    }
    target_rows = [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": list(targets[_key(row)]),
    } for row in base_rows]
    for graph in graphs:
        graph["baseline_objects"] = list(controls[_key(graph)])
        graph.pop("_graph_native_action_cache", None)
    folds = subject_grouped_folds(graphs)
    reports = {
        arm: _nested_boundary_oof(
            graphs, base_rows, controls, targets, gold, folds, arm)
        for arm in ARMS
    }
    base_scores = score(base_rows, gold_rows)
    target_scores = score(target_rows, gold_rows)
    for arm, report in reports.items():
        replacements = {
            (str(value["SubjectEntity"]), str(value["Relation"])):
                value["selected"]
            for value in report["decisions"]
        }
        path = output / f"analysis/BOUNDARY_{arm.upper()}_OOF.jsonl"
        write_jsonl_atomic(
            path, _prediction_rows(base_rows, replacements))
        report["predictions"] = str(path)
        report["predictions_sha256"] = sha256(path)
    local, edge = (
        reports["component_local"], reports["typed_edges"])
    edge_increment = edge["scores"][POOLED] - local["scores"][POOLED]
    edge_delta = edge["scores"][POOLED] - base_scores[POOLED]
    relation_deltas = {
        relation: edge["scores"][relation] - local["scores"][relation]
        for relation in RELATIONS
    }
    fold_increments = [
        right - left for left, right in zip(
            local["fold_scores"], edge["fold_scores"])
    ]
    fold_wins = sum(value > 1e-12 for value in fold_increments)
    gate = bool(
        edge_increment >= MIN_EDGE_INCREMENT
        and edge_delta > 1e-12
        and fold_wins >= MIN_EDGE_FOLD_WINS
        and min(relation_deltas.values()) >= MAX_RELATION_REGRESSION
        and edge["audit"]["helped"] > edge["audit"]["harmed"]
    )
    result = {
        "schema": "cot40-graph-boundary-decoder-result-v1",
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "graph": plan["graph"],
        "graph_sha256": plan["graph_sha256"],
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "base_detail": base_detail,
        "base_scores": base_scores,
        "unconditional_cot40_scores": target_scores,
        "arms": reports,
        "typed_edge_increment_vs_component_local": edge_increment,
        "typed_edge_delta_vs_base": edge_delta,
        "typed_edge_relation_deltas_vs_component_local": relation_deltas,
        "typed_edge_fold_increments": fold_increments,
        "typed_edge_fold_wins": fold_wins,
        "promotion_gate_passed": gate,
        "decision_boundary":
            "KEEP exact pre-CoT40 SOTA set versus accept exact complete "
            "typed-component CoT40 set; no arbitrary add/drop/replace action",
    }
    path = output / "analysis/BOUNDARY_RESULT.json"
    _write_json(path, result)
    print(json.dumps({
        "result": str(path),
        "base": base_scores[POOLED],
        "unconditional_cot40": target_scores[POOLED],
        "boundary_component_local": local["scores"][POOLED],
        "boundary_typed_edges": edge["scores"][POOLED],
        "edge_increment": edge_increment,
        "edge_fold_wins": fold_wins,
        "promotion_gate_passed": gate,
        "validation_opened": False,
    }, indent=2, sort_keys=True))
    return 0


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan, graphs = _validate_prepared(output)
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    gold = {_key(row): row for row in gold_rows}
    if len(gold) != 477 or set(gold) != {_key(row) for row in graphs}:
        raise ContractError("training gold coverage mismatch")
    control_rows, control_detail = compose_competition_train_oof()
    original_controls = {
        _key(row): list(row.get("ObjectEntities", []))
        for row in control_rows
    }
    if set(original_controls) != set(gold):
        raise ContractError("incumbent OOF coverage mismatch")
    graph_by = {_key(row): row for row in graphs}
    count_anchor_rows = [{
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": str(row["Relation"]),
        "ObjectEntities": cot40_count_anchor(
            graph_by[_key(row)], original_controls[_key(row)]),
    } for row in control_rows]
    controls = {
        _key(row): list(row["ObjectEntities"]) for row in count_anchor_rows
    }
    for graph in graphs:
        graph["baseline_objects"] = list(controls[_key(graph)])
    folds = subject_grouped_folds(graphs)
    fold_path = output / "analysis/SUBJECT_GROUPED_FOLDS.jsonl"
    write_jsonl_atomic(fold_path, [{
        "SubjectEntity": key[0],
        "Relation": key[1],
        "fold": value,
    } for key, value in sorted(folds.items())])
    original_control_scores = score(control_rows, gold_rows)
    control_scores = score(count_anchor_rows, gold_rows)
    reports = {
        arm: _nested_oof(
            graphs, count_anchor_rows, controls, gold, folds, arm)
        for arm in ARMS
    }
    for arm, report in reports.items():
        prediction_path = output / f"analysis/{arm.upper()}_OOF.jsonl"
        replacements = {
            (str(value["SubjectEntity"]), str(value["Relation"])):
                value["selected"]
            for value in report["decisions"]
        }
        write_jsonl_atomic(
            prediction_path, _prediction_rows(
                count_anchor_rows, replacements))
        report["predictions"] = str(prediction_path)
        report["predictions_sha256"] = sha256(prediction_path)

    local = reports["component_local"]
    edge = reports["typed_edges"]
    edge_delta = edge["scores"][POOLED] - local["scores"][POOLED]
    delta_vs_control = edge["scores"][POOLED] - control_scores[POOLED]
    relation_deltas = {
        relation: edge["scores"][relation] - local["scores"][relation]
        for relation in RELATIONS
    }
    fold_increments = [
        right - left for left, right in zip(
            local["fold_scores"], edge["fold_scores"])
    ]
    fold_wins = sum(value > 1e-12 for value in fold_increments)
    gate = bool(
        edge_delta >= MIN_EDGE_INCREMENT
        and delta_vs_control > 1e-12
        and fold_wins >= MIN_EDGE_FOLD_WINS
        and min(relation_deltas.values()) >= MAX_RELATION_REGRESSION
        and edge["audit"]["helped"] > edge["audit"]["harmed"]
    )
    result = {
        "schema": RESULT_SCHEMA,
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "graph": plan["graph"],
        "graph_sha256": plan["graph_sha256"],
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "incumbent_detail": control_detail,
        "pre_cot40_incumbent_scores": original_control_scores,
        "incumbent_scores": control_scores,
        "incumbent_policy":
            "typed_component_cot40_support_7_of_10_over_competition_oof",
        "arms": reports,
        "typed_edge_increment_vs_component_local": edge_delta,
        "typed_edge_delta_vs_incumbent": delta_vs_control,
        "typed_edge_relation_deltas_vs_component_local": relation_deltas,
        "typed_edge_fold_increments": fold_increments,
        "typed_edge_fold_wins": fold_wins,
        "promotion_gate_passed": gate,
        "next_stage": (
            "freeze_label_free_validation_graph_and_decoder"
            if gate else
            "reject_current_edge_decoder_and_inspect_failure_ledger"
        ),
        "methodology": (
            "five-fold strict-subject nested CV; each outer fold selects one "
            "global l2 and guard using only inner OOF predictions; matched "
            "arms differ only by typed inter-component edge features"
        ),
    }
    result_path = output / "analysis/RESULT.json"
    _write_json(result_path, result)
    lines = [
        "# CoT40 graph-native decoder",
        "",
        "Training-only nested subject-grouped audit; validation was not opened.",
        "",
        "| arm | pooled OOF F1 | delta vs incumbent | changed | helped | harmed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        report = reports[arm]
        lines.append(
            f"| {arm} | {report['scores'][POOLED]:.6f} | "
            f"{report['scores'][POOLED]-control_scores[POOLED]:+.6f} | "
            f"{report['audit']['changed']} | "
            f"{report['audit']['helped']} | "
            f"{report['audit']['harmed']} |"
        )
    lines.extend([
        "",
        f"- Typed-edge increment over matched component-local arm: "
        f"**{edge_delta:+.6f}**",
        f"- Typed-edge fold wins: **{fold_wins}/5**",
        f"- Promotion gate passed: **{gate}**",
        f"- Next stage: `{result['next_stage']}`",
        "",
        "The graph claim is supported only if the typed-edge arm improves over "
        "the matched component-local arm. Candidate expansion alone does not "
        "count as evidence that edges helped.",
    ])
    (output / "analysis/RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "result": str(result_path),
        "pre_cot40_incumbent": original_control_scores[POOLED],
        "incumbent": control_scores[POOLED],
        "component_local": local["scores"][POOLED],
        "typed_edges": edge["scores"][POOLED],
        "edge_increment": edge_delta,
        "edge_fold_wins": fold_wins,
        "promotion_gate_passed": gate,
        "validation_opened": False,
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    prepare_parser.add_argument("--cot40-run", default=str(DEFAULT_COT40))
    prepare_parser.set_defaults(function=prepare)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    analyze_parser.add_argument("--train-gold", default=str(DEFAULT_GOLD))
    analyze_parser.set_defaults(function=analyze)
    boundary_parser = subparsers.add_parser("boundary-analyze")
    boundary_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT))
    boundary_parser.add_argument(
        "--train-gold", default=str(DEFAULT_GOLD))
    boundary_parser.set_defaults(function=boundary_analyze)
    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
