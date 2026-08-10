#!/usr/bin/env python3
"""Build and test a minimal candidate/evidence bipartite graph.

The historical typed candidate graph contains useful construction-time
bookkeeping (surface nodes, equivalence membership, route nodes, null
hypotheses, contradiction cliques, and several derived edge families).  The
matched edge ablations showed that most of this topology is redundant with
component-local support or actively unstable as selector input.

This experiment materializes the smallest graph that can still express real
row-specific evidence:

* one node per canonical candidate component;
* one node per exact generation when generation provenance is available;
* one aggregate route-evidence node when only route-level support survived;
* one directed ``supports`` edge from evidence to candidate.

Null/cardinality/action legality remain outside the evidence graph as hard
decoder constraints.  Preparation is label-free and proves exact parity for
the count anchor, legal action inventory, and component-table features.
Analysis opens training labels only and compares four otherwise matched arms:

* ``component_table`` -- the edge-free component-support baseline;
* ``event_unweighted`` -- exact event/set agreement only;
* ``event_weighted`` -- exact event agreement plus weighted support messages;
* ``event_subject_shifted`` -- a deterministic row-misaligned negative control.

There is intentionally no validation command and no deployable output.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.baseline_relative_route_decoder import (
    ResidualRidge,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.cot40_graph_native_decoder import (
    GUARDS,
    L2_VALUES,
    LOCAL_NAMES,
    POOLED,
    RELATIONS,
    _prediction_rows,
    _row_f1,
    _score_predictions,
    _validate_prepared,
    cot40_count_anchor,
    legal_actions,
    local_features,
)
from experiments.heterogeneous_agents.heterogeneous_memory_selector import _key
from experiments.heterogeneous_agents.sota_pipeline import (
    compose_competition_train_oof,
)
from experiments.heterogeneous_agents.three_model_component_decoder import (
    subject_grouped_folds,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_SOURCE = RUNS / "cot40_graph_native_decoder_20260729_v1"
DEFAULT_OUTPUT = RUNS / "cot40_minimal_evidence_graph_20260730_v1"
DEFAULT_GOLD = ROOT / "data/train.jsonl"

PLAN_SCHEMA = "cot40-minimal-evidence-graph-plan-v1"
ROW_SCHEMA = "minimal-candidate-evidence-bipartite-row-v1"
GRAPH_SCHEMA = "minimal-candidate-evidence-bipartite-graph-v1"
MANIFEST_SCHEMA = "minimal-candidate-evidence-graph-manifest-v1"
RESULT_SCHEMA = "cot40-minimal-evidence-graph-result-v1"

ARMS = (
    "component_table",
    "event_unweighted",
    "event_weighted",
    "event_subject_shifted",
)

UNWEIGHTED_NAMES = (
    "event_exact_jaccard_mean",
    "event_exact_jaccard_max",
    "event_exact_set_rate",
    "event_exact_selected_recall_mean",
    "event_exact_selected_precision_mean",
    "event_exact_selected_component_coverage",
    "event_exact_boundary_rate",
    "event_explicit_none_agreement",
)
WEIGHTED_NAMES = (
    "event_selected_mass_share",
    "event_omitted_mass_share",
    "event_selected_mass_per_component",
    "event_message_precision_mean",
    "event_message_precision_max",
    "event_complete_message_rate",
    "event_selected_family_coverage",
    "event_selected_family_share_mean",
)
EVENT_NAMES = (*UNWEIGHTED_NAMES, *WEIGHTED_NAMES)

MIN_INCREMENT = 0.003
MIN_FOLD_WINS = 3
MAX_RELATION_REGRESSION = -0.01
MIN_ALIGNED_OVER_SHIFTED = 0.001


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


def _route_families(row: Mapping[str, Any]) -> dict[str, str]:
    result = {
        str(route): str(metadata.get("model_family", ""))
        for route, metadata in row.get("proposal_routes", {}).items()
        if isinstance(metadata, Mapping)
    }
    for node in row["relational_graph"].get("nodes", []):
        if node.get("node_type") == "evidence_route":
            result[str(node["route"])] = str(
                node.get("model_family", result.get(str(node["route"]), "")))
    return result


def _route_samples(
    row: Mapping[str, Any],
    route: str,
    components: Sequence[Mapping[str, Any]],
) -> int:
    route_metadata = row.get("proposal_routes", {}).get(route, {})
    declared = int(route_metadata.get("n_samples", 0) or 0)
    observed = max(
        (
            int(component.get("routes", {}).get(route, {}).get(
                "samples", 0) or 0)
            for component in components
        ),
        default=0,
    )
    return max(declared, observed, 1)


def _exact_routes(row: Mapping[str, Any]) -> set[str]:
    routes = {
        str(route)
        for route, metadata in row.get("proposal_routes", {}).items()
        if (
            isinstance(metadata, Mapping)
            and metadata.get("generation_provenance_available") is True
        )
    }
    provenance = row.get("cot40_generation_provenance", {})
    if isinstance(provenance, Mapping) and provenance.get("route"):
        routes.add(str(provenance["route"]))
    return routes


def _event_id(route: str, suffix: str) -> str:
    return f"evidence:{route}:{suffix}"


def minimalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a typed row into a minimal evidence/component bipartite graph."""
    relation = str(row["Relation"])
    components = copy.deepcopy(row["relational_graph"]["components"])
    route_families = _route_families(row)
    component_routes = {
        str(route)
        for component in components
        for route in component.get("routes", {})
    }
    # A route that produced only an explicit empty answer has no candidate
    # endpoint, but its evidence event is still semantically real.  Enumerate
    # every available sampled route rather than deriving route existence only
    # from positive candidate components.
    available_routes = {
        str(route)
        for route, metadata in row.get("proposal_routes", {}).items()
        if (
            isinstance(metadata, Mapping)
            and metadata.get("available") is True
            and int(metadata.get("n_samples", 0) or 0) > 0
        )
    }
    routes = sorted(component_routes | available_routes)
    missing_families = [
        route for route in routes if not route_families.get(route)
    ]
    if missing_families:
        raise ContractError(
            f"{_key(row)}: routes lack model-family provenance: "
            f"{missing_families}")
    exact_routes = _exact_routes(row)
    none_indices: dict[str, set[int]] = defaultdict(set)
    provenance = row.get("cot40_generation_provenance", {})
    if isinstance(provenance, Mapping) and provenance.get("route"):
        none_indices[str(provenance["route"])] = {
            int(value)
            for value in provenance.get("none_generation_indices", [])
        }

    event_nodes: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []
    route_contract: dict[str, dict[str, Any]] = {}
    exact_event_count = aggregate_event_count = 0
    for route in routes:
        family = route_families[route]
        samples = _route_samples(row, route, components)
        route_components = [
            component for component in components
            if route in component.get("routes", {})
        ]
        inferred_single_generation = (
            route not in exact_routes and samples == 1)
        if route in exact_routes or inferred_single_generation:
            event_members: dict[int, list[tuple[str, float, bool]]] = (
                defaultdict(list)
            )
            for component in route_components:
                metadata = component["routes"][route]
                generations = list(metadata.get(
                    "generation_indices", []))
                support = float(metadata.get(
                    "component_support_rate",
                    metadata.get("max_support_rate", 0.0),
                ))
                # With exactly one sample, positive route support proves that
                # the sole generation contained this component.  This is a
                # logical reconstruction, not a fabricated co-occurrence.
                if (
                    inferred_single_generation
                    and not generations
                    and support > 0.0
                ):
                    generations = [0]
                for generation in generations:
                    event_members[int(generation)].append((
                        str(component["id"]),
                        1.0 / samples,
                        bool(metadata.get("selected", False)),
                    ))
            for generation in range(samples):
                members = event_members.get(generation, [])
                explicit_none = generation in none_indices.get(route, set())
                if (
                    inferred_single_generation
                    and not members
                    and int(row.get("agents", {}).get(
                        family, {}).get("n_samples", 0) or 0) == 1
                    and float(row.get("agents", {}).get(
                        family, {}).get("none_rate", 0.0)) >= 1.0
                ):
                    explicit_none = True
                status = (
                    "candidate_set" if members else
                    "explicit_none" if explicit_none else
                    "unparsed_or_no_candidate"
                )
                event = {
                    "id": _event_id(route, f"generation:{generation}"),
                    "node_type": "evidence_event",
                    "evidence_kind": "exact_generation",
                    "route": route,
                    "model_family": family,
                    "generation_index": generation,
                    "samples": samples,
                    "status": status,
                    "provenance_mode": (
                        "inferred_single_generation"
                        if inferred_single_generation else "recorded_index"
                    ),
                }
                event_nodes.append(event)
                exact_event_count += 1
                for component_id, weight, selected in members:
                    supports.append({
                        "source": event["id"],
                        "target": component_id,
                        "edge_type": "supports",
                        "evidence_kind": "exact_generation",
                        "route": route,
                        "model_family": family,
                        "generation_index": generation,
                        "weight": weight,
                        "selected": selected,
                    })
            route_contract[route] = {
                "model_family": family,
                "evidence_kind": "exact_generation",
                "provenance_mode": (
                    "inferred_single_generation"
                    if inferred_single_generation else "recorded_index"
                ),
                "samples": samples,
                "events": samples,
            }
        else:
            event = {
                "id": _event_id(route, "aggregate"),
                "node_type": "evidence_event",
                "evidence_kind": "aggregate_route",
                "route": route,
                "model_family": family,
                "generation_index": None,
                "samples": samples,
                "status": "aggregate_support",
            }
            event_nodes.append(event)
            aggregate_event_count += 1
            for component in route_components:
                metadata = component["routes"][route]
                weight = float(metadata.get(
                    "component_support_rate",
                    metadata.get("max_support_rate", 0.0),
                ))
                if weight <= 0.0:
                    continue
                supports.append({
                    "source": event["id"],
                    "target": str(component["id"]),
                    "edge_type": "supports",
                    "evidence_kind": "aggregate_route",
                    "route": route,
                    "model_family": family,
                    "generation_index": None,
                    "weight": weight,
                    "selected": bool(metadata.get("selected", False)),
                })
            route_contract[route] = {
                "model_family": family,
                "evidence_kind": "aggregate_route",
                "samples": samples,
                "events": 1,
            }

    for component in components:
        for route, metadata in component.get("routes", {}).items():
            metadata["model_family"] = route_families[str(route)]

    minimal = {
        "schema": ROW_SCHEMA,
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": relation,
        "baseline_objects": list(row.get("baseline_objects", [])),
        "candidates": [
            {"key": str(value["key"]), "item": str(value["item"])}
            for value in row.get("candidates", [])
        ],
        "agents": copy.deepcopy(row.get("agents", {})),
        "proposal_routes": copy.deepcopy(row.get("proposal_routes", {})),
        "relational_graph": {
            "schema": GRAPH_SCHEMA,
            "relation": relation,
            "nodes": [*copy.deepcopy(components), *event_nodes],
            "edges": supports,
            "components": components,
        },
        "minimal_evidence_contract": {
            "candidate_node_type": "candidate_component",
            "evidence_node_type": "evidence_event",
            "edge_types": ["supports"],
            "route_families": route_families,
            "routes": route_contract,
            "exact_event_count": exact_event_count,
            "aggregate_event_count": aggregate_event_count,
            "hard_constraints_outside_graph": [
                "null_legality",
                "relation_cardinality",
                "bounded_action_inventory",
            ],
            "surface_normalization_is_construction_only": True,
            "no_fabricated_generation_provenance": True,
        },
    }
    return minimal


def _compatibility_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Recreate route nodes transiently for exact component-feature parity."""
    relational = graph["relational_graph"]
    route_families = graph["minimal_evidence_contract"]["route_families"]
    compatible = dict(graph)
    compatible["relational_graph"] = {
        "schema": "component-feature-compatibility-view-v1",
        "relation": str(graph["Relation"]),
        "nodes": [
            *[
                {
                    "id": f"route:{route}",
                    "node_type": "evidence_route",
                    "route": route,
                    "model_family": family,
                }
                for route, family in sorted(route_families.items())
            ],
            *relational["components"],
        ],
        "edges": [],
        "components": relational["components"],
    }
    return compatible


def component_features(
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    action: Mapping[str, Any],
) -> list[float]:
    return local_features(_compatibility_graph(graph), incumbent, action)


def _event_sets(
    graph: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, float]],
]:
    nodes = {
        str(node["id"]): node
        for node in graph["relational_graph"]["nodes"]
        if node.get("node_type") == "evidence_event"
    }
    support: dict[str, dict[str, float]] = {
        event_id: {} for event_id in nodes
    }
    for edge in graph["relational_graph"]["edges"]:
        if edge.get("edge_type") != "supports":
            raise ContractError("minimal evidence graph has a non-support edge")
        event_id = str(edge["source"])
        if event_id not in nodes:
            raise ContractError("support edge lacks evidence-event source")
        component_id = str(edge["target"])
        weight = float(edge.get("weight", 0.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ContractError("support edge has invalid weight")
        if component_id in support[event_id]:
            raise ContractError("duplicate evidence/component support edge")
        support[event_id][component_id] = weight
    return nodes, support


def _safe_overlap(
    selected: set[str], supported: set[str],
) -> tuple[float, float, float]:
    if not selected and not supported:
        return 1.0, 1.0, 1.0
    intersection = len(selected & supported)
    union = len(selected | supported)
    jaccard = intersection / union if union else 1.0
    recall = intersection / len(selected) if selected else 0.0
    precision = intersection / len(supported) if supported else 0.0
    return jaccard, recall, precision


def event_features(
    graph: Mapping[str, Any],
    action: Mapping[str, Any],
) -> list[float]:
    cached = action.get("_minimal_event_features")
    if cached is not None:
        if len(cached) != len(EVENT_NAMES):
            raise ContractError("cached event feature schema drift")
        return list(cached)
    selected = {str(value) for value in action["component_ids"]}
    all_components = {
        str(value["id"])
        for value in graph["relational_graph"]["components"]
    }
    nodes, support = _event_sets(graph)
    exact_records = []
    for event_id, node in nodes.items():
        if (
            node.get("evidence_kind") != "exact_generation"
            or node.get("status") not in ("candidate_set", "explicit_none")
        ):
            continue
        supported = set(support[event_id])
        jaccard, recall, precision = _safe_overlap(selected, supported)
        exact_records.append((
            jaccard,
            recall,
            precision,
            float(supported == selected),
            supported,
            str(node.get("status")),
        ))
    exact_count = len(exact_records)
    exact_jaccards = [value[0] for value in exact_records]
    selected_supported = set().union(
        *(value[4] for value in exact_records)
    ) if exact_records else set()
    exact_edges = sum(len(value[4]) for value in exact_records)
    exact_omitted = sum(
        len(value[4] - selected) for value in exact_records)
    explicit_none = [
        value for value in exact_records if value[5] == "explicit_none"
    ]
    unweighted = [
        math.fsum(exact_jaccards) / exact_count if exact_count else 0.0,
        max(exact_jaccards, default=0.0),
        (
            math.fsum(value[3] for value in exact_records) / exact_count
            if exact_count else 0.0
        ),
        (
            math.fsum(value[1] for value in exact_records) / exact_count
            if exact_count else 0.0
        ),
        (
            math.fsum(value[2] for value in exact_records) / exact_count
            if exact_count else 0.0
        ),
        (
            len(selected & selected_supported) / len(selected)
            if selected else (
                len(explicit_none) / exact_count if exact_count else 0.0
            )
        ),
        exact_omitted / exact_edges if exact_edges else 0.0,
        (
            math.fsum(
                float((not selected) == (value[5] == "explicit_none"))
                for value in exact_records
            ) / exact_count if exact_count else 0.0
        ),
    ]

    total_mass = selected_mass = omitted_mass = 0.0
    event_precisions: list[float] = []
    complete_messages = 0
    positive_messages = 0
    family_total: dict[str, float] = defaultdict(float)
    family_selected: dict[str, float] = defaultdict(float)
    for event_id, node in nodes.items():
        edges = support[event_id]
        mass = math.fsum(edges.values())
        if mass <= 0.0:
            continue
        event_selected = math.fsum(
            value for component, value in edges.items()
            if component in selected
        )
        event_omitted = mass - event_selected
        total_mass += mass
        selected_mass += event_selected
        omitted_mass += event_omitted
        precision = event_selected / mass
        event_precisions.append(precision)
        positive_messages += 1
        complete_messages += int(
            event_omitted <= 1e-12
            and selected
            and selected <= set(edges)
        )
        family = str(node.get("model_family", ""))
        family_total[family] += mass
        family_selected[family] += event_selected
    family_shares = [
        family_selected[family] / total
        for family, total in sorted(family_total.items())
        if total > 0.0
    ]
    weighted = [
        selected_mass / total_mass if total_mass else 0.0,
        omitted_mass / total_mass if total_mass else 0.0,
        min(1.0, selected_mass / max(1, len(selected))),
        (
            math.fsum(event_precisions) / len(event_precisions)
            if event_precisions else 0.0
        ),
        max(event_precisions, default=0.0),
        complete_messages / positive_messages if positive_messages else 0.0,
        (
            sum(value > 0.0 for value in family_shares)
            / len(family_shares)
            if family_shares else 0.0
        ),
        (
            math.fsum(family_shares) / len(family_shares)
            if family_shares else 0.0
        ),
    ]
    values = [*unweighted, *weighted]
    if (
        len(values) != len(EVENT_NAMES)
        or not all(math.isfinite(value) for value in values)
        or not selected <= all_components
    ):
        raise ContractError("invalid minimal event feature vector")
    action["_minimal_event_features"] = list(values)
    return values


def _feature_names(arm: str) -> tuple[str, ...]:
    if arm == "component_table":
        return tuple(LOCAL_NAMES)
    if arm == "event_unweighted":
        return tuple((*LOCAL_NAMES, *UNWEIGHTED_NAMES))
    if arm in ("event_weighted", "event_subject_shifted"):
        return tuple((*LOCAL_NAMES, *EVENT_NAMES))
    raise ContractError(f"unknown minimal evidence arm: {arm}")


def _event_values(
    action: Mapping[str, Any], arm: str,
) -> list[float]:
    if arm == "component_table":
        return []
    field = (
        "_minimal_event_features_shifted"
        if arm == "event_subject_shifted"
        else "_minimal_event_features"
    )
    values = action.get(field)
    if not isinstance(values, list) or len(values) != len(EVENT_NAMES):
        raise ContractError(f"{arm}: event feature cache not prepared")
    if arm == "event_unweighted":
        return list(values[:len(UNWEIGHTED_NAMES)])
    return list(values)


def action_features(
    graph: Mapping[str, Any],
    incumbent: Sequence[str],
    action: Mapping[str, Any],
    arm: str,
) -> list[float]:
    values = [
        *component_features(graph, incumbent, action),
        *_event_values(action, arm),
    ]
    if (
        len(values) != len(_feature_names(arm))
        or not all(math.isfinite(value) for value in values)
    ):
        raise ContractError(f"{arm}: invalid action feature vector")
    return values


def parity_audit(
    source_rows: Sequence[Mapping[str, Any]],
    minimal_rows: Sequence[Mapping[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
) -> dict[str, Any]:
    source_by = {_key(row): row for row in source_rows}
    minimal_by = {_key(row): row for row in minimal_rows}
    if set(source_by) != set(minimal_by) or set(source_by) != set(controls):
        raise ContractError("minimal evidence parity coverage mismatch")
    feature_matches = legal_matches = count_matches = 0
    support_edges = exact_events = aggregate_events = 0
    removed_nodes = removed_edges = 0
    for key in sorted(source_by):
        source = source_by[key]
        minimal = minimal_by[key]
        incumbent = list(controls[key])
        if cot40_count_anchor(source, incumbent) != cot40_count_anchor(
            minimal, incumbent
        ):
            raise ContractError(f"{key}: count-anchor parity failed")
        count_matches += 1
        source_actions = {
            (
                str(value["action_type"]),
                tuple(str(item) for item in value["component_ids"]),
            ): value
            for value in legal_actions(source, incumbent)
        }
        minimal_actions = {
            (
                str(value["action_type"]),
                tuple(str(item) for item in value["component_ids"]),
            ): value
            for value in legal_actions(minimal, incumbent)
        }
        if set(source_actions) != set(minimal_actions):
            raise ContractError(f"{key}: legal-action parity failed")
        legal_matches += 1
        for identity in sorted(source_actions):
            left = local_features(
                source, incumbent, source_actions[identity])
            right = component_features(
                minimal, incumbent, minimal_actions[identity])
            if left != right:
                raise ContractError(f"{key}: component feature parity failed")
            feature_matches += 1
        contract = minimal["minimal_evidence_contract"]
        exact_events += int(contract["exact_event_count"])
        aggregate_events += int(contract["aggregate_event_count"])
        support_edges += len(minimal["relational_graph"]["edges"])
        removed_nodes += (
            len(source["relational_graph"]["nodes"])
            - len(minimal["relational_graph"]["nodes"])
        )
        removed_edges += (
            len(source["relational_graph"]["edges"])
            - len(minimal["relational_graph"]["edges"])
        )
    return {
        "rows": len(source_by),
        "count_anchor_matches": count_matches,
        "legal_action_inventory_matches": legal_matches,
        "component_feature_matches": feature_matches,
        "exact_generation_events": exact_events,
        "aggregate_route_events": aggregate_events,
        "support_edges": support_edges,
        "removed_nodes_net": removed_nodes,
        "removed_edges_net": removed_edges,
        "parity_passed": True,
    }


def _prepare_event_cache(
    graphs: Sequence[dict[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
) -> dict[str, Any]:
    strata: dict[
        tuple[str, str],
        list[tuple[dict[str, Any], dict[str, Any], list[float]]],
    ] = defaultdict(list)
    total = 0
    for graph in sorted(graphs, key=_key):
        incumbent = controls[_key(graph)]
        for action in legal_actions(graph, incumbent):
            values = event_features(graph, action)
            strata[(
                str(graph["Relation"]),
                str(action["action_type"]),
            )].append((graph, action, values))
            total += 1
    changed = same_subject = 0
    stratum_audit = []
    for stratum in sorted(strata):
        values = sorted(
            strata[stratum],
            key=lambda item: (
                str(item[0]["SubjectEntity"]),
                tuple(str(value) for value in item[1]["component_ids"]),
            ),
        )
        size = len(values)
        offset = 0
        if size > 1:
            for candidate in range(1, size):
                if all(
                    str(values[index][0]["SubjectEntity"])
                    != str(values[(index + candidate) % size][0][
                        "SubjectEntity"])
                    for index in range(size)
                ):
                    offset = candidate
                    break
            if offset == 0:
                offset = max(1, size // 2)
        for index, (graph, action, original) in enumerate(values):
            source_graph, _, shifted = values[(index + offset) % size]
            action["_minimal_event_features_shifted"] = list(shifted)
            changed += int(list(original) != list(shifted))
            same_subject += int(
                str(graph["SubjectEntity"])
                == str(source_graph["SubjectEntity"])
            )
        stratum_audit.append({
            "relation": stratum[0],
            "action_type": stratum[1],
            "actions": size,
            "offset": offset,
        })
    return {
        "actions": total,
        "strata": len(strata),
        "changed_vectors": changed,
        "same_subject_assignments": same_subject,
        "stratum_offsets": stratum_audit,
    }


def _fit_model(
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
    return ResidualRidge(_feature_names(arm), l2).fit(x, y, weights)


def _propose(
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
    return (
        list(actions[best]["objects"]),
        float(estimates[best] - estimates[keep_index]),
        str(actions[best]["action_type"]),
        len(actions),
    )


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
    oof: dict[
        tuple[str, str], tuple[list[str], float, str, int]
    ] = {}
    diagnostics = []
    for outer in fold_ids:
        print(f"{arm}: outer fold {outer + 1}/{len(fold_ids)}", flush=True)
        outer_fit = [row for row in graphs if folds[_key(row)] != outer]
        outer_hold = [row for row in graphs if folds[_key(row)] == outer]
        fit_keys = {_key(row) for row in outer_fit}
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
                model = _fit_model(
                    inner_train, controls, gold, arm, l2)
                for graph in inner_hold:
                    key = _key(graph)
                    inner_proposals[key] = _propose(
                        model, graph, controls[key], arm)
            if set(inner_proposals) != fit_keys:
                raise ContractError("inner OOF proposal coverage mismatch")
            for guard in GUARDS:
                scores, audit, _ = _score_predictions(
                    inner_controls, inner_proposals, guard, gold)
                candidates.append((scores[POOLED], guard, l2, scores, audit))
        _, guard, l2, inner_scores, inner_audit = max(
            candidates, key=lambda value: (value[0], value[1], value[2]))
        model = _fit_model(outer_fit, controls, gold, arm, l2)
        hold_proposals = {}
        for graph in outer_hold:
            key = _key(graph)
            proposal = _propose(model, graph, controls[key], arm)
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
    if set(oof) != set(graph_by):
        raise ContractError("outer OOF proposal coverage mismatch")
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


def _paired_audit(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    before_by = {
        (str(row["SubjectEntity"]), str(row["Relation"])):
            list(row["selected"])
        for row in before["decisions"]
    }
    after_by = {
        (str(row["SubjectEntity"]), str(row["Relation"])):
            list(row["selected"])
        for row in after["decisions"]
    }
    if set(before_by) != set(after_by) or set(before_by) != set(gold):
        raise ContractError("paired minimal-event audit coverage mismatch")
    changed = helped = harmed = neutral = 0
    by_relation: dict[str, Counter[str]] = defaultdict(Counter)
    for key in sorted(before_by):
        if tuple(before_by[key]) == tuple(after_by[key]):
            continue
        changed += 1
        left = _row_f1(before_by[key], gold[key], key[1])
        right = _row_f1(after_by[key], gold[key], key[1])
        if right > left + 1e-12:
            outcome = "helped"
            helped += 1
        elif right < left - 1e-12:
            outcome = "harmed"
            harmed += 1
        else:
            outcome = "neutral"
            neutral += 1
        by_relation[key[1]]["changed"] += 1
        by_relation[key[1]][outcome] += 1
    return {
        "changed": changed,
        "helped": helped,
        "harmed": harmed,
        "neutral": neutral,
        "by_relation": {
            relation: dict(values)
            for relation, values in sorted(by_relation.items())
        },
    }


def prepare(args: argparse.Namespace) -> int:
    source = Path(args.source_run).resolve()
    output = Path(args.output_dir).resolve()
    source_plan, source_rows = _validate_prepared(source)
    base_rows, _ = compose_competition_train_oof()
    controls = {
        _key(row): list(row.get("ObjectEntities", []))
        for row in base_rows
    }
    minimal_rows = [minimalize_row(row) for row in source_rows]
    parity = parity_audit(source_rows, minimal_rows, controls)
    graph_path = output / "graph/MINIMAL_EVIDENCE_GRAPH.jsonl"
    write_jsonl_atomic(graph_path, minimal_rows)
    manifest_path = graph_path.with_suffix(
        graph_path.suffix + ".manifest.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "deployable": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "rows": len(minimal_rows),
        "node_types": ["candidate_component", "evidence_event"],
        "edge_types": ["supports"],
        "source_graph": str(source_plan["graph"]),
        "source_graph_sha256": sha256(Path(source_plan["graph"])),
        "output": str(graph_path),
        "output_sha256": sha256(graph_path),
        "parity": parity,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(manifest_path, manifest)
    plan = {
        "schema": PLAN_SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "development_only": True,
        "deployable": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "source_run": str(source),
        "source_plan": str(source / "plan/PLAN.json"),
        "source_plan_sha256": sha256(source / "plan/PLAN.json"),
        "source_graph": str(source_plan["graph"]),
        "source_graph_sha256": sha256(Path(source_plan["graph"])),
        "minimal_graph": str(graph_path),
        "minimal_graph_sha256": sha256(graph_path),
        "minimal_manifest": str(manifest_path),
        "minimal_manifest_sha256": sha256(manifest_path),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "arms": list(ARMS),
        "component_feature_names": list(LOCAL_NAMES),
        "unweighted_event_feature_names": list(UNWEIGHTED_NAMES),
        "weighted_event_feature_names": list(WEIGHTED_NAMES),
        "l2_values": list(L2_VALUES),
        "guards": list(GUARDS),
        "folding": "strict_subject_grouped_5_fold_nested_cv",
        "negative_control":
            "deterministic subject shift within relation/action_type",
        "parity": parity,
    }
    plan_path = output / "plan/PLAN.json"
    _write_json(plan_path, plan)
    print(json.dumps({
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "minimal_graph": str(graph_path),
        "minimal_graph_sha256": sha256(graph_path),
        "parity": parity,
    }, indent=2, sort_keys=True))
    return 0


def _validate_plan(
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    required = (
        ("source_plan", "source_plan_sha256"),
        ("source_graph", "source_graph_sha256"),
        ("minimal_graph", "minimal_graph_sha256"),
        ("minimal_manifest", "minimal_manifest_sha256"),
        ("implementation", "implementation_sha256"),
    )
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
        or plan.get("arms") != list(ARMS)
        or not all(
            sha256(Path(plan[path_field])) == plan[hash_field]
            for path_field, hash_field in required
        )
    ):
        raise ContractError("minimal evidence plan contract failed")
    rows = read_jsonl(Path(plan["minimal_graph"]))
    manifest = _json(Path(plan["minimal_manifest"]))
    if (
        len(rows) != 477
        or len({_key(row) for row in rows}) != 477
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("contains_labels") is not False
        or manifest.get("node_types")
            != ["candidate_component", "evidence_event"]
        or manifest.get("edge_types") != ["supports"]
        or manifest.get("output_sha256") != plan["minimal_graph_sha256"]
        or manifest.get("parity", {}).get("parity_passed") is not True
    ):
        raise ContractError("minimal evidence graph contract failed")
    return plan, rows


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan, graphs = _validate_plan(output)
    gold_path = Path(args.train_gold).resolve()
    gold_rows = read_jsonl(gold_path)
    gold = {_key(row): row for row in gold_rows}
    base_rows, base_detail = compose_competition_train_oof()
    base_controls = {
        _key(row): list(row.get("ObjectEntities", []))
        for row in base_rows
    }
    graph_by = {_key(row): row for row in graphs}
    if (
        len(gold) != 477
        or set(gold) != set(base_controls)
        or set(gold) != set(graph_by)
    ):
        raise ContractError("minimal evidence analysis coverage mismatch")
    count_anchor_rows = [{
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": str(row["Relation"]),
        "ObjectEntities": cot40_count_anchor(
            graph_by[_key(row)], base_controls[_key(row)]),
    } for row in base_rows]
    controls = {
        _key(row): list(row["ObjectEntities"])
        for row in count_anchor_rows
    }
    for graph in graphs:
        graph["baseline_objects"] = list(controls[_key(graph)])
        graph.pop("_graph_native_action_cache", None)
    shift_audit = _prepare_event_cache(graphs, controls)
    folds = subject_grouped_folds(graphs)
    fold_path = output / "analysis/SUBJECT_GROUPED_FOLDS.jsonl"
    write_jsonl_atomic(fold_path, [{
        "SubjectEntity": key[0],
        "Relation": key[1],
        "fold": fold,
    } for key, fold in sorted(folds.items())])

    incumbent_scores = score(count_anchor_rows, gold_rows)
    reports = {
        arm: _nested_oof(
            graphs, count_anchor_rows, controls, gold, folds, arm)
        for arm in ARMS
    }
    for arm, report in reports.items():
        replacements = {
            (str(value["SubjectEntity"]), str(value["Relation"])):
                value["selected"]
            for value in report["decisions"]
        }
        path = output / f"analysis/{arm.upper()}_OOF.jsonl"
        write_jsonl_atomic(
            path, _prediction_rows(count_anchor_rows, replacements))
        report["predictions"] = str(path)
        report["predictions_sha256"] = sha256(path)

    baseline = reports["component_table"]
    shifted = reports["event_subject_shifted"]
    comparisons = {}
    for arm in ARMS:
        report = reports[arm]
        fold_deltas = [
            right - left for left, right in zip(
                baseline["fold_scores"], report["fold_scores"])
        ]
        relation_deltas = {
            relation:
                report["scores"][relation] - baseline["scores"][relation]
            for relation in RELATIONS
        }
        comparisons[arm] = {
            "delta_vs_component_table":
                report["scores"][POOLED] - baseline["scores"][POOLED],
            "delta_vs_incumbent":
                report["scores"][POOLED] - incumbent_scores[POOLED],
            "fold_deltas_vs_component_table": fold_deltas,
            "fold_wins_vs_component_table": sum(
                value > 1e-12 for value in fold_deltas),
            "relation_deltas_vs_component_table": relation_deltas,
            "paired_audit_vs_component_table":
                _paired_audit(baseline, report, gold),
        }
    weighted = reports["event_weighted"]
    weighted_comparison = comparisons["event_weighted"]
    weighted_paired = weighted_comparison[
        "paired_audit_vs_component_table"]
    graph_gate_passed = bool(
        weighted_comparison["delta_vs_component_table"] >= MIN_INCREMENT
        and weighted_comparison["fold_wins_vs_component_table"]
            >= MIN_FOLD_WINS
        and min(
            weighted_comparison[
                "relation_deltas_vs_component_table"].values()
        ) >= MAX_RELATION_REGRESSION
        and weighted_paired["helped"] > weighted_paired["harmed"]
        and (
            weighted["scores"][POOLED] - shifted["scores"][POOLED]
            >= MIN_ALIGNED_OVER_SHIFTED
        )
    )
    ranked = sorted(
        ARMS,
        key=lambda arm: (
            reports[arm]["scores"][POOLED],
            comparisons[arm]["fold_wins_vs_component_table"],
        ),
        reverse=True,
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
        "minimal_graph": plan["minimal_graph"],
        "minimal_graph_sha256": plan["minimal_graph_sha256"],
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
        "incumbent_detail": base_detail,
        "incumbent_scores": incumbent_scores,
        "shift_audit": shift_audit,
        "arms": reports,
        "comparisons": comparisons,
        "ranked_arms": ranked,
        "best_arm": ranked[0],
        "graph_gate_passed": graph_gate_passed,
        "gate": {
            "minimum_increment": MIN_INCREMENT,
            "minimum_fold_wins": MIN_FOLD_WINS,
            "maximum_relation_regression": MAX_RELATION_REGRESSION,
            "minimum_aligned_over_shifted": MIN_ALIGNED_OVER_SHIFTED,
            "helpful_edits_must_exceed_harmful": True,
        },
        "methodology":
            "label-free minimal bipartite graph construction; exact parity; "
            "matched strict-subject nested OOF event-message ablation",
        "next_stage": (
            "freeze_minimal_event_decoder_for_separate_confirmation"
            if graph_gate_passed else
            "retain_minimal_graph_as_schema_and_add_new_evidence_not_more_edges"
        ),
    }
    result_path = output / "analysis/RESULT.json"
    _write_json(result_path, result)
    lines = [
        "# Minimal evidence graph ablation",
        "",
        "Train-only nested subject-grouped audit. Validation was not opened.",
        "",
        "| arm | OOF F1 | vs component table | vs incumbent | folds won | helped | harmed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ranked:
        report = reports[arm]
        comparison = comparisons[arm]
        paired = comparison["paired_audit_vs_component_table"]
        lines.append(
            f"| {arm} | {report['scores'][POOLED]:.6f} | "
            f"{comparison['delta_vs_component_table']:+.6f} | "
            f"{comparison['delta_vs_incumbent']:+.6f} | "
            f"{comparison['fold_wins_vs_component_table']}/5 | "
            f"{paired['helped']} | {paired['harmed']} |"
        )
    lines.extend([
        "",
        f"- Structural/action/feature parity: "
        f"**{plan['parity']['parity_passed']}**",
        f"- Exact generation events: "
        f"**{plan['parity']['exact_generation_events']}**",
        f"- Aggregate route events: "
        f"**{plan['parity']['aggregate_route_events']}**",
        f"- `supports` edges: **{plan['parity']['support_edges']}**",
        f"- Weighted aligned graph gate passed: "
        f"**{graph_gate_passed}**",
        f"- Best diagnostic arm: `{ranked[0]}`",
        f"- Next stage: `{result['next_stage']}`",
    ])
    result_md = output / "analysis/RESULT.md"
    result_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "result": str(result_path),
        "result_md": str(result_md),
        "incumbent": incumbent_scores[POOLED],
        "component_table": baseline["scores"][POOLED],
        "event_unweighted": reports["event_unweighted"]["scores"][POOLED],
        "event_weighted": weighted["scores"][POOLED],
        "event_subject_shifted": shifted["scores"][POOLED],
        "best_arm": ranked[0],
        "graph_gate_passed": graph_gate_passed,
        "validation_opened": False,
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source-run", default=str(DEFAULT_SOURCE))
    prepare_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT))
    prepare_parser.set_defaults(function=prepare)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT))
    analyze_parser.add_argument(
        "--train-gold", default=str(DEFAULT_GOLD))
    analyze_parser.set_defaults(function=analyze)
    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
