#!/usr/bin/env python3
"""Three-model component/action decoder over the current typed graph.

This closes the architectural mismatch left by adding Ministral candidate
nodes to a decoder whose feature schema only knew Qwen and Gemma.

The decoder:

* starts from the exact competition SOTA incumbent;
* applies the already train-approved unanimous-area Ministral correction as a
  locked safety anchor;
* enumerates complete row actions over typed candidate components;
* exposes Qwen, Gemma, System-2, and Ministral evidence separately;
* represents Ministral support, unanimity, and admission type explicitly;
* trains a single relation-conditioned residual utility model;
* produces subject-grouped OOF predictions before selecting one global guard;
* freezes validation predictions before opening validation labels.

The learned target is the change in row F1 caused by a complete action, not
candidate AUROC.  Validation is never used to select the model or guard.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate import try_parse_number
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.components.baseline_relative_route_decoder import (
    ResidualRidge,
)
from experiments.heterogeneous_agents.components.component_aware_decoder import (
    _action_tokens,
    _component_by_id,
    _members,
    actions_for,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import _key
from experiments.heterogeneous_agents.components.ministral_candidate_supply import MINISTRAL
from experiments.heterogeneous_agents.components.ministral_consistency_admission import ROUTE
from experiments.heterogeneous_agents.components.ministral_typed_validation_confirmation import (
    apply_frozen_policy,
)
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    _prob,
    _row_f1,
)
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    LIST_RELATIONS,
    NUMERIC_RELATIONS,
    SINGLE_RELATIONS,
    collapse_prediction,
)
from experiments.heterogeneous_agents.components.route_aware_candidate_graph import (
    ROUTE_GEMMA,
    ROUTE_QWEN_SC,
    ROUTE_QWEN_SYSTEM2,
)
from experiments.heterogeneous_agents.components.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    compose_competition_train_oof,
    competition_validation_predictions,
)


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_OUTPUT = RUNS / "three_model_component_decoder_20260729_v5"
DEFAULT_TRAIN_GRAPH = (
    RUNS / "ministral_typed_component_admission_20260729_v3/"
    "graph/TYPED_ADMITTED_GRAPH.jsonl"
)
DEFAULT_VALIDATION_GRAPH = (
    RUNS / "ministral_typed_validation_confirmation_20260729_v3/"
    "graph/TYPED_VALIDATION_GRAPH.jsonl"
)
DEFAULT_SOURCE_PLAN = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/plan/PLAN.json"
)
DEFAULT_TRAIN_GOLD = ROOT / "data/train.jsonl"
DEFAULT_VALIDATION_GOLD = ROOT / "data/val.jsonl"
RELATIONS = tuple(sorted(
    LIST_RELATIONS | SINGLE_RELATIONS | NUMERIC_RELATIONS))
MARGINS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
RESULT_SCHEMA = "three-model-component-decoder-result-v1"
MODEL_SCHEMA = "three-model-component-decoder-model-v1"
PREDICTION_SCHEMA = "three-model-component-decoder-predictions-v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _validate_graph(path: Path, split: str, rows: int) -> list[dict[str, Any]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ContractError(f"missing typed graph or manifest: {path}")
    manifest = _json(manifest_path)
    if (
        manifest.get("contains_labels") is not False
        or manifest.get("validation_labels_used") is not False
        or manifest.get("split") != split
        or manifest.get("rows") != rows
        or manifest.get("output_sha256") != sha256(path)
        or manifest.get("ministral_commitments_preserved") is not True
        or manifest.get("verification_queue_graph_connected") is not True
        or manifest.get("dormant_candidates_directly_outputtable") is not False
        or manifest.get("route_selection_normalization")
        != "canonical-agent-output-selection-v2"
        or manifest.get("route_selection_requires_agent_outputs") is not True
        or manifest.get("legacy_selected_by_fallback_allowed") is not False
    ):
        raise ContractError(f"{path}: invalid typed {split} graph")
    values = read_jsonl(path)
    if len(values) != rows or len({_key(row) for row in values}) != rows:
        raise ContractError(f"{path}: graph coverage mismatch")
    for row in values:
        assert_route_selection_provenance(row)
        agent = row.get("agents", {}).get(MINISTRAL, {})
        if (
            not agent.get("decoder_commitments_enabled", False)
            or not agent.get("existence", {}).get("available", False)
            or not agent.get("cardinality", {}).get("available", False)
        ):
            raise ContractError(
                f"{path}: row lacks enabled Ministral commitments: {_key(row)}")
        if any(
            node.get("output_eligible") is not False
            or node.get("dormant") is not True
            for node in row.get("dormant_candidates", [])
        ):
            raise ContractError(
                f"{path}: unsafe dormant candidate schema: {_key(row)}")
    return values


def assert_route_selection_provenance(
    graph: Mapping[str, Any],
) -> None:
    """Require canonical output-derived route flags with no legacy fallback."""
    key = _key(graph)
    normalization = graph.get("route_selection_normalization", {})
    if (
        normalization.get("schema")
        != "canonical-agent-output-selection-v2"
        or normalization.get("qwen_outputs_required") is not True
        or normalization.get("gemma_outputs_required") is not True
        or normalization.get("qwen_outputs_available") is not True
        or normalization.get("gemma_outputs_available") is not True
        or normalization.get("legacy_selected_by_fallback_allowed") is not False
    ):
        raise ContractError(
            f"{key}: route-selection provenance is not fail-closed")
    relation = str(graph["Relation"])
    outputs = graph.get("agent_outputs", {})
    route_by_agent = {
        QWEN: ROUTE_QWEN_SC,
        GEMMA: ROUTE_GEMMA,
    }
    for agent, route in route_by_agent.items():
        if agent not in outputs or not isinstance(outputs[agent], list):
            raise ContractError(f"{key}: missing canonical output for {agent}")
        selected_keys = {
            candidate_key
            for item in outputs[agent]
            if (candidate_key := canonical_key(str(item), relation))
        }
        for node in graph.get("candidates", []):
            evidence = node.get("routes", {}).get(route)
            if evidence is None:
                continue
            node_key = canonical_key(str(node.get("item", "")), relation)
            expected = bool(node_key and node_key in selected_keys)
            if bool(evidence.get("selected", False)) != expected:
                raise ContractError(
                    f"{key}: {route} selection disagrees with agent_outputs "
                    f"for {node.get('item')!r}")


def route_flag_population(
    graphs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for graph in graphs:
        relation = str(graph["Relation"])
        by_route = result.setdefault(relation, {})
        for route in (ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2, ROUTE_GEMMA, ROUTE):
            counts = by_route.setdefault(route, {"present": 0, "selected": 0})
            for node in graph.get("candidates", []):
                evidence = node.get("routes", {}).get(route)
                if evidence is None:
                    continue
                counts["present"] += 1
                counts["selected"] += int(bool(evidence.get("selected", False)))
    return result


def assert_route_flag_population_parity(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject flags that are degenerate on one split but active on the other."""
    populations = {
        "train": route_flag_population(train),
        "validation": route_flag_population(validation),
    }
    failures = []
    for relation in RELATIONS:
        for route in (ROUTE_QWEN_SC, ROUTE_QWEN_SYSTEM2, ROUTE_GEMMA, ROUTE):
            left = populations["train"].get(
                relation, {}).get(route, {"present": 0, "selected": 0})
            right = populations["validation"].get(
                relation, {}).get(route, {"present": 0, "selected": 0})
            active = (
                right if left["selected"] == 0 else left
                if right["selected"] == 0 else None
            )
            materially_active = bool(
                active is not None
                and active["selected"] >= max(
                    5, math.ceil(0.02 * active["present"]))
            )
            if (
                left["present"] and right["present"]
                and (left["selected"] == 0) != (right["selected"] == 0)
                and materially_active
            ):
                failures.append({
                    "relation": relation,
                    "route": route,
                    "train": left,
                    "validation": right,
                })
    if failures:
        raise ContractError(
            f"train/validation route-selection parity failure: {failures}")
    return populations


def subject_grouped_folds(
    graphs: Sequence[Mapping[str, Any]], n_folds: int = 5,
) -> dict[tuple[str, str], int]:
    """Deterministic relation-balanced folds with strict subject isolation."""
    if n_folds < 2:
        raise ContractError("at least two folds are required")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for graph in graphs:
        groups.setdefault(str(graph["SubjectEntity"]), []).append(graph)
    relation_counts = [Counter() for _ in range(n_folds)]
    totals = [0 for _ in range(n_folds)]
    assignments: dict[tuple[str, str], int] = {}
    ordered = sorted(groups.items(), key=lambda item: (
        -len(item[1]), item[0]))
    for _, rows in ordered:
        additions = Counter(str(row["Relation"]) for row in rows)
        fold = min(range(n_folds), key=lambda value: (
            sum(
                (relation_counts[value][relation] + count) ** 2
                - relation_counts[value][relation] ** 2
                for relation, count in additions.items()
            ),
            totals[value],
            value,
        ))
        for row in rows:
            assignments[_key(row)] = fold
        relation_counts[fold].update(additions)
        totals[fold] += len(rows)
    subject_folds: dict[str, set[int]] = {}
    for key, fold in assignments.items():
        subject_folds.setdefault(key[0], set()).add(fold)
    leaked = {
        subject: sorted(values) for subject, values in subject_folds.items()
        if len(values) != 1
    }
    if leaked:
        raise ContractError(f"subject grouping failed: {leaked}")
    return assignments


def _route_support(
    members: Sequence[Mapping[str, Any]], route: str,
) -> float:
    return max([
        float(node.get("routes", {}).get(
            route, {}).get("support_rate", 0.0))
        for node in members
    ] or [0.0])


def _route_selected(
    members: Sequence[Mapping[str, Any]], route: str,
) -> float:
    return float(any(
        bool(node.get("routes", {}).get(route, {}).get("selected", False))
        for node in members))


SUMMARY_NAMES = (
    "member_count",
    "alias_collapsed",
    "qwen_support",
    "system2_support",
    "gemma_support",
    "ministral_support",
    "qwen_selected",
    "system2_selected",
    "gemma_selected",
    "ministral_selected",
    "route_count",
    "independent_family_count",
    "cross_model",
    "three_model",
    "ministral_present",
    "ministral_two_of_three",
    "ministral_unanimous",
    "ministral_new",
    "ministral_corroborates",
    "ministral_numeric_new",
    "ministral_open_list_new",
    "numeric_spread",
    "component_numeric_surface_spread",
)


def _numeric_spread(values: Sequence[float]) -> float:
    positive = [float(value) for value in values if float(value) > 0]
    if len(positive) < 2:
        return 0.0
    return min(
        1.0,
        math.log(max(positive) / min(positive)) / math.log(1.05),
    )


def component_summary(
    graph: Mapping[str, Any],
    component: Mapping[str, Any] | None,
) -> dict[str, float]:
    members = _members(graph, component) if component is not None else []
    routes = {
        ROUTE_QWEN_SC: _route_support(members, ROUTE_QWEN_SC),
        ROUTE_QWEN_SYSTEM2: _route_support(members, ROUTE_QWEN_SYSTEM2),
        ROUTE_GEMMA: _route_support(members, ROUTE_GEMMA),
        ROUTE: _route_support(members, ROUTE),
    }
    qwen_family = routes[ROUTE_QWEN_SC] > 0 or routes[
        ROUTE_QWEN_SYSTEM2] > 0
    gemma_family = routes[ROUTE_GEMMA] > 0
    ministral_family = routes[ROUTE] > 0
    family_count = sum((qwen_family, gemma_family, ministral_family))
    ministral_evidence = [
        node.get("routes", {}).get(ROUTE, {})
        for node in members if ROUTE in node.get("routes", {})
    ]
    reasons = {
        str(value.get("admission_reason", ""))
        for value in ministral_evidence
    }
    numeric_values = [
        float(value) for node in members
        if (value := try_parse_number(str(node["item"]))) is not None
        and float(value) > 0
    ]
    surface_spread = _numeric_spread(numeric_values)
    cluster_values = []
    for evidence in ministral_evidence:
        recorded = evidence.get("cluster_member_values")
        if recorded is not None:
            cluster_values.extend(float(value) for value in recorded)
            continue
        for item in evidence.get("cluster_members", []):
            value = try_parse_number(str(item))
            if value is not None and float(value) > 0:
                cluster_values.append(float(value))
    cluster_spread = _numeric_spread(cluster_values)
    m_support = routes[ROUTE]
    return {
        "member_count": min(1.0, len(members) / 5.0),
        "alias_collapsed": float(len(members) > 1),
        "qwen_support": routes[ROUTE_QWEN_SC],
        "system2_support": routes[ROUTE_QWEN_SYSTEM2],
        "gemma_support": routes[ROUTE_GEMMA],
        "ministral_support": m_support,
        "qwen_selected": _route_selected(members, ROUTE_QWEN_SC),
        "system2_selected": _route_selected(
            members, ROUTE_QWEN_SYSTEM2),
        "gemma_selected": _route_selected(members, ROUTE_GEMMA),
        "ministral_selected": _route_selected(members, ROUTE),
        "route_count": sum(value > 0 for value in routes.values()) / 4.0,
        "independent_family_count": family_count / 3.0,
        "cross_model": float(family_count >= 2),
        "three_model": float(family_count == 3),
        "ministral_present": float(ministral_family),
        "ministral_two_of_three": float(m_support >= 2.0 / 3.0 - 1e-12),
        "ministral_unanimous": float(m_support >= 1.0 - 1e-12),
        "ministral_new": float(any(reason.endswith("_new") for reason in reasons)),
        "ministral_corroborates": float(any(
            "corroborates_source" in reason for reason in reasons)),
        "ministral_numeric_new": float(
            "numeric_complete_link_self_consistent_new" in reasons),
        "ministral_open_list_new": float(
            "open_list_exact_self_consistent_new" in reasons),
        # Backward-compatible name now has the intended semantics: within-
        # Ministral-proposal-cluster dispersion, not component alias spread.
        "numeric_spread": cluster_spread,
        "component_numeric_surface_spread": surface_spread,
    }


def _mean_summary(
    summaries: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    return {
        name: (
            statistics.mean(summary[name] for summary in summaries)
            if summaries else 0.0
        )
        for name in SUMMARY_NAMES
    }


def _token_summaries(
    graph: Mapping[str, Any], tokens: Sequence[str],
) -> list[dict[str, float]]:
    by_id = _component_by_id(graph)
    return [
        component_summary(graph, by_id.get(token))
        for token in tokens
    ]


def feature_names() -> list[str]:
    return [
        *[f"relation_{relation}" for relation in RELATIONS],
        "control_empty",
        "action_empty",
        "control_size",
        "action_size",
        "size_delta",
        "noop",
        "add",
        "drop",
        "replace",
        "multi_edit",
        "overlap",
        "collapse_action",
        *[f"control_{name}" for name in SUMMARY_NAMES],
        *[f"action_{name}" for name in SUMMARY_NAMES],
        *[f"added_{name}" for name in SUMMARY_NAMES],
        *[f"dropped_{name}" for name in SUMMARY_NAMES],
        "candidate_count",
        "component_count",
        "collapsed_surface_rate",
        "co_support_density",
        "qwen_none_rate",
        "gemma_none_rate",
        "dormant_candidate_count",
        "dormant_max_support",
        "dormant_two_of_three_rate",
        "dormant_numeric_rate",
        "qwen_exist_yes",
        "gemma_exist_yes",
        "ministral_exist_yes",
        "qwen_exist_no",
        "gemma_exist_no",
        "ministral_exist_no",
        "qwen_card_zero",
        "gemma_card_zero",
        "ministral_card_zero",
        "qwen_card_one",
        "gemma_card_one",
        "ministral_card_one",
        "qwen_card_many",
        "gemma_card_many",
        "ministral_card_many",
        "action_cardinality_gap",
        "numeric_log_distance_from_control",
        *[
            f"relation_{relation}_x_added_ministral_support"
            for relation in RELATIONS
        ],
        *[
            f"relation_{relation}_x_action_ministral_unanimous"
            for relation in RELATIONS
        ],
        *[
            f"relation_{relation}_x_size_delta_x_ministral_many"
            for relation in RELATIONS
        ],
        *[
            f"relation_{relation}_x_action_empty_x_ministral_no"
            for relation in RELATIONS
        ],
    ]


def action_features(
    graph: Mapping[str, Any],
    control: Sequence[str],
    action: Sequence[str],
) -> list[float]:
    relation = str(graph["Relation"])
    control_tokens = set(_action_tokens(graph, control, "component"))
    action_tokens = set(_action_tokens(graph, action, "component"))
    added = action_tokens - control_tokens
    dropped = control_tokens - action_tokens
    control_summary = _mean_summary(_token_summaries(
        graph, sorted(control_tokens)))
    action_summary = _mean_summary(_token_summaries(
        graph, sorted(action_tokens)))
    added_summary = _mean_summary(_token_summaries(graph, sorted(added)))
    dropped_summary = _mean_summary(_token_summaries(graph, sorted(dropped)))
    edit_count = len(added) + len(dropped)
    commitment_agents = [
        agent for agent in (QWEN, GEMMA, MINISTRAL)
        if graph["agents"][agent].get(
            "cardinality", {}).get("available", False)
    ]
    if not commitment_agents:
        raise ContractError("row has no available cardinality commitments")
    expected_cardinality = statistics.mean([
        _prob(graph, agent, "cardinality", "ONE")
        + 2.0 * _prob(graph, agent, "cardinality", "MANY")
        for agent in commitment_agents
    ])
    relational = graph["relational_graph"]["statistics"]
    surfaces = int(relational["surface_candidate_count"])
    components = int(relational["component_count"])
    co_support = int(relational["co_support_edge_count"])
    possible_pairs = components * (components - 1) / 2
    dormant = list(graph.get("dormant_candidates", []))
    dormant_support = [
        float(node.get("routes", {}).get(
            ROUTE, {}).get("support_rate", 0.0))
        for node in dormant
    ]
    numeric_distance = 0.0
    if relation in NUMERIC_RELATIONS and control and action:
        before = try_parse_number(str(control[0]))
        after = try_parse_number(str(action[0]))
        if before is not None and after is not None and before > 0 and after > 0:
            numeric_distance = min(
                1.0, abs(math.log(float(after) / float(before))) / 3.0)
    collapsed = collapse_prediction(graph, control)
    size_delta = max(-1.0, min(
        1.0, (len(action_tokens) - len(control_tokens)) / 3.0))
    ministral_many = _prob(
        graph, MINISTRAL, "cardinality", "MANY")
    ministral_no = _prob(graph, MINISTRAL, "existence", "NO")
    values = [
        *[float(relation == value) for value in RELATIONS],
        float(not control_tokens),
        float(not action_tokens),
        min(1.0, len(control_tokens) / 5.0),
        min(1.0, len(action_tokens) / 5.0),
        size_delta,
        float(edit_count == 0),
        float(bool(added) and not dropped),
        float(bool(dropped) and not added),
        float(bool(added) and bool(dropped) and edit_count == 2),
        float(edit_count > 2),
        len(control_tokens & action_tokens)
        / max(1, len(control_tokens | action_tokens)),
        float(
            _action_tokens(graph, action, "surface")
            == _action_tokens(graph, collapsed, "surface")
            and _action_tokens(graph, control, "surface")
            != _action_tokens(graph, collapsed, "surface")
        ),
        *[control_summary[name] for name in SUMMARY_NAMES],
        *[action_summary[name] for name in SUMMARY_NAMES],
        *[added_summary[name] for name in SUMMARY_NAMES],
        *[dropped_summary[name] for name in SUMMARY_NAMES],
        min(1.0, surfaces / 15.0),
        min(1.0, components / 15.0),
        (surfaces - components) / max(1, surfaces),
        co_support / max(1.0, possible_pairs),
        float(graph["agents"][QWEN]["none_rate"]),
        float(graph["agents"][GEMMA]["none_rate"]),
        min(1.0, len(dormant) / 15.0),
        max(dormant_support, default=0.0),
        (
            sum(value >= 2.0 / 3.0 - 1e-12 for value in dormant_support)
            / max(1, len(dormant_support))
        ),
        (
            sum(node.get("type") == "numeric" for node in dormant)
            / max(1, len(dormant))
        ),
        _prob(graph, QWEN, "existence", "YES"),
        _prob(graph, GEMMA, "existence", "YES"),
        _prob(graph, MINISTRAL, "existence", "YES"),
        _prob(graph, QWEN, "existence", "NO"),
        _prob(graph, GEMMA, "existence", "NO"),
        _prob(graph, MINISTRAL, "existence", "NO"),
        _prob(graph, QWEN, "cardinality", "ZERO"),
        _prob(graph, GEMMA, "cardinality", "ZERO"),
        _prob(graph, MINISTRAL, "cardinality", "ZERO"),
        _prob(graph, QWEN, "cardinality", "ONE"),
        _prob(graph, GEMMA, "cardinality", "ONE"),
        _prob(graph, MINISTRAL, "cardinality", "ONE"),
        _prob(graph, QWEN, "cardinality", "MANY"),
        _prob(graph, GEMMA, "cardinality", "MANY"),
        _prob(graph, MINISTRAL, "cardinality", "MANY"),
        min(1.0, abs(len(action_tokens) - expected_cardinality) / 4.0),
        numeric_distance,
        *[
            float(relation == value) * added_summary["ministral_support"]
            for value in RELATIONS
        ],
        *[
            float(relation == value) * action_summary["ministral_unanimous"]
            for value in RELATIONS
        ],
        *[
            float(relation == value) * size_delta * ministral_many
            for value in RELATIONS
        ],
        *[
            float(relation == value) * float(not action_tokens) * ministral_no
            for value in RELATIONS
        ],
    ]
    if len(values) != len(feature_names()):
        raise AssertionError("three-model component feature schema drift")
    if not all(math.isfinite(value) for value in values):
        raise ContractError("non-finite three-model component feature")
    return values


def legal_actions(
    graph: Mapping[str, Any],
    control: Sequence[str],
    locked: bool,
) -> list[list[str]]:
    if locked:
        return [list(control)]
    return actions_for(graph, control, "component")


def fit_model(
    graphs: Sequence[Mapping[str, Any]],
    controls: Mapping[tuple[str, str], Sequence[str]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    locked: set[tuple[str, str]],
    l2: float,
) -> ResidualRidge:
    x: list[list[float]] = []
    y: list[float] = []
    weights: list[float] = []
    for graph in graphs:
        key = _key(graph)
        control = list(controls[key])
        actions = legal_actions(graph, control, key in locked)
        before = _row_f1(control, gold[key], key[1])
        row_weight = 1.0 / len(actions)
        for action in actions:
            x.append(action_features(graph, control, action))
            y.append(_row_f1(action, gold[key], key[1]) - before)
            weights.append(row_weight)
    return ResidualRidge(feature_names(), l2).fit(x, y, weights)


def propose(
    model: ResidualRidge,
    graph: Mapping[str, Any],
    control: Sequence[str],
    locked: bool,
) -> tuple[list[str], float, int]:
    actions = legal_actions(graph, control, locked)
    estimates = model.predict([
        action_features(graph, control, action) for action in actions])
    control_tokens = _action_tokens(graph, control, "component")
    control_indices = [
        index for index, action in enumerate(actions)
        if _action_tokens(graph, action, "component") == control_tokens
    ]
    if len(control_indices) != 1:
        raise ContractError("legal action set lacks one canonical KEEP action")
    control_estimate = float(estimates[control_indices[0]])
    best = max(range(len(actions)), key=lambda index: (
        float(estimates[index]),
        -len(actions[index]),
        _action_tokens(graph, actions[index], "component"),
    ))
    # Absolute residual predictions have a learned intercept whose scale can
    # shift between outer-fold and full-data fits.  Actions are comparative:
    # use the predicted advantage over this row's KEEP action.  This cancels
    # the row/model intercept and makes a train-selected guard portable.
    advantage = float(estimates[best]) - control_estimate
    return list(actions[best]), advantage, len(actions)


def _prediction_rows(
    controls: Sequence[Mapping[str, Any]],
    replacements: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict[str, Any]]:
    return [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": list(replacements.get(
            _key(row), row.get("ObjectEntities", []))),
    } for row in controls]


def _write_predictions(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    model_path: Path,
    source_graph: Path,
    competition_control_detail: Mapping[str, Any],
    immediate_control_path: Path,
    train_selected_margin: float,
    hypothesis_created_after_validation_opened: bool,
) -> None:
    write_jsonl_atomic(path, rows)
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({
        "schema": PREDICTION_SCHEMA,
        "contains_labels": False,
        "gold_aware": True,
        "gold_awareness_scope": "train_oof_model_and_guard_only",
        "validation_opened": False,
        "validation_labels_used": False,
        "hypothesis_created_after_validation_opened":
            bool(hypothesis_created_after_validation_opened),
        "rows": len(rows),
        "output_sha256": sha256(path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "source_graph": str(source_graph),
        "source_graph_sha256": sha256(source_graph),
        "control_pipeline": competition_control_detail["pipeline_id"],
        "competition_control_predictions":
            competition_control_detail["prediction_path"],
        "competition_control_predictions_sha256":
            competition_control_detail["prediction_sha256"],
        "immediate_anchored_control_predictions": str(immediate_control_path),
        "immediate_anchored_control_predictions_sha256":
            sha256(immediate_control_path),
        "train_selected_margin": train_selected_margin,
    }, indent=2, sort_keys=True) + "\n")


def _changed(
    left: Sequence[str], right: Sequence[str], graph: Mapping[str, Any],
) -> bool:
    return _action_tokens(graph, left, "component") != _action_tokens(
        graph, right, "component")


def _evaluate_proposals(
    proposals: Mapping[tuple[str, str], tuple[list[str], float, int]],
    *,
    controls: Mapping[tuple[str, str], Sequence[str]],
    graph_by: Mapping[tuple[str, str], Mapping[str, Any]],
    gold: Mapping[tuple[str, str], Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    margins: Sequence[float],
) -> tuple[
    dict[float, list[dict[str, Any]]],
    dict[float, dict[str, float]],
    dict[float, dict[str, Any]],
]:
    rows_by_margin: dict[float, list[dict[str, Any]]] = {}
    scores_by_margin: dict[float, dict[str, float]] = {}
    audits_by_margin: dict[float, dict[str, Any]] = {}
    expected = {_key(row) for row in control_rows}
    if set(proposals) != expected:
        raise ContractError("proposal/control coverage mismatch")
    for margin in margins:
        replacements: dict[tuple[str, str], list[str]] = {}
        changed = helpful = harmful = neutral = 0
        relation_changes: Counter[str] = Counter()
        for key, (proposal, estimate, _) in proposals.items():
            control = list(controls[key])
            selected = proposal if estimate > margin else control
            replacements[key] = list(selected)
            graph = graph_by[key]
            if _changed(control, selected, graph):
                changed += 1
                relation_changes[key[1]] += 1
                before = _row_f1(control, gold[key], key[1])
                after = _row_f1(selected, gold[key], key[1])
                helpful += int(after > before + 1e-12)
                harmful += int(after < before - 1e-12)
                neutral += int(abs(after - before) <= 1e-12)
        rows = _prediction_rows(control_rows, replacements)
        rows_by_margin[margin] = rows
        scores_by_margin[margin] = score(
            rows, [gold[_key(row)] for row in rows])
        audits_by_margin[margin] = {
            "changed": changed,
            "helpful": helpful,
            "harmful": harmful,
            "neutral": neutral,
            "relation_changes": dict(relation_changes),
        }
    return rows_by_margin, scores_by_margin, audits_by_margin


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train_graph).resolve()
    validation_path = Path(args.validation_graph).resolve()
    source_plan_path = Path(args.source_plan).resolve()
    train_gold_path = Path(args.train_gold).resolve()
    validation_gold_path = Path(args.validation_gold).resolve()
    train_graphs_all = _validate_graph(train_path, "train", 477)
    validation_graphs = _validate_graph(validation_path, "validation", 478)
    train_graphs = [
        row for row in train_graphs_all
        if row.get("calibration_eligible", True) is not False
    ]
    source_plan = _json(source_plan_path)
    folds = subject_grouped_folds(train_graphs_all)
    if set(folds) != {_key(row) for row in train_graphs_all}:
        raise ContractError("folds do not cover typed train graph")
    folds_path = output / "SUBJECT_GROUPED_FOLDS.jsonl"
    write_jsonl_atomic(folds_path, [{
        "SubjectEntity": key[0],
        "Relation": key[1],
        "fold": fold,
    } for key, fold in sorted(folds.items())])
    train_gold = {
        _key(row): row for row in read_jsonl(train_gold_path)}
    if set(train_gold) != set(folds):
        raise ContractError("train gold does not cover folds")

    original_train, original_train_detail = compose_competition_train_oof()
    original_validation, validation_detail = (
        competition_validation_predictions())
    train_graph_by = {_key(row): row for row in train_graphs_all}
    validation_graph_by = {_key(row): row for row in validation_graphs}
    route_populations = assert_route_flag_population_parity(
        train_graphs_all, validation_graphs)
    anchor_train, anchor_train_decisions = apply_frozen_policy(
        original_train, train_graphs_all)
    anchor_validation, anchor_validation_decisions = apply_frozen_policy(
        original_validation, validation_graphs)
    anchor_train_path = output / "ANCHORED_TRAIN_CONTROL.jsonl"
    anchor_validation_path = output / "ANCHORED_VALIDATION_CONTROL.jsonl"
    write_jsonl_atomic(anchor_train_path, anchor_train)
    write_jsonl_atomic(anchor_validation_path, anchor_validation)
    train_control = {
        _key(row): list(row["ObjectEntities"]) for row in anchor_train}
    validation_control = {
        _key(row): list(row["ObjectEntities"]) for row in anchor_validation}
    locked_train = {
        _key(row) for row in anchor_train_decisions if row["changed"]}
    locked_validation = {
        _key(row) for row in anchor_validation_decisions if row["changed"]}
    eligible = {_key(row) for row in train_graphs}

    margins = tuple(float(value) for value in args.guard_margins.split(","))
    if (
        not margins
        or any(value < 0 or not math.isfinite(value) for value in margins)
    ):
        raise ContractError("invalid guard margins")
    fold_ids = sorted({folds[_key(row)] for row in train_graphs})
    oof_proposals: dict[tuple[str, str], tuple[list[str], float, int]] = {}
    oof_margin_by_key: dict[tuple[str, str], float] = {}
    fold_diagnostics: list[dict[str, Any]] = []
    outer_margins = []
    pooled = "*** All Relations ***"
    for outer_fold in fold_ids:
        outer_fit = [
            row for row in train_graphs if folds[_key(row)] != outer_fold]
        outer_hold = [
            row for row in train_graphs if folds[_key(row)] == outer_fold]
        inner_proposals: dict[
            tuple[str, str], tuple[list[str], float, int]] = {}
        for inner_fold in fold_ids:
            if inner_fold == outer_fold:
                continue
            inner_fit = [
                row for row in outer_fit if folds[_key(row)] != inner_fold]
            inner_hold = [
                row for row in outer_fit if folds[_key(row)] == inner_fold]
            inner_model = fit_model(
                inner_fit, train_control, train_gold, locked_train,
                args.residual_l2)
            for graph in inner_hold:
                key = _key(graph)
                inner_proposals[key] = propose(
                    inner_model, graph, train_control[key],
                    key in locked_train)
        inner_controls = [
            row for row in anchor_train if _key(row) in {
                _key(graph) for graph in outer_fit
            }
        ]
        _, inner_scores, inner_audits = _evaluate_proposals(
            inner_proposals,
            controls=train_control,
            graph_by=train_graph_by,
            gold=train_gold,
            control_rows=inner_controls,
            margins=margins,
        )
        inner_control_scores = score(
            inner_controls,
            [train_gold[_key(row)] for row in inner_controls])
        inner_margin = max(margins, key=lambda margin: (
            inner_scores[margin][pooled] - inner_control_scores[pooled],
            margin,
        ))
        outer_margins.append(inner_margin)
        model = fit_model(
            outer_fit, train_control, train_gold, locked_train,
            args.residual_l2)
        for graph in outer_hold:
            key = _key(graph)
            proposal, estimate, action_count = propose(
                model, graph, train_control[key], key in locked_train)
            oof_proposals[key] = (proposal, estimate, action_count)
            oof_margin_by_key[key] = inner_margin
        fold_diagnostics.append({
            "outer_fold": outer_fold,
            "inner_selected_margin": inner_margin,
            "inner_margin_scores": inner_scores,
            "inner_margin_audits": inner_audits,
            "fit_rows": len(outer_fit),
            "hold_rows": len(outer_hold),
        })

    if set(oof_proposals) != eligible:
        raise ContractError("OOF proposals do not cover eligible train rows")
    # Deployment guard is derived solely from inner folds. The outer OOF
    # labels never select it. Use the upper median for deterministic,
    # conservative aggregation across outer folds.
    selected_margin = sorted(outer_margins)[len(outer_margins) // 2]
    eligible_anchor = [
        row for row in anchor_train if _key(row) in eligible]
    (
        deployment_oof_rows_by_margin,
        deployment_oof_scores_by_margin,
        deployment_oof_audits_by_margin,
    ) = _evaluate_proposals(
        oof_proposals,
        controls=train_control,
        graph_by=train_graph_by,
        gold=train_gold,
        control_rows=eligible_anchor,
        margins=(selected_margin,),
    )
    deployment_oof_rows = deployment_oof_rows_by_margin[selected_margin]
    deployment_oof_scores = deployment_oof_scores_by_margin[selected_margin]
    deployment_oof_audit = deployment_oof_audits_by_margin[selected_margin]
    nested_replacements: dict[tuple[str, str], list[str]] = {}
    changed = helpful = harmful = neutral = 0
    relation_changes: Counter[str] = Counter()
    for key, (proposal, estimate, _) in oof_proposals.items():
        control = list(train_control[key])
        selected = (
            proposal if estimate > oof_margin_by_key[key] else control)
        nested_replacements[key] = list(selected)
        if _changed(control, selected, train_graph_by[key]):
            changed += 1
            relation_changes[key[1]] += 1
            before = _row_f1(control, train_gold[key], key[1])
            after = _row_f1(selected, train_gold[key], key[1])
            helpful += int(after > before + 1e-12)
            harmful += int(after < before - 1e-12)
            neutral += int(abs(after - before) <= 1e-12)
    nested_rows = _prediction_rows(eligible_anchor, nested_replacements)
    nested_scores = score(
        nested_rows, [train_gold[_key(row)] for row in nested_rows])
    selected_audit = {
        "changed": changed,
        "helpful": helpful,
        "harmful": harmful,
        "neutral": neutral,
        "relation_changes": dict(relation_changes),
    }
    anchor_train_scores = score(
        eligible_anchor, [train_gold[_key(row)] for row in eligible_anchor])
    selected_oof_delta = nested_scores[pooled] - anchor_train_scores[pooled]
    train_gate_passed = bool(
        selected_oof_delta > 1e-12
        and selected_audit["helpful"] > selected_audit["harmful"]
        and selected_audit["harmful"] <= max(2, selected_audit["helpful"] // 2)
    )

    final_model = fit_model(
        train_graphs, train_control, train_gold, locked_train,
        args.residual_l2)
    model_path = output / "MODEL.json"
    model_value = {
        "schema": MODEL_SCHEMA,
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "hypothesis_created_after_validation_opened":
            bool(args.hypothesis_created_after_validation_opened),
        "feature_schema": "qwen-gemma-system2-ministral-component-action-v2",
        "feature_names": feature_names(),
        "model": final_model.to_dict(),
        "residual_l2": args.residual_l2,
        "guard_margins": list(margins),
        "selected_margin": selected_margin,
        "selected_oof_delta": selected_oof_delta,
        "train_gate_passed": train_gate_passed,
        "selection_rule": (
            "five-fold strict-subject nested CV; each outer fold selects its "
            "guard on inner OOF only; deployment guard is the deterministic "
            "upper median of outer inner-selected guards; deploy only for "
            "positive outer OOF delta with more helpful than harmful edits "
            "and harmful <= max(2, helpful/2); guard applies to predicted "
            "action advantage over row KEEP; fail closed if global or any "
            "per-relation validation action rate exceeds twice the rate "
            "obtained by applying the fixed deployment guard to outer-OOF "
            "predictions"
        ),
        "score_semantics": "predicted_action_utility_minus_predicted_keep_utility",
        "nested_outer_fold_diagnostics": fold_diagnostics,
        "outer_inner_selected_margins": outer_margins,
        "nested_outer_oof_score": nested_scores,
        "nested_outer_oof_action_audit": selected_audit,
        "deployment_margin_oof_score": deployment_oof_scores,
        "deployment_margin_oof_action_audit": deployment_oof_audit,
        "route_flag_populations": route_populations,
        "train_graph": str(train_path),
        "train_graph_sha256": sha256(train_path),
        "train_gold": str(train_gold_path),
        "train_gold_sha256": sha256(train_gold_path),
        "folds": str(folds_path),
        "folds_sha256": sha256(folds_path),
        "train_rows": len(train_graphs),
        "excluded_train_rows": len(train_graphs_all) - len(train_graphs),
        "anchor_policy": "area_unanimous_new_component_replace",
        "anchor_locked_train_rows": len(locked_train),
        "anchored_train_control": str(anchor_train_path),
        "anchored_train_control_sha256": sha256(anchor_train_path),
        "anchored_validation_control": str(anchor_validation_path),
        "anchored_validation_control_sha256": sha256(anchor_validation_path),
        "control_pipeline": COMPETITION_PIPELINE_ID,
        "control_train_detail": original_train_detail,
    }
    model_path.write_text(json.dumps(
        model_value, indent=2, sort_keys=True) + "\n")
    oof_path = output / "TRAIN_OOF_PREDICTIONS.jsonl"
    write_jsonl_atomic(oof_path, nested_rows)
    oof_path.with_suffix(oof_path.suffix + ".manifest.json").write_text(
        json.dumps({
            "schema": "three-model-component-decoder-oof-v1",
            "split": "train",
            "contains_labels": False,
            "gold_aware": True,
            "rows": len(nested_rows),
            "subject_grouped_oof": True,
            "oof_model_excludes_subject": True,
            "output_sha256": sha256(oof_path),
            "model": str(model_path),
            "model_sha256": sha256(model_path),
        }, indent=2, sort_keys=True) + "\n")
    deployment_oof_path = (
        output / "TRAIN_OOF_DEPLOYMENT_MARGIN_PREDICTIONS.jsonl")
    write_jsonl_atomic(deployment_oof_path, deployment_oof_rows)
    deployment_oof_path.with_suffix(
        deployment_oof_path.suffix + ".manifest.json").write_text(
            json.dumps({
                "schema":
                    "three-model-component-decoder-deployment-margin-oof-v1",
                "split": "train",
                "contains_labels": False,
                "gold_aware": True,
                "rows": len(deployment_oof_rows),
                "subject_grouped_oof": True,
                "oof_model_excludes_subject": True,
                "fixed_deployment_margin": selected_margin,
                "purpose": "like-for-like validation action-rate ceiling",
                "output_sha256": sha256(deployment_oof_path),
                "model": str(model_path),
                "model_sha256": sha256(model_path),
            }, indent=2, sort_keys=True) + "\n")

    validation_replacements: dict[tuple[str, str], list[str]] = {}
    validation_diagnostics: list[dict[str, Any]] = []
    for graph in validation_graphs:
        key = _key(graph)
        control = validation_control[key]
        proposal, estimate, action_count = propose(
            final_model, graph, control, key in locked_validation)
        projected_changed = bool(
            estimate > selected_margin
            and _changed(control, proposal, graph)
        )
        selected = (
            proposal
            if train_gate_passed and estimate > selected_margin
            else control
        )
        validation_replacements[key] = list(selected)
        validation_diagnostics.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "control": control,
            "proposal": proposal,
            "selected": selected,
            "estimated_f1_delta": estimate,
            "guard_margin": selected_margin,
            "action_count": action_count,
            "anchor_locked": key in locked_validation,
            "deployment_gate_passed": train_gate_passed,
            "projected_changed": projected_changed,
            "changed": _changed(control, selected, graph),
        })
    projected_changes = sum(
        bool(row["projected_changed"]) for row in validation_diagnostics)
    projected_change_ceiling = math.ceil(
        2.0 * deployment_oof_audit["changed"]
        * len(validation_graphs) / max(1, len(eligible)))
    projected_relation_changes = Counter(
        str(row["Relation"]) for row in validation_diagnostics
        if row["projected_changed"])
    train_relation_rows = Counter(
        str(row["Relation"]) for row in train_graphs)
    validation_relation_rows = Counter(
        str(row["Relation"]) for row in validation_graphs)
    relation_change_ceilings = {
        relation: math.ceil(
            2.0
            * int(deployment_oof_audit["relation_changes"].get(relation, 0))
            * validation_relation_rows[relation]
            / max(1, train_relation_rows[relation])
        )
        for relation in RELATIONS
    }
    relation_action_rate_stability_gate = all(
        projected_relation_changes[relation]
        <= relation_change_ceilings[relation]
        for relation in RELATIONS
    )
    global_action_rate_stability_gate = (
        projected_changes <= projected_change_ceiling)
    action_rate_stability_gate = (
        global_action_rate_stability_gate
        and relation_action_rate_stability_gate
    )
    deploy_model = train_gate_passed and action_rate_stability_gate
    if not deploy_model:
        for row in validation_diagnostics:
            key = _key(row)
            row["selected"] = list(row["control"])
            row["changed"] = False
            row["deployment_gate_passed"] = False
            validation_replacements[key] = list(row["control"])
    model_value.update({
        "projected_validation_changes": projected_changes,
        "projected_validation_change_ceiling": projected_change_ceiling,
        "projected_validation_relation_changes":
            dict(projected_relation_changes),
        "projected_validation_relation_change_ceilings":
            relation_change_ceilings,
        "global_action_rate_stability_gate_passed":
            global_action_rate_stability_gate,
        "relation_action_rate_stability_gate_passed":
            relation_action_rate_stability_gate,
        "action_rate_stability_gate_passed": action_rate_stability_gate,
        "deployment_gate_passed": deploy_model,
        "deployment_margin_oof_predictions": str(deployment_oof_path),
        "deployment_margin_oof_predictions_sha256":
            sha256(deployment_oof_path),
    })
    model_path.write_text(json.dumps(
        model_value, indent=2, sort_keys=True) + "\n")
    # The model gained only label-free stability metadata. Refresh the OOF
    # manifest so its model hash continues to certify the exact artifact.
    oof_path.with_suffix(oof_path.suffix + ".manifest.json").write_text(
        json.dumps({
            "schema": "three-model-component-decoder-oof-v1",
            "split": "train",
            "contains_labels": False,
            "gold_aware": True,
            "rows": len(nested_rows),
            "subject_grouped_oof": True,
            "oof_model_excludes_subject": True,
            "output_sha256": sha256(oof_path),
            "model": str(model_path),
            "model_sha256": sha256(model_path),
        }, indent=2, sort_keys=True) + "\n")
    deployment_oof_path.with_suffix(
        deployment_oof_path.suffix + ".manifest.json").write_text(
            json.dumps({
                "schema":
                    "three-model-component-decoder-deployment-margin-oof-v1",
                "split": "train",
                "contains_labels": False,
                "gold_aware": True,
                "rows": len(deployment_oof_rows),
                "subject_grouped_oof": True,
                "oof_model_excludes_subject": True,
                "fixed_deployment_margin": selected_margin,
                "purpose": "like-for-like validation action-rate ceiling",
                "output_sha256": sha256(deployment_oof_path),
                "model": str(model_path),
                "model_sha256": sha256(model_path),
            }, indent=2, sort_keys=True) + "\n")
    predictions = _prediction_rows(
        anchor_validation, validation_replacements)
    prediction_path = output / "VALIDATION_PREDICTIONS.jsonl"
    _write_predictions(
        prediction_path,
        predictions,
        model_path=model_path,
        source_graph=validation_path,
        competition_control_detail=validation_detail,
        immediate_control_path=anchor_validation_path,
        train_selected_margin=selected_margin,
        hypothesis_created_after_validation_opened=
            bool(args.hypothesis_created_after_validation_opened),
    )
    diagnostics_path = output / "VALIDATION_DIAGNOSTICS.jsonl"
    write_jsonl_atomic(diagnostics_path, validation_diagnostics)

    # Only now may validation labels be opened.
    validation_gold = read_jsonl(validation_gold_path)
    scores = {
        "competition_control": score(original_validation, validation_gold),
        "anchored_control": score(anchor_validation, validation_gold),
        "integrated_decoder": score(predictions, validation_gold),
        "train_original_control": score(
            original_train,
            [train_gold[_key(row)] for row in original_train]),
        "train_anchored_control": anchor_train_scores,
        "train_integrated_oof":
            nested_scores,
    }
    changed_rows = [
        row for row in validation_diagnostics if row["changed"]]
    helpful = harmful = neutral = 0
    for row in changed_rows:
        key = _key(row)
        before = _row_f1(
            row["control"],
            {_key(value): value for value in validation_gold}[key],
            key[1],
        )
        after = _row_f1(
            row["selected"],
            {_key(value): value for value in validation_gold}[key],
            key[1],
        )
        helpful += int(after > before + 1e-12)
        harmful += int(after < before - 1e-12)
        neutral += int(abs(after - before) <= 1e-12)
    result = {
        "schema": RESULT_SCHEMA,
        "contains_labels": True,
        "gold_aware": True,
        "development_only": True,
        "deployable": False,
        "validation_opened": True,
        "validation_labels_used_for_selection": False,
        "hypothesis_created_after_validation_opened":
            bool(args.hypothesis_created_after_validation_opened),
        "feature_schema": model_value["feature_schema"],
        "scores": scores,
        "train_oof_delta_over_anchor": selected_oof_delta,
        "validation_delta_over_competition":
            scores["integrated_decoder"][pooled]
            - scores["competition_control"][pooled],
        "validation_delta_over_anchor":
            scores["integrated_decoder"][pooled]
            - scores["anchored_control"][pooled],
        "selected_margin": selected_margin,
        "deployment_gate_passed": deploy_model,
        "train_gate_passed": train_gate_passed,
        "action_rate_stability_gate_passed": action_rate_stability_gate,
        "global_action_rate_stability_gate_passed":
            global_action_rate_stability_gate,
        "relation_action_rate_stability_gate_passed":
            relation_action_rate_stability_gate,
        "projected_validation_changes": projected_changes,
        "projected_validation_change_ceiling":
            projected_change_ceiling,
        "projected_validation_relation_changes":
            dict(projected_relation_changes),
        "projected_validation_relation_change_ceilings":
            relation_change_ceilings,
        "nested_outer_inner_selected_margins": outer_margins,
        "nested_policy_oof_action_audit": selected_audit,
        "deployment_margin_oof_action_audit": deployment_oof_audit,
        "deployment_margin_oof_score": deployment_oof_scores,
        "deployment_margin_oof_predictions": str(deployment_oof_path),
        "deployment_margin_oof_predictions_sha256":
            sha256(deployment_oof_path),
        "validation_action_audit": {
            "changed": len(changed_rows),
            "helpful": helpful,
            "harmful": harmful,
            "neutral": neutral,
            "relation_changes": dict(Counter(
                row["Relation"] for row in changed_rows)),
        },
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "validation_diagnostics": str(diagnostics_path),
        "validation_diagnostics_sha256": sha256(diagnostics_path),
        "train_graph": str(train_path),
        "train_graph_sha256": sha256(train_path),
        "validation_graph": str(validation_path),
        "validation_graph_sha256": sha256(validation_path),
        "validation_gold": str(validation_gold_path),
        "validation_gold_sha256": sha256(validation_gold_path),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(
        result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Three-model component decoder",
        "",
        "The decoder was trained and its validation predictions were frozen "
        "before validation labels were opened.",
        (
            "This specific hypothesis was created after earlier validation "
            "outcomes had been inspected and is post-hoc development evidence."
            if args.hypothesis_created_after_validation_opened
            else "The hypothesis predates validation inspection."
        ),
        "",
        f"- Competition control: "
        f"**{scores['competition_control'][pooled]:.6f}**",
        f"- Frozen Ministral anchor: "
        f"**{scores['anchored_control'][pooled]:.6f}**",
        f"- Integrated three-model decoder: "
        f"**{scores['integrated_decoder'][pooled]:.6f}**",
        f"- Integrated delta over competition: "
        f"**{result['validation_delta_over_competition']:+.6f}**",
        f"- Integrated delta over anchor: "
        f"**{result['validation_delta_over_anchor']:+.6f}**",
        f"- Train OOF delta over anchor: "
        f"**{selected_oof_delta:+.6f}**",
        f"- Selected guard: **{selected_margin:.3f}**",
        f"- Fixed-guard OOF edits used for rate ceilings: "
        f"**{deployment_oof_audit['changed']}**",
        f"- Deployment gate: **{'PASS' if deploy_model else 'FAIL'}**",
        f"- Train/action-rate gates: "
        f"**{'PASS' if train_gate_passed else 'FAIL'} / "
        f"{'PASS' if action_rate_stability_gate else 'FAIL'}**",
        f"- Projected validation edits / ceiling: "
        f"**{projected_changes} / {projected_change_ceiling}**",
        f"- Validation edits (help/harm/neutral): "
        f"**{len(changed_rows)} "
        f"({helpful}/{harmful}/{neutral})**",
        "",
        "| relation | competition | anchor | integrated |",
        "|---|---:|---:|---:|",
    ]
    for relation in sorted(scores["competition_control"]):
        lines.append(
            f"| {relation} | "
            f"{scores['competition_control'][relation]:.4f} | "
            f"{scores['anchored_control'][relation]:.4f} | "
            f"{scores['integrated_decoder'][relation]:.4f} |")
    lines.extend([
        "",
        "The competition incumbent has validation-selected lineage, so this "
        "is a development comparison rather than a blind-test claim.",
        "",
    ])
    (output / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps({
        "competition": scores["competition_control"][pooled],
        "anchor": scores["anchored_control"][pooled],
        "integrated": scores["integrated_decoder"][pooled],
        "delta_over_competition":
            result["validation_delta_over_competition"],
        "delta_over_anchor": result["validation_delta_over_anchor"],
        "train_oof_delta_over_anchor": selected_oof_delta,
        "selected_margin": selected_margin,
        "deployment_gate_passed": deploy_model,
        "validation_edits": len(changed_rows),
        "result": str(output / "RESULT.md"),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    result.add_argument("--train-graph", default=str(DEFAULT_TRAIN_GRAPH))
    result.add_argument(
        "--validation-graph", default=str(DEFAULT_VALIDATION_GRAPH))
    result.add_argument("--source-plan", default=str(DEFAULT_SOURCE_PLAN))
    result.add_argument("--train-gold", default=str(DEFAULT_TRAIN_GOLD))
    result.add_argument(
        "--validation-gold", default=str(DEFAULT_VALIDATION_GOLD))
    result.add_argument("--residual-l2", type=float, default=10.0)
    result.add_argument(
        "--guard-margins",
        default=",".join(str(value) for value in MARGINS))
    result.add_argument(
        "--hypothesis-created-after-validation-opened",
        action="store_true",
        help=(
            "Mark post-hoc development runs whose design followed inspection "
            "of validation outcomes. This never makes validation a selector."
        ),
    )
    return result


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
