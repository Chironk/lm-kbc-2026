#!/usr/bin/env python3
"""Leakage-controlled surface-versus-component decoder ablation.

The experiment holds the candidate reservoir, proposal routes, OOF training
incumbents, validation incumbent, folds, and residual learner fixed.  Its two
arms differ only in the unit on which edits and evidence are represented:

``surface``
    One action/evidence record per raw candidate surface.
``component``
    Alias or numeric-equivalent surfaces are one action.  Route evidence is
    pooled across every member of the component.

System-2-only identities are excluded in both arms because the preceding
route-residual ablation found that expanding the reservoir with those
identities did not pass its training gate.  Validation labels are opened only
after all arm and train-selected prediction files have been written.
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluate import try_parse_number
from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.components.baseline_relative_route_decoder import (
    ResidualRidge,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    _key,
    _load_graph,
)
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    _prob,
    _row_f1,
    _selection,
)
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    LIST_RELATIONS,
    NUMERIC_RELATIONS,
    SINGLE_RELATIONS,
    _component_for_prediction,
    collapse_prediction,
)
from experiments.heterogeneous_agents.components.route_aware_candidate_graph import (
    ROUTE_GEMMA,
    ROUTE_QWEN_SC,
    ROUTE_QWEN_SYSTEM2,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "relational_graph_v1_20260723")
DEFAULT_CONTROL = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "route_residual_decoder_20260723_v1/"
    "validation_train_selected.jsonl")
ARMS = ("surface", "component")
RELATIONS = tuple(sorted(
    LIST_RELATIONS | SINGLE_RELATIONS | NUMERIC_RELATIONS))
DEFAULT_MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _object_key(item: str, relation: str) -> str:
    return canonical_key(str(item), relation)


def _surface_action_key(
        objects: Sequence[str], relation: str,
) -> tuple[str, ...]:
    return tuple(sorted({
        _object_key(str(item), relation) for item in objects
        if _object_key(str(item), relation)
    }))


def _component_token(
        graph: Mapping[str, Any], item: str,
) -> str:
    component = _component_for_prediction(graph, str(item))
    if component is not None:
        return str(component["id"])
    return f"surface:{_object_key(str(item), str(graph['Relation']))}"


def _action_tokens(
        graph: Mapping[str, Any], objects: Sequence[str], arm: str,
) -> tuple[str, ...]:
    if arm == "surface":
        return _surface_action_key(objects, str(graph["Relation"]))
    if arm != "component":
        raise ContractError(f"unknown decoder arm {arm}")
    return tuple(sorted({
        _component_token(graph, str(item)) for item in objects
    }))


def _dedupe_actions(
        graph: Mapping[str, Any], actions: Sequence[Sequence[str]], arm: str,
) -> list[list[str]]:
    unique: dict[tuple[str, ...], list[str]] = {}
    for action in actions:
        values = list(dict.fromkeys(str(item) for item in action))
        unique.setdefault(_action_tokens(graph, values, arm), values)
    return list(unique.values())


def _system2_only_candidate(node: Mapping[str, Any]) -> bool:
    summary = node.get("route_summary", {})
    if isinstance(summary, Mapping):
        return bool(summary.get("system2_only", False))
    routes = set(node.get("routes", {}))
    return routes == {ROUTE_QWEN_SYSTEM2}


def _eligible_candidates(
        graph: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    return [
        node for node in graph.get("candidates", [])
        if not _system2_only_candidate(node)
    ]


def _candidate_by_id(
        graph: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        f"candidate:{index}": node
        for index, node in enumerate(graph.get("candidates", []))
    }


def _eligible_components(
        graph: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    candidate_by_id = _candidate_by_id(graph)
    output = []
    for component in graph["relational_graph"]["components"]:
        members = [
            candidate_by_id[candidate_id]
            for candidate_id in component["member_candidate_ids"]]
        if any(not _system2_only_candidate(node) for node in members):
            output.append(component)
    return output


def actions_for(
        graph: Mapping[str, Any], control: Sequence[str], arm: str,
) -> list[list[str]]:
    """Enumerate bounded, inference-legal edits around the incumbent."""
    relation = str(graph["Relation"])
    if arm == "surface":
        representatives = [
            str(node["item"]) for node in _eligible_candidates(graph)]
        normalized_control = list(control)
    elif arm == "component":
        representatives = [
            str(component["representative"])
            for component in _eligible_components(graph)]
        normalized_control = collapse_prediction(graph, control)
    else:
        raise ContractError(f"unknown decoder arm {arm}")

    actions: list[list[str]] = [list(control), [], normalized_control]
    if relation in SINGLE_RELATIONS | NUMERIC_RELATIONS:
        actions.extend([[item] for item in representatives])
    elif relation in LIST_RELATIONS:
        current_tokens = set(_action_tokens(graph, normalized_control, arm))
        for item in representatives:
            if _action_tokens(graph, [item], arm)[0] not in current_tokens:
                actions.append([*normalized_control, item])
        for index in range(len(normalized_control)):
            actions.append(
                normalized_control[:index] + normalized_control[index + 1:])
    else:
        raise ContractError(f"unsupported relation {relation}")
    return _dedupe_actions(graph, actions, arm)


def _members(
        graph: Mapping[str, Any], component: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    candidate_by_id = _candidate_by_id(graph)
    return [
        candidate_by_id[candidate_id]
        for candidate_id in component["member_candidate_ids"]
        if candidate_id in candidate_by_id
        and not _system2_only_candidate(candidate_by_id[candidate_id])
    ]


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


def _component_summary(
        graph: Mapping[str, Any], component: Mapping[str, Any] | None,
        surface_node: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    if component is not None:
        members = _members(graph, component)
        alias_collapsed = float(len(members) > 1)
    elif surface_node is not None:
        members = [surface_node]
        alias_collapsed = 0.0
    else:
        members, alias_collapsed = [], 0.0
    qwen = _route_support(members, ROUTE_QWEN_SC)
    system2 = _route_support(members, ROUTE_QWEN_SYSTEM2)
    gemma = _route_support(members, ROUTE_GEMMA)
    routes_present = sum(value > 0 for value in (qwen, system2, gemma))
    independent_families = int(qwen > 0 or system2 > 0) + int(gemma > 0)
    numeric_values = [
        float(value) for node in members
        if (value := try_parse_number(str(node["item"]))) is not None
        and float(value) > 0]
    numeric_spread = 0.0
    if len(numeric_values) > 1:
        numeric_spread = min(
            1.0, math.log(max(numeric_values) / min(numeric_values))
            / math.log(1.05))
    return {
        "member_count": min(1.0, len(members) / 5.0),
        "alias_collapsed": alias_collapsed,
        "qwen_support": qwen,
        "system2_support": system2,
        "gemma_support": gemma,
        "qwen_selected": _route_selected(members, ROUTE_QWEN_SC),
        "system2_selected": _route_selected(
            members, ROUTE_QWEN_SYSTEM2),
        "gemma_selected": _route_selected(members, ROUTE_GEMMA),
        "route_count": routes_present / 3.0,
        "independent_family_count": independent_families / 2.0,
        "cross_model": float(independent_families == 2),
        "within_qwen": float(qwen > 0 and system2 > 0),
        "numeric_spread": numeric_spread,
    }


def _component_by_id(
        graph: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(component["id"]): component
        for component in graph["relational_graph"]["components"]
    }


def _surface_by_key(
        graph: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        _object_key(str(node["item"]), str(graph["Relation"])): node
        for node in _eligible_candidates(graph)
    }


SUMMARY_NAMES = (
    "member_count", "alias_collapsed", "qwen_support",
    "system2_support", "gemma_support", "qwen_selected",
    "system2_selected", "gemma_selected", "route_count",
    "independent_family_count", "cross_model", "within_qwen",
    "numeric_spread",
)


def feature_names() -> list[str]:
    return [
        "control_empty", "action_empty", "control_size", "action_size",
        "size_delta", "noop", "add", "drop", "replace", "multi_edit",
        "overlap", "component_arm", "collapse_action",
        *[f"added_{name}" for name in SUMMARY_NAMES],
        *[f"dropped_{name}" for name in SUMMARY_NAMES],
        "added_component_count", "dropped_component_count",
        "candidate_count", "component_count",
        "collapsed_surface_rate", "co_support_density",
        "qwen_none_rate", "gemma_none_rate",
        "qwen_exist_yes", "gemma_exist_yes",
        "qwen_exist_no", "gemma_exist_no",
        "qwen_card_zero", "gemma_card_zero",
        "qwen_card_one", "gemma_card_one",
        "qwen_card_many", "gemma_card_many",
        "action_cardinality_gap", "numeric_log_distance_from_control",
    ]


def _mean_summary(
        summaries: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    return {
        name: statistics.mean(summary[name] for summary in summaries)
        if summaries else 0.0
        for name in SUMMARY_NAMES
    }


def _token_summaries(
        graph: Mapping[str, Any], tokens: Sequence[str], arm: str,
) -> list[dict[str, float]]:
    if arm == "component":
        by_id = _component_by_id(graph)
        return [
            _component_summary(graph, by_id.get(token))
            for token in tokens]
    by_key = _surface_by_key(graph)
    return [
        _component_summary(
            graph, None,
            by_key.get(token.removeprefix("surface:")))
        for token in tokens]


def action_features(
        graph: Mapping[str, Any], control: Sequence[str],
        action: Sequence[str], arm: str,
) -> list[float]:
    relation = str(graph["Relation"])
    control_tokens = set(_action_tokens(graph, control, arm))
    action_tokens = set(_action_tokens(graph, action, arm))
    added, dropped = action_tokens - control_tokens, control_tokens - action_tokens
    edit_count = len(added) + len(dropped)
    added_summaries = _token_summaries(
        graph, sorted(added), arm)
    dropped_summaries = _token_summaries(
        graph, sorted(dropped), arm)
    added_summary = _mean_summary(added_summaries)
    dropped_summary = _mean_summary(dropped_summaries)
    expected_cardinality = statistics.mean([
        _prob(graph, agent, "cardinality", "ONE")
        + 2.0 * _prob(graph, agent, "cardinality", "MANY")
        for agent in (QWEN, GEMMA)
    ])
    relational_stats = graph["relational_graph"]["statistics"]
    surface_count = int(relational_stats["surface_candidate_count"])
    component_count = int(relational_stats["component_count"])
    co_support = int(relational_stats["co_support_edge_count"])
    possible_pairs = component_count * (component_count - 1) / 2
    numeric_distance = 0.0
    if relation in NUMERIC_RELATIONS and control and action:
        before = try_parse_number(str(control[0]))
        after = try_parse_number(str(action[0]))
        if before is not None and after is not None and before > 0 and after > 0:
            numeric_distance = min(
                1.0, abs(math.log(float(after) / float(before))) / 3.0)
    collapsed_control = collapse_prediction(graph, control)
    values = [
        float(not control_tokens), float(not action_tokens),
        min(1.0, len(control_tokens) / 5.0),
        min(1.0, len(action_tokens) / 5.0),
        max(-1.0, min(
            1.0, (len(action_tokens) - len(control_tokens)) / 3.0)),
        float(edit_count == 0),
        float(bool(added) and not dropped),
        float(bool(dropped) and not added),
        float(bool(added) and bool(dropped) and edit_count == 2),
        float(edit_count > 2),
        len(control_tokens & action_tokens)
        / max(1, len(control_tokens | action_tokens)),
        float(arm == "component"),
        float(
            arm == "component"
            and _surface_action_key(action, relation)
            == _surface_action_key(collapsed_control, relation)
            and _surface_action_key(control, relation)
            != _surface_action_key(collapsed_control, relation)),
        *[added_summary[name] for name in SUMMARY_NAMES],
        *[dropped_summary[name] for name in SUMMARY_NAMES],
        min(1.0, len(added_summaries) / 3.0),
        min(1.0, len(dropped_summaries) / 3.0),
        min(1.0, surface_count / 15.0),
        min(1.0, component_count / 15.0),
        (surface_count - component_count) / max(1, surface_count),
        co_support / max(1.0, possible_pairs),
        float(graph["agents"][QWEN]["none_rate"]),
        float(graph["agents"][GEMMA]["none_rate"]),
        _prob(graph, QWEN, "existence", "YES"),
        _prob(graph, GEMMA, "existence", "YES"),
        _prob(graph, QWEN, "existence", "NO"),
        _prob(graph, GEMMA, "existence", "NO"),
        _prob(graph, QWEN, "cardinality", "ZERO"),
        _prob(graph, GEMMA, "cardinality", "ZERO"),
        _prob(graph, QWEN, "cardinality", "ONE"),
        _prob(graph, GEMMA, "cardinality", "ONE"),
        _prob(graph, QWEN, "cardinality", "MANY"),
        _prob(graph, GEMMA, "cardinality", "MANY"),
        min(1.0, abs(len(action_tokens) - expected_cardinality) / 4.0),
        numeric_distance,
    ]
    if len(values) != len(feature_names()):
        raise AssertionError("component action feature schema drift")
    return values


def fit_residual(
        graphs: Sequence[Mapping[str, Any]],
        gold: Mapping[tuple[str, str], Mapping[str, Any]],
        relation: str, arm: str, l2: float,
) -> ResidualRidge:
    x: list[list[float]] = []
    y: list[float] = []
    weights: list[float] = []
    for graph in graphs:
        if graph["Relation"] != relation:
            continue
        control = list(graph.get("baseline_objects", []))
        actions = actions_for(graph, control, arm)
        before = _row_f1(control, gold[_key(graph)], relation)
        row_weight = 1.0 / len(actions)
        for action in actions:
            x.append(action_features(graph, control, action, arm))
            y.append(_row_f1(
                action, gold[_key(graph)], relation) - before)
            weights.append(row_weight)
    if not x:
        raise ContractError(f"no residual examples for {relation}/{arm}")
    return ResidualRidge(feature_names(), l2).fit(x, y, weights)


def decode(
        model: ResidualRidge, graph: Mapping[str, Any],
        control: Sequence[str], arm: str, margin: float,
) -> tuple[list[str], dict[str, Any]]:
    actions = actions_for(graph, control, arm)
    estimates = model.predict([
        action_features(graph, control, action, arm) for action in actions])
    best = max(range(len(actions)), key=lambda index: (
        float(estimates[index]), -len(actions[index]),
        _action_tokens(graph, actions[index], arm)))
    proposed = actions[best]
    estimated_delta = float(estimates[best])
    use_control = estimated_delta <= margin
    return (list(control) if use_control else list(proposed)), {
        "arm": arm,
        "control_objects": list(control),
        "proposed_objects": list(proposed),
        "estimated_f1_delta": estimated_delta,
        "guard_margin": float(margin),
        "used_control": use_control,
        "action_count": len(actions),
    }


def _prediction_rows(
        control_rows: Sequence[Mapping[str, Any]],
        replacements: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict[str, Any]]:
    return [{
        "SubjectEntity": row["SubjectEntity"],
        "Relation": row["Relation"],
        "ObjectEntities": list(replacements.get(
            _key(row), row.get("ObjectEntities", []))),
    } for row in control_rows]


def _write_prediction(
        path: Path, rows: Sequence[Mapping[str, Any]], *,
        arm: str, control_path: Path, model_path: Path,
) -> None:
    write_jsonl_atomic(path, rows)
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({
        "schema": "component-aware-decoder-predictions-v1",
        "contains_labels": False,
        "gold_aware": False,
        "arm": arm,
        "rows": len(rows),
        "output_sha256": sha256(path),
        "control_predictions": str(control_path),
        "control_predictions_sha256": sha256(control_path),
        "models": str(model_path),
        "models_sha256": sha256(model_path),
        "validation_labels_used_for_selection": False,
        "validation_labels_used_for_decoding": False,
    }, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = _json(source / "plan/PLAN.json")
    train_path = source / "graphs/train_graph.jsonl"
    validation_path = source / "graphs/validation_graph.jsonl"
    all_train = _load_graph(train_path, expected_split="train")
    train = [
        graph for graph in all_train
        if graph.get("calibration_eligible", True) is not False]
    validation = _load_graph(validation_path, expected_split="validation")
    train_by = {_key(graph): graph for graph in train}
    validation_by = {_key(graph): graph for graph in validation}
    train_gold = {
        _key(row): row for row in read_jsonl(Path(plan["train_gold"]))}
    folds_path = Path(plan["folds"])
    if sha256(folds_path) != plan["folds_sha256"]:
        raise ContractError("fold hash mismatch")
    folds = {_key(row): int(row["fold"]) for row in read_jsonl(folds_path)}
    eligible_keys = sorted(train_by)
    if not set(eligible_keys) <= set(folds):
        raise ContractError("fold map does not cover eligible training graph")
    margins = tuple(float(value) for value in args.guard_margins.split(","))
    if (not margins or any(
            value < 0 or not math.isfinite(value) for value in margins)):
        raise ContractError("invalid guard margins")

    fold_scores = {
        arm: {
            relation: {margin: {} for margin in margins}
            for relation in RELATIONS}
        for arm in ARMS}
    control_scores = {relation: {} for relation in RELATIONS}
    fold_diagnostics: list[dict[str, Any]] = []
    for fold in sorted(set(folds[key] for key in eligible_keys)):
        fit_keys = [key for key in eligible_keys if folds[key] != fold]
        hold_keys = [key for key in eligible_keys if folds[key] == fold]
        for relation in RELATIONS:
            relation_hold = [
                key for key in hold_keys if key[1] == relation]
            if not relation_hold:
                raise ContractError(
                    f"fold {fold} has no {relation} holdout rows")
            control_scores[relation][fold] = statistics.mean(
                _row_f1(
                    train_by[key].get("baseline_objects", []),
                    train_gold[key], relation)
                for key in relation_hold)
        for arm in ARMS:
            for relation in RELATIONS:
                model = fit_residual(
                    [train_by[key] for key in fit_keys],
                    train_gold, relation, arm, args.residual_l2)
                relation_hold = [
                    train_by[key] for key in hold_keys
                    if key[1] == relation]
                # The best proposal and estimated delta do not depend on the
                # guard margin.  Compute them once per row rather than
                # rebuilding identical action features for every margin.
                proposals = []
                for graph in relation_hold:
                    key = _key(graph)
                    control = list(graph.get("baseline_objects", []))
                    proposed, detail = decode(
                        model, graph, control, arm, float("-inf"))
                    before = _row_f1(
                        control, train_gold[key], relation)
                    after = _row_f1(
                        proposed, train_gold[key], relation)
                    proposals.append((
                        control, proposed,
                        float(detail["estimated_f1_delta"]),
                        before, after))
                for margin in margins:
                    values = []
                    changed = helpful = harmful = equal = 0
                    for control, proposed, estimate, before, proposed_f1 in (
                            proposals):
                        objects = control if estimate <= margin else proposed
                        after = before if estimate <= margin else proposed_f1
                        values.append(float(after))
                        if (_surface_action_key(objects, relation)
                                != _surface_action_key(control, relation)):
                            changed += 1
                            helpful += int(after > before + 1e-12)
                            harmful += int(after < before - 1e-12)
                            equal += int(abs(after - before) <= 1e-12)
                    fold_scores[arm][relation][margin][fold] = (
                        statistics.mean(values))
                    fold_diagnostics.append({
                        "arm": arm, "relation": relation,
                        "fold": fold, "margin": margin,
                        "changed": changed, "helpful": helpful,
                        "harmful": harmful, "equal": equal,
                    })

    selection: dict[str, dict[str, Any]] = {}
    selected_margin: dict[str, dict[str, float]] = {}
    enabled: dict[str, dict[str, bool]] = {}
    for arm in ARMS:
        selection[arm], selected_margin[arm], enabled[arm] = {}, {}, {}
        for relation in RELATIONS:
            margin, gate, detail = _selection(
                fold_scores[arm][relation], control_scores[relation])
            selected_margin[arm][relation] = margin
            enabled[arm][relation] = gate
            selection[arm][relation] = detail

    # Arm choice uses only paired train-OOF evidence.  Ties prefer the
    # simpler surface representation.
    chosen_arm: dict[str, str | None] = {}
    for relation in RELATIONS:
        candidates = [arm for arm in ARMS if enabled[arm][relation]]
        chosen_arm[relation] = (
            max(candidates, key=lambda arm: (
                selection[arm][relation]["margins"][
                    str(selected_margin[arm][relation])
                ]["mean_paired_delta"],
                -ARMS.index(arm)))
            if candidates else None)

    final_models: dict[str, dict[str, ResidualRidge]] = {}
    serialized: dict[str, Any] = {}
    for arm in ARMS:
        final_models[arm], serialized[arm] = {}, {}
        for relation in RELATIONS:
            model = fit_residual(
                train, train_gold, relation, arm, args.residual_l2)
            final_models[arm][relation] = model
            serialized[arm][relation] = model.to_dict()

    model_path = output / "models.json"
    model_path.write_text(json.dumps({
        "schema": "component-aware-decoder-models-v1",
        "validation_labels_used_for_selection": False,
        "training_incumbents": (
            "frozen OOF baseline_objects from the relational train graph"),
        "source_train_graph": str(train_path),
        "source_train_graph_sha256": sha256(train_path),
        "folds": str(folds_path),
        "folds_sha256": sha256(folds_path),
        "train_rows_calibration_eligible": len(train),
        "guard_margins": list(margins),
        "selection": selection,
        "selected_margins": selected_margin,
        "enabled": enabled,
        "chosen_arm": chosen_arm,
        "models": serialized,
    }, indent=2, sort_keys=True) + "\n")

    # Emit production-policy incumbents for downstream cross-fitted stages.
    # Every row is decoded by a residual model that excluded its fold.  The
    # arm and margin are the already-selected train-only component policy,
    # rather than being reselected for the downstream learner.
    oof_replacements: dict[tuple[str, str], list[str]] = {}
    oof_details: list[dict[str, Any]] = []
    for fold in sorted(set(folds[key] for key in eligible_keys)):
        fit_keys = [key for key in eligible_keys if folds[key] != fold]
        hold_keys = [key for key in eligible_keys if folds[key] == fold]
        for relation in RELATIONS:
            relation_keys = [key for key in hold_keys if key[1] == relation]
            arm = chosen_arm[relation]
            if arm is None:
                for key in relation_keys:
                    oof_replacements[key] = list(
                        train_by[key].get("baseline_objects", []))
                continue
            model = fit_residual(
                [train_by[key] for key in fit_keys],
                train_gold, relation, arm, args.residual_l2)
            margin = selected_margin[arm][relation]
            for key in relation_keys:
                graph = train_by[key]
                control = list(graph.get("baseline_objects", []))
                objects, detail = decode(
                    model, graph, control, arm, margin)
                oof_replacements[key] = objects
                oof_details.append({
                    "SubjectEntity": key[0],
                    "Relation": relation,
                    "fold": fold,
                    "arm": arm,
                    **detail,
                })
    if set(oof_replacements) != set(eligible_keys):
        raise ContractError("OOF component predictions do not cover training")
    oof_path = output / "train_oof_selected.jsonl"
    write_jsonl_atomic(oof_path, [{
        "SubjectEntity": key[0],
        "Relation": key[1],
        "ObjectEntities": oof_replacements[key],
    } for key in eligible_keys])
    write_jsonl_atomic(output / "train_oof_diagnostics.jsonl", oof_details)
    oof_path.with_suffix(oof_path.suffix + ".manifest.json").write_text(
        json.dumps({
            "schema": "component-aware-oof-incumbents-v1",
            "split": "train",
            "contains_labels": False,
            "gold_aware": True,
            "deployable": False,
            "selection_uses_train_labels": True,
            "oof_model_excludes_row": True,
            "rows": len(eligible_keys),
            "output_sha256": sha256(oof_path),
            "models": str(model_path),
            "models_sha256": sha256(model_path),
            "folds": str(folds_path),
            "folds_sha256": sha256(folds_path),
        }, indent=2, sort_keys=True) + "\n")

    control_path = Path(args.control_predictions).resolve()
    control_rows = read_jsonl(control_path)
    control_by = {_key(row): row for row in control_rows}
    if set(control_by) != set(validation_by):
        raise ContractError("validation control does not cover graph")

    arm_predictions: dict[str, list[dict[str, Any]]] = {}
    diagnostics: list[dict[str, Any]] = []
    for arm in ARMS:
        replacements: dict[tuple[str, str], list[str]] = {}
        for key, graph in validation_by.items():
            relation = key[1]
            objects, detail = decode(
                final_models[arm][relation], graph,
                control_by[key]["ObjectEntities"], arm,
                selected_margin[arm][relation])
            replacements[key] = objects
            diagnostics.append({
                "SubjectEntity": key[0], "Relation": relation,
                "deployment_gate_enabled": enabled[arm][relation],
                **detail})
        rows = _prediction_rows(control_rows, replacements)
        path = output / f"validation_{arm}.jsonl"
        _write_prediction(
            path, rows, arm=arm, control_path=control_path,
            model_path=model_path)
        arm_predictions[arm] = rows

    selected_replacements: dict[tuple[str, str], list[str]] = {}
    for key, graph in validation_by.items():
        relation = key[1]
        arm = chosen_arm[relation]
        if arm is None:
            continue
        objects, _ = decode(
            final_models[arm][relation], graph,
            control_by[key]["ObjectEntities"], arm,
            selected_margin[arm][relation])
        selected_replacements[key] = objects
    selected_rows = _prediction_rows(control_rows, selected_replacements)
    selected_path = output / "validation_train_selected.jsonl"
    _write_prediction(
        selected_path, selected_rows, arm="train_selected",
        control_path=control_path, model_path=model_path)
    write_jsonl_atomic(output / "validation_diagnostics.jsonl", diagnostics)
    write_jsonl_atomic(output / "fold_diagnostics.jsonl", fold_diagnostics)

    # Open validation labels only after all prediction artifacts are frozen.
    validation_gold = read_jsonl(Path(plan["validation_gold"]))
    scores = {"control": score(control_rows, validation_gold)}
    scores.update({
        arm: score(rows, validation_gold)
        for arm, rows in arm_predictions.items()})
    scores["train_selected"] = score(selected_rows, validation_gold)
    pooled_control = scores["control"]["*** All Relations ***"]
    result = {
        "schema": "component-aware-decoder-ablation-v1",
        "development_only": True,
        "validation_labels_used_for_selection": False,
        "validation_labels_used_for_posthoc_evaluation": True,
        "arms": {
            "surface": (
                "train-margin candidate policy: raw surface actions and "
                "per-surface evidence"),
            "component": (
                "train-margin candidate policy: component actions and "
                "evidence pooled over equivalent surface members"),
        },
        "arm_scores_ignore_deployment_gate": True,
        "train_selected_respects_deployment_gate": True,
        "chosen_arm": chosen_arm,
        "selected_margins": selected_margin,
        "enabled": enabled,
        "selection": selection,
        "scores": scores,
        "pooled_deltas": {
            name: values["*** All Relations ***"] - pooled_control
            for name, values in scores.items()},
        "predictions": str(selected_path),
        "predictions_sha256": sha256(selected_path),
        "train_oof_predictions": str(oof_path),
        "train_oof_predictions_sha256": sha256(oof_path),
        "model_artifact": str(model_path),
        "model_artifact_sha256": sha256(model_path),
    }
    result_path = output / "RESULT.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Component-aware decoder ablation",
        "",
        "Surface and component arms use the same frozen graphs, OOF training "
        "incumbents, folds, learner, margins, and validation control. Arm and "
        "margin decisions use training labels only.",
        "",
        "| policy | pooled | award | company | borders | area | capacity | "
        "city | pooled delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("control", *ARMS, "train_selected"):
        values = scores[name]
        lines.append(
            f"| {name} | {values['*** All Relations ***']:.9f} | "
            f"{values['awardWonBy']:.6f} | "
            f"{values['companyTradesAtStockExchange']:.6f} | "
            f"{values['countryLandBordersCountry']:.6f} | "
            f"{values['hasArea']:.6f} | "
            f"{values['hasCapacity']:.6f} | "
            f"{values['personHasCityOfDeath']:.6f} | "
            f"{values['*** All Relations ***'] - pooled_control:+.9f} |")
    lines += [
        "",
        "| relation | arm | margin | enabled | mean train-OOF delta |",
        "|---|---|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        for arm in ARMS:
            margin = selected_margin[arm][relation]
            mean_delta = selection[arm][relation]["margins"][
                str(margin)]["mean_paired_delta"]
            lines.append(
                f"| {relation} | {arm} | {margin:.3f} | "
                f"{enabled[arm][relation]} | {mean_delta:+.6f} |")
    lines += [
        "",
        f"Train-selected arms: `{json.dumps(chosen_arm, sort_keys=True)}`.",
        "",
        "The clean representation test is `component - surface`. A relation "
        "that fails the train-OOF deployment gate remains unchanged in "
        "`train_selected`; the two arm rows intentionally show their "
        "train-margin candidate policies before that final gate.",
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(
        f"complete: pooled="
        f"{scores['train_selected']['*** All Relations ***']:.9f}; "
        f"chosen={chosen_arm}; report={output / 'RESULT.md'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-output-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument(
        "--control-predictions", default=str(DEFAULT_CONTROL))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--residual-l2", type=float, default=10.0)
    parser.add_argument(
        "--guard-margins",
        default=",".join(str(value) for value in DEFAULT_MARGINS))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
