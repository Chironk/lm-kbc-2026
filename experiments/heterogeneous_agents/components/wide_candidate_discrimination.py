#!/usr/bin/env python3
"""Wide anchored candidate discrimination for the retained linear selector.

The earlier candidate tournament placed at most three factual candidates in a
prompt.  Although every challenger was compared with an incumbent anchor, the
small contexts did not ask either memory to discriminate among most of the
plausible values at once.  This experiment uses the expanded A--H choice
contract to place as many as seven factual candidates plus UNKNOWN in one
balanced choice task.

Every group contains the same incumbent anchor.  Consequently the evidence
consumed downstream is the same-context log odds

    log P(candidate | group) - log P(anchor | group)

and remains comparable across groups.  ``prepare`` is label-free. ``analyze``
opens train labels only after both frozen response manifests are complete,
adds the wide evidence to the retained relation-conditioned paired linear
head, and evaluates it with nested subject-grouped out-of-fold fitting.
Validation is structurally absent.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.components.coherent_candidate_selector import (
    L2_GRID,
    MIN_FOLD_DELTA,
    MIN_PROMOTION_DELTA,
    MIN_RELATION_DELTA,
    MIN_WINNING_FOLDS,
    PAIRED_RELATION_FEATURE_NAMES,
    THRESHOLD_GRID,
    _candidate_weights,
    _cross_fit as _paired_cross_fit,
    _decode,
    _features as _paired_features,
    _quality,
)
from experiments.heterogeneous_agents.components.coherent_row_selector import (
    CoherentAction,
    CoherentRow,
    load_coherent_rows,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    RELATION_QUESTIONS,
    balanced_choice_codebooks,
    canonical_key,
    read_jsonl,
    sha256,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.crossfit_action_utility_selector import (
    DEFAULT_GOLD,
    DEFAULT_SOURCE,
    _key,
    _write_json,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.components.full_candidate_tournament import (
    AGENTS,
    RELATION_GUIDANCE,
    UNKNOWN,
    _graph_manifest,
    _json,
    _surface_key,
    row_nodes,
)
from experiments.heterogeneous_agents.components.multi_challenger_graph_decoder import (
    EPSILON,
)
from experiments.heterogeneous_agents.run_agent import (
    validate_tasks,
)
from experiments.heterogeneous_agents.components.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    DEFAULT_TRAIN_OOF,
    validate_registered_predictions,
)
from experiments.heterogeneous_agents.components.truth_calibrated_action_decoder import (
    StandardizedLinear,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import (
    RELATIONS,
)


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/graphs/train_graph.jsonl")
DEFAULT_AGENTS = (
    ROOT / "experiments/heterogeneous_agents/"
    "agents_qwen_gemma_n1_frozen.json")
DEFAULT_OUTPUT = RUNS / "wide_candidate_discrimination_20260728_v1"
MAX_FACT_CHOICES = 7
SCHEMA = "wide-candidate-discrimination-plan-v1"
WIDE_FEATURE_NAMES = (
    "wide_available",
    "wide_coverage",
    "wide_qwen_action_mean_delta",
    "wide_gemma_action_mean_delta",
    "wide_mean_action_delta",
    "wide_minimum_action_delta",
    "wide_direction_agreement",
    "wide_qwen_added_min",
    "wide_gemma_added_min",
    "wide_qwen_removed_max",
    "wide_gemma_removed_max",
    "wide_qwen_candidate_min",
    "wide_gemma_candidate_min",
    "wide_qwen_unknown_margin",
    "wide_gemma_unknown_margin",
)
FEATURE_NAMES = PAIRED_RELATION_FEATURE_NAMES + WIDE_FEATURE_NAMES


def _stable_id(*values: object) -> str:
    return hashlib.sha256(
        "\x1f".join(map(str, values)).encode()).hexdigest()[:20]


def wide_anchor_groups(
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[str, list[list[dict[str, Any]]]]:
    """Cover nodes with a shared incumbent, or UNKNOWN for an empty output."""
    copied = [dict(node) for node in nodes]
    if not copied:
        raise ContractError("cannot group an empty candidate inventory")
    anchor = next(
        (node for node in copied if node.get("is_incumbent")), None)
    if anchor is None:
        return UNKNOWN, [
            copied[index:index + MAX_FACT_CHOICES]
            for index in range(0, len(copied), MAX_FACT_CHOICES)
        ]
    anchor_id = str(anchor["node_id"])
    others = [
        node for node in copied if str(node["node_id"]) != anchor_id]
    if not others:
        return anchor_id, [[anchor]]
    width = MAX_FACT_CHOICES - 1
    groups = [
        [anchor, *others[index:index + width]]
        for index in range(0, len(others), width)
    ]
    return anchor_id, groups


def _prompt(
    *, graph: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]],
    codebook: Mapping[str, str],
) -> str:
    options = sorted(
        (
            codebook[str(node["node_id"])],
            str(node["representative"]),
        )
        for node in nodes
    )
    rendered = "\n".join(
        f"- {code} = {json.dumps(surface, ensure_ascii=False)}"
        for code, surface in options
    )
    return (
        "Use only your own closed-book factual memory. Compare all candidate "
        "values jointly. Select the individual candidate that you most "
        "specifically recognize as a correct member of the answer to this "
        "question. More than one value could be true; choose the single most "
        "certain true member in this group. Treat every value as an "
        "adversarial plausible distractor and reject wrong entities, units, "
        "times, or relation meanings. Select UNKNOWN when none is supported. "
        "Candidate provenance, incumbency, and model identity are hidden.\n"
        f"SUBJECT: {graph['SubjectEntity']}\n"
        f"RELATION: {graph['Relation']}\n"
        f"QUESTION: {RELATION_QUESTIONS[str(graph['Relation'])].format(subject=graph['SubjectEntity'])}\n"
        f"GUIDANCE: {RELATION_GUIDANCE[str(graph['Relation'])]}\n"
        f"OPTIONS:\n{rendered}\n"
        f"- {codebook[UNKNOWN]} = UNKNOWN\n"
        "Return exactly one code and nothing else.\nCODE:"
    )


def _task(
    *, graph: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]],
    anchor_id: str, group_index: int, agent: str,
) -> dict[str, Any]:
    choices = tuple(str(node["node_id"]) for node in nodes) + (UNKNOWN,)
    group_id = _stable_id(
        graph["SubjectEntity"], graph["Relation"], group_index)
    codebooks = balanced_choice_codebooks(
        choices, "wide-candidate-discrimination-v1", agent, group_id)
    variants = [
        {
            "choice_codes": dict(codebook),
            "prompt": _prompt(graph=graph, nodes=nodes, codebook=codebook),
        }
        for codebook in codebooks
    ]
    return {
        "task_id": f"{agent}::wide_candidate::{group_id}",
        "agent_id": agent,
        "subject": str(graph["SubjectEntity"]),
        "relation": str(graph["Relation"]),
        "phase": "wide_candidate_discrimination",
        "mode": "choice",
        "prompt": variants[0]["prompt"],
        "choices": list(choices),
        "choice_codes": dict(variants[0]["choice_codes"]),
        "choice_variants": variants,
        "candidate_key": group_id,
        "candidate_item": "wide_anchor_group",
        "group_index": group_index,
        "group_node_ids": [str(node["node_id"]) for node in nodes],
        "anchor_node_id": anchor_id,
        "excluded_proposer_agents": list(AGENTS),
        "contains_labels": False,
        "gold_aware": False,
        "prompt_masks_provenance": True,
        "prompt_masks_incumbency": True,
    }


def prepare(args: argparse.Namespace) -> int:
    graph_path = Path(args.graph).resolve()
    _graph_manifest(graph_path)
    graphs = read_jsonl(graph_path)
    control_path = Path(args.control).resolve()
    validate_registered_predictions(
        control_path, pipeline_id=COMPETITION_PIPELINE_ID, split="train")
    control = {_key(row): row for row in read_jsonl(control_path)}
    if set(control) != {_key(row) for row in graphs}:
        raise ContractError("wide graph/control coverage mismatch")
    output = Path(args.output_dir).resolve()
    registry, tasks_by_agent = [], {agent: [] for agent in AGENTS}
    skipped = 0
    for graph in graphs:
        incumbent_objects = list(map(
            str, control[_key(graph)]["ObjectEntities"]))
        nodes = row_nodes(graph, incumbent_objects)
        if not nodes:
            skipped += 1
            continue
        anchor_id, groups = wide_anchor_groups(nodes)
        registry.append({
            "SubjectEntity": str(graph["SubjectEntity"]),
            "Relation": str(graph["Relation"]),
            "incumbent_objects": incumbent_objects,
            "anchor_node_id": anchor_id,
            "nodes": nodes,
            "groups": [
                [str(node["node_id"]) for node in group] for group in groups
            ],
            "contains_labels": False,
            "gold_aware": False,
        })
        for group_index, group in enumerate(groups):
            for agent in AGENTS:
                tasks_by_agent[agent].append(_task(
                    graph=graph,
                    nodes=group,
                    anchor_id=anchor_id,
                    group_index=group_index,
                    agent=agent,
                ))
    registry_path = output / "plan/ROWS.jsonl"
    write_jsonl_atomic(registry_path, registry)
    jobs = {}
    for agent, tasks in tasks_by_agent.items():
        task_path = output / f"plan/tasks/{agent}.jsonl"
        smoke_path = output / f"plan/smoke/{agent}.jsonl"
        smoke = []
        for relation in RELATIONS:
            match = next(
                (task for task in tasks if task["relation"] == relation),
                None,
            )
            if match:
                smoke.append(match)
        write_jsonl_atomic(task_path, tasks)
        write_jsonl_atomic(smoke_path, smoke)
        validate_tasks(tasks, agent)
        jobs[agent] = {
            "tasks": len(tasks),
            "task_path": str(task_path),
            "task_sha256": sha256(task_path),
            "smoke_path": str(smoke_path),
            "smoke_sha256": sha256(smoke_path),
            "response_path": str(output / f"responses/{agent}.jsonl"),
        }
    plan = {
        "schema": SCHEMA,
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "graph": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "starting_predictions": str(control_path),
        "starting_predictions_sha256": sha256(control_path),
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "rows": len(registry),
        "skipped_zero_cardinality_rows": skipped,
        "max_fact_choices": MAX_FACT_CHOICES,
        "shared_incumbent_anchor": True,
        "balanced_code_cycles": True,
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "jobs": jobs,
        "agents": str(Path(args.agents).resolve()),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(output / "plan/PLAN.json", plan)
    print(json.dumps({
        "rows": len(registry),
        "skipped_zero_cardinality_rows": skipped,
        "jobs": {agent: job["tasks"] for agent, job in jobs.items()},
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


def _validated_responses(
    plan: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    tasks, responses = [], {}
    for agent, job in plan["jobs"].items():
        task_path = Path(job["task_path"])
        response_path = Path(job["response_path"])
        manifest_path = response_path.with_suffix(
            response_path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise ContractError(f"missing complete response manifest: {agent}")
        manifest = _json(manifest_path)
        if (
            sha256(task_path) != job["task_sha256"]
            or manifest.get("task_sha256") != job["task_sha256"]
            or manifest.get("output_sha256") != sha256(response_path)
            or manifest.get("agent_id") != agent
            or int(manifest.get("tasks", -1)) != int(job["tasks"])
        ):
            raise ContractError(f"stale wide responses: {agent}")
        agent_tasks = read_jsonl(task_path)
        task_by_id = validate_tasks(agent_tasks, agent)
        agent_responses = read_jsonl(response_path)
        if len(agent_responses) != len(agent_tasks):
            raise ContractError(f"incomplete wide responses: {agent}")
        for response in agent_responses:
            task_id = str(response["task_id"])
            if task_id not in task_by_id or task_id in responses:
                raise ContractError("invalid or duplicate wide response")
            validate_task_response(task_by_id[task_id], response)
            responses[task_id] = response
        tasks.extend(agent_tasks)
    return tasks, responses


def aggregate_anchor_margins(
    tasks: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    node_values: dict[
        tuple[str, str], dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    unknown_values: dict[
        tuple[str, str], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        response = responses[str(task["task_id"])]
        probabilities = response["choice_probabilities"]
        anchor_id = str(task["anchor_node_id"])
        anchor = max(float(probabilities[anchor_id]), EPSILON)
        key = (str(task["subject"]), str(task["relation"]))
        agent = str(task["agent_id"])
        for node_id in task["group_node_ids"]:
            value = max(float(probabilities[str(node_id)]), EPSILON)
            node_values[key][agent][str(node_id)].append(
                math.log(value) - math.log(anchor))
        unknown_values[key][agent].append(
            math.log(max(float(probabilities[UNKNOWN]), EPSILON))
            - math.log(anchor))
    return {
        key: {
            agent: {
                "nodes": {
                    node_id: statistics.mean(values)
                    for node_id, values in nodes.items()
                },
                "unknown_margin": statistics.mean(
                    unknown_values[key][agent]),
            }
            for agent, nodes in agents.items()
        }
        for key, agents in node_values.items()
    }


class WideEvidence:
    """Validated label-free wide evidence and action feature projection."""

    def __init__(self, run: Path):
        self.run = run.resolve()
        plan_path = self.run / "plan/PLAN.json"
        plan = _json(plan_path)
        registry_path = Path(plan["registry"])
        if (
            plan.get("schema") != SCHEMA
            or plan.get("contains_labels")
            or plan.get("gold_aware")
            or plan.get("validation_opened")
            or plan.get("validation_labels_used")
            or sha256(registry_path) != plan["registry_sha256"]
        ):
            raise ContractError("wide evidence is not certified label-free")
        tasks, responses = _validated_responses(plan)
        self.scores = aggregate_anchor_margins(tasks, responses)
        self.nodes: dict[
            tuple[str, str], dict[str, dict[str, str]]
        ] = {}
        for row in read_jsonl(registry_path):
            raw_to_node: dict[str, str] = {}
            canonical_candidates: dict[str, set[str]] = defaultdict(set)
            relation = str(row["Relation"])
            for node in row["nodes"]:
                node_id = str(node["node_id"])
                for surface in node.get(
                    "member_items", [node["representative"]],
                ):
                    raw = str(surface)
                    raw_key = _surface_key(raw)
                    old = raw_to_node.setdefault(raw_key, node_id)
                    if old != node_id:
                        raise ContractError(
                            f"{_key(row)}: duplicate raw wide surface")
                    canonical_candidates[_surface_key(
                        canonical_key(raw, relation))].add(node_id)
            # Exact strings are authoritative. A canonical alias is only a
            # fallback when it identifies one graph node. For example, NYSE
            # and New York Stock Exchange can legitimately occur as distinct
            # prompt surfaces while sharing one relation canonical key.
            self.nodes[_key(row)] = {
                "raw": raw_to_node,
                "canonical": {
                    normalized: next(iter(node_ids))
                    for normalized, node_ids in canonical_candidates.items()
                    if len(node_ids) == 1
                },
            }
        self.provenance = {
            "wide_plan": str(plan_path),
            "wide_plan_sha256": sha256(plan_path),
            "wide_registry": str(registry_path),
            "wide_registry_sha256": sha256(registry_path),
        }

    def features(
        self, row: CoherentRow, action: CoherentAction,
    ) -> list[float]:
        if row.key not in self.nodes or row.key not in self.scores:
            return [0.0] * len(WIDE_FEATURE_NAMES)
        node_indexes = self.nodes[row.key]

        def node_set(objects: Sequence[str]) -> tuple[set[str], int]:
            found, total = set(), 0
            unique_objects: dict[str, str] = {}
            for item in objects:
                raw = str(item)
                unique_objects.setdefault(
                    canonical_key(raw, row.relation), raw)
            for canonical, raw in unique_objects.items():
                total += 1
                node_id = node_indexes["raw"].get(_surface_key(raw))
                if node_id is None:
                    node_id = node_indexes["canonical"].get(
                        _surface_key(canonical))
                if node_id is not None:
                    found.add(node_id)
            return found, total

        incumbent, incumbent_total = node_set(row.source.keep["objects"])
        candidate, candidate_total = node_set(
            action.source.action["objects"])
        total = incumbent_total + candidate_total
        coverage = (
            (len(incumbent) + len(candidate)) / total if total else 1.0)
        if coverage < 1.0:
            return [
                0.0, coverage,
                *([0.0] * (len(WIDE_FEATURE_NAMES) - 2)),
            ]
        added = candidate - incumbent
        removed = incumbent - candidate

        def agent_stats(agent: str) -> tuple[float, float, float, float, float]:
            evidence = self.scores[row.key][agent]
            scores = evidence["nodes"]
            incumbent_values = [float(scores[node]) for node in incumbent]
            candidate_values = [float(scores[node]) for node in candidate]
            unknown_margin = float(evidence["unknown_margin"])
            baseline_mean = (
                statistics.mean(incumbent_values)
                if incumbent_values else unknown_margin
            )
            action_delta = (
                statistics.mean(candidate_values) - baseline_mean
                if candidate_values else unknown_margin - baseline_mean
            )
            added_min = min(
                (float(scores[node]) for node in added), default=0.0)
            removed_max = max(
                (float(scores[node]) for node in removed), default=0.0)
            candidate_min = min(candidate_values, default=0.0)
            return (
                action_delta,
                added_min,
                removed_max,
                candidate_min,
                unknown_margin,
            )

        q = agent_stats(QWEN)
        g = agent_stats(GEMMA)
        values = [
            1.0,
            coverage,
            q[0],
            g[0],
            (q[0] + g[0]) / 2.0,
            min(q[0], g[0]),
            float((q[0] > 0.0) == (g[0] > 0.0)),
            q[1],
            g[1],
            q[2],
            g[2],
            q[3],
            g[3],
            q[4],
            g[4],
        ]
        return [
            value if index < 2 else max(-12.0, min(12.0, value)) / 12.0
            for index, value in enumerate(values)
        ]


def _features(
    evidence: WideEvidence, row: CoherentRow, action: CoherentAction,
) -> list[float]:
    values = [
        *_paired_features(
            row,
            action,
            hierarchical=False,
            include_truth=False,
            paired=True,
            relation_bias=True,
        ),
        *evidence.features(row, action),
    ]
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("wide selector feature schema drift")
    return values


def _fit(
    evidence: WideEvidence, rows: Sequence[CoherentRow], l2: float,
) -> StandardizedLinear:
    usable = [row for row in rows if row.alternatives]
    return StandardizedLinear(
        FEATURE_NAMES, l2, logistic=True,
    ).fit(
        [
            _features(evidence, row, action)
            for row in usable for action in row.alternatives
        ],
        [
            float(action.delta > EPSILON)
            for row in usable for action in row.alternatives
        ],
        _candidate_weights(usable),
    )


def _probabilities(
    evidence: WideEvidence, model: StandardizedLinear,
    rows: Sequence[CoherentRow],
) -> dict[tuple[str, str], list[float]]:
    return {
        row.key: (
            list(map(float, model.predict([
                _features(evidence, row, action)
                for action in row.alternatives
            ])))
            if row.alternatives else []
        )
        for row in rows
    }


def _inner_select(
    evidence: WideEvidence, rows: Sequence[CoherentRow],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[float, float, dict[str, Any]]:
    grid = {}
    for l2 in L2_GRID:
        probabilities = {}
        for fold in sorted({row.fold for row in rows}):
            fit = [row for row in rows if row.fold != fold]
            hold = [row for row in rows if row.fold == fold]
            probabilities.update(_probabilities(
                evidence, _fit(evidence, fit, l2), hold))
        for threshold in THRESHOLD_GRID:
            predictions, diagnostics = _decode(
                rows, probabilities, threshold)
            grid[(l2, threshold)] = _quality(
                predictions, diagnostics, rows, gold_by)
    selected = max(
        grid,
        key=lambda item: (
            grid[item]["score"],
            -grid[item]["harmed"],
            -grid[item]["changed"],
            item[1],
            item[0],
        ),
    )
    return selected[0], selected[1], {
        "selected_l2": selected[0],
        "selected_threshold": selected[1],
        "quality": grid[selected],
    }


def _cross_fit(
    evidence: WideEvidence, rows: Sequence[CoherentRow],
    gold_by: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions, diagnostics, reports = [], [], []
    for outer in sorted({row.fold for row in rows}):
        fit = [row for row in rows if row.fold != outer]
        hold = [row for row in rows if row.fold == outer]
        l2, threshold, inner = _inner_select(evidence, fit, gold_by)
        model = _fit(evidence, fit, l2)
        fold_predictions, fold_diagnostics = _decode(
            hold, _probabilities(evidence, model, hold), threshold)
        quality = _quality(
            fold_predictions, fold_diagnostics, hold, gold_by)
        predictions.extend(fold_predictions)
        diagnostics.extend(fold_diagnostics)
        reports.append({
            "fold": outer,
            "fit_rows": len(fit),
            "hold_rows": len(hold),
            "inner_selection": inner,
            "holdout": quality,
            "model": model.to_dict(),
        })
    prediction_by = {_key(row): row for row in predictions}
    diagnostic_by = {_key(row): row for row in diagnostics}
    expected = {row.key for row in rows}
    if set(prediction_by) != expected or set(diagnostic_by) != expected:
        raise ContractError("incomplete wide selector OOF")
    return (
        [prediction_by[row.key] for row in rows],
        [diagnostic_by[row.key] for row in rows],
        reports,
    )


def _binary_auroc(
    labels: Sequence[bool], values: Sequence[float],
) -> float | None:
    positives = [
        value for label, value in zip(labels, values, strict=True) if label]
    negatives = [
        value for label, value in zip(labels, values, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        float(positive > negative)
        + 0.5 * float(positive == negative)
        for positive in positives for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _discrimination_audit(
    evidence: WideEvidence, rows: Sequence[CoherentRow],
) -> dict[str, Any]:
    arms = {
        "qwen": 2,
        "gemma": 3,
        "mean": 4,
        "minimum": 5,
    }
    records = []
    top_helpful = {arm: 0 for arm in arms}
    rows_with_alternatives = 0
    for row in rows:
        if not row.alternatives:
            continue
        rows_with_alternatives += 1
        row_scores = {arm: [] for arm in arms}
        for action in row.alternatives:
            wide = evidence.features(row, action)
            label = action.delta > EPSILON
            records.append({
                "relation": row.relation,
                "label": label,
                **{arm: float(wide[index]) for arm, index in arms.items()},
            })
            for arm, index in arms.items():
                row_scores[arm].append(float(wide[index]))
        for arm, values in row_scores.items():
            selected = max(
                range(len(values)),
                key=lambda index: (
                    values[index],
                    row.alternatives[index].canonical_output,
                ),
            )
            top_helpful[arm] += int(
                row.alternatives[selected].delta > EPSILON)

    def audit(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        labels = [bool(record["label"]) for record in subset]
        return {
            arm: _binary_auroc(
                labels, [float(record[arm]) for record in subset])
            for arm in arms
        }

    return {
        "actions": len(records),
        "positive_actions": sum(record["label"] for record in records),
        "negative_actions": sum(not record["label"] for record in records),
        "overall_auroc": audit(records),
        "relation_aurocs": {
            relation: audit([
                record for record in records
                if record["relation"] == relation
            ])
            for relation in RELATIONS
        },
        "top_candidate_helpful": top_helpful,
        "rows_with_alternatives": rows_with_alternatives,
    }


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    source = Path(args.source_run).resolve()
    gold_path = Path(args.gold).resolve()
    evidence = WideEvidence(output)
    rows, provenance = load_coherent_rows(source, gold_path)
    gold_by = {_key(row): row for row in read_jsonl(gold_path)}
    predictions, diagnostics, folds = _cross_fit(
        evidence, rows, gold_by)
    quality = _quality(predictions, diagnostics, rows, gold_by)
    reference_predictions, reference_diagnostics, _ = _paired_cross_fit(
        rows,
        gold_by,
        hierarchical=False,
        include_truth=False,
        paired=True,
        relation_bias=True,
    )
    reference = _quality(
        reference_predictions, reference_diagnostics, rows, gold_by)
    discrimination = _discrimination_audit(evidence, rows)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "analysis/TRAIN_OOF_PREDICTIONS.jsonl"
    diagnostic_path = output / "analysis/TRAIN_OOF_DIAGNOSTICS.jsonl"
    fold_path = output / "analysis/FOLDS.json"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(diagnostic_path, diagnostics)
    _write_json(fold_path, {"folds": folds})
    fold_deltas = [item["holdout"]["delta"] for item in folds]
    checks = {
        "beats_retained_linear_head": (
            quality["score"] > reference["score"] + EPSILON),
        "minimum_delta": quality["delta"] >= MIN_PROMOTION_DELTA,
        "minimum_winning_folds": (
            sum(delta > 0.0 for delta in fold_deltas)
            >= MIN_WINNING_FOLDS),
        "minimum_fold_delta": min(fold_deltas) >= MIN_FOLD_DELTA,
        "minimum_relation_delta": (
            min(quality["relation_deltas"].values())
            >= MIN_RELATION_DELTA),
        "no_harmful_switches": quality["harmed"] == 0,
    }
    result = {
        "schema": "wide-candidate-discrimination-result-v1",
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "retained_linear_reference": reference,
        "wide_augmented_linear": quality,
        "increment_over_retained_linear": (
            quality["score"] - reference["score"]),
        "candidate_discrimination": discrimination,
        "fold_deltas": fold_deltas,
        "winning_folds": sum(delta > 0.0 for delta in fold_deltas),
        "feature_count": len(FEATURE_NAMES),
        "promotion_gate": {"passed": all(checks.values()), "checks": checks},
        "provenance": {**provenance, **evidence.provenance},
        "artifacts": {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
            "folds": str(fold_path),
            "folds_sha256": sha256(fold_path),
        },
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "development_only": True,
        "deployable": False,
        "next_stage": (
            "freeze_and_prepare_one_validation_confirmation"
            if all(checks.values())
            else "retain_existing_linear_head"
        ),
    }
    result_path = output / "analysis/RESULT.json"
    _write_json(result_path, result)
    (output / "analysis/RESULT.md").write_text(
        "# Wide candidate discrimination\n\n"
        "Nested subject-grouped train OOF; validation was not opened.\n\n"
        f"- Exact control: **{quality['control']:.9f}**\n"
        f"- Retained linear head: **{reference['score']:.9f}** "
        f"({reference['delta']:+.9f})\n"
        f"- Wide-augmented linear head: **{quality['score']:.9f}** "
        f"({quality['delta']:+.9f})\n"
        f"- Increment over retained head: "
        f"**{quality['score'] - reference['score']:+.9f}**\n"
        f"- Changed/helped/harmed: **{quality['changed']} / "
        f"{quality['helped']} / {quality['harmed']}**\n"
        f"- Promotion gate: **{all(checks.values())}**\n"
    )
    print(json.dumps({
        "retained_linear_reference": reference,
        "wide_augmented_linear": quality,
        "increment_over_retained_linear": (
            quality["score"] - reference["score"]),
        "promotion_gate": result["promotion_gate"],
        "result": str(result_path),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("prepare", "analyze"))
    value.add_argument("--graph", default=str(DEFAULT_GRAPH))
    value.add_argument("--gold", default=str(DEFAULT_GOLD))
    value.add_argument("--control", default=str(DEFAULT_TRAIN_OOF))
    value.add_argument("--agents", default=str(DEFAULT_AGENTS))
    value.add_argument("--source-run", default=str(DEFAULT_SOURCE))
    value.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return value


def main() -> int:
    args = parser().parse_args()
    return prepare(args) if args.command == "prepare" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
