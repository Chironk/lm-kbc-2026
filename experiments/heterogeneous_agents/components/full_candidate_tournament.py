#!/usr/bin/env python3
"""Broad anonymized within-question candidate tournament.

This experiment adds the signal missing from the current heterogeneous graph:
each frozen parametric memory directly compares competing candidate components
for one subject.  It never exposes proposer identity or labels to a model.

Two complementary three-candidate views share UNKNOWN as a cross-group anchor:

* ``global3``: disjoint groups for broad candidate coverage;
* ``incumbent3``: the incumbent anchor against pairs of challengers.

``prepare`` is label-free. ``analyze`` opens train labels only after both
response manifests are complete, evaluates all predeclared arms, and performs
subject-grouped nested arm selection against the exact registered SOTA OOF
starting predictions. Validation is structurally absent.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.assemble_and_audit import score
from experiments.heterogeneous_agents.core import (
    ContractError,
    RELATION_QUESTIONS,
    balanced_choice_codebooks,
    read_jsonl,
    sha256,
    validate_task_response,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.run_agent import validate_tasks
from experiments.heterogeneous_agents.components.relation_specific_structured_decoder import (
    _row_f1,
)
from experiments.heterogeneous_agents.components.sota_pipeline import (
    COMPETITION_PIPELINE_ID,
    DEFAULT_TRAIN_OOF,
    validate_registered_predictions,
)
from experiments.heterogeneous_agents.components.unified_memory_action_graph import (
    RELATIONS,
    _key,
    _relation_deltas,
)


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
DEFAULT_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/graphs/train_graph.jsonl")
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_AGENTS = (
    ROOT / "experiments/heterogeneous_agents/"
    "agents_qwen_gemma_n1_frozen.json")
DEFAULT_OUTPUT = RUNS / "full_candidate_tournament_20260727_v1"
AGENTS = (QWEN, GEMMA)
UNKNOWN = "UNKNOWN"
VIEWS = ("global3", "incumbent3")
PROMPT_ARMS = ("recognition", "skeptical", "submission")
ARMS = ("qwen", "gemma", "mean", "minimum", "agreement_penalized")
GATES = (
    "unconditional",
    "advantage",
    "dual_advantage",
    "unknown",
    "dual_unknown",
    "consensus",
)
EVIDENCE_ARMS = (
    *(f"{view}:{prompt}" for view in VIEWS for prompt in PROMPT_ARMS),
    *(f"combined:{prompt}" for prompt in PROMPT_ARMS),
    "ensemble",
)
MIN_INCREMENTAL_DELTA = 0.005
MAX_REGRESSION_PER_RELATION = -0.02
EPSILON = 1e-12


RELATION_GUIDANCE = {
    "countryLandBordersCountry": (
        "Select countries that truly share a land border with the subject; "
        "exclude maritime-only borders."),
    "personHasCityOfDeath": (
        "Select the city or most specific publicly known locality where the "
        "person died, not a country or broad region."),
    "hasCapacity": (
        "Select the normal maximum spectator capacity; exclude attendance, "
        "temporary layouts, area, years, and obsolete configurations."),
    "awardWonBy": (
        "Select actual recipient entities of this exact award, not winning "
        "works or recipients of predecessor/successor awards."),
    "companyTradesAtStockExchange": (
        "Select exchanges where this exact company has publicly traded shares, "
        "not a parent, subsidiary, or merely plausible market."),
    "hasArea": (
        "Select surface area in square kilometres, using total area for "
        "countries; reject population, land-only area, hectares, and miles."),
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        dict(value), indent=2, sort_keys=True) + "\n")


def _stable_id(*values: object) -> str:
    return hashlib.sha256(
        "\x1f".join(map(str, values)).encode()).hexdigest()[:20]


def _surface_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _graph_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ContractError("missing tournament graph manifest")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != "heterogeneous-memory-graph-manifest-v1"
        or manifest.get("split") != "train"
        or manifest.get("contains_labels")
        or manifest.get("gold_aware")
        or manifest.get("output_sha256") != sha256(path)
    ):
        raise ContractError("tournament graph is not certified label-free")
    return manifest


def row_nodes(
    graph: Mapping[str, Any], incumbent_objects: Sequence[str],
) -> list[dict[str, Any]]:
    """Return unique graph components plus incumbent-only components."""
    output, seen = [], set()
    incumbent_keys = {
        _surface_key(item) for item in incumbent_objects}
    for component in graph["relational_graph"]["components"]:
        representative = str(component["representative"])
        keys = {
            _surface_key(item)
            for item in component.get("member_items", [representative])}
        node_id = str(component["id"])
        output.append({
            "node_id": node_id,
            "representative": representative,
            "member_items": list(map(
                str, component.get("member_items", [representative]))),
            "is_incumbent": bool(keys & incumbent_keys),
        })
        seen.update(keys)
    for item in incumbent_objects:
        rendered = str(item)
        key = _surface_key(rendered)
        if key in seen:
            continue
        seen.add(key)
        output.append({
            "node_id": f"incumbent:{_stable_id(graph['SubjectEntity'], rendered)}",
            "representative": rendered,
            "member_items": [rendered],
            "is_incumbent": True,
        })
    return output


def partition_groups(
    nodes: Sequence[Mapping[str, Any]], view: str,
) -> list[list[dict[str, Any]]]:
    """Create legal three-candidate groups plus the shared UNKNOWN choice."""
    copied = [dict(item) for item in nodes]
    if view == "global3":
        return [copied[index:index + 3]
                for index in range(0, len(copied), 3)]
    if view == "incumbent3":
        anchor = next(
            (item for item in copied if item["is_incumbent"]),
            copied[0] if copied else None)
        if anchor is None:
            return []
        others = [
            item for item in copied
            if str(item["node_id"]) != str(anchor["node_id"])]
        if not others:
            return [[anchor]]
        return [
            [anchor, *others[index:index + 2]]
            for index in range(0, len(others), 2)
        ]
    raise ContractError(f"unknown tournament view: {view}")


def _task(
    *, graph: Mapping[str, Any], view: str, group_index: int,
    nodes: Sequence[Mapping[str, Any]], agent: str, prompt_arm: str,
) -> dict[str, Any]:
    choices = tuple(str(node["node_id"]) for node in nodes) + (UNKNOWN,)
    group_id = _stable_id(
        graph["SubjectEntity"], graph["Relation"], view, group_index,
        prompt_arm)
    codebooks = balanced_choice_codebooks(
        choices, "full-candidate-tournament-v1", agent, group_id)
    instruction = {
        "recognition": (
            "Select the single candidate most likely to be factually correct "
            "for the question. Require specific factual recognition."),
        "skeptical": (
            "Treat the options as adversarial plausible distractors. Select a "
            "candidate only when you specifically remember the fact; actively "
            "reject values with the wrong entity, unit, time, or relation."),
        "submission": (
            "Act as the final knowledge-base editor. If forced to submit one "
            "of these values under the exact relation definition, select the "
            "value you would enter; otherwise select UNKNOWN."),
    }.get(prompt_arm)
    if instruction is None:
        raise ContractError(f"unknown tournament prompt arm: {prompt_arm}")
    variants = []
    for codebook in codebooks:
        options = "; ".join(
            f"{codebook[str(node['node_id'])]} = "
            f"{json.dumps(str(node['representative']), ensure_ascii=False)}"
            for node in nodes)
        variants.append({
            "choice_codes": dict(codebook),
            "prompt": (
                "Use only your own closed-book factual memory. The candidates "
                "are anonymous competing knowledge-base values. "
                f"{instruction} Do not infer from option order, generic "
                "plausibility, or repetition.\n"
                f"SUBJECT: {graph['SubjectEntity']}\n"
                f"RELATION: {graph['Relation']}\n"
                f"QUESTION: {RELATION_QUESTIONS[str(graph['Relation'])].format(subject=graph['SubjectEntity'])}\n"
                f"GUIDANCE: {RELATION_GUIDANCE[str(graph['Relation'])]}\n"
                f"OPTIONS: {options}; {codebook[UNKNOWN]} = UNKNOWN.\n"
                "Return exactly one code and nothing else.\nCODE:"
            ),
        })
    return {
        "task_id": (
            f"{agent}::full_tournament::{view}::{prompt_arm}::{group_id}"),
        "agent_id": agent,
        "subject": str(graph["SubjectEntity"]),
        "relation": str(graph["Relation"]),
        "phase": "full_candidate_tournament",
        "mode": "choice",
        "prompt": variants[0]["prompt"],
        "choices": list(choices),
        "choice_codes": dict(variants[0]["choice_codes"]),
        "choice_variants": variants,
        "candidate_key": group_id,
        "candidate_item": view,
        "view": view,
        "prompt_arm": prompt_arm,
        "group_index": group_index,
        "group_node_ids": [str(node["node_id"]) for node in nodes],
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
        raise ContractError("tournament graph/control coverage mismatch")
    output = Path(args.output_dir).resolve()
    registry, tasks_by_agent = [], {agent: [] for agent in AGENTS}
    skipped_zero_cardinality = 0
    for graph in graphs:
        if str(graph["Relation"]) not in RELATIONS:
            raise ContractError("unknown tournament relation")
        incumbent_objects = list(map(
            str, control[_key(graph)]["ObjectEntities"]))
        cardinality = len({
            _surface_key(item) for item in incumbent_objects})
        nodes = row_nodes(graph, incumbent_objects)
        if cardinality == 0 or not nodes:
            skipped_zero_cardinality += 1
            continue
        view_groups = {}
        for view in VIEWS:
            groups = partition_groups(nodes, view)
            view_groups[view] = []
            for group_index, group in enumerate(groups):
                group_id = _stable_id(
                    graph["SubjectEntity"], graph["Relation"],
                    view, group_index)
                view_groups[view].append({
                    "group_id": group_id,
                    "node_ids": [str(node["node_id"]) for node in group],
                })
                for prompt_arm in PROMPT_ARMS:
                    for agent in AGENTS:
                        tasks_by_agent[agent].append(_task(
                            graph=graph, view=view, group_index=group_index,
                            nodes=group, agent=agent,
                            prompt_arm=prompt_arm))
        registry.append({
            "SubjectEntity": str(graph["SubjectEntity"]),
            "Relation": str(graph["Relation"]),
            "cardinality": cardinality,
            "incumbent_objects": incumbent_objects,
            "nodes": nodes,
            "views": view_groups,
            "contains_labels": False,
            "gold_aware": False,
        })
    registry_path = output / "plan/ROWS.jsonl"
    write_jsonl_atomic(registry_path, registry)
    jobs = {}
    for agent, tasks in tasks_by_agent.items():
        task_path = output / f"plan/tasks/{agent}.jsonl"
        smoke_path = output / f"plan/smoke/{agent}.jsonl"
        smoke = []
        for relation in RELATIONS:
            match = next((
                row for row in tasks if row["relation"] == relation), None)
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
        "schema": "full-candidate-tournament-plan-v1",
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
        "skipped_zero_cardinality_rows": skipped_zero_cardinality,
        "views": list(VIEWS),
        "prompt_arms": list(PROMPT_ARMS),
        "unknown_anchor": True,
        "balanced_code_cycles": True,
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "jobs": jobs,
        "arms": list(ARMS),
        "evidence_arms": list(EVIDENCE_ARMS),
        "gates": list(GATES),
        "minimum_incremental_delta": MIN_INCREMENTAL_DELTA,
        "agents": str(Path(args.agents).resolve()),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    _write_json(output / "plan/PLAN.json", plan)
    print(json.dumps({
        "rows": len(registry),
        "skipped_zero_cardinality_rows": skipped_zero_cardinality,
        "jobs": {key: value["tasks"] for key, value in jobs.items()},
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


def _validated_responses(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    responses, all_tasks = {}, []
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
            raise ContractError(f"stale tournament responses: {agent}")
        tasks = read_jsonl(task_path)
        by_id = validate_tasks(tasks, agent)
        rows = read_jsonl(response_path)
        if len(rows) != len(tasks):
            raise ContractError(f"incomplete tournament responses: {agent}")
        for response in rows:
            task_id = str(response["task_id"])
            if task_id not in by_id or task_id in responses:
                raise ContractError("invalid tournament response id")
            validate_task_response(by_id[task_id], response)
            responses[task_id] = response
        all_tasks.extend(tasks)
    return responses, all_tasks


def aggregate_scores(
    tasks: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Mapping[str, Any]],
) -> dict[
    tuple[str, str],
    dict[str, dict[str, dict[str, dict[str, float]]]],
]:
    accum: dict[
        tuple[str, str],
        dict[str, dict[str, dict[str, dict[str, list[float]]]]]
    ] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(list)))))
    for task in tasks:
        response = responses[str(task["task_id"])]
        probabilities = response["choice_probabilities"]
        unknown = max(float(probabilities[UNKNOWN]), EPSILON)
        key = (str(task["subject"]), str(task["relation"]))
        for node_id in task["group_node_ids"]:
            probability = max(float(probabilities[str(node_id)]), EPSILON)
            accum[key][str(task["view"])][str(task["prompt_arm"])][
                str(task["agent_id"])][str(node_id)].append(
                    math.log(probability) - math.log(unknown))
    return {
        key: {
            view: {
                prompt: {
                    agent: {
                        node_id: statistics.mean(values)
                        for node_id, values in nodes.items()
                    }
                    for agent, nodes in agents.items()
                }
                for prompt, agents in prompts.items()
            }
            for view, prompts in views.items()
        }
        for key, views in accum.items()
    }


def _agent_score(
    evidence: Mapping[
        str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    evidence_arm: str, agent: str, node_id: str,
) -> float:
    if evidence_arm == "ensemble":
        return statistics.mean(
            float(evidence[view][prompt][agent][node_id])
            for view in VIEWS for prompt in PROMPT_ARMS)
    view, prompt = evidence_arm.split(":", 1)
    if view == "combined":
        return statistics.mean(
            float(evidence[item][prompt][agent][node_id])
            for item in VIEWS)
    return float(evidence[view][prompt][agent][node_id])


def _arm_score(qwen: float, gemma: float, arm: str) -> float:
    if arm == "qwen":
        return qwen
    if arm == "gemma":
        return gemma
    if arm == "mean":
        return (qwen + gemma) / 2.0
    if arm == "minimum":
        return min(qwen, gemma)
    if arm == "agreement_penalized":
        return (qwen + gemma) / 2.0 - abs(qwen - gemma) / 2.0
    raise ContractError(f"unknown tournament arm: {arm}")


def decode_row(
    registry: Mapping[str, Any],
    evidence: Mapping[
        str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    *, evidence_arm: str, arm: str, gate: str,
) -> tuple[list[str], dict[str, Any]]:
    nodes = list(registry["nodes"])
    cardinality = int(registry["cardinality"])
    q = {
        str(node["node_id"]): _agent_score(
            evidence, evidence_arm, QWEN, str(node["node_id"]))
        for node in nodes}
    g = {
        str(node["node_id"]): _agent_score(
            evidence, evidence_arm, GEMMA, str(node["node_id"]))
        for node in nodes}
    combined = {
        node_id: _arm_score(q[node_id], g[node_id], arm) for node_id in q}
    ranked = sorted(
        nodes,
        key=lambda node: (
            combined[str(node["node_id"])],
            str(node["representative"])),
        reverse=True)
    proposal_nodes = ranked[:cardinality]
    incumbent_nodes = [node for node in nodes if node["is_incumbent"]]
    proposal_ids = {str(node["node_id"]) for node in proposal_nodes}
    incumbent_ids = {str(node["node_id"]) for node in incumbent_nodes}
    changed = proposal_ids != incumbent_ids
    added = [node for node in proposal_nodes
             if str(node["node_id"]) not in incumbent_ids]
    removed = [node for node in incumbent_nodes
               if str(node["node_id"]) not in proposal_ids]
    advantage = (
        min((combined[str(node["node_id"])] for node in added), default=0.0)
        - max((combined[str(node["node_id"])] for node in removed), default=0.0)
    )
    q_advantage = (
        min((q[str(node["node_id"])] for node in added), default=0.0)
        - max((q[str(node["node_id"])] for node in removed), default=0.0)
    )
    g_advantage = (
        min((g[str(node["node_id"])] for node in added), default=0.0)
        - max((g[str(node["node_id"])] for node in removed), default=0.0)
    )
    q_top = {
        str(node["node_id"]) for node in sorted(
            nodes, key=lambda node: q[str(node["node_id"])], reverse=True
        )[:cardinality]}
    g_top = {
        str(node["node_id"]) for node in sorted(
            nodes, key=lambda node: g[str(node["node_id"])], reverse=True
        )[:cardinality]}
    allowed = {
        "unconditional": changed,
        "advantage": changed and advantage > 0.0,
        "dual_advantage": changed and q_advantage > 0.0 and g_advantage > 0.0,
        "unknown": changed and all(
            combined[str(node["node_id"])] > 0.0 for node in added),
        "dual_unknown": changed and all(
            q[str(node["node_id"])] > 0.0
            and g[str(node["node_id"])] > 0.0 for node in added),
        "consensus": changed and proposal_ids == q_top == g_top,
    }[gate]
    selected = (
        [str(node["representative"]) for node in proposal_nodes]
        if allowed else list(registry["incumbent_objects"]))
    return selected, {
        "changed": allowed,
        "proposal": [str(node["representative"]) for node in proposal_nodes],
        "advantage": advantage,
        "qwen_advantage": q_advantage,
        "gemma_advantage": g_advantage,
    }


def _predictions(
    registry: Sequence[Mapping[str, Any]],
    evidence: Mapping[
        tuple[str, str], Mapping[str, Mapping[str, Mapping[str, float]]]],
    starting: Mapping[tuple[str, str], Mapping[str, Any]],
    config: tuple[str, str, str],
    keys: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_arm, arm, gate = config
    registry_by = {_key(row): row for row in registry}
    predictions, diagnostics = [], []
    for key, baseline in starting.items():
        if keys is not None and key not in keys:
            continue
        if key not in registry_by:
            objects, detail = list(baseline["ObjectEntities"]), {
                "changed": False, "proposal": [], "advantage": 0.0,
                "qwen_advantage": 0.0, "gemma_advantage": 0.0,
            }
        else:
            objects, detail = decode_row(
                registry_by[key], evidence[key],
                evidence_arm=evidence_arm, arm=arm, gate=gate)
        predictions.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": objects,
        })
        diagnostics.append({
            "SubjectEntity": key[0], "Relation": key[1],
            "evidence_arm": evidence_arm, "arm": arm, "gate": gate, **detail,
        })
    return predictions, diagnostics


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    graph_path = Path(plan["graph"])
    registry_path = Path(plan["registry"])
    if (
        plan.get("schema") != "full-candidate-tournament-plan-v1"
        or plan.get("contains_labels")
        or plan.get("gold_aware")
        or plan.get("validation_opened")
        or sha256(graph_path) != plan["graph_sha256"]
        or sha256(registry_path) != plan["registry_sha256"]
        or plan.get("starting_pipeline_id") != COMPETITION_PIPELINE_ID
    ):
        raise ContractError("invalid tournament plan")
    responses, tasks = _validated_responses(plan)
    evidence = aggregate_scores(tasks, responses)
    registry = read_jsonl(registry_path)
    if set(evidence) != {_key(row) for row in registry}:
        raise ContractError("tournament evidence coverage failure")

    control_path = Path(args.control).resolve()
    if (
        control_path != Path(plan["starting_predictions"]).resolve()
        or sha256(control_path) != plan["starting_predictions_sha256"]
    ):
        raise ContractError("tournament starting predictions changed")
    control_manifest = validate_registered_predictions(
        control_path, pipeline_id=COMPETITION_PIPELINE_ID, split="train")
    starting_rows = read_jsonl(control_path)
    starting = {_key(row): row for row in starting_rows}
    gold_rows = read_jsonl(Path(args.gold).resolve())
    gold_by = {_key(row): row for row in gold_rows}
    if set(starting) != set(gold_by):
        raise ContractError("tournament control/gold mismatch")
    fold_path = Path(control_manifest["folds"])
    if sha256(fold_path) != control_manifest["folds_sha256"]:
        raise ContractError("stale exact-SOTA folds")
    folds = {_key(row): int(row["fold"]) for row in read_jsonl(fold_path)}
    configs = [
        (evidence_arm, arm, gate)
        for evidence_arm in EVIDENCE_ARMS
        for arm in ARMS for gate in GATES]

    starting_scores = score(starting_rows, gold_rows)
    cache = {}
    utility_cache: dict[tuple[tuple[str, str], tuple[str, ...]], float] = {}
    def row_utility(
        key: tuple[str, str], prediction: Mapping[str, Any],
    ) -> float:
        state = tuple(map(str, prediction["ObjectEntities"]))
        cache_key = (key, state)
        if cache_key not in utility_cache:
            utility_cache[cache_key] = _row_f1(
                state, gold_by[key], key[1])
        return utility_cache[cache_key]

    def macro_f1(
        keys: set[tuple[str, str]],
        utility_by: Mapping[tuple[str, str], float],
    ) -> float:
        by_relation: dict[str, list[float]] = defaultdict(list)
        for key in keys:
            by_relation[key[1]].append(utility_by[key])
        return statistics.mean(
            statistics.mean(values) for values in by_relation.values())

    config_results = []
    for config in configs:
        predictions, diagnostics = _predictions(
            registry, evidence, starting, config)
        prediction_by = {_key(row): row for row in predictions}
        utility_by = {
            key: row_utility(key, row)
            for key, row in prediction_by.items()}
        selected = macro_f1(set(starting), utility_by)
        cache[config] = (
            prediction_by,
            {_key(row): row for row in diagnostics},
            utility_by,
        )
        config_results.append({
            "config": config,
            "scores": {"*** All Relations ***": selected},
            "delta": selected - starting_scores["*** All Relations ***"],
            "changed": sum(row["changed"] for row in diagnostics),
        })
    best_global = max(
        config_results,
        key=lambda row: (row["delta"], -row["changed"], row["config"]))

    oof_by, nested_diagnostics, selections = {}, [], []
    for fold in sorted(set(folds.values())):
        fit_keys = {key for key, value in folds.items() if value != fold}
        hold_keys = {key for key, value in folds.items() if value == fold}
        candidates = []
        for config in configs:
            prediction_by, diagnostic_by, utility_by = cache[config]
            candidates.append((
                macro_f1(fit_keys, utility_by),
                -sum(
                    diagnostic_by[key]["changed"] for key in fit_keys),
                config,
            ))
        selected_config = max(candidates)[2]
        prediction_by, diagnostic_by, _ = cache[selected_config]
        for key in hold_keys:
            row = prediction_by[key]
            if _key(row) in oof_by:
                raise ContractError("duplicate tournament OOF row")
            oof_by[_key(row)] = row
        nested_diagnostics.extend(
            {**diagnostic_by[key], "outer_fold": fold} for key in hold_keys)
        selections.append({
            "fold": fold,
            "fit_rows": len(fit_keys),
            "hold_rows": len(hold_keys),
            "selected_config": list(selected_config),
        })
    if set(oof_by) != set(starting):
        raise ContractError("tournament nested OOF coverage failure")
    predictions = [oof_by[_key(row)] for row in starting_rows]
    selected_scores = score(predictions, gold_rows)
    delta = (
        selected_scores["*** All Relations ***"]
        - starting_scores["*** All Relations ***"])
    relation_deltas = _relation_deltas(selected_scores, starting_scores)
    passed = (
        delta >= MIN_INCREMENTAL_DELTA
        and min(relation_deltas.values()) >= MAX_REGRESSION_PER_RELATION)

    prediction_path = output / "analysis/TRAIN_OOF_PREDICTIONS.jsonl"
    diagnostic_path = output / "analysis/TRAIN_OOF_DIAGNOSTICS.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(diagnostic_path, nested_diagnostics)
    result = {
        "schema": "full-candidate-tournament-result-v1",
        "development_only": True,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "starting_pipeline_id": COMPETITION_PIPELINE_ID,
        "starting_scores": starting_scores,
        "selected_scores": selected_scores,
        "incremental_delta": delta,
        "relation_deltas": relation_deltas,
        "changed_rows": sum(row["changed"] for row in nested_diagnostics),
        "fold_selections": selections,
        "predeclared_configs": len(configs),
        "config_results": config_results,
        "best_in_sample_diagnostic": best_global,
        "deployment_gate": {
            "passed": passed,
            "minimum_incremental_delta": MIN_INCREMENTAL_DELTA,
            "minimum_relation_delta": MAX_REGRESSION_PER_RELATION,
        },
        "artifacts": {
            "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path),
            "diagnostics": str(diagnostic_path),
            "diagnostics_sha256": sha256(diagnostic_path),
        },
    }
    _write_json(output / "analysis/RESULT.json", result)
    lines = [
        "# Full candidate tournament", "",
        f"- Exact SOTA start: "
        f"**{starting_scores['*** All Relations ***']:.9f}**",
        f"- Nested OOF tournament: "
        f"**{selected_scores['*** All Relations ***']:.9f}**",
        f"- Incremental delta: **{delta:+.9f}**",
        f"- Changed rows: **{result['changed_rows']}**",
        f"- Promotion gate: **{passed}**", "",
        "## Relation deltas", "",
        "| relation | delta |", "|---|---:|",
    ]
    lines.extend(
        f"| {relation} | {value:+.6f} |"
        for relation, value in relation_deltas.items())
    lines.extend(["", "## Fold-selected arms", ""])
    lines.extend(
        f"- fold {row['fold']}: `{' / '.join(row['selected_config'])}`"
        for row in selections)
    (output / "analysis/RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "starting": starting_scores["*** All Relations ***"],
        "selected": selected_scores["*** All Relations ***"],
        "incremental_delta": delta,
        "gate_passed": passed,
        "best_in_sample": best_global,
        "output": str(output / "analysis"),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("prepare", "analyze"))
    value.add_argument("--graph", default=str(DEFAULT_GRAPH))
    value.add_argument("--gold", default=str(DEFAULT_GOLD))
    value.add_argument("--control", default=str(DEFAULT_TRAIN_OOF))
    value.add_argument("--agents", default=str(DEFAULT_AGENTS))
    value.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return value


def main() -> int:
    args = parser().parse_args()
    return prepare(args) if args.command == "prepare" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
