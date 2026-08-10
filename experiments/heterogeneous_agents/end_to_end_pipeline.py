#!/usr/bin/env python3
"""Run the whole heterogeneous system on a split, from raw rows to a score.

Every previous runner in this repo executes one *experiment*: it consumes
frozen generations produced weeks earlier by some other runner, and edits a
prediction artifact inherited from a chain of predecessors.  Nothing here
could regenerate the system from `data/<split>.jsonl`.  That is the gap this
module closes, and it is the path a blind test submission has to take.

Stages
------
``plan``      Build the model-facing task files for every evidence route of
              the deployed portfolio.  Label-free by construction: only
              (SubjectEntity, Relation) reach a prompt.
``assemble``  Turn the four response files into one candidate graph plus the
              chain's own aggregate answer, with no historical incumbent.
``graph``     Type the graph (components, equivalence, typed edges) and
              attach every model family as a first-class evidence route.
``decode``    Apply the frozen policy stack in one pass over that graph.
``score``     Official scorer.  Refuses to run on a blind split.

The routes are exactly the deployed portfolio, 30,515,165,024 parameters
against the 32B cap:

    qwen_recall           Qwen3.5-9B        CoT-5,  N=10   self-consistency
    gemma_independent     Gemma-3-12B       CoT-5,  N=1    independent
    ministral_independent Ministral-8B      zero-shot N=3  typed admission
    ministral_independent Ministral-8B      CoT-5 cap40 N=10  support rule

What this pipeline deliberately does NOT carry is the historical incumbent.
It starts from the chain's own aggregate over freshly generated evidence, so
its output is a genuine end-to-end result rather than a replay.  Two layers
of the deployed artifact are therefore out of scope here and are recorded as
such by ``decode``: the route-residual company arm (its train gate disables
itself under current gold) and the capacity precision veto (it needs a
separate review-generation pass that has no end-to-end runner yet).
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.assemble_and_audit import (
    assemble_graphs,
    load_responses,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    build_agent_tasks,
    load_agent_config,
    load_synthetic_by_relation,
    read_jsonl,
    sha256,
    validate_inputs,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.heterogeneous_memory_selector import _key
from experiments.heterogeneous_agents.relational_candidate_graph import (
    augment_relational_graph,
)
from experiments.heterogeneous_agents.unified_graph_decoder import (
    MINISTRAL_COT40,
    MINISTRAL_N3,
    apply_area_unanimity,
    apply_component_residual,
    apply_cot40_support,
    cot40_route_evidence,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FROZEN_ARTIFACTS = ROOT / "artifacts/frozen"

DEFAULT_OUTPUT = RUNS / "end_to_end_validation_20260730_v1"
DEFAULT_SYNTHETIC = ROOT / "data/synthetic_cot_faithful.jsonl"
COMPONENT_MODELS = FROZEN_ARTIFACTS / "component_models.json"

# Frozen portfolio configs.  The CoT config carries all three checkpoints at
# their deployed prompt settings; the supply config carries the zero-shot
# Ministral used by the typed admission route.
COT_AGENTS = ROOT / "configs/final/portfolio_cot.json"
SUPPLY_AGENTS = ROOT / "configs/final/portfolio_supply.json"

QWEN = "qwen_recall"
GEMMA = "gemma_independent"
MINISTRAL = "ministral_independent"

SEED = 20260730
PARAMETER_CAP = 32_000_000_000
EXPECTED_PARAMETER_TOTAL = 30_515_165_024

PLAN_SCHEMA = "end-to-end-pipeline-plan-v1"
GRAPH_SCHEMA = "end-to-end-unified-graph-v1"
DECISION_SCHEMA = "end-to-end-decisions-v1"

# One entry per generation pass: an evidence route of the deployed system.
ROUTES = (
    {
        "route": "qwen:self_consistency",
        "agent_id": QWEN,
        "config": "cot",
        "n_proposals": 10,
        # Qwen and Gemma build the base graph, so their existence and
        # cardinality commitments are consumed by the assembler.
        "phases": ("commit_existence", "commit_cardinality", "propose"),
        "purpose": "primary candidate reservoir (self-consistency)",
    },
    {
        "route": "gemma:independent",
        "agent_id": GEMMA,
        "config": "cot",
        "n_proposals": 1,
        "phases": ("commit_existence", "commit_cardinality", "propose"),
        "purpose": "independent second opinion (best-calibrated support)",
    },
    {
        "route": MINISTRAL_N3,
        "agent_id": MINISTRAL,
        "config": "supply",
        "n_proposals": 3,
        # Both Ministral routes are candidate supply only: no deployed policy
        # reads their commitments, and generating them would cost ~950 extra
        # model calls per route for nothing.  This matches the frozen CoT40
        # task file, which contains propose tasks and nothing else.
        "phases": ("propose",),
        "purpose": "zero-shot typed admission (area unanimity rule)",
    },
    {
        "route": MINISTRAL_COT40,
        "agent_id": MINISTRAL,
        "config": "cot",
        "n_proposals": 10,
        "phases": ("propose",),
        "purpose": "synthetic-CoT recall (two-thirds support rule)",
    },
)
ROUTE_BY_NAME = {route["route"]: route for route in ROUTES}

OUT_OF_SCOPE = (
    {
        "layer": "route_residual_company_arm",
        "reason": "its train gate selects a pass-through arm under current gold",
        "deployed_rows": 1,
    },
    {
        "layer": "capacity_qwen_precision_veto",
        "reason": "requires a review-generation pass with no end-to-end runner",
        "deployed_rows": 11,
    },
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _route_slug(route: str) -> str:
    return route.replace(":", "__")


def _agent_of(config: Mapping[str, Any], agent_id: str) -> Mapping[str, Any]:
    for agent in config["agents"]:
        if str(agent["id"]) == agent_id:
            return agent
    raise ContractError(f"agent {agent_id} absent from portfolio config")


def _verify_portfolio(configs: Mapping[str, Mapping[str, Any]]) -> int:
    """Confirm the deployed three-checkpoint budget before any generation."""
    checkpoints: dict[str, int] = {}
    for route in ROUTES:
        agent = _agent_of(configs[route["config"]], route["agent_id"])
        model = str(agent["model"])
        count = int(agent["verified_parameter_count"])
        if checkpoints.setdefault(model, count) != count:
            raise ContractError(f"inconsistent parameter count for {model}")
    total = sum(checkpoints.values())
    if total > PARAMETER_CAP:
        raise ContractError(
            f"portfolio exceeds the parameter cap: {total} > {PARAMETER_CAP}")
    if total != EXPECTED_PARAMETER_TOTAL:
        raise ContractError(
            f"portfolio total changed: {total} != {EXPECTED_PARAMETER_TOTAL}")
    return total


# ------------------------------------------------------------------- plan --
def plan(args: argparse.Namespace) -> int:
    split = str(args.split)
    output = Path(args.output_dir).resolve()
    input_path = Path(args.input).resolve()

    rows = read_jsonl(input_path)
    validate_inputs(rows)
    # Only the key reaches a prompt; object entities never leave this scope.
    task_rows = [{
        "SubjectEntity": str(row["SubjectEntity"]),
        "Relation": str(row["Relation"]),
    } for row in rows]
    if len({(r["SubjectEntity"], r["Relation"]) for r in task_rows}) != len(
            task_rows):
        raise ContractError("duplicate (subject, relation) in the input split")

    configs = {
        "cot": load_agent_config(Path(args.cot_agents).resolve()),
        "supply": load_agent_config(Path(args.supply_agents).resolve()),
    }
    total_parameters = _verify_portfolio(configs)
    synthetic = load_synthetic_by_relation(Path(args.synthetic_cot).resolve())

    rows_path = output / "plan/INPUT_ROWS.jsonl"
    write_jsonl_atomic(rows_path, task_rows)

    jobs = {}
    for route in ROUTES:
        agent = _agent_of(configs[route["config"]], route["agent_id"])
        tasks = build_agent_tasks(
            task_rows, agent, synthetic,
            seed=SEED, n_proposals=int(route["n_proposals"]),
            question_contract=str(args.question_contract))
        wanted = set(route["phases"])
        tasks = [task for task in tasks if str(task["phase"]) in wanted]
        expected = len(task_rows) * len(wanted)
        if len(tasks) != expected:
            raise ContractError(
                f"{route['route']}: expected {expected} tasks, got {len(tasks)}")
        slug = _route_slug(str(route["route"]))
        task_path = output / f"plan/tasks/{slug}.jsonl"
        write_jsonl_atomic(task_path, tasks)
        # One smoke task per relation keeps a cheap pre-flight honest.
        smoke, seen = [], set()
        for task in tasks:
            if task.get("phase") != "propose":
                continue
            relation = str(task["relation"])
            if relation not in seen:
                seen.add(relation)
                smoke.append(task)
        smoke_path = output / f"plan/smoke/{slug}.jsonl"
        write_jsonl_atomic(smoke_path, smoke)
        jobs[str(route["route"])] = {
            "route": route["route"],
            "agent_id": route["agent_id"],
            "agent_config": str(Path(
                args.cot_agents if route["config"] == "cot"
                else args.supply_agents).resolve()),
            "model": str(agent["model"]),
            "revision": agent.get("revision"),
            "synthetic_shots": agent.get("synthetic_shots"),
            "question_contract": str(args.question_contract),
            "reasoning_words": agent.get("proposal_reasoning_words"),
            "n_proposals": int(route["n_proposals"]),
            "phases": list(route["phases"]),
            "generation_calls": len(task_rows) * (
                int(route["n_proposals"]) if "propose" in route["phases"] else 0),
            "purpose": route["purpose"],
            "tasks": len(tasks),
            "task_path": str(task_path),
            "task_sha256": sha256(task_path),
            "smoke_path": str(smoke_path),
            "smoke_sha256": sha256(smoke_path),
            "response_path": str(output / f"responses/{slug}.jsonl"),
            "smoke_response_path": str(
                output / f"smoke_responses/{slug}.jsonl"),
        }

    blind = split == "test"
    plan_record = {
        "schema": PLAN_SCHEMA,
        "split": split,
        "blind": blind,
        "contains_labels": False,
        "gold_aware": False,
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "input_rows": str(rows_path),
        "input_rows_sha256": sha256(rows_path),
        "rows": len(task_rows),
        "synthetic_cot": str(Path(args.synthetic_cot).resolve()),
        "synthetic_cot_sha256": sha256(Path(args.synthetic_cot)),
        "cot_agents": str(Path(args.cot_agents).resolve()),
        "cot_agents_sha256": sha256(Path(args.cot_agents)),
        "supply_agents": str(Path(args.supply_agents).resolve()),
        "supply_agents_sha256": sha256(Path(args.supply_agents)),
        "component_models": str(Path(args.component_models).resolve()),
        "component_models_sha256": sha256(Path(args.component_models)),
        "verified_parameter_total": total_parameters,
        "question_contract": str(args.question_contract),
        "parameter_cap": PARAMETER_CAP,
        "seed": SEED,
        "routes": [dict(route) for route in ROUTES],
        "jobs": jobs,
        "out_of_scope_layers": [dict(item) for item in OUT_OF_SCOPE],
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    plan_path = output / "plan/PLAN.json"
    _write_json(plan_path, plan_record)
    print(json.dumps({
        "plan": str(plan_path),
        "split": split,
        "rows": len(task_rows),
        "verified_parameter_total": total_parameters,
        "parameter_cap": PARAMETER_CAP,
        "generation_passes": {
            name: job["tasks"] for name, job in jobs.items()},
    }, indent=2, sort_keys=True))
    return 0


def _validate_plan(output: Path) -> dict[str, Any]:
    record = _json(output / "plan/PLAN.json")
    if record.get("schema") != PLAN_SCHEMA:
        raise ContractError("end-to-end plan schema mismatch")
    for field in ("input", "input_rows", "synthetic_cot", "cot_agents",
                  "supply_agents", "component_models", "implementation"):
        if sha256(Path(record[field])) != record[f"{field}_sha256"]:
            raise ContractError(f"frozen plan artifact changed: {field}")
    for name, job in record["jobs"].items():
        if sha256(Path(job["task_path"])) != job["task_sha256"]:
            raise ContractError(f"{name}: task file changed since planning")
    return record


def _validated_responses(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    response_path = Path(job["response_path"])
    manifest_path = response_path.with_suffix(
        response_path.suffix + ".manifest.json")
    if not response_path.is_file() or not manifest_path.is_file():
        raise ContractError(
            f"{job['route']}: missing responses; run the generate stage")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != "heterogeneous-agent-responses-v1"
        or manifest.get("agent_id") != job["agent_id"]
        or manifest.get("task_sha256") != job["task_sha256"]
        or manifest.get("output_sha256") != sha256(response_path)
        or int(manifest.get("tasks", -1)) != int(job["tasks"])
    ):
        raise ContractError(f"{job['route']}: stale response manifest")
    return read_jsonl(response_path)


# --------------------------------------------------------------- assemble --
def assemble(args: argparse.Namespace) -> int:
    """Build the candidate graph and the chain's own aggregate answer."""
    output = Path(args.output_dir).resolve()
    record = _validate_plan(output)
    rows = read_jsonl(Path(record["input_rows"]))

    cot_config = load_agent_config(Path(record["cot_agents"]))
    # The base graph is the primary reservoir: Qwen self-consistency plus the
    # independent Gemma opinion.  Ministral enters later as typed routes so
    # that policies fitted before it existed keep their evidence scope.
    base_agents = [
        _agent_of(cot_config, QWEN), _agent_of(cot_config, GEMMA)]
    responses: dict[str, dict[tuple[str, str], dict]] = {}
    for route_name in ("qwen:self_consistency", "gemma:independent"):
        job = record["jobs"][route_name]
        by_key: dict[tuple[str, str], dict] = {}
        for response in _validated_responses(job):
            key = (str(response["subject"]), str(response["relation"]))
            by_key.setdefault(key, {})[str(response["phase"])] = response
        responses[str(job["agent_id"])] = by_key

    # assemble_graphs expects {agent_id: {(subject, relation, phase): row}}
    flattened = {
        agent_id: {
            (key[0], key[1], phase): value
            for key, phases in by_key.items()
            for phase, value in phases.items()
        }
        for agent_id, by_key in responses.items()
    }
    graphs = assemble_graphs(rows, base_agents, flattened)
    graph_path = output / "graph/BASE_GRAPH.jsonl"
    write_jsonl_atomic(graph_path, graphs)
    print(json.dumps({
        "base_graph": str(graph_path),
        "rows": len(graphs),
        "candidates": sum(len(row["candidates"]) for row in graphs),
    }, indent=2, sort_keys=True))
    return 0


# ------------------------------------------------------------------ graph --
def _numeric(value: str) -> float | None:
    from evaluate import try_parse_number
    return try_parse_number(str(value))


def _within_tolerance(left: float, right: float) -> bool:
    return abs(left - right) <= 0.05 * max(abs(left), abs(right), 1e-12)


def _attach_supply_route(
    row: dict[str, Any],
    response: Mapping[str, Any],
    *,
    route_name: str,
    samples: int,
) -> None:
    """Attach a Ministral candidate-supply route with typed admission.

    The area policy fires only on a *new* numeric component that the zero-shot
    route proposes unanimously.  Novelty is decided against the candidates the
    base chain already holds, under the evaluator's own 5% tolerance, so the
    rule cannot fire on a value the chain already knows.
    """
    relation = str(row["Relation"])
    occurrences, _ = cot40_route_evidence(
        list(response["generations"])[:samples], relation)
    existing = [
        _numeric(str(node["item"])) for node in row.get("candidates", [])]
    existing = [value for value in existing if value is not None]

    candidates = list(row.get("candidates", []))
    by_canonical = {}
    for node in candidates:
        from experiments.heterogeneous_agents.core import canonical_key
        key = canonical_key(str(node["item"]), relation)
        if key and key not in by_canonical:
            by_canonical[key] = node

    for canonical, value in sorted(occurrences.items()):
        support = sorted({int(i) for i in value["generation_indices"]})
        node = by_canonical.get(canonical)
        is_new = node is None
        if node is None:
            node = {
                "key": canonical,
                "item": str(value["item"]),
                "routes": {},
                "sources": {},
                "selected_by": {},
                "output_eligible": True,
                "admitted_by": route_name,
            }
            candidates.append(node)
            by_canonical[canonical] = node
        route: dict[str, Any] = {
            "model_family": MINISTRAL,
            "route_type": "independent-recall",
            "support": len(support),
            "samples": samples,
            "support_rate": len(support) / samples,
            "selected": len(support) >= (samples + 1) // 2,
            "generation_indices": support,
            "display_item": str(value["item"]),
        }
        if route_name == MINISTRAL_N3:
            number = _numeric(str(value["item"]))
            unanimous = len(support) == samples
            novel = number is not None and not any(
                _within_tolerance(number, other) for other in existing)
            if (
                relation in ("hasArea", "hasCapacity")
                and unanimous and novel and is_new
            ):
                route["admission_reason"] = (
                    "numeric_complete_link_self_consistent_new")
        node.setdefault("routes", {})[route_name] = route
    row["candidates"] = candidates
    row.setdefault("proposal_routes", {})[route_name] = {
        "available": True,
        "model_family": MINISTRAL,
        "n_samples": samples,
        "generation_provenance_available": True,
    }


def graph(args: argparse.Namespace) -> int:
    """Type the assembled graph and attach every model family as a route."""
    from experiments.heterogeneous_agents.assemble_and_audit import (
        heterogeneous_prediction,
        prediction_for_agent,
    )
    from experiments.heterogeneous_agents.route_aware_candidate_graph import (
        augment_graph,
    )

    output = Path(args.output_dir).resolve()
    record = _validate_plan(output)
    base = read_jsonl(output / "graph/BASE_GRAPH.jsonl")

    ministral_responses: dict[str, dict[tuple[str, str], dict]] = {}
    for route_name in (MINISTRAL_N3, MINISTRAL_COT40):
        job = record["jobs"][route_name]
        ministral_responses[route_name] = {
            (str(r["subject"]), str(r["relation"])): r
            for r in _validated_responses(job)
            if str(r.get("phase")) == "propose"
        }

    unified = []
    for source in base:
        row = copy.deepcopy(dict(source))
        relation = str(row["Relation"])
        key = _key(row)

        # Per-agent view: each family's own answer and commitment summary.
        agents: dict[str, Any] = {}
        agent_outputs: dict[str, list[str]] = {}
        for agent_id in (QWEN, GEMMA):
            commitments = row["commitments"][agent_id]
            n_samples = int(row["proposal_sample_counts"][agent_id])
            diagnostics = row["proposal_parse_diagnostics"][agent_id]
            agents[agent_id] = {
                "existence": {
                    "available": True,
                    "selected": commitments["existence"],
                    "probabilities": commitments.get(
                        "existence_probabilities", {}),
                },
                "cardinality": {
                    "available": True,
                    "selected": commitments["cardinality"],
                    "probabilities": commitments.get(
                        "cardinality_probabilities", {}),
                },
                "n_samples": n_samples,
                "none_count": int(diagnostics.get("explicit_none", 0)),
                "none_rate": (
                    int(diagnostics.get("explicit_none", 0)) / n_samples
                    if n_samples else 0.0),
                "parse_failures": int(sum(
                    count for status, count in diagnostics.items()
                    if status not in ("parsed_nonempty", "explicit_none"))),
            }
            agent_outputs[agent_id] = [
                str(item) for item in prediction_for_agent(source, agent_id)]
        row["agents"] = agents
        row["agent_outputs"] = agent_outputs

        # Candidate-level evidence in the shape the route builder expects.
        for node in row["candidates"]:
            node["sources"] = {
                agent_id: {
                    "support": int(support),
                    "samples": int(
                        row["proposal_sample_counts"].get(agent_id, 0)),
                    "support_rate": (
                        int(support)
                        / max(int(row["proposal_sample_counts"].get(
                            agent_id, 0)), 1)),
                }
                for agent_id, support in node["proposal_support"].items()
            }
            node.setdefault("output_eligible", True)

        # The chain's own aggregate is the incumbent: no historical artifact.
        row["baseline_objects"] = [
            str(item) for item in heterogeneous_prediction(
                source, use_reviews=False)]
        row["baseline_agent"] = "heterogeneous_chain_aggregate"

        # No system-2 route exists in the end-to-end portfolio.
        row = augment_graph(row, [])
        for route_name, samples in (
            (MINISTRAL_N3, 3), (MINISTRAL_COT40, 10),
        ):
            response = ministral_responses[route_name].get(key)
            if response is None:
                raise ContractError(f"{key}: missing {route_name} response")
            _attach_supply_route(
                row, response, route_name=route_name, samples=samples)
        row.pop("relational_graph", None)
        row.pop("relational_graph_schema", None)
        row = augment_relational_graph(row)
        row["schema"] = GRAPH_SCHEMA
        row["contains_labels"] = False
        row["gold_aware"] = False
        unified.append(row)

    graph_path = output / "graph/UNIFIED_GRAPH.jsonl"
    write_jsonl_atomic(graph_path, unified)
    route_counts: Counter[str] = Counter()
    for row in unified:
        for node in row.get("candidates", []):
            for name in node.get("routes", {}):
                route_counts[name] += 1
    _write_json(graph_path.with_suffix(graph_path.suffix + ".manifest.json"), {
        "schema": "end-to-end-unified-graph-manifest-v1",
        "contains_labels": False,
        "gold_aware": False,
        "split": record["split"],
        "rows": len(unified),
        "output": str(graph_path),
        "output_sha256": sha256(graph_path),
        "route_candidate_counts": dict(sorted(route_counts.items())),
    })
    print(json.dumps({
        "unified_graph": str(graph_path),
        "rows": len(unified),
        "route_candidate_counts": dict(sorted(route_counts.items())),
    }, indent=2, sort_keys=True))
    return 0


# ----------------------------------------------------------------- decode --
def decode(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    record = _validate_plan(output)
    models = _json(Path(record["component_models"]))
    graphs = read_jsonl(output / "graph/UNIFIED_GRAPH.jsonl")

    predictions, decisions = [], []
    layer_changes: Counter[str] = Counter()
    for row in graphs:
        objects = [str(item) for item in row["baseline_objects"]]
        trace = []
        stack = (
            ("component_surface_residual_ridge",
             lambda g, o: apply_component_residual(g, o, models)),
            ("area_unanimous_new_component_replace", apply_area_unanimity),
            ("cot40_two_thirds_support_7", apply_cot40_support),
        )
        for policy_id, function in stack:
            before = list(objects)
            objects, detail = function(row, objects)
            objects = [str(item) for item in objects]
            if objects != before:
                layer_changes[policy_id] += 1
            trace.append({
                "policy": policy_id, "before": before,
                "after": list(objects), "changed": objects != before,
                **detail,
            })
        key = _key(row)
        predictions.append({
            "SubjectEntity": key[0], "Relation": key[1],
            "ObjectEntities": objects,
        })
        decisions.append({
            "schema": DECISION_SCHEMA,
            "SubjectEntity": key[0], "Relation": key[1],
            "chain_baseline": [
                str(item) for item in row["baseline_objects"]],
            "prediction": objects,
            "layers": trace,
        })
    prediction_path = output / "PREDICTIONS.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(output / "DECISIONS.jsonl", decisions)
    print(json.dumps({
        "predictions": str(prediction_path),
        "rows": len(predictions),
        "layer_changed_rows": dict(sorted(layer_changes.items())),
    }, indent=2, sort_keys=True))
    return 0


# ------------------------------------------------------------------ score --
def score(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    record = _validate_plan(output)
    if record.get("blind"):
        raise ContractError(
            "refusing to score a blind split; test labels must stay unopened")
    import evaluate as official

    gold = official.read_jsonl_file(record["input"])
    rows = official.read_jsonl_file(str(output / "PREDICTIONS.jsonl"))
    per_relation = official.macro_average_per_relation(
        official.evaluate_per_sr_pair(rows, gold, official.RELATION_TYPE))
    pooled = per_relation["*** All Relations ***"]

    lines = [
        f"# End-to-end heterogeneous system — {record['split']}",
        "",
        f"Generated from `{record['input']}` by this pipeline: four evidence "
        "routes produced fresh, one typed graph, one decode pass. No "
        "historical incumbent and no inherited prediction artifact.",
        "",
        f"- Pooled macro-F1: **{pooled['macro-f1']:.6f}**",
        f"- Rows: **{len(rows)}**",
        f"- Verified parameters: **{record['verified_parameter_total']:,}** "
        f"of {record['parameter_cap']:,}",
        "",
        "| relation | macro-P | macro-R | macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(per_relation):
        if name.startswith("***"):
            continue
        value = per_relation[name]
        lines.append(
            f"| {name} | {value['macro-p']:.4f} | {value['macro-r']:.4f} "
            f"| {value['macro-f1']:.4f} |")
    lines.extend([
        f"| **pooled** | {pooled['macro-p']:.4f} | {pooled['macro-r']:.4f} "
        f"| **{pooled['macro-f1']:.6f}** |",
        "",
        "## Layers not carried by this run",
        "",
    ])
    for item in record["out_of_scope_layers"]:
        lines.append(
            f"- `{item['layer']}` ({item['deployed_rows']} deployed rows): "
            f"{item['reason']}")
    (output / "analysis").mkdir(parents=True, exist_ok=True)
    (output / "analysis/RESULT.md").write_text("\n".join(lines) + "\n")
    _write_json(output / "analysis/RESULT.json", {
        "schema": "end-to-end-result-v1",
        "contains_labels": True,
        "gold_aware": True,
        "split": record["split"],
        "pooled_macro_f1": pooled["macro-f1"],
        "per_relation": per_relation,
        "rows": len(rows),
        "verified_parameter_total": record["verified_parameter_total"],
        "out_of_scope_layers": record["out_of_scope_layers"],
    })
    print(json.dumps({
        "pooled_macro_f1": pooled["macro-f1"],
        "rows": len(rows),
        "result": str(output / "analysis/RESULT.md"),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--split", default="validation",
                             choices=["train", "validation", "test"])
    plan_parser.add_argument("--input", default=str(ROOT / "data/val.jsonl"))
    plan_parser.add_argument("--cot-agents", default=str(COT_AGENTS))
    plan_parser.add_argument("--supply-agents", default=str(SUPPLY_AGENTS))
    plan_parser.add_argument("--synthetic-cot", default=str(DEFAULT_SYNTHETIC))
    plan_parser.add_argument(
        "--question-contract", default="legacy",
        choices=["legacy", "official-v1"],
        help=(
            "Versioned question wording. Use official-v1 with the capacity "
            "maximum-aligned SyntheticCoT pool for new experiments."
        ),
    )
    plan_parser.add_argument(
        "--component-models", default=str(COMPONENT_MODELS))
    plan_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    plan_parser.set_defaults(function=plan)

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    assemble_parser.set_defaults(function=assemble)

    graph_parser = subparsers.add_parser("graph")
    graph_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    graph_parser.set_defaults(function=graph)

    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    decode_parser.set_defaults(function=decode)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    score_parser.set_defaults(function=score)
    return value


def main() -> int:
    arguments = parser().parse_args()
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
