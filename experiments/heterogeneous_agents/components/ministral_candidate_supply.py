#!/usr/bin/env python3
"""Independent Ministral candidate-supply bakeoff.

``prepare`` creates target-label-free proposal and commitment tasks for every
training row.  It never opens train or validation labels.  ``analyze`` first
validates the immutable response manifest, then opens train labels solely to
measure standalone accuracy, unique correct supply, complementarity, and the
candidate-union oracle delta.  No selector is trained and validation is never
opened by this experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate import RELATION_TYPE, true_positives
from experiments.heterogeneous_agents.assemble_and_audit import (
    _gold_aliases,
    assemble_graphs,
    load_responses,
    oracle_rows,
    portfolio_diagnostics,
    prediction_rows,
    score,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    build_agent_tasks,
    canonical_key,
    load_agent_config,
    load_synthetic_by_relation,
    read_jsonl,
    sha256,
    validate_inputs,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.run_agent import validate_tasks


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
DEFAULT_OUTPUT = RUNS / "ministral_candidate_supply_20260728_v1"
DEFAULT_CONFIG = ROOT / "configs/final/portfolio_supply.json"
DEFAULT_INPUT = ROOT / "data/train.jsonl"
DEFAULT_GOLD = ROOT / "data/train.jsonl"
DEFAULT_SYNTHETIC = ROOT / "data/synthetic_cot_faithful.jsonl"
DEFAULT_SOURCE_GRAPH = (
    RUNS / "targeted_company_gemma_n3_20260724_v1/graphs/train_graph.jsonl"
)
MINISTRAL = "ministral_independent"
QWEN = "qwen_recall"
GEMMA = "gemma_independent"
EXPECTED_MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
EXPECTED_REVISION = "f6fae9795746f63c9be8344932f01275f3c63734"
SEED = 20260728
N_PROPOSALS = 1
MIN_ORACLE_DELTA = 0.015
MIN_UNIQUE_ROWS = 8
MIN_UNIQUE_RELATIONS = 4
MAX_PARSE_FAILURE_RATE = 0.10


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _agent(config: Mapping[str, Any], agent_id: str) -> dict[str, Any]:
    matches = [
        dict(agent) for agent in config["agents"] if agent["id"] == agent_id
    ]
    if len(matches) != 1:
        raise ContractError(f"expected exactly one {agent_id!r} agent")
    return matches[0]


def _response_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ContractError(f"missing response or manifest: {path}")
    manifest = _json(manifest_path)
    if manifest.get("output_sha256") != sha256(path):
        raise ContractError(f"response hash mismatch: {path}")
    return manifest


def _source_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = _json(manifest_path)
    if (
        manifest.get("contains_labels") is not False
        or manifest.get("gold_aware") is not False
        or manifest.get("split") != "train"
        or manifest.get("output_sha256") != sha256(path)
    ):
        raise ContractError("source graph is not a valid label-free train graph")
    return manifest


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    config_path = Path(args.agents).resolve()
    input_path = Path(args.input).resolve()
    synthetic_path = Path(args.synthetic_cot).resolve()
    source_graph = Path(args.source_graph).resolve()
    config = load_agent_config(config_path)
    agent = _agent(config, MINISTRAL)
    if (
        agent["model"] != EXPECTED_MODEL
        or agent.get("revision") != EXPECTED_REVISION
    ):
        raise ContractError("Ministral config is not the audited pinned checkpoint")
    _source_manifest(source_graph)

    labeled_rows = read_jsonl(input_path)
    validate_inputs(labeled_rows)
    rows = [
        {
            "SubjectEntity": str(row["SubjectEntity"]),
            "Relation": str(row["Relation"]),
        }
        for row in labeled_rows
    ]
    if len(rows) != 477 or len({_key(row) for row in rows}) != len(rows):
        raise ContractError(
            f"expected 477 unique train rows, found {len(rows)}")
    source_rows = read_jsonl(source_graph)
    if {_key(row) for row in source_rows} != {_key(row) for row in rows}:
        raise ContractError("source graph does not cover the frozen train rows")

    synthetic = load_synthetic_by_relation(synthetic_path)
    tasks = build_agent_tasks(
        rows, agent, synthetic, seed=SEED, n_proposals=N_PROPOSALS)
    # Ministral is deliberately an independent zero-shot route.  The generic
    # builder already excludes target labels; this assertion prevents a future
    # config edit from silently adding demonstrations.
    proposal_tasks = [task for task in tasks if task["phase"] == "propose"]
    if (
        len(proposal_tasks) != len(rows)
        or any(task.get("shot_subjects") for task in proposal_tasks)
    ):
        raise ContractError("Ministral supply route must remain zero-shot N=1")
    validate_tasks(tasks, MINISTRAL)

    plan_dir = output / "plan"
    row_path = plan_dir / "INPUT_ROWS.jsonl"
    task_path = plan_dir / f"tasks/{MINISTRAL}.jsonl"
    smoke_path = plan_dir / f"smoke/{MINISTRAL}.jsonl"
    response_path = output / f"responses/{MINISTRAL}.jsonl"
    write_jsonl_atomic(row_path, rows)
    write_jsonl_atomic(task_path, tasks)

    smoke_keys: set[tuple[str, str]] = set()
    seen_relations: set[str] = set()
    for row in rows:
        if row["Relation"] not in seen_relations:
            smoke_keys.add(_key(row))
            seen_relations.add(row["Relation"])
    smoke = [task for task in tasks if
             (str(task["subject"]), str(task["relation"])) in smoke_keys]
    if len(smoke_keys) != 6 or len(smoke) != 18:
        raise ContractError("expected one complete three-task smoke row per relation")
    write_jsonl_atomic(smoke_path, smoke)

    plan = {
        "schema": "ministral-candidate-supply-plan-v1",
        "contains_labels": False,
        "gold_aware": False,
        "validation_opened": False,
        "validation_labels_used": False,
        "rows": len(rows),
        "n_proposals": N_PROPOSALS,
        "seed": SEED,
        "input_rows": str(row_path),
        "input_rows_sha256": sha256(row_path),
        "source_graph": str(source_graph),
        "source_graph_sha256": sha256(source_graph),
        "source_graph_manifest_sha256": sha256(
            source_graph.with_suffix(source_graph.suffix + ".manifest.json")),
        "synthetic_cot": str(synthetic_path),
        "synthetic_cot_sha256": sha256(synthetic_path),
        "agents": str(config_path),
        "agents_sha256": sha256(config_path),
        "model": agent["model"],
        "revision": agent["revision"],
        "task_path": str(task_path),
        "task_sha256": sha256(task_path),
        "task_count": len(tasks),
        "smoke_path": str(smoke_path),
        "smoke_sha256": sha256(smoke_path),
        "response_path": str(response_path),
        "declared_parameter_total": config["declared_parameter_total"],
        "verified_parameter_total": config["verified_parameter_total"],
        "parameter_cap": config["parameter_cap"],
        "parameter_headroom": config["declared_parameter_headroom"],
        "promotion_gate": {
            "candidate_union_oracle_delta_minimum": MIN_ORACLE_DELTA,
            "unique_correct_rows_minimum": MIN_UNIQUE_ROWS,
            "unique_correct_relations_minimum": MIN_UNIQUE_RELATIONS,
            "parse_failure_rate_maximum": MAX_PARSE_FAILURE_RATE,
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    plan_path = plan_dir / "PLAN.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "plan": str(plan_path),
        "rows": len(rows),
        "tasks": len(tasks),
        "proposal_tasks": len(proposal_tasks),
        "smoke_tasks": len(smoke),
        "verified_parameter_total": config["verified_parameter_total"],
        "parameter_headroom": config["declared_parameter_headroom"],
    }, indent=2, sort_keys=True))
    return 0


def _items(
    graph: Mapping[str, Any], agent_id: str | None = None,
) -> list[str]:
    result = []
    for node in graph.get("candidates", []):
        if agent_id is not None:
            sources = set(node.get("sources", {}))
            proposers = set(node.get("proposer_agents", []))
            if agent_id not in sources | proposers:
                continue
        result.append(str(node["item"]))
    return result


def _has_correct(
    items: Sequence[str], gold: Mapping[str, Any], relation: str,
) -> bool:
    aliases = _gold_aliases(gold)
    return bool(
        aliases
        and true_positives(
            list(items),
            aliases,
            RELATION_TYPE.get(relation, "string"),
            0.05,
        ) > 0
    )


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ContractError("invalid correlation vectors")
    mean_left, mean_right = statistics.mean(left), statistics.mean(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right)
        for x, y in zip(left, right)
    )
    left_scale = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    return (
        numerator / (left_scale * right_scale)
        if left_scale and right_scale else None
    )


def _combined_graph(
    base: Mapping[str, Any], ministral: Mapping[str, Any],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, str]] = {}
    relation = str(base["Relation"])
    for item in _items(base) + _items(ministral):
        candidates.setdefault(
            canonical_key(item, relation), {"item": item})
    return {
        "SubjectEntity": str(base["SubjectEntity"]),
        "Relation": relation,
        "candidates": list(candidates.values()),
    }


def _parse_failure_rate(graphs: Sequence[Mapping[str, Any]]) -> tuple[int, int, float]:
    counts = Counter()
    for graph in graphs:
        counts.update(
            graph.get("proposal_parse_diagnostics", {}).get(MINISTRAL, {}))
    total = sum(counts.values())
    failures = total - counts.get("parsed_nonempty", 0) - counts.get(
        "explicit_none", 0)
    return failures, total, failures / total if total else 1.0


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_path = output / "plan/PLAN.json"
    plan = _json(plan_path)
    if (
        plan.get("schema") != "ministral-candidate-supply-plan-v1"
        or plan.get("contains_labels") is not False
        or plan.get("gold_aware") is not False
        or plan.get("validation_opened") is not False
        or plan.get("validation_labels_used") is not False
    ):
        raise ContractError("invalid or label-contaminated Ministral plan")
    for field in ("input_rows", "source_graph", "synthetic_cot", "agents"):
        path = Path(plan[field])
        if sha256(path) != plan[f"{field}_sha256"]:
            raise ContractError(f"frozen plan artifact changed: {field}")
    source_graph_path = Path(plan["source_graph"])
    _source_manifest(source_graph_path)

    task_path = Path(plan["task_path"])
    response_path = Path(plan["response_path"])
    manifest = _response_manifest(response_path)
    if (
        sha256(task_path) != plan["task_sha256"]
        or manifest.get("task_sha256") != plan["task_sha256"]
        or manifest.get("agent_id") != MINISTRAL
        or manifest.get("model") != EXPECTED_MODEL
        or manifest.get("revision") != EXPECTED_REVISION
    ):
        raise ContractError("stale or foreign Ministral responses")
    tasks = read_jsonl(task_path)
    validate_tasks(tasks, MINISTRAL)

    config = load_agent_config(Path(plan["agents"]))
    agent = _agent(config, MINISTRAL)
    rows = read_jsonl(Path(plan["input_rows"]))
    responses = load_responses(
        response_path.parent, [agent])
    ministral_graphs = assemble_graphs(
        rows, [agent], responses)
    base_graphs = read_jsonl(source_graph_path)
    base_by = {_key(row): row for row in base_graphs}
    ministral_by = {_key(row): row for row in ministral_graphs}
    expected = {_key(row) for row in rows}
    if set(base_by) != expected or set(ministral_by) != expected:
        raise ContractError("candidate graph coverage mismatch")

    gold_path = Path(args.train_gold).resolve()
    gold_by = {_key(row): row for row in read_jsonl(gold_path)}
    if set(gold_by) != expected:
        raise ContractError("train label coverage mismatch")
    ordered_gold = [gold_by[_key(row)] for row in rows]
    ordered_base = [base_by[_key(row)] for row in rows]
    ordered_ministral = [ministral_by[_key(row)] for row in rows]
    combined = [
        _combined_graph(base_by[_key(row)], ministral_by[_key(row)])
        for row in rows
    ]

    standalone_predictions = prediction_rows(
        ordered_ministral, f"agent:{MINISTRAL}", [MINISTRAL])
    standalone_scores = score(standalone_predictions, ordered_gold)
    base_oracle_scores = score(
        oracle_rows(ordered_base, ordered_gold), ordered_gold)
    combined_oracle_scores = score(
        oracle_rows(combined, ordered_gold), ordered_gold)
    oracle_deltas = {
        relation: combined_oracle_scores[relation] - base_oracle_scores[relation]
        for relation in base_oracle_scores
    }

    vectors = {agent_id: [] for agent_id in (QWEN, GEMMA, MINISTRAL)}
    unique_rows, unique_by_relation = [], Counter()
    new_candidate_rows = 0
    corroboration_rows = 0
    row_audit = []
    for row in rows:
        key = _key(row)
        relation = key[1]
        base = base_by[key]
        third = ministral_by[key]
        gold = gold_by[key]
        base_items = _items(base)
        third_items = _items(third)
        qwen_items = _items(base, QWEN)
        gemma_items = _items(base, GEMMA)
        correctness = {
            QWEN: _has_correct(qwen_items, gold, relation),
            GEMMA: _has_correct(gemma_items, gold, relation),
            MINISTRAL: _has_correct(third_items, gold, relation),
        }
        for agent_id, value in correctness.items():
            vectors[agent_id].append(float(value))
        base_correct = _has_correct(base_items, gold, relation)
        unique = correctness[MINISTRAL] and not base_correct
        if unique:
            unique_rows.append([key[0], key[1]])
            unique_by_relation[relation] += 1
        base_keys = {canonical_key(item, relation) for item in base_items}
        third_keys = {canonical_key(item, relation) for item in third_items}
        if third_keys - base_keys:
            new_candidate_rows += 1
        if third_keys & base_keys:
            corroboration_rows += 1
        row_audit.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ministral_candidate_count": len(third_keys),
            "new_candidate_count": len(third_keys - base_keys),
            "corroborated_candidate_count": len(third_keys & base_keys),
            "base_has_correct_candidate": base_correct,
            "ministral_has_correct_candidate": correctness[MINISTRAL],
            "ministral_unique_correct": unique,
        })

    correlations = {}
    agent_ids = (QWEN, GEMMA, MINISTRAL)
    for left_index, left in enumerate(agent_ids):
        for right in agent_ids[left_index + 1:]:
            correlations[f"{left}__{right}"] = _correlation(
                vectors[left], vectors[right])
    failures, parsed, failure_rate = _parse_failure_rate(ordered_ministral)
    gate_checks = {
        "candidate_union_oracle_delta": (
            oracle_deltas["*** All Relations ***"] >= MIN_ORACLE_DELTA),
        "unique_correct_rows": len(unique_rows) >= MIN_UNIQUE_ROWS,
        "unique_correct_relations": (
            len(unique_by_relation) >= MIN_UNIQUE_RELATIONS),
        "parse_failure_rate": failure_rate <= MAX_PARSE_FAILURE_RATE,
    }

    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    row_audit_path = analysis / "ROW_SUPPLY_AUDIT.jsonl"
    write_jsonl_atomic(row_audit_path, row_audit)
    result = {
        "schema": "ministral-candidate-supply-result-v1",
        "development_only": True,
        "deployable": False,
        "contains_labels": True,
        "gold_aware": True,
        "validation_opened": False,
        "validation_labels_used": False,
        "model": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "verified_parameter_total": config["verified_parameter_total"],
        "parameter_headroom": config["declared_parameter_headroom"],
        "standalone_scores": standalone_scores,
        "base_candidate_union_oracle_scores": base_oracle_scores,
        "three_model_candidate_union_oracle_scores": combined_oracle_scores,
        "candidate_union_oracle_deltas": oracle_deltas,
        "unique_correct_rows": unique_rows,
        "unique_correct_rows_count": len(unique_rows),
        "unique_correct_by_relation": dict(unique_by_relation),
        "correct_candidate_rows": {
            agent_id: int(sum(vectors[agent_id]))
            for agent_id in agent_ids
        },
        "candidate_correctness_correlations": correlations,
        "rows_with_new_ministral_candidates": new_candidate_rows,
        "rows_with_ministral_corroboration": corroboration_rows,
        "parse": {
            "failures": failures,
            "samples": parsed,
            "failure_rate": failure_rate,
        },
        "ministral_diagnostics": portfolio_diagnostics(
            ordered_ministral, ordered_gold, [MINISTRAL]),
        "promotion_gate_checks": gate_checks,
        "promotion_gate_passed": all(gate_checks.values()),
        "next_stage": (
            "freeze_and_test_consensus_only_integration"
            if all(gate_checks.values())
            else "reject_ministral_candidate_supply"
        ),
        "row_audit": str(row_audit_path),
        "row_audit_sha256": sha256(row_audit_path),
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "train_gold": str(gold_path),
        "train_gold_sha256": sha256(gold_path),
    }
    result_path = analysis / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    relations = sorted(standalone_scores)
    lines = [
        "# Ministral independent candidate-supply bakeoff",
        "",
        "Development-only train audit. Validation was not opened.",
        "",
        f"- Model: `{EXPECTED_MODEL}`",
        f"- Legal portfolio: **{config['verified_parameter_total']:,} / "
        f"{config['parameter_cap']:,}**",
        f"- Standalone F1: "
        f"**{standalone_scores['*** All Relations ***']:.6f}**",
        f"- Candidate-union oracle delta: "
        f"**{oracle_deltas['*** All Relations ***']:+.6f}**",
        f"- Unique correct rows beyond the Qwen/Gemma graph: "
        f"**{len(unique_rows)}**",
        f"- Unique-correct relation coverage: "
        f"**{len(unique_by_relation)}/6**",
        f"- Parse failure rate: **{failure_rate:.2%}**",
        f"- Promotion gate: **{'PASS' if all(gate_checks.values()) else 'FAIL'}**",
        "",
        "| relation | Ministral standalone | two-model oracle | "
        "three-model oracle | oracle delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for relation in relations:
        lines.append(
            f"| {relation} | {standalone_scores[relation]:.4f} | "
            f"{base_oracle_scores[relation]:.4f} | "
            f"{combined_oracle_scores[relation]:.4f} | "
            f"{oracle_deltas[relation]:+.4f} |")
    lines.extend([
        "",
        "The oracle is gold-aware and nondeployable. It measures candidate "
        "supply only; no selector was trained or changed.",
        "",
        "## Promotion checks",
        "",
    ])
    lines.extend(
        f"- `{name}`: **{'PASS' if passed else 'FAIL'}**"
        for name, passed in gate_checks.items()
    )
    (analysis / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "standalone_f1": standalone_scores["*** All Relations ***"],
        "candidate_union_oracle_delta":
            oracle_deltas["*** All Relations ***"],
        "unique_correct_rows": len(unique_rows),
        "unique_correct_relations": len(unique_by_relation),
        "parse_failure_rate": failure_rate,
        "promotion_gate_passed": all(gate_checks.values()),
        "result": str(analysis / "RESULT.md"),
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    prep.add_argument("--agents", default=str(DEFAULT_CONFIG))
    prep.add_argument("--input", default=str(DEFAULT_INPUT))
    prep.add_argument("--synthetic-cot", default=str(DEFAULT_SYNTHETIC))
    prep.add_argument("--source-graph", default=str(DEFAULT_SOURCE_GRAPH))
    prep.set_defaults(function=prepare)
    audit = subparsers.add_parser("analyze")
    audit.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    audit.add_argument("--train-gold", default=str(DEFAULT_GOLD))
    audit.set_defaults(function=analyze)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
