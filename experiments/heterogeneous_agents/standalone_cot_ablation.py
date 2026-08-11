#!/usr/bin/env python3
"""Score model-only CoT routes from a completed validation generation run.

This analysis consumes only immutable response caches produced by
``run_end_to_end_pipeline.sh``.  It never generates text, changes the frozen
final decoder, or scores a blind/test plan.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import evaluate as official
from sample_evidence import classify_samples

from experiments.heterogeneous_agents import end_to_end_pipeline as e2e
from experiments.heterogeneous_agents.assemble_and_audit import (
    assemble_graphs,
    prediction_for_agent,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    load_agent_config,
    proposal_parse_status,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)


POOLED = "*** All Relations ***"
SCHEMA = "standalone-cot-ablation-v1"
NUMERIC_RELATIONS = frozenset(("hasArea", "hasCapacity"))


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def _response_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["subject"]), str(row["relation"])


def _thresholds(n: int) -> dict[str, int]:
    return {
        "any": 1,
        "majority": n // 2 + 1,
        "two_thirds": math.ceil(2 * n / 3),
        "unanimous": n,
    }


def _qwen_items(text: str, relation: str) -> Sequence[str]:
    return classify_samples([text], relation, "legacy-cot")[0].items


def _generic_items(text: str, relation: str) -> Sequence[str]:
    return proposal_parse_status(text, relation)[1]


def _supported_objects(
    response: Mapping[str, Any],
    *,
    parser: Callable[[str, str], Sequence[str]],
    threshold: int,
) -> list[str]:
    relation = str(response["relation"])
    counts: Counter[str] = Counter()
    displays: dict[str, str] = {}
    for generation in response.get("generations", []):
        seen: set[str] = set()
        for raw_item in parser(str(generation), relation):
            item = str(raw_item)
            canonical = canonical_key(item, relation)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            counts[canonical] += 1
            displays.setdefault(canonical, item)
    accepted = {
        canonical: displays[canonical]
        for canonical, support in counts.items()
        if support >= threshold
    }
    if relation not in NUMERIC_RELATIONS:
        return list(accepted.values())
    if not accepted:
        return []
    highest = max(counts[canonical] for canonical in accepted)
    winners = [
        accepted[canonical] for canonical in accepted
        if counts[canonical] == highest
    ]
    return winners if len(winners) == 1 else []


def _score(
    predictions: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return official.macro_average_per_relation(
        official.evaluate_per_sr_pair(
            predictions, gold, official.RELATION_TYPE))


def _route_native_base_predictions(
    plan: Mapping[str, Any],
    source: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Use the repository's established Qwen/Gemma standalone aggregator."""
    config = load_agent_config(Path(plan["cot_agents"]))
    agents = [
        e2e._agent_of(config, e2e.QWEN),
        e2e._agent_of(config, e2e.GEMMA),
    ]
    responses: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for route in ("qwen:self_consistency", "gemma:independent"):
        job = plan["jobs"][route]
        agent_id = str(job["agent_id"])
        by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for response in e2e._validated_responses(job):
            key = (
                str(response["subject"]),
                str(response["relation"]),
                str(response["phase"]),
            )
            if key in by_key:
                raise ContractError(f"{route}: duplicate response key {key}")
            by_key[key] = response
        responses[agent_id] = by_key
    graphs = assemble_graphs(source, agents, responses)
    return {
        agent_id: [
            {
                "SubjectEntity": str(graph["SubjectEntity"]),
                "Relation": str(graph["Relation"]),
                "ObjectEntities": prediction_for_agent(graph, agent_id),
            }
            for graph in graphs
        ]
        for agent_id in (e2e.QWEN, e2e.GEMMA)
    }


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = e2e._validate_plan(output)
    if plan.get("blind") or plan.get("split") != "validation":
        raise ContractError("standalone ablations require a validation plan")
    if Path(plan["input"]).resolve() != Path(args.gold).resolve():
        raise ContractError("gold path must be the validation plan input")

    source = read_jsonl(Path(plan["input_rows"]))
    gold = official.read_jsonl_file(str(Path(args.gold).resolve()))
    if [_key(row) for row in source] != [_key(row) for row in gold]:
        raise ContractError("validation source/gold row order mismatch")
    route_native = _route_native_base_predictions(plan, source)

    arm_specs = (
        ("qwen_cot5_n10", "qwen:self_consistency", 10, _qwen_items, None),
        ("gemma_cot5_n1", "gemma:independent", 1, _generic_items, None),
        ("ministral_cot5_n10", e2e.MINISTRAL_COT40, 10, _generic_items, None),
        ("ministral_zeroshot_n3_area", e2e.MINISTRAL_N3, 3,
         _generic_items, "hasArea"),
    )
    analysis_dir = output / "analysis/standalone_ablation"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    arms: dict[str, Any] = {}
    for arm_name, route, n, parser, relation_filter in arm_specs:
        job = plan["jobs"][route]
        if int(job["n_proposals"]) != n:
            raise ContractError(f"{route}: planned sample count drift")
        rows = e2e._validated_responses(job)
        proposal_rows = [row for row in rows if row.get("phase") == "propose"]
        responses = {
            _response_key(row): row
            for row in proposal_rows
        }
        if (
            len(responses) != len(proposal_rows)
            or set(responses) != {_key(row) for row in source}
        ):
            raise ContractError(f"{route}: proposal response coverage mismatch")
        if any(len(row.get("generations", [])) != n for row in proposal_rows):
            raise ContractError(f"{route}: proposal generation count mismatch")
        policy_results: dict[str, Any] = {}
        for policy_name, threshold in _thresholds(n).items():
            predictions = []
            selected_gold = []
            for source_row, gold_row in zip(source, gold, strict=True):
                key = _key(source_row)
                if relation_filter and key[1] != relation_filter:
                    continue
                predictions.append({
                    "SubjectEntity": key[0],
                    "Relation": key[1],
                    "ObjectEntities": _supported_objects(
                        responses[key], parser=parser, threshold=threshold),
                })
                selected_gold.append(gold_row)
            prediction_path = analysis_dir / f"{arm_name}.{policy_name}.jsonl"
            write_jsonl_atomic(prediction_path, predictions)
            scores = _score(predictions, selected_gold)
            policy_results[policy_name] = {
                "kind": "exact_surface_support_threshold_diagnostic",
                "threshold": threshold,
                "rows": len(predictions),
                "pooled_macro_f1": scores[POOLED]["macro-f1"],
                "scores": scores,
                "predictions": str(prediction_path),
                "predictions_sha256": sha256(prediction_path),
            }
        agent_id = str(plan["jobs"][route]["agent_id"])
        if agent_id in route_native:
            predictions = route_native[agent_id]
            prediction_path = analysis_dir / f"{arm_name}.route_native.jsonl"
            write_jsonl_atomic(prediction_path, predictions)
            scores = _score(predictions, gold)
            policy_results["route_native"] = {
                "kind": "repository_standalone_aggregator",
                "threshold": None,
                "rows": len(predictions),
                "pooled_macro_f1": scores[POOLED]["macro-f1"],
                "scores": scores,
                "predictions": str(prediction_path),
                "predictions_sha256": sha256(prediction_path),
            }
        arms[arm_name] = {
            "route": route,
            "samples": n,
            "synthetic_shots": plan["jobs"][route]["synthetic_shots"],
            "relation_filter": relation_filter,
            "policies": policy_results,
        }

    result = {
        "schema": SCHEMA,
        "development_only": True,
        "split": "validation",
        "contains_labels": True,
        "gold_aware_analysis_only": True,
        "selection_warning": (
            "Report all predeclared thresholds; do not select a threshold "
            "from validation and present it as test-independent."
        ),
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "gold": str(Path(args.gold).resolve()),
        "gold_sha256": sha256(Path(args.gold).resolve()),
        "arms": arms,
    }
    result_path = analysis_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "result": str(result_path),
        "arms": {
            name: {
                policy: values["pooled_macro_f1"]
                for policy, values in arm["policies"].items()
            }
            for name, arm in arms.items()
        },
    }, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--gold", default="data/val.jsonl")
    return value


def main() -> int:
    return analyze(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
