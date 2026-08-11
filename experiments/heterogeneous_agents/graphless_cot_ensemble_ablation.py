#!/usr/bin/env python3
"""Evaluate direct-output CoT ensembles without constructing a graph.

The analysis consumes completed response caches from the current end-to-end
validation plan.  It decodes only each route's ``propose`` generations and
combines the resulting model-level answers with predeclared source-vote rules.
It does not use commitments, graph components, routers, or the final decoder.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate as official

from experiments.heterogeneous_agents import end_to_end_pipeline as e2e
from experiments.heterogeneous_agents.components.dual_model_validation import (
    proposal_only_prediction,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    NUMERIC_RELATIONS,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)


POOLED = "*** All Relations ***"
SCHEMA = "graphless-cot-ensemble-ablation-v1"
ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_graphless_cot_ensemble_ablation.sh")
PROPOSAL_DECODER = (
    Path(__file__).with_name("components") / "dual_model_validation.py"
)
ROUTES = {
    "qwen": ("qwen:self_consistency", 10, 5),
    "gemma": ("gemma:independent", 1, 5),
    "ministral": (e2e.MINISTRAL_COT40, 10, 5),
}
ARMS = {
    "qwen_gemma_union": (("qwen", "gemma"), 1),
    "qwen_gemma_agreement": (("qwen", "gemma"), 2),
    "qwen_gemma_ministral_union": (("qwen", "gemma", "ministral"), 1),
    "qwen_gemma_ministral_majority": (
        ("qwen", "gemma", "ministral"), 2),
    "qwen_gemma_ministral_unanimous": (
        ("qwen", "gemma", "ministral"), 3),
}


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def _response_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["subject"]), str(row["relation"])


def _number(objects: Sequence[str]) -> float | None:
    if len(objects) != 1:
        return None
    try:
        value = float(str(objects[0]).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def fuse_source_outputs(
    source_outputs: Sequence[Sequence[str]],
    relation: str,
    *,
    minimum_sources: int,
) -> list[str]:
    """Fuse already-decoded model outputs with no graph representation.

    Strings are retained when they occur in at least ``minimum_sources``
    distinct model outputs after canonicalization.  Numeric relations follow
    the repository's existing graph-free control: if enough models emitted one
    valid scalar, return their plain median.  Numeric values are not clustered
    or required to agree within the official tolerance.
    """
    if minimum_sources < 1 or minimum_sources > len(source_outputs):
        raise ValueError("minimum_sources must be within the source count")
    if relation in NUMERIC_RELATIONS:
        values = [
            value for objects in source_outputs
            if (value := _number(objects)) is not None
        ]
        if len(values) < minimum_sources:
            return []
        return [format(statistics.median(values), ".12g")]

    counts: Counter[str] = Counter()
    displays: dict[str, str] = {}
    for objects in source_outputs:
        seen: set[str] = set()
        for raw_item in objects:
            item = str(raw_item)
            key = canonical_key(item, relation)
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            displays.setdefault(key, item)
    return [
        item for key, item in displays.items()
        if counts[key] >= minimum_sources
    ]


def _route_predictions(
    plan: Mapping[str, Any],
    source: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    expected_keys = [_key(row) for row in source]
    expected_key_set = set(expected_keys)
    predictions: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for source_name, (route, expected_n, expected_shots) in ROUTES.items():
        job = plan["jobs"][route]
        if int(job["n_proposals"]) != expected_n:
            raise ContractError(f"{route}: planned sample count drift")
        if int(job["synthetic_shots"]) != expected_shots:
            raise ContractError(f"{route}: synthetic-shot count drift")
        proposal_rows = [
            row for row in e2e._validated_responses(job)
            if row.get("phase") == "propose"
        ]
        response_map = {_response_key(row): row for row in proposal_rows}
        if len(response_map) != len(proposal_rows):
            raise ContractError(f"{route}: duplicate proposal response key")
        if set(response_map) != expected_key_set:
            raise ContractError(f"{route}: proposal response coverage mismatch")
        if any(
            len(row.get("generations", [])) != expected_n
            for row in proposal_rows
        ):
            raise ContractError(f"{route}: proposal generation count mismatch")
        predictions[source_name] = [
            {
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": proposal_only_prediction(response_map[key]),
            }
            for key in expected_keys
        ]
        response_path = Path(job["response_path"])
        provenance[source_name] = {
            "route": route,
            "model": job["model"],
            "revision": job["revision"],
            "samples": expected_n,
            "synthetic_shots": expected_shots,
            "reasoning_words": job["reasoning_words"],
            "response_path": str(response_path),
            "response_sha256": sha256(response_path),
        }
    return predictions, provenance


def _score(
    predictions: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return official.macro_average_per_relation(
        official.evaluate_per_sr_pair(
            predictions, gold, official.RELATION_TYPE))


def analyze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan = e2e._validate_plan(output)
    if plan.get("blind") or plan.get("split") != "validation":
        raise ContractError("graphless ensemble ablations require validation")
    gold_path = Path(args.gold).resolve()
    if Path(plan["input"]).resolve() != gold_path:
        raise ContractError("gold path must be the validation plan input")

    source = read_jsonl(Path(plan["input_rows"]))
    gold = official.read_jsonl_file(str(gold_path))
    expected_keys = [_key(row) for row in source]
    if expected_keys != [_key(row) for row in gold]:
        raise ContractError("validation source/gold row order mismatch")

    source_predictions, provenance = _route_predictions(plan, source)
    analysis_dir = output / "analysis/graphless_cot_ensembles"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    policies: dict[str, Any] = {}

    for source_name, rows in source_predictions.items():
        path = analysis_dir / f"source_{source_name}.jsonl"
        write_jsonl_atomic(path, rows)
        scores = _score(rows, gold)
        policies[f"source_{source_name}"] = {
            "kind": "proposal_only_source_control",
            "sources": [source_name],
            "minimum_sources": 1,
            "rows": len(rows),
            "pooled_macro_f1": scores[POOLED]["macro-f1"],
            "scores": scores,
            "predictions": str(path),
            "predictions_sha256": sha256(path),
        }

    indexed = {
        name: {_key(row): row for row in rows}
        for name, rows in source_predictions.items()
    }
    for arm_name, (source_names, minimum_sources) in ARMS.items():
        predictions = []
        for key in expected_keys:
            predictions.append({
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": fuse_source_outputs(
                    [
                        indexed[name][key].get("ObjectEntities", [])
                        for name in source_names
                    ],
                    key[1],
                    minimum_sources=minimum_sources,
                ),
            })
        path = analysis_dir / f"{arm_name}.jsonl"
        write_jsonl_atomic(path, predictions)
        scores = _score(predictions, gold)
        policies[arm_name] = {
            "kind": "direct_output_source_vote",
            "sources": list(source_names),
            "minimum_sources": minimum_sources,
            "rows": len(predictions),
            "pooled_macro_f1": scores[POOLED]["macro-f1"],
            "scores": scores,
            "predictions": str(path),
            "predictions_sha256": sha256(path),
        }

    result = {
        "schema": SCHEMA,
        "development_only": True,
        "split": "validation",
        "contains_labels": True,
        "gold_used_in_decoding": False,
        "graph_constructed": False,
        "commitments_used": False,
        "router_used": False,
        "selection_warning": (
            "All predeclared graph-free policies are reported. Do not select "
            "the best validation policy and describe it as test-independent."
        ),
        "numeric_policy_note": (
            "Numeric arms return a plain median when the required number of "
            "models emit valid scalars; values are not clustered or checked "
            "for tolerance agreement."
        ),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "runner": str(RUNNER.resolve()),
        "runner_sha256": sha256(RUNNER.resolve()),
        "proposal_decoder_implementation": str(PROPOSAL_DECODER.resolve()),
        "proposal_decoder_implementation_sha256": sha256(
            PROPOSAL_DECODER.resolve()),
        "requirements_lock": str((ROOT / "requirements-lock.txt").resolve()),
        "requirements_lock_sha256": sha256(ROOT / "requirements-lock.txt"),
        "plan": str(output / "plan/PLAN.json"),
        "plan_sha256": sha256(output / "plan/PLAN.json"),
        "gold": str(gold_path),
        "gold_sha256": sha256(gold_path),
        "routes": provenance,
        "policies": policies,
    }
    result_path = analysis_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    relation_order = [
        POOLED,
        "awardWonBy",
        "companyTradesAtStockExchange",
        "countryLandBordersCountry",
        "hasArea",
        "hasCapacity",
        "personHasCityOfDeath",
    ]
    lines = [
        "# Graph-free CoT ensemble ablation",
        "",
        "This is a development result on the current 475-row validation split. "
        "It reuses the completed CoT-5 response caches and never reads or "
        "scores the blind test.",
        "",
        "Only proposal generations are decoded. No commitment, graph "
        "component, router, or final graph decoder is used.",
        "",
        "| policy | " + " | ".join(relation_order) + " |",
        "|---|" + "|".join(["---:"] * len(relation_order)) + "|",
    ]
    for name, record in policies.items():
        lines.append(
            "| " + name + " | " + " | ".join(
                f"{record['scores'][relation]['macro-f1']:.6f}"
                for relation in relation_order
            ) + " |"
        )
    lines.extend([
        "",
        "For string relations, the threshold counts distinct source models "
        "after canonicalization. For numeric relations, the scorer returns "
        "the plain median when enough models emitted one valid scalar; it "
        "does not cluster values or require tolerance agreement.",
        "",
        "All predeclared policies are reported. Validation scores must not be "
        "used to present one policy as test-independent.",
        "",
    ])
    (analysis_dir / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps({
        "result": str(result_path),
        "graph_constructed": False,
        "pooled_macro_f1": {
            name: record["pooled_macro_f1"]
            for name, record in policies.items()
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
