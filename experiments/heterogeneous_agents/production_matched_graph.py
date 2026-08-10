#!/usr/bin/env python3
"""Rebuild training graphs from the exact frozen production evidence routes.

The first expanded-calibration graph used old 4-bit Qwen fold artifacts and
could not apply System-2 corroboration on training rows.  Validation, however,
used the frozen v0495 production stack: relation-routed System-1 precision,
recovery, and System-2 city/company corroboration.  This script closes that
train/validation evidence mismatch without opening validation labels.

It is intentionally an offline, fail-closed transform.  Existing model
generations are hash-checked against their manifests; no GPU is used.  Qwen
candidate nodes are rebuilt with ``sample_evidence.evidence_summary`` so
explicit abstentions and malformed outputs are not conflated.  Gemma evidence
and all candidate-blind commitments are preserved byte-for-byte at the row
level.

The ten training rows used as System-2 golden demonstrations are retained for
coverage but marked ``calibration_eligible=false``.  Downstream calibrators
must exclude them because their targets occur verbatim in the System-2 prompt.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from architecture_candidate_v3 import company_prediction, city_prediction
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.dual_model_validation import GEMMA, QWEN
from experiments.heterogeneous_agents.heterogeneous_memory_selector import (
    _key,
    _load_graph,
)
from numeric_aggregation import aggregate_quantile
from run_inference import (
    aggregate,
    aggregate_numeric_cluster,
    drop_self_reference,
    extract_after_think,
)
from sample_evidence import evidence_summary


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "archive/experiments/architecture_overnight_20260713"
DEFAULT_SOURCE = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "expanded_calibration_n1_20260723_v1")
DEFAULT_FP16 = ARCHIVE / "arms/precision_fp16_recovery/raw.jsonl"
DEFAULT_BORDER = ARCHIVE / "arms/recovery_4bit/raw.jsonl"
DEFAULT_SYSTEM2 = ARCHIVE / "system2_fp16/predictions.jsonl"
DEFAULT_GOLDEN = ROOT / "prompt_templates/golden_few_shot_examples.json"

PRODUCTION_RELATIONS = {
    "countryLandBordersCountry",
    "companyTradesAtStockExchange",
    "personHasCityOfDeath",
    "hasArea",
    "hasCapacity",
}
SYSTEM2_RELATIONS = {
    "companyTradesAtStockExchange",
    "personHasCityOfDeath",
}


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _manifest_for_raw(path: Path) -> Path:
    direct = path.parent / "manifest.json"
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if direct.is_file():
        return direct
    if sidecar.is_file():
        return sidecar
    raise ContractError(f"missing raw manifest for {path}")


def _validate_system1(
        path: Path, *, precision: str, required_relations: set[str],
        required_recovery: set[str],
) -> tuple[dict[tuple[str, str], dict], dict]:
    manifest_path = _manifest_for_raw(path)
    manifest = _json(manifest_path)
    if manifest.get("raw_cache_sha256") != sha256(path):
        raise ContractError(f"System-1 raw hash mismatch: {path}")
    if manifest.get("precision") != precision:
        raise ContractError(
            f"{path}: precision {manifest.get('precision')!r} != {precision!r}")
    if int(manifest.get("n_consistency", -1)) != 10:
        raise ContractError(f"{path}: expected N=10")
    if not bool(manifest.get("exclude_target_from_shots")):
        raise ContractError(f"{path}: target demonstrations were not excluded")
    recovery = set(manifest.get("recover_unclosed_relations", []))
    if not required_recovery <= recovery:
        raise ContractError(
            f"{path}: missing recovery relations "
            f"{sorted(required_recovery - recovery)}")
    rows = read_jsonl(path)
    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ContractError(f"duplicate System-1 key {key}")
        if row["Relation"] in required_relations:
            indexed[key] = row
    present = {relation for _, relation in indexed}
    if present != required_relations:
        raise ContractError(
            f"{path}: relation coverage {sorted(present)} != "
            f"{sorted(required_relations)}")
    return indexed, {
        "path": str(path),
        "sha256": sha256(path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "precision": precision,
        "recovery": sorted(recovery),
        "seed": manifest.get("seed"),
        "model_revision": manifest.get("model_revision"),
    }


def _validate_system2(
        path: Path,
) -> tuple[dict[tuple[str, str], dict], dict]:
    manifest_path = path.parent / "manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("output_sha256") != sha256(path):
        raise ContractError(f"System-2 prediction hash mismatch: {path}")
    if int(manifest.get("n_rows", -1)) != len(read_jsonl(path)):
        raise ContractError(f"System-2 row count mismatch: {path}")
    indexed: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(path):
        key = _key(row)
        if key in indexed:
            raise ContractError(f"duplicate System-2 key {key}")
        indexed[key] = row
    for relation in SYSTEM2_RELATIONS:
        count = sum(key[1] == relation for key in indexed)
        if count != 100:
            raise ContractError(
                f"System-2 {relation} coverage is {count}, expected 100")
    return indexed, {
        "path": str(path),
        "sha256": sha256(path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "config_sha256": manifest.get("config_sha256"),
        "seed": manifest.get("seed"),
        "model_revision": manifest.get("model_revision"),
    }


def _golden_keys(path: Path) -> set[tuple[str, str]]:
    golden = _json(path)
    keys = set()
    for relation in SYSTEM2_RELATIONS:
        rows = golden.get(relation)
        if not isinstance(rows, list) or len(rows) != 5:
            raise ContractError(
                f"expected five golden demonstrations for {relation}")
        for row in rows:
            subject = row.get("SubjectEntity")
            if not isinstance(subject, str):
                raise ContractError(f"invalid golden row for {relation}")
            keys.add((subject, relation))
    return keys


def _numeric_log_mad(candidates: Sequence[Mapping[str, Any]]) -> float | None:
    values = []
    for candidate in candidates:
        try:
            value = float(str(candidate["item"]).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            values.extend([value] * max(1, int(candidate["votes"])))
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    logs = [math.log(value) for value in values]
    center = statistics.median(logs)
    return statistics.median(abs(value - center) for value in logs)


def _production_objects(
        raw: Mapping[str, Any], system2: Mapping[str, Any] | None,
) -> list[str]:
    subject = str(raw["SubjectEntity"])
    relation = str(raw["Relation"])
    samples = list(raw.get("raw_samples", []))
    system2_items = (
        list(system2.get("ObjectEntities", [])) if system2 is not None else [])
    if relation == "countryLandBordersCountry":
        return aggregate(
            relation, subject, samples, response_protocol="legacy-cot",
            aggregation_profile="relation-v1")
    if relation == "companyTradesAtStockExchange":
        return company_prediction(subject, samples, system2_items)
    if relation == "personHasCityOfDeath":
        return city_prediction(subject, samples, system2_items)
    answers = [extract_after_think(sample) for sample in samples]
    if relation == "hasArea":
        return drop_self_reference(
            subject, aggregate_quantile(answers, 0.55))
    if relation == "hasCapacity":
        return drop_self_reference(
            subject, aggregate_numeric_cluster(answers, 0.30))
    raise ContractError(f"no production decoder for {relation}")


def _rebase_qwen(
        original: Mapping[str, Any], raw: Mapping[str, Any],
        baseline_objects: Sequence[str],
) -> dict:
    """Replace only Qwen proposal evidence; preserve Gemma and commitments."""
    row = copy.deepcopy(original)
    relation = str(row["Relation"])
    summary = evidence_summary(
        raw.get("raw_samples", []), relation,
        response_protocol="legacy-cot")
    nodes: dict[str, dict] = {}
    for old in row.get("candidates", []):
        gemma_source = old.get("sources", {}).get(GEMMA)
        if gemma_source is None:
            continue
        node = copy.deepcopy(old)
        node["sources"] = {GEMMA: copy.deepcopy(gemma_source)}
        node["selected_by"] = {
            GEMMA: bool(old.get("selected_by", {}).get(GEMMA, False)),
            QWEN: False,
        }
        nodes[str(node["key"])] = node
    n_samples = int(summary["n_samples"])
    for candidate in summary["candidates"]:
        key = str(candidate["key"])
        node = nodes.setdefault(key, {
            "key": key,
            "item": str(candidate["item"]),
            "type": (
                "numeric" if relation in {"hasArea", "hasCapacity"}
                else "string"),
            "sources": {},
            "selected_by": {GEMMA: False, QWEN: False},
        })
        votes = int(candidate["votes"])
        node["sources"][QWEN] = {
            "support": votes,
            "samples": n_samples,
            "support_rate": votes / n_samples if n_samples else 0.0,
        }
    selected = {
        canonical_key(str(item), relation)
        for item in baseline_objects
        if canonical_key(str(item), relation)
    }
    for node in nodes.values():
        node["selected_by"][QWEN] = node["key"] in selected
    row["candidates"] = sorted(nodes.values(), key=lambda node: (
        -len(node["sources"]),
        -sum(float(source["support_rate"])
             for source in node["sources"].values()),
        str(node["key"]),
    ))
    row["agents"][QWEN].update({
        "n_samples": n_samples,
        "none_count": int(summary["explicit_abstentions"]),
        "none_rate": (
            float(summary["explicit_abstentions"]) / n_samples
            if n_samples else 0.0),
        "parse_failures": int(summary["invalid_samples"]),
        "numeric_log_mad": _numeric_log_mad(summary["candidates"]),
    })
    row["baseline_objects"] = list(baseline_objects)
    row["agent_outputs"][QWEN] = list(baseline_objects)
    row["production_match"] = {
        "qwen_evidence": "frozen-v0495-relation-route",
        "system2_corroboration": relation in SYSTEM2_RELATIONS,
        "candidate_extraction": "relation-aware-sample-evidence-v1",
    }
    return row


def _write_graph(
        path: Path, rows: Sequence[Mapping[str, Any]], *, split: str,
        sources: Mapping[str, Any],
) -> None:
    write_jsonl_atomic(path, rows)
    manifest = {
        "schema": "heterogeneous-memory-graph-manifest-v1",
        "split": split,
        "rows": len(rows),
        "contains_labels": False,
        "gold_aware": False,
        "output_sha256": sha256(path),
        "production_matched": split == "train",
        "sources": sources,
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    source = Path(args.source_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    fp16_path = Path(args.fp16_raw).resolve()
    border_path = Path(args.border_raw).resolve()
    system2_path = Path(args.system2_predictions).resolve()
    golden_path = Path(args.golden_few_shot).resolve()
    train_path = source / "graphs/train_graph.jsonl"
    validation_path = source / "graphs/validation_graph.jsonl"
    train = _load_graph(train_path, expected_split="train")
    validation = _load_graph(validation_path, expected_split="validation")
    if len(train) != 477 or len(validation) != 478:
        raise ContractError("unexpected train/validation graph coverage")

    fp16, fp16_provenance = _validate_system1(
        fp16_path, precision="fp16",
        required_relations={
            "companyTradesAtStockExchange", "personHasCityOfDeath",
            "hasArea", "hasCapacity"},
        required_recovery={
            "companyTradesAtStockExchange", "countryLandBordersCountry"},
    )
    border, border_provenance = _validate_system1(
        border_path, precision="4bit",
        required_relations={"countryLandBordersCountry"},
        required_recovery={
            "companyTradesAtStockExchange", "countryLandBordersCountry"},
    )
    system2, system2_provenance = _validate_system2(system2_path)
    golden_keys = _golden_keys(golden_path)

    train_keys = {_key(row) for row in train}
    exact_raw = {**fp16, **border}
    expected = {
        key for key in train_keys if key[1] in PRODUCTION_RELATIONS}
    if set(exact_raw) != expected:
        raise ContractError(
            f"production raw coverage mismatch: missing "
            f"{len(expected - set(exact_raw))}, extra "
            f"{len(set(exact_raw) - expected)}")
    missing_system2 = {
        key for key in expected
        if key[1] in SYSTEM2_RELATIONS and key not in system2}
    if missing_system2:
        raise ContractError(
            f"System-2 misses {len(missing_system2)} city/company rows")

    rebuilt = []
    for graph in train:
        key = _key(graph)
        if key[1] in PRODUCTION_RELATIONS:
            raw = exact_raw[key]
            objects = _production_objects(raw, system2.get(key))
            row = _rebase_qwen(graph, raw, objects)
        else:
            row = copy.deepcopy(graph)
            row["production_match"] = {
                "qwen_evidence": "preserved-existing-award-evidence",
                "system2_corroboration": False,
                "candidate_extraction": "preserved",
            }
        if key in golden_keys:
            row["calibration_eligible"] = False
            row["calibration_exclusion_reason"] = (
                "subject appears in System-2 golden few-shot demonstrations")
        else:
            row["calibration_eligible"] = True
        rebuilt.append(row)
    if sum(not row["calibration_eligible"] for row in rebuilt) != 10:
        raise ContractError("expected exactly ten golden-demo exclusions")

    graph_dir = output / "graphs"
    plan_dir = output / "plan"
    graph_dir.mkdir(parents=True, exist_ok=True)
    plan_dir.mkdir(parents=True, exist_ok=True)
    source_plan = source / "plan/PLAN.json"
    if not source_plan.is_file():
        raise ContractError(f"missing source plan {source_plan}")
    shutil.copy2(source_plan, plan_dir / "PLAN.json")
    sources = {
        "source_train_graph": {
            "path": str(train_path), "sha256": sha256(train_path)},
        "fp16_system1": fp16_provenance,
        "border_system1": border_provenance,
        "system2": system2_provenance,
        "golden_few_shot": {
            "path": str(golden_path), "sha256": sha256(golden_path)},
    }
    new_train = graph_dir / "train_graph.jsonl"
    new_validation = graph_dir / "validation_graph.jsonl"
    _write_graph(new_train, rebuilt, split="train", sources=sources)
    _write_graph(
        new_validation, validation, split="validation",
        sources={
            "source_validation_graph": {
                "path": str(validation_path),
                "sha256": sha256(validation_path)},
            "note": (
                "preserved exactly: source validation graph already uses "
                "frozen production Qwen/Gemma evidence"),
        })
    audit = {
        "schema": "production-matched-graph-audit-v1",
        "labels_opened": False,
        "validation_labels_opened": False,
        "train_rows": len(rebuilt),
        "validation_rows": len(validation),
        "production_rebased_rows": sum(
            row["Relation"] in PRODUCTION_RELATIONS for row in rebuilt),
        "preserved_award_rows": sum(
            row["Relation"] == "awardWonBy" for row in rebuilt),
        "calibration_eligible_rows": sum(
            bool(row["calibration_eligible"]) for row in rebuilt),
        "golden_demo_exclusions": sorted(
            [list(key) for key in golden_keys]),
        "sources": sources,
        "train_graph": str(new_train),
        "train_graph_sha256": sha256(new_train),
        "validation_graph": str(new_validation),
        "validation_graph_sha256": sha256(new_validation),
    }
    (output / "PRODUCTION_MATCH_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        f"production-matched graphs ready: {len(rebuilt)} train rows, "
        f"{audit['calibration_eligible_rows']} calibration eligible")
    print(f"audit={output / 'PRODUCTION_MATCH_AUDIT.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fp16-raw", default=str(DEFAULT_FP16))
    parser.add_argument("--border-raw", default=str(DEFAULT_BORDER))
    parser.add_argument("--system2-predictions", default=str(DEFAULT_SYSTEM2))
    parser.add_argument("--golden-few-shot", default=str(DEFAULT_GOLDEN))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
