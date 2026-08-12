#!/usr/bin/env python3
"""Blind-safe replay of the final test decoder with one Ministral route.

The replay reuses a completed test run's frozen Qwen, Gemma, and Ministral
SyntheticCoT N=10 responses.  It removes the zero-shot Ministral N=3 route at
graph construction time, applies the retained N=10 component rule for area,
then runs the existing relation-typed graph correction.  No labels are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

from experiments.heterogeneous_agents import end_to_end_pipeline as e2e
from experiments.heterogeneous_agents import final_submission_pipeline as final
from experiments.heterogeneous_agents.analysis import ministral_route1_ablation as route1
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT
    / "experiments/heterogeneous_agents/runs/"
    "final_test_release_candidate_20260809_v2"
)
DEFAULT_ARCHIVED = (
    ROOT
    / "submissions/official_test/"
    "heterogeneous_final_strict_proof_20260803_v1_test.zip"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/heterogeneous_agents/runs/"
    "ministral_single_route_test_replay_20260810_v1"
)
HISTORICAL_FINAL_IMPLEMENTATION_SHA256 = (
    "d72b91a814c2fdb5b34c8784155daa7af935a34d4d417ec173d18464610788c8"
)
HISTORICAL_PRIMARY_RUNNER_SHA256 = (
    "ecddca9a6395118b7beece60f5724f5b5109da4af53f43a1b369c55978e479fd"
)


def _objects(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    result = {
        final._key(row): [str(value) for value in row["ObjectEntities"]]
        for row in rows
    }
    if len(result) != len(rows):
        raise ContractError("duplicate subject-relation key")
    return result


def _read_archived_predictions(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != ["predictions.jsonl"]:
            raise ContractError(
                f"unexpected archived submission members: {names}")
        payload = archive.read("predictions.jsonl").decode("utf-8")
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _write_deterministic_zip(path: Path, prediction_path: Path) -> None:
    payload = prediction_path.read_bytes()
    info = zipfile.ZipInfo("predictions.jsonl", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        archive.writestr(info, payload)
    temporary.replace(path)


def _changed_by_relation(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    before_by = _objects(before)
    after_by = _objects(after)
    if set(before_by) != set(after_by):
        raise ContractError("prediction key coverage differs")
    counts: dict[str, int] = {}
    for key in before_by:
        if before_by[key] == after_by[key]:
            continue
        counts[key[1]] = counts.get(key[1], 0) + 1
    return dict(sorted(counts.items()))


def _validate_historical_policy(
    source: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    """Validate the immutable pre-stable-key test policy and its artifacts.

    The only later changes to ``final_submission_pipeline.py`` and
    ``run_submission.py`` added configurable seed plumbing.  The completed
    source run used the legacy scheme, and this replay performs no generation.
    We therefore bind to the recorded historical hashes while still checking
    every frozen model and unchanged decoder artifact byte-for-byte.
    """
    policy_path = source / "plan/FINAL_POLICY.json"
    policy = json.loads(policy_path.read_text())
    source_plan = e2e._validate_plan(source)
    if (
        policy.get("schema") != final.POLICY_SCHEMA
        or policy.get("policy_id") != final.POLICY_ID
        or policy.get("source_plan_sha256")
            != sha256(source / "plan/PLAN.json")
        or policy.get("implementation_sha256")
            != HISTORICAL_FINAL_IMPLEMENTATION_SHA256
        or policy.get("primary_runner_sha256")
            != HISTORICAL_PRIMARY_RUNNER_SHA256
        or policy.get("primary_policy") != final.PRIMARY_POLICY
        or policy.get("split") != "test"
        or not policy.get("blind")
        or source_plan.get("split") != "test"
        or not source_plan.get("blind")
    ):
        raise ContractError("historical frozen test policy contract failed")

    model_paths: dict[str, Path] = {}
    for name in final.MODEL_ARTIFACTS:
        record = policy.get("model_artifacts", {}).get(name, {})
        path = Path(str(record.get("path", "")))
        if not path.is_file() or record.get("sha256") != sha256(path):
            raise ContractError(f"frozen model contract failed: {name}")
        model_paths[name] = path

    for name, record in policy.get("decoder_implementations", {}).items():
        if name == "final_submission":
            if record.get("sha256") != HISTORICAL_FINAL_IMPLEMENTATION_SHA256:
                raise ContractError("historical decoder hash mismatch")
            continue
        path = Path(str(record.get("path", "")))
        if not path.is_file() or record.get("sha256") != sha256(path):
            raise ContractError(f"frozen decoder contract failed: {name}")
    return policy, model_paths, source_plan


def _primary_inputs_legacy(
    output: Path,
    source_plan: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str], list[str]],
    dict[tuple[str, str], list[str]],
    dict[tuple[str, str], list[str]],
]:
    """Load the completed run's historical legacy-seeded Qwen artifacts."""
    primary_dir = output / "primary_qwen"
    arguments = argparse.Namespace(
        policy=final.PRIMARY_POLICY,
        input=str(Path(source_plan["input"]).resolve()),
        output_dir=str(primary_dir),
        dry_run=False,
        skip_inference=True,
        stage="compose",
        seed_scheme="legacy",
    )
    submission = final.PrimarySubmission(arguments)
    submission.preflight()
    submission.split_inputs()
    submission.validate_all_bundles()
    manifest = json.loads((primary_dir / "MANIFEST.json").read_text())
    prediction_path = (
        primary_dir / f"submission_{final.PRIMARY_POLICY}.jsonl")
    if (
        manifest.get("policy") != final.PRIMARY_POLICY
        or manifest.get("input_sha256") != source_plan["input_sha256"]
        or manifest.get("submission_sha256") != sha256(prediction_path)
    ):
        raise ContractError("historical primary Qwen manifest mismatch")

    predictions = {
        final._key(row): [str(value) for value in row["ObjectEntities"]]
        for row in read_jsonl(prediction_path)
    }
    raw: dict[tuple[str, str], list[str]] = {}
    for path in (
        primary_dir / "raw_borders.jsonl",
        primary_dir / "raw_fp16.jsonl",
    ):
        for row in read_jsonl(path):
            key = final._key(row)
            samples = [str(value) for value in row.get("raw_samples", [])]
            if len(samples) != 10 or key in raw:
                raise ContractError(f"invalid primary Qwen raw row: {key}")
            raw[key] = samples
    for key, objects in predictions.items():
        if key[1] == "awardWonBy":
            raw[key] = [
                "ANSWER: " + ("; ".join(objects) if objects else "None")]
    if len(predictions) != int(source_plan["rows"]):
        raise ContractError("primary Qwen prediction coverage mismatch")
    if len(raw) != int(source_plan["rows"]):
        raise ContractError("primary Qwen raw coverage mismatch")
    system2 = {
        final._key(row): [str(value) for value in row["ObjectEntities"]]
        for row in read_jsonl(primary_dir / "pred_system2.jsonl")
    }
    return predictions, raw, system2


def run(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output_dir).resolve()
    archived_path = Path(args.archived_submission).resolve()

    policy, model_paths, source_plan = _validate_historical_policy(source)
    primary, qwen_raw, system2 = _primary_inputs_legacy(source, source_plan)
    base_rows = final._assemble_from_primary(
        source, source_plan, primary, qwen_raw)
    gemma = final._response_map(source_plan, "gemma:independent")
    ministral_cot40 = final._response_map(
        source_plan, e2e.MINISTRAL_COT40)

    decision_graphs: list[dict[str, Any]] = []
    full_graphs: list[dict[str, Any]] = []
    for source_row in base_rows:
        key = final._key(source_row)
        base, qwen_texts, gemma_texts = final._prepare_base_row(
            source_row,
            {"generations": list(qwen_raw[key])},
            gemma[key],
            primary_objects=primary[key],
            system2_objects=system2.get(key, ()),
        )
        graph = route1.build_graph_without_route1(
            base,
            qwen_texts=qwen_texts,
            gemma_texts=gemma_texts,
            ministral_cot40=ministral_cot40[key],
        )
        if e2e.MINISTRAL_N3 in graph.get("proposal_routes", {}):
            raise ContractError(f"N=3 route survived for {key}")
        decision_graphs.append(base)
        full_graphs.append(graph)

    predictions, decisions = final._apply_frozen_stack(
        decision_graphs, full_graphs, model_paths)

    # Replace the removed N=3 area stage with the already frozen N=10
    # complete-link component rule, then reapply the graph correction to area.
    area_graphs: list[dict[str, Any]] = []
    area_incumbents: dict[tuple[str, str], list[str]] = {}
    area_details: dict[tuple[str, str], dict[str, Any]] = {}
    for graph, decision in zip(full_graphs, decisions, strict=True):
        if graph["Relation"] != "hasArea":
            continue
        key = final._key(graph)
        before = [str(value) for value in decision["layers"][-1]["before"]]
        selected, detail = route1.component_cot40_area(graph, before)
        area_graphs.append(graph)
        area_incumbents[key] = selected
        area_details[key] = detail
    area_predictions, area_proof = final._apply_relation_typed_graph_correction(
        area_graphs, area_incumbents)

    final_predictions = [dict(row) for row in predictions]
    index = {final._key(row): i for i, row in enumerate(final_predictions)}
    for row in area_predictions:
        final_predictions[index[final._key(row)]] = row

    if len(final_predictions) != 475:
        raise ContractError(
            f"expected 475 official-test rows, got {len(final_predictions)}")
    if any("ObjectEntities" not in row for row in final_predictions):
        raise ContractError("prediction row lacks ObjectEntities")

    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.jsonl"
    decision_path = output / "DECISIONS.jsonl"
    graph_path = output / "EVIDENCE_GRAPH.jsonl"
    archive_path = output / "ministral_single_route_test.zip"
    write_jsonl_atomic(prediction_path, final_predictions)
    write_jsonl_atomic(decision_path, decisions)
    write_jsonl_atomic(graph_path, full_graphs)
    _write_deterministic_zip(archive_path, prediction_path)

    source_predictions = read_jsonl(source / "FINAL_PREDICTIONS.jsonl")
    archived_predictions = _read_archived_predictions(archived_path)
    source_changes = _changed_by_relation(source_predictions, final_predictions)
    archived_changes = _changed_by_relation(
        archived_predictions, final_predictions)
    result = {
        "schema": "ministral-single-route-blind-test-replay-v1",
        "blind": True,
        "contains_labels": False,
        "gold_aware": False,
        "rows": len(final_predictions),
        "source_run": str(source),
        "removed_route": e2e.MINISTRAL_N3,
        "retained_ministral_route": e2e.MINISTRAL_COT40,
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "submission": str(archive_path),
        "submission_sha256": sha256(archive_path),
        "source_prediction_sha256": sha256(source / "FINAL_PREDICTIONS.jsonl"),
        "archived_submission": str(archived_path),
        "archived_submission_sha256": sha256(archived_path),
        "changed_vs_source": sum(source_changes.values()),
        "changed_vs_source_by_relation": source_changes,
        "changed_vs_archived_0_4845": sum(archived_changes.values()),
        "changed_vs_archived_0_4845_by_relation": archived_changes,
        "area_component_replacements": sum(
            bool(detail.get("applied")) for detail in area_details.values()),
        "area_graph_corrections": sum(
            bool(detail.get("changed")) for detail in area_proof),
        "score_available_locally": False,
        "score_requires_codabench": True,
    }
    result_path = output / "RESULT.json"
    final._write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", default=str(DEFAULT_SOURCE))
    value.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    value.add_argument(
        "--archived-submission", default=str(DEFAULT_ARCHIVED))
    value.set_defaults(function=run)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.function(arguments))
