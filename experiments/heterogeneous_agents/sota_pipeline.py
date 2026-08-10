#!/usr/bin/env python3
"""Canonical, fail-closed definitions of the current heterogeneous pipeline.

Accuracy experiments must be residualized against a prediction artifact, not
against an informal stage name.  This module records the exact artifacts and
hashes for the current clean and competition pipelines and constructs the
subject-grouped OOF counterpart of the competition pipeline.

The competition pipeline is deliberately *not* called blind-safe: its final
capacity veto was formulated after validation had been opened.  Keeping that
fact in the registry prevents an exact Codabench reproduction from silently
being presented as untouched validation evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.heterogeneous_memory_selector import _key


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/heterogeneous_agents/runs"
CANONICAL_ARTIFACTS = (
    ROOT / "results/heterogeneous/canonical_runtime/sota_pipeline_20260726")

COMPETITION_PIPELINE_ID = "competition_sota_0_511138728_20260726"
CLEAN_PIPELINE_ID = "clean_train_selected_0_509046679_20260725"

COMPONENT_OOF = (
    CANONICAL_ARTIFACTS / "component/"
    "train_oof_selected.jsonl")
COMPONENT_VALIDATION = (
    CANONICAL_ARTIFACTS / "component/"
    "validation_train_selected.jsonl")
COMPONENT_MODEL = (
    CANONICAL_ARTIFACTS / "component/models.json")
COMPONENT_SOURCE_GRAPH = (
    CANONICAL_ARTIFACTS / "component/source_train_graph.jsonl")
COMPONENT_FOLDS = (
    CANONICAL_ARTIFACTS / "component/FOLDS.jsonl")
CAPACITY_VETO_OOF = (
    CANONICAL_ARTIFACTS / "capacity_veto/TRAIN_OOF.jsonl")
CAPACITY_VETO_GATE = (
    CANONICAL_ARTIFACTS / "capacity_veto/TRAIN_GATE.json")
COMPETITION_VALIDATION = (
    CANONICAL_ARTIFACTS / "capacity_veto/"
    "VALIDATION_PREDICTIONS.jsonl")
COMPETITION_RESULT = (
    CANONICAL_ARTIFACTS / "capacity_veto/RESULT.json")

DEFAULT_OUTPUT = RUNS / "canonical_sota_pipeline_20260727_v1"
DEFAULT_TRAIN_OOF = DEFAULT_OUTPUT / "TRAIN_OOF_PREDICTIONS.jsonl"

EXPECTED = {
    "component_oof": {
        "path": COMPONENT_OOF,
        "sha256": "3a849305bc0f62d2866c67edc4e25ac7029a5e3544b57939e6cbff68fa0b201d",
        "rows": 467,
    },
    "component_validation": {
        "path": COMPONENT_VALIDATION,
        "sha256": "ae303a5832fdcf4aa33b3a45bd4fdc05d35f5b3837b125a20bfa89ef27e6854c",
        "rows": 478,
    },
    "component_model": {
        "path": COMPONENT_MODEL,
        "sha256": "f026a55f22a00125ca6991df046a674fc72e951bb740859196769961d90526bd",
    },
    "component_source_graph": {
        "path": COMPONENT_SOURCE_GRAPH,
        "sha256": "5d203e00aa128c744c5f57b6dedabe58d5f3aac75dda515f14e4921e18559855",
        "rows": 477,
    },
    "component_folds": {
        "path": COMPONENT_FOLDS,
        "sha256": "c234d3b33b2b3f6dbfe941d77edeb34f09bc8cb268a4d9442579e57262f2963e",
    },
    "capacity_veto_oof": {
        "path": CAPACITY_VETO_OOF,
        "sha256": "8bc86d92999cab5efb25ba828500a8e230e775ed15462b7ac3c3693e4ea265a9",
        "rows": 387,
    },
    "capacity_veto_gate": {
        "path": CAPACITY_VETO_GATE,
        "sha256": "211ce820646500e4b7b2bcc1577a9a07fcb336a00580ffb49eb70cbe2a91e4d6",
    },
    "competition_validation": {
        "path": COMPETITION_VALIDATION,
        "sha256": "4943fda7755c3686b7f5309683e5d94a7b4f6a95a757d790bac7ce9fa8d381b1",
        "rows": 478,
        "score": 0.511138728718981,
    },
    "competition_result": {
        "path": COMPETITION_RESULT,
        "sha256": "7a673b90c8d41b14e5552d235a8476853c0cd19447fa50aa37283c4a4f436e2f",
    },
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _verify_file(name: str) -> Path:
    specification = EXPECTED[name]
    path = Path(specification["path"]).resolve()
    if not path.is_file():
        raise ContractError(f"missing registered SOTA artifact: {path}")
    actual = sha256(path)
    if actual != specification["sha256"]:
        raise ContractError(
            f"stale registered SOTA artifact {name}: "
            f"expected {specification['sha256']}, got {actual}")
    return path


def _prediction_rows(name: str) -> list[dict[str, Any]]:
    path = _verify_file(name)
    rows = read_jsonl(path)
    expected_rows = int(EXPECTED[name]["rows"])
    keys = [_key(row) for row in rows]
    if len(rows) != expected_rows or len(set(keys)) != len(keys):
        raise ContractError(
            f"{name}: expected {expected_rows} unique rows, got {len(rows)}")
    return rows


def competition_validation_predictions() -> tuple[
    list[dict[str, Any]], dict[str, Any]
]:
    """Load and certify the exact Codabench-matched validation predictions."""
    rows = _prediction_rows("competition_validation")
    result_path = _verify_file("competition_result")
    result = _json(result_path)
    validation = result.get("validation")
    if (
        result.get("schema") != "capacity-qwen-precision-veto-experiment-v1"
        or not isinstance(validation, Mapping)
        or validation.get("predictions_sha256")
        != EXPECTED["competition_validation"]["sha256"]
        or float(validation["scores"]["*** All Relations ***"])
        != EXPECTED["competition_validation"]["score"]
        or validation.get("validation_labels_used_for_selection") is not True
        or validation.get("hypothesis_created_after_validation_opened")
        is not True
    ):
        raise ContractError("competition SOTA result provenance is invalid")
    return rows, {
        "pipeline_id": COMPETITION_PIPELINE_ID,
        "prediction_path": str(
            Path(EXPECTED["competition_validation"]["path"]).resolve()),
        "prediction_sha256": EXPECTED["competition_validation"]["sha256"],
        "result_path": str(result_path),
        "result_sha256": EXPECTED["competition_result"]["sha256"],
        "reported_score": EXPECTED["competition_validation"]["score"],
        "validation_selected_lineage": True,
        "blind_safe": False,
    }


def compose_competition_train_oof() -> tuple[
    list[dict[str, Any]], dict[str, Any]
]:
    """Compose the exact stage ordering as subject-grouped train OOF rows.

    The component decoder supplies the full OOF incumbent.  Only retained
    hasCapacity decisions from the Qwen precision-veto ledger replace it.
    Other relation rows in that ledger are diagnostics and must not overwrite
    the component decoder.
    """
    component = _prediction_rows("component_oof")
    component_model_path = _verify_file("component_model")
    component_model = _json(component_model_path)
    # Historical manifests contain machine-local absolute paths.  Artifact
    # identity comes from their pinned hashes, so use the repository-local
    # registered copies and validate the embedded hashes instead of following
    # paths that cannot survive a clone.
    source_graph_path = _verify_file("component_source_graph")
    if (
        component_model.get("schema") != "component-aware-decoder-models-v1"
        or component_model.get("source_train_graph_sha256")
        != EXPECTED["component_source_graph"]["sha256"]
    ):
        raise ContractError("component model source graph is stale")
    source_graph = read_jsonl(source_graph_path)
    if len(source_graph) != int(EXPECTED["component_source_graph"]["rows"]):
        raise ContractError("expected complete 477-row component source graph")
    component_manifest_path = COMPONENT_OOF.with_suffix(
        COMPONENT_OOF.suffix + ".manifest.json")
    component_manifest = _json(component_manifest_path)
    folds_path = _verify_file("component_folds")
    if (
        component_manifest.get("schema")
        != "component-aware-oof-incumbents-v1"
        or component_manifest.get("oof_model_excludes_row") is not True
        or component_manifest.get("output_sha256")
        != EXPECTED["component_oof"]["sha256"]
        or component_manifest.get("folds_sha256")
        != EXPECTED["component_folds"]["sha256"]
    ):
        raise ContractError("component OOF manifest provenance is invalid")
    ledger_path = _verify_file("capacity_veto_oof")
    ledger = read_jsonl(ledger_path)
    if len(ledger) != int(EXPECTED["capacity_veto_oof"]["rows"]):
        raise ContractError("capacity veto OOF row-count mismatch")
    gate_path = _verify_file("capacity_veto_gate")
    gate = _json(gate_path)
    if (
        gate.get("schema") != "capacity-qwen-precision-veto-train-gate-v1"
        or gate.get("train_oof_supported") is not True
        or gate.get("oof_sha256") != EXPECTED["capacity_veto_oof"]["sha256"]
        or gate.get("hypothesis_created_after_validation_opened") is not True
    ):
        raise ContractError("capacity veto train gate provenance is invalid")

    # The historical component OOF writer emitted only its 467 calibration-
    # eligible rows. Validation composition preserved all rows from control.
    # Reproduce that behavior by carrying the source graph's certified OOF
    # baseline through for the ten ineligible rows before applying component
    # replacements.
    by_key = {
        _key(row): {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": [
                str(item) for item in row.get("baseline_objects", [])],
        }
        for row in source_graph
    }
    if len(by_key) != len(source_graph):
        raise ContractError("duplicate component source-graph key")
    for row in component:
        if _key(row) not in by_key:
            raise ContractError("component OOF row absent from source graph")
        by_key[_key(row)] = dict(row)
    capacity = [
        row for row in ledger if str(row.get("Relation")) == "hasCapacity"]
    if len(capacity) != 100:
        raise ContractError("expected 100 capacity OOF decisions")
    missing = [_key(row) for row in capacity if _key(row) not in by_key]
    if missing:
        raise ContractError("capacity veto does not cover component OOF")

    switched = 0
    for decision in capacity:
        if decision.get("action_id") is None:
            continue
        proposal = decision.get("proposal")
        if not isinstance(proposal, list):
            raise ContractError("selected capacity OOF decision lacks proposal")
        by_key[_key(decision)]["ObjectEntities"] = [
            str(item) for item in proposal]
        switched += 1
    if switched != int(gate["retained_switches"]):
        raise ContractError("capacity veto retained-switch count mismatch")

    output = [by_key[_key(row)] for row in source_graph]
    return output, {
        "pipeline_id": COMPETITION_PIPELINE_ID,
        "split": "train",
        "rows": len(output),
        "capacity_switches": switched,
        "component_rows_decoded": len(component),
        "component_rows_carried_forward": len(source_graph) - len(component),
        "subject_grouped_oof": True,
        "oof_model_excludes_subject": True,
        "contains_labels": False,
        "gold_aware": True,
        "deployable": False,
        "validation_selected_lineage": True,
        "blind_safe": False,
        "folds": str(folds_path),
        "folds_sha256": sha256(folds_path),
        "stages": [
            {
                "name": "july_component_decoder",
                "artifact": str(Path(COMPONENT_OOF).resolve()),
                "sha256": EXPECTED["component_oof"]["sha256"],
                "model": str(component_model_path),
                "model_sha256": EXPECTED["component_model"]["sha256"],
                "source_graph": str(source_graph_path),
                "source_graph_sha256": sha256(source_graph_path),
            },
            {
                "name": "capacity_qwen_precision_veto",
                "artifact": str(ledger_path),
                "sha256": EXPECTED["capacity_veto_oof"]["sha256"],
                "gate": str(gate_path),
                "gate_sha256": EXPECTED["capacity_veto_gate"]["sha256"],
            },
        ],
    }


def write_competition_train_oof(output_dir: Path) -> Path:
    rows, detail = compose_competition_train_oof()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "TRAIN_OOF_PREDICTIONS.jsonl"
    write_jsonl_atomic(path, rows)
    manifest = {
        "schema": "heterogeneous-sota-pipeline-predictions-manifest-v1",
        **detail,
        "output_sha256": sha256(path),
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def validate_registered_predictions(
    path: Path, *, pipeline_id: str, split: str,
) -> dict[str, Any]:
    """Validate a canonical pipeline artifact before downstream use."""
    path = Path(path).resolve()
    if pipeline_id != COMPETITION_PIPELINE_ID:
        raise ContractError(f"unknown pipeline id: {pipeline_id}")
    if split == "validation":
        _, detail = competition_validation_predictions()
        if path != Path(detail["prediction_path"]):
            raise ContractError(
                "validation start is not the registered competition SOTA")
        return detail
    if split != "train":
        raise ContractError(f"unsupported pipeline split: {split}")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ContractError("missing canonical train OOF manifest")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema")
        != "heterogeneous-sota-pipeline-predictions-manifest-v1"
        or manifest.get("pipeline_id") != pipeline_id
        or manifest.get("split") != "train"
        or manifest.get("oof_model_excludes_subject") is not True
        or manifest.get("output_sha256") != sha256(path)
    ):
        raise ContractError("invalid canonical train OOF artifact")
    expected_rows, expected_detail = compose_competition_train_oof()
    rows = read_jsonl(path)
    if rows != expected_rows:
        raise ContractError("canonical train OOF content is stale")
    if manifest.get("stages") != expected_detail["stages"]:
        raise ContractError("canonical train OOF stage lineage is stale")
    return manifest


def registry() -> dict[str, Any]:
    validation_rows, validation = competition_validation_predictions()
    train_rows, train = compose_competition_train_oof()
    return {
        "schema": "heterogeneous-pipeline-registry-v1",
        "default_pipeline": COMPETITION_PIPELINE_ID,
        "pipelines": {
            COMPETITION_PIPELINE_ID: {
                "description": (
                    "Exact Codabench-matched heterogeneous system; "
                    "development/competition SOTA, not blind-safe."),
                "train_oof": train,
                "validation": validation,
                "row_counts": {
                    "train": len(train_rows),
                    "validation": len(validation_rows),
                },
            },
            CLEAN_PIPELINE_ID: {
                "description": (
                    "July component decoder selected using train OOF only; "
                    "excludes the post-validation capacity veto."),
                "blind_safe": True,
                "validation_selected_lineage": False,
                "train_oof": {
                    "path": str(Path(COMPONENT_OOF).resolve()),
                    "sha256": EXPECTED["component_oof"]["sha256"],
                },
                "validation": {
                    "path": str(Path(COMPONENT_VALIDATION).resolve()),
                    "sha256": EXPECTED["component_validation"]["sha256"],
                    "reported_score": 0.509046678509776,
                },
            },
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("materialize", "verify"))
    result.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    if args.command == "materialize":
        path = write_competition_train_oof(output)
        (output / "PIPELINE_REGISTRY.json").write_text(
            json.dumps(registry(), indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "pipeline_id": COMPETITION_PIPELINE_ID,
            "train_oof": str(path),
            "train_oof_sha256": sha256(path),
            "validation": str(Path(COMPETITION_VALIDATION).resolve()),
            "validation_sha256":
                EXPECTED["competition_validation"]["sha256"],
            "blind_safe": False,
        }, indent=2, sort_keys=True))
        return 0
    validate_registered_predictions(
        DEFAULT_TRAIN_OOF if output == DEFAULT_OUTPUT
        else output / "TRAIN_OOF_PREDICTIONS.jsonl",
        pipeline_id=COMPETITION_PIPELINE_ID,
        split="train")
    competition_validation_predictions()
    print(json.dumps(registry(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
