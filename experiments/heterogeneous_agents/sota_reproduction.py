#!/usr/bin/env python3
"""Portable, fail-closed reproduction of the 0.518450 development SOTA.

This is the canonical entrypoint for the final development pipeline.  It
does not follow machine-local paths in historical run manifests.  Instead it
verifies the compact evidence snapshot checked into
``results/heterogeneous/canonical_runtime/sota_0_518450_20260729`` and then
executes the graph construction and frozen decoder stack in one workflow.

Two parser modes are intentional:

``legacy-20260729``
    Replays the parser behavior used by the original development run and must
    reproduce its prediction artifact byte-for-byte.

``corrected``
    Removes repeated ``ANSWER:`` markers.  It differs on six audited rows but
    has the same official pooled development F1.  This mode is diagnostic and
    is not the byte-level parity target.

The evidence snapshot is sufficient for deterministic pipeline replay.  It
does not claim bitwise GPU regeneration of frozen model responses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate as official
import numpy as np

from experiments.heterogeneous_agents.components import (
    unified_graph_decoder as unified,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.baseline_relative_route_decoder import (
    ResidualRidge,
    _prediction_rows as route_prediction_rows,
    decode as decode_route,
)
from experiments.heterogeneous_agents.components.explicit_cardinality_ablation import (
    ExplicitCardinalityModel,
    _prediction_rows as cardinality_prediction_rows,
)
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    LogisticCalibrator,
    _key,
)
from experiments.heterogeneous_agents.components.relation_specific_numeric_decoder import (
    RelationSpecificNumericModel,
    _merge_numeric,
)


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = (
    ROOT / "results/heterogeneous/canonical_runtime/"
    "sota_0_518450_20260729"
)
ARTIFACT_MANIFEST = SNAPSHOT / "ARTIFACTS.json"
DEFAULT_OUTPUT = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "sota_reproduction_20260729_v1"
)

PIPELINE_ID = "development_sota_0_518450_20260729"
MANIFEST_SCHEMA = "heterogeneous-sota-reproduction-artifacts-v1"
RESULT_SCHEMA = "heterogeneous-sota-reproduction-result-v1"
EXPECTED_ROWS = 478
EXPECTED_SCORE = 0.5184496147269507
EXPECTED_LEGACY_GRAPH_SHA256 = (
    "6f65087ba9b8a03a5dbd401424af99ed69840a55328f1b4914c7d6b145c57486"
)
POOLED = "*** All Relations ***"


def _calibrator(value: Mapping[str, Any]) -> LogisticCalibrator:
    return LogisticCalibrator.from_dict(value)


def _cardinality_model(value: Mapping[str, Any]) -> ExplicitCardinalityModel:
    if value.get("schema") != "explicit-cardinality-ovr-v1":
        raise ContractError("foreign explicit-cardinality model")
    model = ExplicitCardinalityModel(float(value["l2"]))
    model.many_mean = {
        str(relation): float(mean)
        for relation, mean in value["many_mean"].items()
    }
    model.models = {
        str(relation): {
            str(label): _calibrator(calibrator)
            for label, calibrator in labels.items()
        }
        for relation, labels in value["models"].items()
    }
    return model


def _numeric_model(value: Mapping[str, Any]) -> RelationSpecificNumericModel:
    if value.get("schema") != "relation-specific-numeric-model-v1":
        raise ContractError("foreign relation-specific numeric model")
    model = RelationSpecificNumericModel(float(value["l2"]))
    model.models = {
        str(relation): _calibrator(calibrator)
        for relation, calibrator in value["models"].items()
    }
    return model


def _residual_model(value: Mapping[str, Any]) -> ResidualRidge:
    if value.get("schema") != "signed-standardized-residual-ridge-v1":
        raise ContractError("foreign route-residual model")
    model = ResidualRidge(value["feature_names"], float(value["l2"]))
    model.mean = np.asarray(value["mean"], dtype=np.float64)
    model.scale = np.asarray(value["scale"], dtype=np.float64)
    model.coefficients = np.asarray(value["coefficients"], dtype=np.float64)
    if (
        model.mean.shape != (len(model.names),)
        or model.scale.shape != (len(model.names),)
        or model.coefficients.shape != (len(model.names) + 1,)
    ):
        raise ContractError("route-residual model shape mismatch")
    return model


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _snapshot_path(relative: str) -> Path:
    path = (SNAPSHOT / relative).resolve()
    try:
        path.relative_to(SNAPSHOT.resolve())
    except ValueError as exc:
        raise ContractError(f"snapshot path escapes its root: {relative}") from exc
    return path


def _root_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ContractError(f"tracked path escapes repository root: {relative}") from exc
    return path


def _key_set(rows: Sequence[Mapping[str, Any]], *, response: bool = False) -> set[
    tuple[str, str]
]:
    if response:
        keys = {
            (str(row.get("subject")), str(row.get("relation")))
            for row in rows
        }
    else:
        keys = {_key(row) for row in rows}
    if len(keys) != len(rows):
        raise ContractError("duplicate or malformed subject-relation key")
    return keys


def verify_snapshot() -> dict[str, Any]:
    """Verify every portable input before any graph or prediction is written."""
    manifest = _json(ARTIFACT_MANIFEST)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("pipeline_id") != PIPELINE_ID
        or manifest.get("split") != "validation"
        or int(manifest.get("rows", -1)) != EXPECTED_ROWS
        or float(manifest.get("reported_pooled_macro_f1", -1))
        != EXPECTED_SCORE
        or int(manifest.get("verified_parameter_total", -1))
        != 30_515_165_024
        or int(manifest.get("parameter_cap", -1)) != 32_000_000_000
        or manifest.get("validation_selected_lineage") is not True
        or manifest.get("blind_safe") is not False
        or manifest.get("contains_labels") is not False
        or manifest.get("gold_aware") is not False
    ):
        raise ContractError("invalid SOTA artifact manifest contract")
    portfolio = manifest.get("model_portfolio")
    if (
        not isinstance(portfolio, Mapping)
        or sum(
            int(value.get("verified_parameter_count", -1))
            for value in portfolio.values()
            if isinstance(value, Mapping)
        ) != int(manifest["verified_parameter_total"])
    ):
        raise ContractError("invalid SOTA parameter-accounting contract")

    artifacts = manifest.get("artifacts")
    external = manifest.get("external_tracked_inputs")
    if not isinstance(artifacts, Mapping) or not isinstance(external, Mapping):
        raise ContractError("SOTA manifest lacks artifact inventories")

    paths: dict[str, Path] = {}
    for name, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            raise ContractError(f"invalid artifact specification: {name}")
        path = _snapshot_path(str(raw["path"]))
        if not path.is_file() or sha256(path) != raw.get("sha256"):
            raise ContractError(f"missing or stale SOTA artifact: {name}")
        rows_expected = raw.get("rows")
        if rows_expected is not None:
            rows = read_jsonl(path)
            if len(rows) != int(rows_expected):
                raise ContractError(
                    f"{name}: expected {rows_expected} rows, got {len(rows)}")
        paths[str(name)] = path

    external_paths: dict[str, Path] = {}
    for name, raw in external.items():
        if not isinstance(raw, Mapping):
            raise ContractError(f"invalid tracked-input specification: {name}")
        path = _root_path(str(raw["path"]))
        if not path.is_file() or sha256(path) != raw.get("sha256"):
            raise ContractError(f"missing or stale tracked input: {name}")
        external_paths[str(name)] = path

    typed = read_jsonl(paths["typed_graph"])
    responses = read_jsonl(paths["cot40_responses"])
    base = read_jsonl(paths["base_validation_graph"])
    route = read_jsonl(paths["route_validation_graph"])
    l1_target = read_jsonl(paths["l1_cardinality_target"])
    l2_target = read_jsonl(paths["l2_numeric_target"])
    incumbent = read_jsonl(paths["chain_incumbent"])
    deployed = read_jsonl(paths["deployed_predictions"])
    decisions = read_jsonl(paths["deployed_decisions"])
    expected_keys = _key_set(typed)
    if (
        len(expected_keys) != EXPECTED_ROWS
        or _key_set(responses, response=True) != expected_keys
        or _key_set(base) != expected_keys
        or _key_set(route) != expected_keys
        or _key_set(l1_target) != expected_keys
        or _key_set(l2_target) != expected_keys
        or _key_set(incumbent) != expected_keys
        or _key_set(deployed) != expected_keys
        or _key_set(decisions) != expected_keys
    ):
        raise ContractError("portable SOTA artifacts have different row coverage")
    if any(
        ("contains_labels" in row and row.get("contains_labels") is not False)
        or ("gold_aware" in row and row.get("gold_aware") is not False)
        for row in typed
    ):
        raise ContractError("typed validation graph is not certified label-free")

    model = _json(paths["component_models"])
    if model.get("schema") != "component-aware-decoder-models-v1":
        raise ContractError("foreign component model artifact")
    selector = _json(paths["candidate_selector"])
    if not isinstance(selector.get("candidate_model"), Mapping):
        raise ContractError("foreign candidate selector artifact")
    cardinality = _json(paths["cardinality_result"])
    if cardinality.get("cardinality_model", {}).get(
            "schema") != "explicit-cardinality-ovr-v1":
        raise ContractError("foreign cardinality model artifact")
    numeric = _json(paths["numeric_model"])
    if numeric.get("schema") != "relation-specific-numeric-model-v1":
        raise ContractError("foreign numeric model artifact")
    route_models = _json(paths["route_models"])
    if route_models.get(
            "schema") != "baseline-relative-route-decoder-models-v1":
        raise ContractError("foreign route model artifact")
    capacity = read_jsonl(paths["capacity_ledger"])
    if (
        len(capacity) != 100
        or any(str(row.get("Relation")) != "hasCapacity" for row in capacity)
    ):
        raise ContractError("capacity ledger does not cover 100 capacity rows")

    return {
        "manifest": manifest,
        "manifest_sha256": sha256(ARTIFACT_MANIFEST),
        "artifacts": {name: str(path) for name, path in paths.items()},
        "external": {
            name: str(path) for name, path in external_paths.items()},
        "rows": len(expected_keys),
    }


def _require_hash(path: Path, expected: Path, stage: str) -> None:
    actual_sha = sha256(path)
    expected_sha = sha256(expected)
    if actual_sha != expected_sha:
        raise ContractError(
            f"{stage} reconstruction drift: expected {expected_sha}, "
            f"got {actual_sha}")


def _changed_rows(
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]],
) -> int:
    before_by = {_key(row): list(row["ObjectEntities"]) for row in before}
    after_by = {_key(row): list(row["ObjectEntities"]) for row in after}
    if set(before_by) != set(after_by):
        raise ContractError("decoder stages have different row coverage")
    return sum(before_by[key] != after_by[key] for key in before_by)


def reconstruct_chain_incumbent(
        artifacts: Mapping[str, str], output: Path,
) -> tuple[Path, dict[str, Any]]:
    """Re-execute frozen L1--L3 models from the L0 validation graph.

    This deliberately performs inference only.  Training and policy selection
    already happened on the labeled training split; their exact fitted model
    parameters, enabled relations, selected arms, and margins are immutable
    inputs to this development-SOTA replay.
    """
    stage_dir = output / "incumbent"
    stage_dir.mkdir(parents=True, exist_ok=True)
    base_graphs = read_jsonl(Path(artifacts["base_validation_graph"]))
    if len(base_graphs) != EXPECTED_ROWS:
        raise ContractError("L0 graph does not cover the validation split")
    l0_rows = [
        {
            "SubjectEntity": graph["SubjectEntity"],
            "Relation": graph["Relation"],
            "ObjectEntities": list(graph["baseline_objects"]),
        }
        for graph in base_graphs
    ]

    # L1: explicit set cardinality, guarded against the L0 baseline.
    selector = _json(Path(artifacts["candidate_selector"]))
    cardinality_result = _json(Path(artifacts["cardinality_result"]))
    candidate_model = _calibrator(selector["candidate_model"])
    cardinality_model = _cardinality_model(
        cardinality_result["cardinality_model"])
    cardinality_margin = float(cardinality_result["guard_margin"])
    l1_rows, _ = cardinality_prediction_rows(
        base_graphs, candidate_model, cardinality_model, cardinality_margin)
    l1_path = stage_dir / "L1_CARDINALITY.jsonl"
    write_jsonl_atomic(l1_path, l1_rows)
    _require_hash(
        l1_path, Path(artifacts["l1_cardinality_target"]), "L1 cardinality")

    # L2: relation-specific numeric option selection.  The stable policy is
    # encoded by relation in the saved model; disabled relations retain L1.
    numeric_payload = _json(Path(artifacts["numeric_model"]))
    numeric_model = _numeric_model(numeric_payload)
    numeric_replacements: dict[tuple[str, str], list[str]] = {}
    l1_by = {_key(row): row for row in l1_rows}
    for graph in base_graphs:
        relation = str(graph["Relation"])
        if relation not in numeric_model.models:
            continue
        if not bool(numeric_payload["stable_relations"][relation]):
            numeric_replacements[_key(graph)] = list(
                l1_by[_key(graph)]["ObjectEntities"])
            continue
        objects, _ = numeric_model.decode(
            graph, float(numeric_payload["best_mean_margins"][relation]))
        numeric_replacements[_key(graph)] = objects
    l2_rows = _merge_numeric(l1_rows, numeric_replacements)
    l2_path = stage_dir / "L2_NUMERIC.jsonl"
    write_jsonl_atomic(l2_path, l2_rows)
    _require_hash(l2_path, Path(artifacts["l2_numeric_target"]), "L2 numeric")

    # L3: baseline-relative route residual.  Only train-OOF-enabled relation
    # arms execute; every other row is an identity fallback to L2.
    route_graphs = read_jsonl(Path(artifacts["route_validation_graph"]))
    route_by = {_key(row): row for row in route_graphs}
    base_by = {_key(row): row for row in base_graphs}
    if set(route_by) != set(base_by):
        raise ContractError("L0 and route graphs have different row coverage")
    route_payload = _json(Path(artifacts["route_models"]))
    chosen = route_payload["chosen_arm"]
    margins = route_payload["selected_margins"]
    serialized = route_payload["models"]
    l2_by = {_key(row): row for row in l2_rows}
    route_replacements: dict[tuple[str, str], list[str]] = {}
    for key in base_by:
        relation = key[1]
        arm = chosen.get(relation)
        if arm is None:
            continue
        source = base_by if arm == "base_residual" else route_by
        model = _residual_model(serialized[arm][relation])
        objects, _ = decode_route(
            model, source[key], l2_by[key]["ObjectEntities"], arm,
            float(margins[arm][relation]))
        route_replacements[key] = objects
    l3_rows = route_prediction_rows(l2_rows, route_replacements)
    l3_path = stage_dir / "L3_CHAIN_INCUMBENT.jsonl"
    write_jsonl_atomic(l3_path, l3_rows)
    _require_hash(
        l3_path, Path(artifacts["chain_incumbent"]), "L3 route residual")

    report = {
        "schema": "heterogeneous-sota-incumbent-reconstruction-v1",
        "rows": EXPECTED_ROWS,
        "inference_only": True,
        "validation_labels_opened": False,
        "stages": {
            "L0_base_graph": {
                "path": str(Path(artifacts["base_validation_graph"])),
                "sha256": sha256(Path(artifacts["base_validation_graph"])),
            },
            "L1_cardinality": {
                "path": str(l1_path), "sha256": sha256(l1_path),
                "guard_margin": cardinality_margin,
                "changed_rows_vs_L0": _changed_rows(l0_rows, l1_rows),
                "byte_identical": True,
            },
            "L2_numeric": {
                "path": str(l2_path), "sha256": sha256(l2_path),
                "stable_relations": numeric_payload["stable_relations"],
                "margins": numeric_payload["best_mean_margins"],
                "changed_rows_vs_L1": _changed_rows(l1_rows, l2_rows),
                "byte_identical": True,
            },
            "L3_route_residual": {
                "path": str(l3_path), "sha256": sha256(l3_path),
                "chosen_arm": chosen,
                "margins": margins,
                "changed_rows_vs_L2": _changed_rows(l2_rows, l3_rows),
                "byte_identical": True,
            },
        },
    }
    _write_json(stage_dir / "RECONSTRUCTION.json", report)
    return l3_path, report


def audit(args: argparse.Namespace) -> int:
    verified = verify_snapshot()
    output = Path(args.output_dir).resolve()
    report = {
        "schema": "heterogeneous-sota-reproduction-audit-v1",
        "pipeline_id": PIPELINE_ID,
        "snapshot": str(SNAPSHOT.resolve()),
        "snapshot_manifest": str(ARTIFACT_MANIFEST.resolve()),
        "snapshot_manifest_sha256": verified["manifest_sha256"],
        "rows": verified["rows"],
        "verified": True,
        "reproduction_tier": "frozen_model_inference_and_evidence_replay",
        "gpu_regeneration_claimed": False,
    }
    _write_json(output / "audit/INPUT_AUDIT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build(args: argparse.Namespace) -> int:
    verified = verify_snapshot()
    paths = verified["artifacts"]
    output = Path(args.output_dir).resolve()
    chain_incumbent, _ = reconstruct_chain_incumbent(paths, output)
    return unified.build(argparse.Namespace(
        output_dir=str(output),
        typed_graph=paths["typed_graph"],
        cot40_responses=paths["cot40_responses"],
        chain_incumbent=str(chain_incumbent),
        capacity_ledger=paths["capacity_ledger"],
        component_models=paths["component_models"],
        deployed=paths["deployed_predictions"],
        parser_mode=str(args.parser_mode),
    ))


def decode(args: argparse.Namespace) -> int:
    verify_snapshot()
    output = Path(args.output_dir).resolve()
    plan = _json(output / "plan/PLAN.json")
    if plan.get("cot40_parser_mode") != str(args.parser_mode):
        raise ContractError("requested parser mode differs from the frozen plan")
    return unified.decode(argparse.Namespace(output_dir=str(output)))


def _official_scores(prediction_path: Path, gold_path: Path) -> dict[str, Any]:
    predictions = official.read_jsonl_file(prediction_path)
    gold = official.read_jsonl_file(gold_path)
    return official.macro_average_per_relation(
        official.evaluate_per_sr_pair(
            predictions, gold, official.RELATION_TYPE))


def verify(args: argparse.Namespace) -> int:
    verified = verify_snapshot()
    output = Path(args.output_dir).resolve()
    plan = _json(output / "plan/PLAN.json")
    parser_mode = str(plan.get("cot40_parser_mode"))
    if parser_mode != str(args.parser_mode):
        raise ContractError("requested parser mode differs from the frozen plan")
    unified_status = unified.verify(argparse.Namespace(
        output_dir=str(output),
        gold=verified["external"]["validation_gold"],
        deployed_decisions=verified["artifacts"]["deployed_decisions"],
    ))
    parity = _json(output / "analysis/PARITY.json")
    prediction_path = output / "VALIDATION_PREDICTIONS.jsonl"
    scores = _official_scores(
        prediction_path,
        Path(verified["external"]["validation_gold"]),
    )
    score = float(scores[POOLED]["macro-f1"])
    if score != EXPECTED_SCORE:
        raise ContractError(
            f"official score drift: expected {EXPECTED_SCORE}, got {score}")

    exact_required = parser_mode == "legacy-20260729"
    graph_path = output / "graph/UNIFIED_VALIDATION_GRAPH.jsonl"
    if exact_required and (
        unified_status != 0
        or parity.get("byte_identical") is not True
        or int(parity.get("divergent_rows", -1)) != 0
        or sha256(graph_path) != EXPECTED_LEGACY_GRAPH_SHA256
        or sha256(prediction_path)
        != verified["manifest"]["artifacts"]["deployed_predictions"]["sha256"]
    ):
        raise ContractError("legacy compatibility failed byte-level parity")
    if not exact_required and (
        int(parity.get("divergences_unexplained", -1)) != 0
        or float(parity.get("score_delta", 1.0)) != 0.0
    ):
        raise ContractError("corrected parser has unexplained parity drift")

    result = {
        "schema": RESULT_SCHEMA,
        "pipeline_id": PIPELINE_ID,
        "parser_mode": parser_mode,
        "reproduction_tier": "frozen_model_inference_and_evidence_replay",
        "gpu_regeneration_claimed": False,
        "rows": EXPECTED_ROWS,
        "prediction": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "unified_graph": str(graph_path),
        "unified_graph_sha256": sha256(graph_path),
        "byte_identical_to_original": bool(parity["byte_identical"]),
        "divergent_rows": int(parity["divergent_rows"]),
        "unexplained_divergences": int(parity["divergences_unexplained"]),
        "official_pooled_macro_f1": score,
        "official_per_relation": scores,
        "snapshot_manifest": str(ARTIFACT_MANIFEST.resolve()),
        "snapshot_manifest_sha256": verified["manifest_sha256"],
        "validation_selected_lineage": True,
        "blind_safe": False,
    }
    _write_json(output / "REPRODUCTION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def all_stages(args: argparse.Namespace) -> int:
    audit(args)
    build(args)
    decode(args)
    return verify(args)


def status(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    names = (
        "audit/INPUT_AUDIT.json",
        "incumbent/L1_CARDINALITY.jsonl",
        "incumbent/L2_NUMERIC.jsonl",
        "incumbent/L3_CHAIN_INCUMBENT.jsonl",
        "incumbent/RECONSTRUCTION.json",
        "plan/PLAN.json",
        "graph/UNIFIED_VALIDATION_GRAPH.jsonl",
        "VALIDATION_PREDICTIONS.jsonl",
        "DECISIONS.jsonl",
        "analysis/PARITY.json",
        "REPRODUCTION.json",
    )
    for name in names:
        path = output / name
        print(f"{name}: {'ready' if path.is_file() else 'pending'}")
    result_path = output / "REPRODUCTION.json"
    if result_path.is_file():
        result = _json(result_path)
        print(
            f"score={result['official_pooled_macro_f1']:.6f} "
            f"byte_identical={result['byte_identical_to_original']} "
            f"parser={result['parser_mode']}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    for command, function in (
        ("audit", audit),
        ("build", build),
        ("decode", decode),
        ("verify", verify),
        ("all", all_stages),
        ("status", status),
    ):
        current = subparsers.add_parser(command)
        current.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
        if command not in {"audit", "status"}:
            current.add_argument(
                "--parser-mode", choices=unified.PARSER_MODES,
                default="legacy-20260729")
        current.set_defaults(function=function)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not hasattr(args, "parser_mode"):
        args.parser_mode = "legacy-20260729"
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
