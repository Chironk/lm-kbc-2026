#!/usr/bin/env python3
"""Reproducible validation pipeline with one Ministral SyntheticCoT route.

The Qwen primary route, Gemma route, frozen staged decoder, and final graph
correction are unchanged.  The obsolete zero-shot Ministral N=3 route is not
read.  The retained N=10 SyntheticCoT route supplies the graph and, for
``hasArea``, its unique component with support in at least seven samples may
replace the staged incumbent.

This module deliberately separates inference from replay.  A completed run
directory contains task files, response files, and response manifests.  The
``all`` command below deterministically rebuilds the graph, predictions,
official validation score, and ZIP from those frozen inference artifacts.
It never reads validation labels until the explicit ``score`` command.
"""
from __future__ import annotations

import argparse
import copy
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.heterogeneous_agents import end_to_end_pipeline as e2e
from experiments.heterogeneous_agents import final_submission_pipeline as final
from experiments.heterogeneous_agents.core import (
    ContractError,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    augment_relational_graph,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "experiments/heterogeneous_agents/runs/"
    "single_ministral_validation_20260810_v1"
)
POLICY_SCHEMA = "single-ministral-cot40-policy-v1"
RESULT_SCHEMA = "single-ministral-cot40-validation-result-v1"
MANIFEST_SCHEMA = "single-ministral-cot40-manifest-v1"
POLICY_ID = "heterogeneous_single_ministral_cot40_component_v1"
AREA = "hasArea"
SUPPORT = 7
ACTIVE_GENERATED_ROUTES = (
    "gemma:independent",
    e2e.MINISTRAL_COT40,
)
IMPLEMENTATION_DEPENDENCIES = {
    **final.DECODER_IMPLEMENTATIONS,
    "official_evaluator": ROOT / "evaluate.py",
}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def freeze(args: argparse.Namespace) -> int:
    """Freeze the single-route policy before model responses are consumed."""
    output = Path(args.output_dir).resolve()
    plan = e2e._validate_plan(output)
    snapshot, model_paths = final._snapshot_artifacts()
    if str(plan.get("split")) != "validation" or bool(plan.get("blind")):
        raise ContractError("single-Ministral validation requires validation plan")
    required_jobs = set(ACTIVE_GENERATED_ROUTES)
    if not required_jobs.issubset(plan.get("jobs", {})):
        raise ContractError("validation plan lacks a required active route")
    policy = {
        "schema": POLICY_SCHEMA,
        "policy_id": POLICY_ID,
        "split": "validation",
        "contains_labels": False,
        "gold_aware": False,
        "source_plan": str((output / "plan/PLAN.json").resolve()),
        "source_plan_sha256": sha256(output / "plan/PLAN.json"),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "implementation_dependencies": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in IMPLEMENTATION_DEPENDENCIES.items()
        },
        "primary_runner": str((ROOT / "run_submission.py").resolve()),
        "primary_runner_sha256": sha256(ROOT / "run_submission.py"),
        "primary_policy": final.PRIMARY_POLICY,
        "primary_seed_scheme": str(args.primary_seed_scheme),
        "active_evidence_routes": [
            "qwen:self_consistency",
            "gemma:independent",
            e2e.MINISTRAL_COT40,
        ],
        "inactive_route": e2e.MINISTRAL_N3,
        "area_rule": {
            "route": e2e.MINISTRAL_COT40,
            "minimum_distinct_generation_support": SUPPORT,
            "numeric_component_tolerance": 0.05,
            "unique_winner_required": True,
        },
        "snapshot_manifest": str(final.SNAPSHOT_MANIFEST.resolve()),
        "snapshot_manifest_sha256": sha256(final.SNAPSHOT_MANIFEST),
        "model_artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in model_paths.items()
        },
        "model_portfolio": snapshot["model_portfolio"],
        "verified_parameter_total": int(plan["verified_parameter_total"]),
        "parameter_cap": int(plan["parameter_cap"]),
        "replay_contract": (
            "task, response, model, implementation, graph, prediction, and "
            "package hashes are verified; GPU kernel determinism is not claimed"
        ),
    }
    if policy["primary_seed_scheme"] not in final.PRIMARY_SEED_SCHEMES:
        raise ContractError("invalid primary seed scheme")
    path = output / "plan/SINGLE_MINISTRAL_POLICY.json"
    _write_json(path, policy)
    print(json.dumps({
        "policy": str(path),
        "policy_sha256": sha256(path),
        "active_evidence_routes": policy["active_evidence_routes"],
        "inactive_route": policy["inactive_route"],
    }, indent=2, sort_keys=True))
    return 0


def _validate_policy(
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    plan = e2e._validate_plan(output)
    policy_path = output / "plan/SINGLE_MINISTRAL_POLICY.json"
    policy = _json(policy_path)
    _, model_paths = final._snapshot_artifacts()
    if (
        policy.get("schema") != POLICY_SCHEMA
        or policy.get("policy_id") != POLICY_ID
        or policy.get("split") != "validation"
        or bool(policy.get("contains_labels"))
        or bool(policy.get("gold_aware"))
        or policy.get("source_plan_sha256") != sha256(output / "plan/PLAN.json")
        or policy.get("implementation_sha256") != sha256(Path(__file__).resolve())
        or policy.get("primary_runner_sha256") != sha256(ROOT / "run_submission.py")
        or policy.get("snapshot_manifest_sha256")
            != sha256(final.SNAPSHOT_MANIFEST)
        or policy.get("active_evidence_routes") != [
            "qwen:self_consistency",
            "gemma:independent",
            e2e.MINISTRAL_COT40,
        ]
        or policy.get("inactive_route") != e2e.MINISTRAL_N3
    ):
        raise ContractError("single-Ministral frozen policy contract failed")
    for name, path in model_paths.items():
        if policy.get("model_artifacts", {}).get(name, {}).get("sha256") != sha256(path):
            raise ContractError(f"frozen model artifact changed: {name}")
    for name, path in IMPLEMENTATION_DEPENDENCIES.items():
        record = policy.get("implementation_dependencies", {}).get(name, {})
        if (
            record.get("path") != str(path.resolve())
            or record.get("sha256") != sha256(path)
        ):
            raise ContractError(f"decoder implementation changed: {name}")
    return policy, plan, model_paths


def _single_route_graph(
    base: Mapping[str, Any],
    *,
    qwen_texts: Sequence[str],
    gemma_texts: Sequence[str],
    ministral_cot40: Mapping[str, Any],
) -> dict[str, Any]:
    graph = copy.deepcopy(dict(base))
    e2e._attach_supply_route(
        graph,
        ministral_cot40,
        route_name=e2e.MINISTRAL_COT40,
        samples=10,
    )
    graph.pop("relational_graph", None)
    graph.pop("relational_graph_schema", None)
    graph = augment_relational_graph(graph)
    final._replace_route_events(
        graph,
        route="qwen:self_consistency",
        family=e2e.QWEN,
        records=final._qwen_records(graph, qwen_texts),
        raw_texts=qwen_texts,
        provenance="frozen_split_generation",
    )
    final._replace_route_events(
        graph,
        route="gemma:independent",
        family=e2e.GEMMA,
        records=[final._generic_record(graph, text) for text in gemma_texts],
        raw_texts=gemma_texts,
        provenance="frozen_split_generation",
    )
    cot40_texts = [str(value) for value in ministral_cot40["generations"]]
    final._replace_route_events(
        graph,
        route=e2e.MINISTRAL_COT40,
        family=e2e.MINISTRAL,
        records=[final._generic_record(graph, text) for text in cot40_texts],
        raw_texts=cot40_texts,
        provenance="frozen_split_generation",
    )
    final._state_and_relation_edges(graph)
    graph["schema"] = final.GRAPH_SCHEMA
    graph["contains_labels"] = False
    graph["gold_aware"] = False
    _assert_no_n3(graph)
    return graph


def _assert_no_n3(graph: Mapping[str, Any]) -> None:
    if e2e.MINISTRAL_N3 in graph.get("proposal_routes", {}):
        raise ContractError(f"{_key(graph)}: inactive N=3 proposal route survived")
    for field in ("candidates", "dormant_candidates"):
        if any(e2e.MINISTRAL_N3 in node.get("routes", {})
               for node in graph.get(field, [])):
            raise ContractError(f"{_key(graph)}: inactive N=3 candidate survived")
    if any(
        str(node.get("route")) == e2e.MINISTRAL_N3
        for node in graph.get("relational_graph", {}).get("nodes", [])
    ):
        raise ContractError(f"{_key(graph)}: inactive N=3 event survived")


def _component_area(
    graph: Mapping[str, Any], incumbent: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    if str(graph["Relation"]) != AREA:
        return list(incumbent), {"applied": False, "reason": "out_of_scope"}
    components: list[tuple[int, str]] = []
    for node in graph.get("relational_graph", {}).get("nodes", []):
        if node.get("node_type") != "candidate_component":
            continue
        route = node.get("routes", {}).get(e2e.MINISTRAL_COT40)
        if isinstance(route, Mapping):
            components.append((
                int(route.get("distinct_generation_support", 0)),
                str(node["representative"]),
            ))
    if not components:
        return list(incumbent), {
            "applied": False,
            "reason": "no_cot40_numeric_component",
        }
    highest = max(value for value, _ in components)
    winners = [item for value, item in components
               if value == highest and value >= SUPPORT]
    if len(winners) != 1:
        return list(incumbent), {
            "applied": False,
            "reason": "no_unique_7_of_10_component",
            "highest_support": highest,
            "winner_count": len(winners),
        }
    selected = [winners[0]]
    return selected, {
        "applied": selected != list(incumbent),
        "reason": "unique_7_of_10_numeric_component",
        "highest_support": highest,
        "selected": selected,
    }


def build(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    policy, plan, model_paths = _validate_policy(output)
    primary, qwen_raw, system2 = final._primary_inputs(output, plan, policy)
    base_rows = final._assemble_from_primary(output, plan, primary, qwen_raw)
    gemma = final._response_map(plan, "gemma:independent")
    ministral = final._response_map(plan, e2e.MINISTRAL_COT40)

    decision_graphs: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for source in base_rows:
        key = _key(source)
        base, qwen_texts, gemma_texts = final._prepare_base_row(
            source,
            {"generations": list(qwen_raw[key])},
            gemma[key],
            primary_objects=primary[key],
            system2_objects=system2.get(key, ()),
        )
        graphs.append(_single_route_graph(
            base,
            qwen_texts=qwen_texts,
            gemma_texts=gemma_texts,
            ministral_cot40=ministral[key],
        ))
        decision_graphs.append(base)

    # Reuse all frozen stages.  Because N=3 is absent, its area stage is an
    # identity operation.  We then replace only the area CoT40 exact-surface
    # decision with the predeclared numeric-component decision.
    predictions, decisions = final._apply_frozen_stack(
        decision_graphs, graphs, model_paths)
    prediction_by = {_key(row): dict(row) for row in predictions}
    area_graphs: list[dict[str, Any]] = []
    area_incumbents: dict[tuple[str, str], list[str]] = {}
    component_details: dict[tuple[str, str], dict[str, Any]] = {}
    for graph, decision in zip(graphs, decisions, strict=True):
        if str(graph["Relation"]) != AREA:
            continue
        key = _key(graph)
        layers = list(decision["layers"])
        # CoT40 is the final staged layer.  Its detail payload intentionally
        # records the short policy name ``two_thirds`` and therefore replaces
        # the stack label in the flattened trace.
        if not layers or layers[-1].get("policy") != "two_thirds":
            raise ContractError(f"{key}: missing CoT40 staged decision")
        before = [str(value) for value in layers[-1]["before"]]
        area_incumbents[key], component_details[key] = _component_area(
            graph, before)
        area_graphs.append(graph)
    area_predictions, area_proof = final._apply_relation_typed_graph_correction(
        area_graphs, area_incumbents)
    proof_by = {_key(row): detail for row, detail in zip(
        area_predictions, area_proof, strict=True)}
    for row in area_predictions:
        prediction_by[_key(row)] = dict(row)
    ordered = [prediction_by[_key(graph)] for graph in graphs]
    for decision in decisions:
        key = _key(decision)
        if key in component_details:
            decision["single_ministral_component_area"] = component_details[key]
            decision["proof"] = proof_by[key]
            decision["prediction"] = list(prediction_by[key]["ObjectEntities"])

    graph_path = output / "single_ministral/GRAPH.jsonl"
    prediction_path = output / "single_ministral/PREDICTIONS.jsonl"
    decision_path = output / "single_ministral/DECISIONS.jsonl"
    write_jsonl_atomic(graph_path, graphs)
    write_jsonl_atomic(prediction_path, ordered)
    write_jsonl_atomic(decision_path, decisions)

    response_artifacts = {}
    for route in ACTIVE_GENERATED_ROUTES:
        job = plan["jobs"][route]
        response = Path(job["response_path"])
        manifest = response.with_suffix(response.suffix + ".manifest.json")
        # _response_map already verified these, but pin both bytes explicitly.
        response_artifacts[route] = {
            "task_sha256": job["task_sha256"],
            "response": str(response.resolve()),
            "response_sha256": sha256(response),
            "response_manifest": str(manifest.resolve()),
            "response_manifest_sha256": sha256(manifest),
        }
    primary_manifest = output / "primary_qwen/MANIFEST.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "policy_id": POLICY_ID,
        "split": "validation",
        "contains_labels": False,
        "gold_aware": False,
        "rows": len(ordered),
        "verified_parameter_total": policy["verified_parameter_total"],
        "parameter_cap": policy["parameter_cap"],
        "active_evidence_routes": policy["active_evidence_routes"],
        "inactive_route": e2e.MINISTRAL_N3,
        "policy": str((output / "plan/SINGLE_MINISTRAL_POLICY.json").resolve()),
        "policy_sha256": sha256(output / "plan/SINGLE_MINISTRAL_POLICY.json"),
        "primary_manifest": str(primary_manifest.resolve()),
        "primary_manifest_sha256": sha256(primary_manifest),
        "response_artifacts": response_artifacts,
        "graph": str(graph_path.resolve()),
        "graph_sha256": sha256(graph_path),
        "predictions": str(prediction_path.resolve()),
        "predictions_sha256": sha256(prediction_path),
        "decisions": str(decision_path.resolve()),
        "decisions_sha256": sha256(decision_path),
        "component_area_replacements": sum(
            bool(value.get("applied")) for value in component_details.values()),
    }
    _write_json(output / "single_ministral/MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def score(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    _validate_policy(output)
    prediction_path = output / "single_ministral/PREDICTIONS.jsonl"
    rows = read_jsonl(prediction_path)
    scores = final._score(rows, Path(args.gold).resolve())
    result = {
        "schema": RESULT_SCHEMA,
        "development_only": True,
        "contains_labels": True,
        "gold_aware_evaluation": True,
        "rows": len(rows),
        "predictions_sha256": sha256(prediction_path),
        "pooled_macro_f1": scores[final.POOLED]["macro-f1"],
        "per_relation": scores,
    }
    _write_json(output / "single_ministral/RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _write_deterministic_zip(archive: Path, predictions: bytes) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("predictions.jsonl", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, predictions)


def package(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    policy, plan, _ = _validate_policy(output)
    manifest = _json(output / "single_ministral/MANIFEST.json")
    predictions = output / "single_ministral/PREDICTIONS.jsonl"
    rows = read_jsonl(predictions)
    source = read_jsonl(Path(plan["input_rows"]))
    if (
        manifest.get("predictions_sha256") != sha256(predictions)
        or len(rows) != len(source)
        or [_key(row) for row in rows] != [_key(row) for row in source]
    ):
        raise ContractError("single-Ministral package coverage contract failed")
    package_dir = output / "single_ministral/submission"
    staged = package_dir / "predictions.jsonl"
    package_dir.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(predictions.read_bytes())
    archive = package_dir / f"{POLICY_ID}_validation.zip"
    _write_deterministic_zip(archive, predictions.read_bytes())
    package_manifest = {
        "schema": "single-ministral-cot40-package-v1",
        "split": policy["split"],
        "rows": len(rows),
        "member": "predictions.jsonl",
        "predictions_sha256": sha256(predictions),
        "archive": str(archive.resolve()),
        "archive_sha256": sha256(archive),
    }
    _write_json(package_dir / "PACKAGE.json", package_manifest)
    print(json.dumps(package_manifest, indent=2, sort_keys=True))
    return 0


def verify(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    policy, plan, _ = _validate_policy(output)
    manifest = _json(output / "single_ministral/MANIFEST.json")
    result = _json(output / "single_ministral/RESULT.json")
    package_manifest = _json(output / "single_ministral/submission/PACKAGE.json")
    predictions = output / "single_ministral/PREDICTIONS.jsonl"
    archive = Path(package_manifest["archive"])
    if (
        policy.get("inactive_route") != e2e.MINISTRAL_N3
        or manifest.get("inactive_route") != e2e.MINISTRAL_N3
        or manifest.get("predictions_sha256") != sha256(predictions)
        or result.get("predictions_sha256") != sha256(predictions)
        or package_manifest.get("predictions_sha256") != sha256(predictions)
        or package_manifest.get("archive_sha256") != sha256(archive)
        or int(package_manifest.get("rows", -1)) != int(plan["rows"])
    ):
        raise ContractError("single-Ministral final verification failed")
    for graph in read_jsonl(Path(manifest["graph"])):
        _assert_no_n3(graph)
    with zipfile.ZipFile(archive) as handle:
        if handle.namelist() != ["predictions.jsonl"]:
            raise ContractError("ZIP must contain one root predictions.jsonl")
        if handle.read("predictions.jsonl") != predictions.read_bytes():
            raise ContractError("ZIP prediction bytes differ")
    verification = {
        "verified": True,
        "policy_id": POLICY_ID,
        "split": "validation",
        "rows": int(plan["rows"]),
        "pooled_macro_f1": result["pooled_macro_f1"],
        "predictions_sha256": sha256(predictions),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "active_evidence_routes": policy["active_evidence_routes"],
        "inactive_route": policy["inactive_route"],
    }
    _write_json(output / "single_ministral/VERIFICATION.json", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


def all_stages(args: argparse.Namespace) -> int:
    build(args)
    score(args)
    package(args)
    return verify(args)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    freeze_parser.add_argument(
        "--primary-seed-scheme",
        choices=sorted(final.PRIMARY_SEED_SCHEMES),
        default="stable-key",
    )
    freeze_parser.set_defaults(function=freeze)
    for name, function in (
        ("build", build),
        ("package", package),
        ("verify", verify),
        ("all", all_stages),
    ):
        current = subparsers.add_parser(name)
        current.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
        current.add_argument("--gold", default=str(ROOT / "data/val.jsonl"))
        current.set_defaults(function=function)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    score_parser.add_argument("--gold", default=str(ROOT / "data/val.jsonl"))
    score_parser.set_defaults(function=score)
    return value


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
