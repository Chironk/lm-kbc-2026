#!/usr/bin/env python3
"""Paired official-test runner for the archived 0.4845 SOTA lineage.

This module has two deliberately separate outputs built from one set of fresh
model responses:

``historical_control``
    Reconstructs the decoder frozen in
    ``heterogeneous_final_strict_proof_20260803_v1``.  It uses the legacy
    Qwen v0495 seed scheme, the faithful SyntheticCoT pool, the zero-shot
    Ministral N=3 area rule, the CoT-5/N=10 Ministral rule, and the generic
    strict set proof.  It is the architecture that produced the archived
    official-test submission reported as 0.4845.

``paper_single_ministral``
    Holds that architecture fixed, removes only the zero-shot Ministral N=3
    route, and applies the already train-tested 7/10 rule to the N=10
    Ministral numeric component for ``hasArea``.  All other relations and
    decoder stages are identical to the historical control.

The archived sampled responses were not retained, so fresh stochastic model
inference is not claimed to reproduce the old prediction bytes.  The exact
submitted zip is retained as a read-only audit target: it is compared only
after decoding and is never used to choose or modify an answer.
"""
from __future__ import annotations

import argparse
import copy
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from sample_evidence import classify_samples

from experiments.heterogeneous_agents import end_to_end_pipeline as e2e
from experiments.heterogeneous_agents import final_submission_pipeline as current
from experiments.heterogeneous_agents.assemble_and_audit import (
    prediction_for_agent,
)
from experiments.heterogeneous_agents.components.proof_carrying_graph_decoder import (
    IDENTITY_RELATIONS,
    PRIMARY_ARM,
    _decode as proof_decode,
)
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    augment_relational_graph,
)
from experiments.heterogeneous_agents.components.route_aware_candidate_graph import (
    augment_graph,
)
from experiments.heterogeneous_agents.core import (
    ContractError,
    canonical_key,
    proposal_parse_status,
    read_jsonl,
    sha256,
    write_jsonl_atomic,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

DEFAULT_OUTPUT = (
    HERE / "runs/historical_sota_single_ministral_test_20260810_v1"
)
DEFAULT_INPUT = ROOT / "data/test.jsonl"
DEFAULT_SYNTHETIC = ROOT / "data/synthetic_cot_faithful.jsonl"
DEFAULT_COT_AGENTS = ROOT / "configs/final/portfolio_cot.json"
DEFAULT_SUPPLY_AGENTS = ROOT / "configs/final/portfolio_supply.json"
ARCHIVED_SUBMISSION = (
    ROOT / "submissions/official_test/"
    "heterogeneous_final_strict_proof_20260803_v1_test.zip"
)
ARCHIVED_MANIFEST = ARCHIVED_SUBMISSION.with_suffix(".manifest.json")

HISTORICAL_POLICY_ID = "heterogeneous_final_strict_proof_20260803_v1"
PAPER_POLICY_ID = "heterogeneous_final_single_ministral_n10_20260810_v1"
POLICY_SCHEMA = "heterogeneous-historical-sota-paired-policy-v1"
RESULT_SCHEMA = "heterogeneous-historical-sota-paired-result-v1"
GRAPH_SCHEMA = "heterogeneous-historical-sota-exact-evidence-graph-v1"

EXPECTED_TEST_SHA256 = (
    "67c31c8388c585634df55500612f522ad42da6735d4c89eb59a9ef5a39f043f1"
)
EXPECTED_TEST_ROWS = 475
EXPECTED_SYNTHETIC_SHA256 = (
    "72f9974c355dd98eab9d13e61a6b2e120a8e9fcc40e39fb8251b54ab8d01aacb"
)
EXPECTED_COT_CONFIG_SHA256 = (
    "c40d26d710c8aee2a94317c0bec9540691c05a6cce6771fa4c7fdecaab79822d"
)
EXPECTED_SUPPLY_CONFIG_SHA256 = (
    "601b943e19611883dd60c761ce1ac28d8e254edc4f52e550cb5788d627e93802"
)
EXPECTED_ARCHIVE_SHA256 = (
    "3f73d01fe5d4b3c9b9cc7e2f5dba8348d0e1fec19fc0ddb797ff2e0f460b11e4"
)
EXPECTED_ARCHIVED_MEMBER_SHA256 = (
    "73621130839b572a7fdfdc2f8a58c4bf3f00beece4be86ff4a7874c96b63bb53"
)
HISTORICAL_SOURCE_COMMIT = "130b9a0c02ba0d190f9b33cd61cd74156daf6625"
HISTORICAL_SOURCE_BLOB = "e199d6c1a09deb41e291d8be7f74933283cd3e73"

MODEL_ARTIFACTS = current.MODEL_ARTIFACTS
POOLED = "*** All Relations ***"
IMPLEMENTATIONS = {
    "paired_pipeline": Path(__file__).resolve(),
    "end_to_end_planner": Path(e2e.__file__).resolve(),
    "current_final_helpers": Path(current.__file__).resolve(),
    "agent_runner": HERE / "run_agent.py",
    "core_prompts_and_contracts": HERE / "core.py",
    "graph_assembler": HERE / "assemble_and_audit.py",
    "route_graph": HERE / "components/route_aware_candidate_graph.py",
    "relational_graph": HERE / "components/relational_candidate_graph.py",
    "cardinality_decoder": HERE / "components/explicit_cardinality_ablation.py",
    "numeric_decoder": HERE / "components/relation_specific_numeric_decoder.py",
    "route_decoder": HERE / "components/baseline_relative_route_decoder.py",
    "staged_decoder": HERE / "components/unified_graph_decoder.py",
    "set_proof": HERE / "components/proof_carrying_graph_decoder.py",
    "event_graph": HERE / "components/cot40_evidence_edge_ablation.py",
    "model_deserializer": HERE / "frozen_model_loader.py",
    "primary_orchestrator": ROOT / "run_submission.py",
    "primary_inference": ROOT / "run_inference.py",
    "primary_artifact_contract": ROOT / "artifact_contract.py",
}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _archived_prediction_bytes() -> bytes:
    if sha256(ARCHIVED_SUBMISSION) != EXPECTED_ARCHIVE_SHA256:
        raise ContractError("archived official-test zip hash drift")
    manifest = _json(ARCHIVED_MANIFEST)
    if (
        manifest.get("policy_id") != HISTORICAL_POLICY_ID
        or manifest.get("member") != "predictions.jsonl"
        or manifest.get("member_sha256") != EXPECTED_ARCHIVED_MEMBER_SHA256
    ):
        raise ContractError("archived official-test manifest drift")
    with zipfile.ZipFile(ARCHIVED_SUBMISSION) as handle:
        if handle.namelist() != ["predictions.jsonl"]:
            raise ContractError("archived zip member contract drift")
        payload = handle.read("predictions.jsonl")
    import hashlib
    if hashlib.sha256(payload).hexdigest() != EXPECTED_ARCHIVED_MEMBER_SHA256:
        raise ContractError("archived prediction member hash drift")
    if len([line for line in payload.splitlines() if line]) != EXPECTED_TEST_ROWS:
        raise ContractError("archived prediction row count drift")
    return payload


def _validate_plan_contract(output: Path) -> dict[str, Any]:
    plan = e2e._validate_plan(output)
    required_jobs = {
        "qwen:self_consistency": (
            e2e.QWEN, 10,
            ("commit_existence", "commit_cardinality", "propose"),
            5, None,
        ),
        "gemma:independent": (
            e2e.GEMMA, 1,
            ("commit_existence", "commit_cardinality", "propose"),
            5, 20,
        ),
        e2e.MINISTRAL_N3: (
            e2e.MINISTRAL, 3, ("propose",), 0, 20,
        ),
        e2e.MINISTRAL_COT40: (
            e2e.MINISTRAL, 10, ("propose",), 5, 40,
        ),
    }
    if (
        plan.get("split") != "test"
        or not bool(plan.get("blind"))
        or int(plan.get("rows", -1)) != EXPECTED_TEST_ROWS
        or plan.get("input_sha256") != EXPECTED_TEST_SHA256
        or plan.get("synthetic_cot_sha256") != EXPECTED_SYNTHETIC_SHA256
        or plan.get("cot_agents_sha256") != EXPECTED_COT_CONFIG_SHA256
        or plan.get("supply_agents_sha256") != EXPECTED_SUPPLY_CONFIG_SHA256
        or plan.get("question_contract") != "legacy"
        or int(plan.get("seed", -1)) != e2e.SEED
        or int(plan.get("verified_parameter_total", -1))
            != e2e.EXPECTED_PARAMETER_TOTAL
    ):
        raise ContractError("historical SOTA test-plan contract failed")
    for name, (agent, samples, phases, shots, reasoning_words) in required_jobs.items():
        job = plan.get("jobs", {}).get(name, {})
        if (
            job.get("agent_id") != agent
            or int(job.get("n_proposals", -1)) != samples
            or tuple(job.get("phases", ())) != phases
            or int(job.get("synthetic_shots", -1)) != shots
            or job.get("reasoning_words") != reasoning_words
        ):
            raise ContractError(f"historical route contract failed: {name}")
        tasks = read_jsonl(Path(job["task_path"]))
        expected_tasks = EXPECTED_TEST_ROWS * len(phases)
        if len(tasks) != expected_tasks:
            raise ContractError(f"historical task-count contract failed: {name}")
        proposal_tasks = [task for task in tasks if task.get("phase") == "propose"]
        if len(proposal_tasks) != EXPECTED_TEST_ROWS or any(
            int(task.get("n_samples", -1)) != samples for task in proposal_tasks
        ):
            raise ContractError(f"historical proposal-sampling contract failed: {name}")
        for task in proposal_tasks:
            prompt = str(task.get("prompt", ""))
            subjects = [str(value) for value in task.get("shot_subjects", [])]
            if len(subjects) != shots or str(task["subject"]) in subjects:
                raise ContractError(f"historical shot-assignment contract failed: {name}")
            if shots and "PRIVATE RECALL DEMONSTRATIONS:" not in prompt:
                raise ContractError(f"historical demonstrations missing: {name}")
            if not shots and "PRIVATE RECALL DEMONSTRATIONS:" in prompt:
                raise ContractError(f"unexpected demonstrations: {name}")
            if reasoning_words is not None and (
                f"at most {reasoning_words} words" not in prompt
            ):
                raise ContractError(f"historical reasoning-budget contract failed: {name}")
    return plan


def plan(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    if sha256(Path(args.input).resolve()) != EXPECTED_TEST_SHA256:
        raise ContractError("refusing noncanonical official-test input")
    e2e.plan(argparse.Namespace(
        split="test",
        input=str(Path(args.input).resolve()),
        output_dir=str(output),
        cot_agents=str(DEFAULT_COT_AGENTS),
        supply_agents=str(DEFAULT_SUPPLY_AGENTS),
        synthetic_cot=str(DEFAULT_SYNTHETIC),
        question_contract="legacy",
        component_models=str(current.SNAPSHOT / "component_models.json"),
    ))
    source = _validate_plan_contract(output)
    snapshot, model_paths = current._snapshot_artifacts()
    _archived_prediction_bytes()
    policy = {
        "schema": POLICY_SCHEMA,
        "historical_policy_id": HISTORICAL_POLICY_ID,
        "paper_policy_id": PAPER_POLICY_ID,
        "split": "test",
        "blind": True,
        "contains_labels": False,
        "gold_aware": False,
        "source_plan": str((output / "plan/PLAN.json").resolve()),
        "source_plan_sha256": sha256(output / "plan/PLAN.json"),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "implementation_dependencies": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in IMPLEMENTATIONS.items()
        },
        "historical_source_commit": HISTORICAL_SOURCE_COMMIT,
        "historical_source_blob": HISTORICAL_SOURCE_BLOB,
        "primary_qwen_policy": current.PRIMARY_POLICY,
        "primary_qwen_seed_scheme": "legacy",
        "question_contract": "legacy",
        "synthetic_cot_sha256": EXPECTED_SYNTHETIC_SHA256,
        "archived_submission": str(ARCHIVED_SUBMISSION.resolve()),
        "archived_submission_sha256": EXPECTED_ARCHIVE_SHA256,
        "archived_member_sha256": EXPECTED_ARCHIVED_MEMBER_SHA256,
        "model_portfolio": snapshot["model_portfolio"],
        "model_artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in model_paths.items()
        },
        "verified_parameter_total": source["verified_parameter_total"],
        "parameter_cap": source["parameter_cap"],
        "historical_decoder_order": [
            "qwen_v0495_incumbent",
            "explicit_cardinality",
            "relation_specific_numeric",
            "route_residual",
            "component_surface_residual",
            "ministral_zero_shot_n3_area_unanimity",
            "ministral_cot5_n10_support_7",
            "generic_strict_set_proof",
        ],
        "paper_change_only": {
            "removed": "ministral_zero_shot_n3_area_unanimity",
            "replacement": (
                "ministral_cot5_n10_unique_numeric_component_support_7"
            ),
            "relations_changed": ["hasArea"],
        },
        "fresh_inference_exact_byte_claim": False,
        "reason": "historical sampled intermediate responses were not retained",
    }
    _write_json(output / "plan/HISTORICAL_SOTA_POLICY.json", policy)
    print(json.dumps({
        "policy": str(output / "plan/HISTORICAL_SOTA_POLICY.json"),
        "rows": source["rows"],
        "parameters": source["verified_parameter_total"],
        "historical_control": HISTORICAL_POLICY_ID,
        "paper_policy": PAPER_POLICY_ID,
    }, indent=2, sort_keys=True))
    return 0


def _validate_policy(output: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    policy_path = output / "plan/HISTORICAL_SOTA_POLICY.json"
    policy = _json(policy_path)
    source = _validate_plan_contract(output)
    _, model_paths = current._snapshot_artifacts()
    if (
        policy.get("schema") != POLICY_SCHEMA
        or policy.get("historical_policy_id") != HISTORICAL_POLICY_ID
        or policy.get("paper_policy_id") != PAPER_POLICY_ID
        or policy.get("source_plan_sha256") != sha256(output / "plan/PLAN.json")
        or policy.get("implementation_sha256") != sha256(Path(__file__).resolve())
        or policy.get("primary_qwen_seed_scheme") != "legacy"
        or policy.get("archived_submission_sha256") != EXPECTED_ARCHIVE_SHA256
        or source.get("split") != "test"
    ):
        raise ContractError("paired historical-SOTA policy contract failed")
    for name, path in model_paths.items():
        record = policy.get("model_artifacts", {}).get(name, {})
        if record.get("path") != str(path.resolve()) or record.get("sha256") != sha256(path):
            raise ContractError(f"frozen model artifact drift: {name}")
    for name, path in IMPLEMENTATIONS.items():
        record = policy.get("implementation_dependencies", {}).get(name, {})
        if record.get("path") != str(path.resolve()) or record.get("sha256") != sha256(path):
            raise ContractError(f"implementation dependency drift: {name}")
    _archived_prediction_bytes()
    return policy, model_paths


def _add_exact_base_candidates(
    row: dict[str, Any],
    *,
    agent_id: str,
    raw_texts: Sequence[str],
    parser: Callable[[str, str], Sequence[str]],
    selected_objects: Sequence[str],
) -> None:
    """Historical exact-candidate recovery (one count per generation)."""
    relation = str(row["Relation"])
    selected = {canonical_key(str(value), relation) for value in selected_objects}
    occurrences: Counter[str] = Counter()
    displays: dict[str, str] = {}
    for text in raw_texts:
        seen: set[str] = set()
        for item in parser(str(text), relation):
            key = canonical_key(str(item), relation)
            if not key or key in seen:
                continue
            seen.add(key)
            occurrences[key] += 1
            displays.setdefault(key, str(item))
    by_key = {str(node["key"]): node for node in row["candidates"]}
    for key, support in occurrences.items():
        node = by_key.get(key)
        if node is None:
            node = {
                "key": key,
                "item": displays[key],
                "type": "numeric" if relation in ("hasArea", "hasCapacity") else "entity",
                "proposal_support": {},
                "selected_by": {e2e.QWEN: False, e2e.GEMMA: False},
                "proposer_agents": [agent_id],
            }
            row["candidates"].append(node)
            by_key[key] = node
        elif agent_id not in node.setdefault("proposer_agents", []):
            node["proposer_agents"].append(agent_id)
        node.setdefault("proposal_support", {})[agent_id] = int(support)
        node.setdefault("selected_by", {})[agent_id] = key in selected


def _prepare_base_row_historical(
    source: Mapping[str, Any],
    qwen_response: Mapping[str, Any],
    gemma_response: Mapping[str, Any],
    *,
    primary_objects: Sequence[str],
    system2_objects: Sequence[str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Exact Aug-3 adapter, intentionally excluding later parser realignment."""
    row = copy.deepcopy(dict(source))
    selected = {
        agent: [str(value) for value in prediction_for_agent(row, agent)]
        for agent in (e2e.QWEN, e2e.GEMMA)
    }
    selected[e2e.QWEN] = [str(value) for value in primary_objects]
    qwen_texts = [str(value) for value in qwen_response["generations"]]
    gemma_texts = [str(value) for value in gemma_response["generations"]]
    _add_exact_base_candidates(
        row,
        agent_id=e2e.QWEN,
        raw_texts=qwen_texts,
        parser=lambda text, relation: classify_samples(
            [text], relation, "legacy-cot")[0].items,
        selected_objects=selected[e2e.QWEN],
    )
    _add_exact_base_candidates(
        row,
        agent_id=e2e.GEMMA,
        raw_texts=gemma_texts,
        parser=lambda text, relation: proposal_parse_status(text, relation)[1],
        selected_objects=selected[e2e.GEMMA],
    )
    agents: dict[str, Any] = {}
    for agent in (e2e.QWEN, e2e.GEMMA):
        commitments = row["commitments"][agent]
        samples = int(row["proposal_sample_counts"][agent])
        diagnostics = row["proposal_parse_diagnostics"][agent]
        agents[agent] = {
            "existence": {
                "available": True,
                "selected": commitments["existence"],
                "probabilities": commitments.get("existence_probabilities", {}),
            },
            "cardinality": {
                "available": True,
                "selected": commitments["cardinality"],
                "probabilities": commitments.get("cardinality_probabilities", {}),
            },
            "n_samples": samples,
            "none_count": int(diagnostics.get("explicit_none", 0)),
            "none_rate": int(diagnostics.get("explicit_none", 0)) / samples if samples else 0.0,
            "parse_failures": int(sum(
                count for status, count in diagnostics.items()
                if status not in ("parsed_nonempty", "explicit_none")
            )),
        }
    row["agents"] = agents
    row["agent_outputs"] = selected
    for node in row["candidates"]:
        node["sources"] = {
            agent: {
                "support": int(support),
                "samples": int(row["proposal_sample_counts"].get(agent, 0)),
                "support_rate": int(support) / max(
                    int(row["proposal_sample_counts"].get(agent, 0)), 1),
            }
            for agent, support in node.get("proposal_support", {}).items()
        }
        node.setdefault("output_eligible", True)
    row["baseline_objects"] = list(selected[e2e.QWEN])
    row["baseline_agent"] = e2e.QWEN
    row = augment_graph(row, system2_objects)
    row.pop("relational_graph", None)
    row.pop("relational_graph_schema", None)
    row = augment_relational_graph(row)
    return row, qwen_texts, gemma_texts


def _evidence_graph(
    base: Mapping[str, Any],
    *,
    qwen_texts: Sequence[str],
    gemma_texts: Sequence[str],
    ministral_cot40: Mapping[str, Any],
    ministral_n3: Mapping[str, Any] | None,
) -> dict[str, Any]:
    graph = copy.deepcopy(dict(base))
    if ministral_n3 is not None:
        e2e._attach_supply_route(
            graph, ministral_n3, route_name=e2e.MINISTRAL_N3, samples=3)
    e2e._attach_supply_route(
        graph, ministral_cot40, route_name=e2e.MINISTRAL_COT40, samples=10)
    graph.pop("relational_graph", None)
    graph.pop("relational_graph_schema", None)
    graph = augment_relational_graph(graph)
    current._replace_route_events(
        graph,
        route="qwen:self_consistency",
        family=e2e.QWEN,
        records=current._qwen_records(graph, qwen_texts),
        raw_texts=qwen_texts,
        provenance="fresh_historical_replication",
    )
    current._replace_route_events(
        graph,
        route="gemma:independent",
        family=e2e.GEMMA,
        records=[current._generic_record(graph, text) for text in gemma_texts],
        raw_texts=gemma_texts,
        provenance="fresh_historical_replication",
    )
    cot40_texts = [str(value) for value in ministral_cot40["generations"]]
    current._replace_route_events(
        graph,
        route=e2e.MINISTRAL_COT40,
        family=e2e.MINISTRAL,
        records=[current._generic_record(graph, text) for text in cot40_texts],
        raw_texts=cot40_texts,
        provenance="fresh_historical_replication",
    )
    current._state_and_relation_edges(graph)
    graph["schema"] = GRAPH_SCHEMA
    graph["contains_labels"] = False
    graph["gold_aware"] = False
    return graph


def _component_cot40_area(
    graph: Mapping[str, Any], objects: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Unique 7/10 complete-link numeric component; trained before test use."""
    if str(graph["Relation"]) != "hasArea":
        return current.apply_cot40_support(graph, objects)
    candidates: list[tuple[int, str]] = []
    for node in graph.get("relational_graph", {}).get("nodes", []):
        if node.get("node_type") != "candidate_component":
            continue
        route = node.get("routes", {}).get(e2e.MINISTRAL_COT40)
        if not isinstance(route, Mapping):
            continue
        candidates.append((
            int(route.get("distinct_generation_support", 0)),
            str(node["representative"]),
        ))
    if not candidates:
        return list(objects), {"applied": False, "reason": "no_numeric_components"}
    highest = max(support for support, _ in candidates)
    winners = [value for support, value in candidates if support == highest and support >= 7]
    if len(winners) != 1:
        return list(objects), {
            "applied": False,
            "reason": "no_unique_7_of_10_numeric_component",
            "highest_support": highest,
            "winner_count": len(winners),
        }
    selected = [winners[0]]
    return selected, {
        "applied": selected != list(objects),
        "reason": "unique_7_of_10_numeric_component",
        "highest_support": highest,
        "selected": selected,
        "evidence_routes": [e2e.MINISTRAL_COT40],
    }


def _decode_stack(
    base_graphs: Sequence[Mapping[str, Any]],
    full_graphs: Sequence[Mapping[str, Any]],
    models: Mapping[str, Path],
    *,
    paper_single_ministral: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aug-3 decoder with one selectable, predeclared Ministral area stage."""
    selector = current._json(models["candidate_selector"])
    cardinality_payload = current._json(models["cardinality_result"])
    candidate_model = current._calibrator(selector["candidate_model"])
    cardinality_model = current._cardinality_model(
        cardinality_payload["cardinality_model"])
    l1_rows, l1_details = current.cardinality_prediction_rows(
        base_graphs,
        candidate_model,
        cardinality_model,
        float(cardinality_payload["guard_margin"]),
    )
    numeric_payload = current._json(models["numeric_model"])
    numeric_model = current._numeric_model(numeric_payload)
    l1_by = current._prediction_map(l1_rows)
    numeric_replacements: dict[tuple[str, str], list[str]] = {}
    for graph in base_graphs:
        relation = str(graph["Relation"])
        if relation not in numeric_model.models:
            continue
        if not bool(numeric_payload["stable_relations"][relation]):
            numeric_replacements[current._key(graph)] = l1_by[current._key(graph)]
            continue
        decoded, _ = numeric_model.decode(
            graph, float(numeric_payload["best_mean_margins"][relation]))
        numeric_replacements[current._key(graph)] = decoded
    l2_rows = current._merge_numeric(l1_rows, numeric_replacements)

    route_payload = current._json(models["route_models"])
    base_by = {current._key(row): row for row in base_graphs}
    full_by = {current._key(row): row for row in full_graphs}
    l2_by = current._prediction_map(l2_rows)
    route_replacements: dict[tuple[str, str], list[str]] = {}
    for key in base_by:
        relation = key[1]
        arm = route_payload["chosen_arm"].get(relation)
        if arm is None:
            continue
        source = base_by if arm == "base_residual" else full_by
        model = current._residual_model(route_payload["models"][arm][relation])
        objects, _ = current.decode_route(
            model,
            source[key],
            l2_by[key],
            arm,
            float(route_payload["selected_margins"][arm][relation]),
        )
        route_replacements[key] = objects
    l3_rows = current.route_prediction_rows(l2_rows, route_replacements)

    component_models = current._json(models["component_models"])
    l3_by = current._prediction_map(l3_rows)
    preproof: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for graph in full_graphs:
        key = current._key(graph)
        objects = list(l3_by[key])
        if paper_single_ministral:
            stages = (
                (
                    "component_surface_residual",
                    lambda value, answer: current.apply_component_residual(
                        value, answer, component_models),
                ),
                ("ministral_cot5_n10_component_support_7", _component_cot40_area),
            )
        else:
            stages = (
                (
                    "component_surface_residual",
                    lambda value, answer: current.apply_component_residual(
                        value, answer, component_models),
                ),
                ("ministral_zero_shot_n3_area_unanimity", current.apply_area_unanimity),
                ("ministral_cot5_n10_support_7", current.apply_cot40_support),
            )
        layer_trace: list[dict[str, Any]] = []
        for name, function in stages:
            before = list(objects)
            objects, detail = function(graph, objects)
            objects = [str(value) for value in objects]
            layer_trace.append({
                "policy": name,
                "before": before,
                "after": list(objects),
                "changed": before != objects,
                **detail,
            })
        preproof.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": objects,
        })
        traces.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "qwen_incumbent": list(graph["baseline_objects"]),
            "l1_cardinality": l1_by[key],
            "l2_numeric": l2_by[key],
            "l3_route": l3_by[key],
            "layers": layer_trace,
        })
    incumbents = current._prediction_map(preproof)
    predictions, proof_decisions = proof_decode(
        full_graphs,
        incumbents,
        PRIMARY_ARM,
        fail_closed_invalid_evidence=True,
    )
    for index, graph in enumerate(full_graphs):
        key = current._key(graph)
        if str(graph["Relation"]) in IDENTITY_RELATIONS:
            predictions[index] = {
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": list(incumbents[key]),
            }
            proof_decisions[index]["identity_fallback"] = True
            proof_decisions[index]["changed"] = False
        traces[index]["proof"] = proof_decisions[index]
        traces[index]["prediction"] = list(predictions[index]["ObjectEntities"])
        traces[index]["l1_detail"] = l1_details[index]
    return predictions, traces


def _objects(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[str]]:
    result = {
        current._key(row): [str(value) for value in row["ObjectEntities"]]
        for row in rows
    }
    if len(result) != len(rows):
        raise ContractError("duplicate prediction keys")
    return result


def build(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    policy, model_paths = _validate_policy(output)
    source_plan = _validate_plan_contract(output)
    primary, qwen_raw, system2 = current._primary_inputs(
        output, source_plan, {"primary_seed_scheme": "legacy"})
    assembled = current._assemble_from_primary(
        output, source_plan, primary, qwen_raw)
    gemma = current._response_map(source_plan, "gemma:independent")
    n3 = current._response_map(source_plan, e2e.MINISTRAL_N3)
    cot40 = current._response_map(source_plan, e2e.MINISTRAL_COT40)

    base_graphs: list[dict[str, Any]] = []
    control_graphs: list[dict[str, Any]] = []
    paper_graphs: list[dict[str, Any]] = []
    for source in assembled:
        key = current._key(source)
        base, qwen_texts, gemma_texts = _prepare_base_row_historical(
            source,
            {"generations": list(qwen_raw[key])},
            gemma[key],
            primary_objects=primary[key],
            system2_objects=system2.get(key, ()),
        )
        control_graphs.append(_evidence_graph(
            base,
            qwen_texts=qwen_texts,
            gemma_texts=gemma_texts,
            ministral_n3=n3[key],
            ministral_cot40=cot40[key],
        ))
        paper_graph = _evidence_graph(
            base,
            qwen_texts=qwen_texts,
            gemma_texts=gemma_texts,
            ministral_n3=None,
            ministral_cot40=cot40[key],
        )
        if e2e.MINISTRAL_N3 in paper_graph.get("proposal_routes", {}):
            raise ContractError(f"N=3 route survived paper graph: {key}")
        base_graphs.append(base)
        paper_graphs.append(paper_graph)

    control_predictions, control_decisions = _decode_stack(
        base_graphs, control_graphs, model_paths, paper_single_ministral=False)
    paper_predictions, paper_decisions = _decode_stack(
        base_graphs, paper_graphs, model_paths, paper_single_ministral=True)

    graph_dir = output / "graph"
    write_jsonl_atomic(graph_dir / "HISTORICAL_CONTROL_GRAPH.jsonl", control_graphs)
    write_jsonl_atomic(graph_dir / "PAPER_SINGLE_MINISTRAL_GRAPH.jsonl", paper_graphs)
    control_path = output / "HISTORICAL_CONTROL_PREDICTIONS.jsonl"
    control_decisions_path = output / "HISTORICAL_CONTROL_DECISIONS.jsonl"
    paper_path = output / "FINAL_PREDICTIONS.jsonl"
    paper_decisions_path = output / "FINAL_DECISIONS.jsonl"
    write_jsonl_atomic(control_path, control_predictions)
    write_jsonl_atomic(control_decisions_path, control_decisions)
    write_jsonl_atomic(paper_path, paper_predictions)
    write_jsonl_atomic(paper_decisions_path, paper_decisions)

    archived_payload = _archived_prediction_bytes()
    archived_rows = [json.loads(line) for line in archived_payload.splitlines() if line]
    archived_by = _objects(archived_rows)
    control_by = _objects(control_predictions)
    paper_by = _objects(paper_predictions)
    control_divergences = [key for key in archived_by if archived_by[key] != control_by[key]]
    paper_changes = [key for key in control_by if control_by[key] != paper_by[key]]
    relation_control_divergences = Counter(key[1] for key in control_divergences)
    relation_paper_changes = Counter(key[1] for key in paper_changes)
    result = {
        "schema": RESULT_SCHEMA,
        "split": "test",
        "blind": True,
        "contains_labels": False,
        "gold_aware": False,
        "rows": len(paper_predictions),
        "verified_parameter_total": policy["verified_parameter_total"],
        "parameter_cap": policy["parameter_cap"],
        "historical_control": {
            "policy_id": HISTORICAL_POLICY_ID,
            "predictions": str(control_path.resolve()),
            "predictions_sha256": sha256(control_path),
            "archived_submission_sha256": EXPECTED_ARCHIVE_SHA256,
            "archived_member_sha256": EXPECTED_ARCHIVED_MEMBER_SHA256,
            "byte_identical_to_archived_member": control_path.read_bytes() == archived_payload,
            "divergent_rows_from_archived": len(control_divergences),
            "divergent_rows_by_relation": dict(sorted(relation_control_divergences.items())),
            "exact_byte_reproduction_claimed": False,
        },
        "paper_single_ministral": {
            "policy_id": PAPER_POLICY_ID,
            "predictions": str(paper_path.resolve()),
            "predictions_sha256": sha256(paper_path),
            "changed_rows_from_fresh_control": len(paper_changes),
            "changed_rows_by_relation": dict(sorted(relation_paper_changes.items())),
            "change_is_label_free": True,
            "removed_route": e2e.MINISTRAL_N3,
            "retained_route": e2e.MINISTRAL_COT40,
        },
        "graphs": {
            "historical_control": {
                "path": str((graph_dir / "HISTORICAL_CONTROL_GRAPH.jsonl").resolve()),
                "sha256": sha256(graph_dir / "HISTORICAL_CONTROL_GRAPH.jsonl"),
            },
            "paper_single_ministral": {
                "path": str((graph_dir / "PAPER_SINGLE_MINISTRAL_GRAPH.jsonl").resolve()),
                "sha256": sha256(graph_dir / "PAPER_SINGLE_MINISTRAL_GRAPH.jsonl"),
            },
        },
        "policy_sha256": sha256(output / "plan/HISTORICAL_SOTA_POLICY.json"),
        "test_labels_opened": False,
    }
    _write_json(output / "FINAL_MANIFEST.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _package_one(output: Path, predictions: Path, archive_name: str) -> dict[str, Any]:
    rows = read_jsonl(predictions)
    source = read_jsonl(Path(_validate_plan_contract(output)["input_rows"]))
    if len(rows) != EXPECTED_TEST_ROWS or [current._key(row) for row in rows] != [
        current._key(row) for row in source
    ]:
        raise ContractError(f"prediction coverage/order mismatch: {predictions}")
    package_dir = output / "submission"
    package_dir.mkdir(parents=True, exist_ok=True)
    archive = package_dir / archive_name
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(predictions, arcname="predictions.jsonl")
    with zipfile.ZipFile(archive) as handle:
        if (
            handle.namelist() != ["predictions.jsonl"]
            or handle.read("predictions.jsonl") != predictions.read_bytes()
        ):
            raise ContractError(f"package round-trip failed: {archive}")
    return {
        "archive": str(archive.resolve()),
        "archive_sha256": sha256(archive),
        "member": "predictions.jsonl",
        "predictions_sha256": sha256(predictions),
        "rows": len(rows),
    }


def package(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    _validate_policy(output)
    result = _json(output / "FINAL_MANIFEST.json")
    if result.get("schema") != RESULT_SCHEMA:
        raise ContractError("missing paired decode manifest")
    packages = {
        "historical_control": _package_one(
            output,
            output / "HISTORICAL_CONTROL_PREDICTIONS.jsonl",
            "historical_sota_replication_control_test.zip",
        ),
        "paper_single_ministral": _package_one(
            output,
            output / "FINAL_PREDICTIONS.jsonl",
            "paper_single_ministral_n10_test.zip",
        ),
    }
    _write_json(output / "submission/PACKAGES.json", packages)
    print(json.dumps(packages, indent=2, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_path = output / "plan/PLAN.json"
    if not plan_path.is_file():
        print(f"plan: missing ({plan_path})")
        return 0
    plan = _json(plan_path)
    print(
        f"split={plan.get('split')} rows={plan.get('rows')} "
        f"params={plan.get('verified_parameter_total')}/{plan.get('parameter_cap')}")
    primary = output / "primary_qwen/MANIFEST.json"
    print(f"primary_qwen_v0495: {'ready' if primary.is_file() else 'pending'}")
    for route in ("gemma:independent", e2e.MINISTRAL_N3, e2e.MINISTRAL_COT40):
        job = plan.get("jobs", {}).get(route, {})
        path = Path(str(job.get("response_path", "")))
        done = sum(1 for line in path.open() if line.strip()) if path.is_file() else 0
        total = int(job.get("tasks", 0))
        print(f"{route:34s} {done:5d}/{total:<5d}")
    for path in (
        "FINAL_MANIFEST.json",
        "submission/historical_sota_replication_control_test.zip",
        "submission/paper_single_ministral_n10_test.zip",
    ):
        print(f"{path}: {'ready' if (output / path).is_file() else 'pending'}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.set_defaults(function=plan)
    for name, function in (("build", build), ("package", package), ("status", status)):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
        p.set_defaults(function=function)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.function(arguments))
