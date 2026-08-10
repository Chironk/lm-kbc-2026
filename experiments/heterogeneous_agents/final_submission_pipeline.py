#!/usr/bin/env python3
"""Frozen, split-independent submission decoder for the heterogeneous system.

This module is intentionally downstream of :mod:`end_to_end_pipeline`.
That runner owns label-free task planning, checkpoint-pinned generation, and
the initial two-family candidate assembly.  This module closes the remaining
submission gap:

* recover every candidate occurring in an exact Qwen/Gemma generation;
* materialize exact generation-event ``supports`` edges for all three model
  families;
* apply the frozen cardinality, numeric, route, component, and Ministral
  policies in their declared order; and
* finish with relation-typed symbolic graph correction: the strict set proof
  for entity-valued relations and a singleton-numeric proof for capacity.

No command in this module reads a gold file except ``score-validation``.
That command refuses a plan whose split is ``test``.  The final package
contains one file named ``predictions.jsonl`` at the zip root, as required by
the competition submission format.

The frozen artifact manifest contains model and decoder parameters only.  It
does not contain split-specific predictions, so a newly planned run cannot
inherit answers from a historical development artifact.  Fresh generations
are stochastic; reproducing the architecture does not imply byte-identical
model responses.
"""
from __future__ import annotations

import argparse
import copy
import json
import zipfile
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
from experiments.heterogeneous_agents.components.baseline_relative_route_decoder import (
    _prediction_rows as route_prediction_rows,
    decode as decode_route,
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
from run_submission import Submission as PrimarySubmission
from experiments.heterogeneous_agents.components.cot40_evidence_edge_ablation import (
    _generic_record,
    _qwen_records,
    _replace_route_events,
    _state_and_relation_edges,
)
from experiments.heterogeneous_agents.components.explicit_cardinality_ablation import (
    _prediction_rows as cardinality_prediction_rows,
)
from experiments.heterogeneous_agents.components.heterogeneous_memory_selector import (
    _key,
)
from experiments.heterogeneous_agents.components.proof_carrying_graph_decoder import (
    IDENTITY_RELATIONS,
    PRIMARY_ARM,
    _decode as proof_decode,
)
from experiments.heterogeneous_agents.components.strict_numeric_proof import (
    decode_row as strict_numeric_decode,
)
from experiments.heterogeneous_agents.components.relation_specific_numeric_decoder import (
    _merge_numeric,
)
from experiments.heterogeneous_agents.components.relational_candidate_graph import (
    augment_relational_graph,
)
from experiments.heterogeneous_agents.components.route_aware_candidate_graph import (
    augment_graph,
)
from experiments.heterogeneous_agents.frozen_model_loader import (
    calibrator as _calibrator,
    cardinality_model as _cardinality_model,
    numeric_model as _numeric_model,
    residual_model as _residual_model,
)
from experiments.heterogeneous_agents.components.unified_graph_decoder import (
    apply_area_unanimity,
    apply_component_residual,
    apply_cot40_support,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE / "runs/final_test_submission_20260809_v3"
)
SNAPSHOT = ROOT / "artifacts/frozen"
SNAPSHOT_MANIFEST = SNAPSHOT / "MANIFEST.json"

POLICY_ID = "heterogeneous_final_relation_typed_proof_20260809_v3"
POLICY_SCHEMA = "heterogeneous-final-submission-policy-v1"
GRAPH_SCHEMA = "heterogeneous-final-exact-evidence-graph-v1"
RESULT_SCHEMA = "heterogeneous-final-submission-result-v1"

POOLED = "*** All Relations ***"

MODEL_ARTIFACTS = (
    "candidate_selector",
    "cardinality_result",
    "numeric_model",
    "route_models",
    "component_models",
)
PRIMARY_POLICY = "v0495"
STRICT_NUMERIC_RELATIONS = frozenset({"hasCapacity"})
DECODER_IMPLEMENTATIONS = {
    "final_submission": Path(__file__).resolve(),
    "base_graph_assembler": HERE / "assemble_and_audit.py",
    "route_graph": HERE / "components/route_aware_candidate_graph.py",
    "relational_graph": HERE / "components/relational_candidate_graph.py",
    "cardinality_decoder": HERE / "components/explicit_cardinality_ablation.py",
    "numeric_decoder": HERE / "components/relation_specific_numeric_decoder.py",
    "route_decoder": HERE / "components/baseline_relative_route_decoder.py",
    "staged_decoder": HERE / "components/unified_graph_decoder.py",
    "model_deserializer": HERE / "frozen_model_loader.py",
    "set_proof": HERE / "components/proof_carrying_graph_decoder.py",
    "singleton_numeric_proof": HERE / "components/strict_numeric_proof.py",
    "singleton_numeric_features": HERE / "components/capacity_graph_decoder.py",
    "event_graph_contract": HERE / "components/graph_event_contract.py",
    "exact_event_graph": HERE / "components/cot40_evidence_edge_ablation.py",
    "core_contracts": HERE / "core.py",
    "primary_artifact_contract": ROOT / "artifact_contract.py",
}
OFFICIAL_TEST_SHA256 = (
    "67c31c8388c585634df55500612f522ad42da6735d4c89eb59a9ef5a39f043f1"
)


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


def _snapshot_artifacts() -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = _json(SNAPSHOT_MANIFEST)
    if (
        manifest.get("schema") != "heterogeneous-final-artifacts-v1"
        or int(manifest.get("verified_parameter_total", -1))
            != e2e.EXPECTED_PARAMETER_TOTAL
        or int(manifest.get("parameter_cap", -1)) != e2e.PARAMETER_CAP
    ):
        raise ContractError("foreign SOTA snapshot manifest")
    paths: dict[str, Path] = {}
    for name in MODEL_ARTIFACTS:
        record = manifest.get("artifacts", {}).get(name)
        if not isinstance(record, dict):
            raise ContractError(f"snapshot is missing {name}")
        path = SNAPSHOT / str(record["path"])
        if not path.is_file() or sha256(path) != str(record["sha256"]):
            raise ContractError(f"snapshot hash mismatch: {name}")
        paths[name] = path
    return manifest, paths


def _score(rows: Sequence[Mapping[str, Any]], gold_path: Path) -> dict[str, Any]:
    gold = official.read_jsonl_file(str(gold_path))
    return official.macro_average_per_relation(
        official.evaluate_per_sr_pair(rows, gold, official.RELATION_TYPE))


def freeze(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    source_plan = e2e._validate_plan(output)
    snapshot, model_paths = _snapshot_artifacts()
    policy = {
        "schema": POLICY_SCHEMA,
        "policy_id": POLICY_ID,
        "split": source_plan["split"],
        "blind": bool(source_plan["blind"]),
        "contains_labels": False,
        "gold_aware": False,
        "validation_selected_lineage": True,
        "development_reference_is_not_copied": True,
        "source_plan": str((output / "plan/PLAN.json").resolve()),
        "source_plan_sha256": sha256(output / "plan/PLAN.json"),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "decoder_implementations": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in DECODER_IMPLEMENTATIONS.items()
        },
        "primary_runner": str((ROOT / "run_submission.py").resolve()),
        "primary_runner_sha256": sha256(ROOT / "run_submission.py"),
        "primary_policy": PRIMARY_POLICY,
        "snapshot_manifest": str(SNAPSHOT_MANIFEST.resolve()),
        "snapshot_manifest_sha256": sha256(SNAPSHOT_MANIFEST),
        "model_artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in model_paths.items()
        },
        "model_portfolio": snapshot["model_portfolio"],
        "verified_parameter_total": source_plan["verified_parameter_total"],
        "parameter_cap": source_plan["parameter_cap"],
        "decoder_order": [
            "qwen_incumbent",
            "explicit_cardinality",
            "relation_specific_numeric",
            "route_residual",
            "component_surface_residual",
            "area_unanimous_new_component",
            "ministral_cot40_two_thirds",
            "strict_set_proof_graph_noncapacity",
            "strict_singleton_numeric_proof_capacity",
        ],
        "fail_closed": {
            "invalid_exact_event_graph": "identity",
            "proof_contract_failure": "identity",
            "award_proof": "identity",
            "capacity_graph_correction": (
                "strict singleton-numeric proof; invalid evidence falls back "
                "to the preceding incumbent"
            ),
            "qwen_system2": (
                "production v0495 relation-aware System-2 route"
            ),
        },
    }
    path = output / "plan/FINAL_POLICY.json"
    _write_json(path, policy)
    print(json.dumps({
        "policy": str(path),
        "policy_sha256": sha256(path),
        "split": policy["split"],
        "verified_parameter_total": policy["verified_parameter_total"],
        "parameter_cap": policy["parameter_cap"],
        "frozen_artifact_manifest": str(SNAPSHOT_MANIFEST),
    }, indent=2, sort_keys=True))
    return 0


def _validate_policy(output: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    policy = _json(output / "plan/FINAL_POLICY.json")
    source = e2e._validate_plan(output)
    _, model_paths = _snapshot_artifacts()
    if (
        policy.get("schema") != POLICY_SCHEMA
        or policy.get("policy_id") != POLICY_ID
        or policy.get("source_plan_sha256")
            != sha256(output / "plan/PLAN.json")
        or policy.get("implementation_sha256")
            != sha256(Path(__file__).resolve())
        or policy.get("primary_runner_sha256")
            != sha256(ROOT / "run_submission.py")
        or policy.get("primary_policy") != PRIMARY_POLICY
        or policy.get("split") != source.get("split")
        or bool(policy.get("blind")) != bool(source.get("blind"))
    ):
        raise ContractError("final frozen policy contract failed")
    for name, path in model_paths.items():
        record = policy.get("model_artifacts", {}).get(name, {})
        if record.get("sha256") != sha256(path):
            raise ContractError(f"frozen model contract failed: {name}")
    for name, path in DECODER_IMPLEMENTATIONS.items():
        record = policy.get("decoder_implementations", {}).get(name, {})
        if (
            record.get("path") != str(path.resolve())
            or record.get("sha256") != sha256(path)
        ):
            raise ContractError(
                f"frozen decoder implementation contract failed: {name}")
    return policy, model_paths


def _response_map(
    source_plan: Mapping[str, Any], route: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    job = source_plan["jobs"][route]
    rows = e2e._validated_responses(job)
    result = {
        (str(row["subject"]), str(row["relation"])): row
        for row in rows if row.get("phase") == "propose"
    }
    if len(result) != int(source_plan["rows"]):
        raise ContractError(f"{route}: proposal response coverage mismatch")
    return result


def _add_exact_base_candidates(
    row: dict[str, Any],
    *,
    agent_id: str,
    raw_texts: Sequence[str],
    parser: Callable[[str, str], Sequence[str]],
    selected_objects: Sequence[str],
) -> None:
    """Recover exact generation candidates before constructing components."""
    relation = str(row["Relation"])
    selected = {
        canonical_key(str(value), relation) for value in selected_objects
    }
    occurrences: Counter[str] = Counter()
    displays: dict[str, str] = {}
    for text in raw_texts:
        seen = set()
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
                "type": (
                    "numeric" if relation in ("hasArea", "hasCapacity")
                    else "entity"
                ),
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


def _prepare_base_row(
    source: Mapping[str, Any],
    qwen_response: Mapping[str, Any],
    gemma_response: Mapping[str, Any],
    *,
    primary_objects: Sequence[str] | None = None,
    system2_objects: Sequence[str] = (),
) -> tuple[dict[str, Any], list[str], list[str]]:
    row = copy.deepcopy(dict(source))
    relation = str(row["Relation"])
    selected = {
        agent: [str(value) for value in prediction_for_agent(row, agent)]
        for agent in (e2e.QWEN, e2e.GEMMA)
    }
    if primary_objects is not None:
        selected[e2e.QWEN] = [str(value) for value in primary_objects]
    qwen_texts = [str(value) for value in qwen_response["generations"]]
    gemma_texts = [str(value) for value in gemma_response["generations"]]
    # ``assemble_graphs`` carries a legacy permissive parser inventory.  The
    # exact-event graph below uses the stricter production parsers, so remove
    # route support that those parsers cannot reproduce.  Otherwise malformed
    # remnants such as ``Moscow Exchange</think>`` become unsupported graph
    # components with fabricated historical support.  A selected incumbent is
    # retained even when it has no raw support; it remains a legal incumbent
    # but is not turned into an evidence event.
    exact_keys = {
        e2e.QWEN: {
            canonical_key(str(item), relation)
            for text in qwen_texts
            for item in classify_samples(
                [text], relation, "legacy-cot")[0].items
        },
        e2e.GEMMA: {
            canonical_key(str(item), relation)
            for text in gemma_texts
            for item in proposal_parse_status(text, relation)[1]
        },
    }
    selected_keys = {
        agent: {canonical_key(str(item), relation) for item in objects}
        for agent, objects in selected.items()
    }
    aligned_candidates = []
    for candidate in row["candidates"]:
        key = str(candidate["key"])
        support = dict(candidate.get("proposal_support", {}))
        proposers = list(candidate.get("proposer_agents", []))
        selected_by = dict(candidate.get("selected_by", {}))
        for agent in (e2e.QWEN, e2e.GEMMA):
            if (
                agent in support
                and key not in exact_keys[agent]
                and key not in selected_keys[agent]
            ):
                support.pop(agent, None)
                proposers = [value for value in proposers if value != agent]
                selected_by[agent] = False
        candidate["proposal_support"] = support
        candidate["proposer_agents"] = proposers
        candidate["selected_by"] = selected_by
        if support or any(selected_by.values()):
            aligned_candidates.append(candidate)
    row["candidates"] = aligned_candidates
    _add_exact_base_candidates(
        row,
        agent_id=e2e.QWEN,
        raw_texts=qwen_texts,
        parser=lambda text, rel: classify_samples(
            [text], rel, "legacy-cot")[0].items,
        selected_objects=selected[e2e.QWEN],
    )
    _add_exact_base_candidates(
        row,
        agent_id=e2e.GEMMA,
        raw_texts=gemma_texts,
        parser=lambda text, rel: proposal_parse_status(text, rel)[1],
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
                "probabilities": commitments.get(
                    "existence_probabilities", {}),
            },
            "cardinality": {
                "available": True,
                "selected": commitments["cardinality"],
                "probabilities": commitments.get(
                    "cardinality_probabilities", {}),
            },
            "n_samples": samples,
            "none_count": int(diagnostics.get("explicit_none", 0)),
            "none_rate": (
                int(diagnostics.get("explicit_none", 0)) / samples
                if samples else 0.0
            ),
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
    # This is the historical L0 contract.  The discarded end-to-end runner
    # used an uncalibrated union consensus here, which caused its 0.068 loss.
    row["baseline_objects"] = list(selected[e2e.QWEN])
    row["baseline_agent"] = e2e.QWEN
    row = augment_graph(row, system2_objects)
    row.pop("relational_graph", None)
    row.pop("relational_graph_schema", None)
    row = augment_relational_graph(row)
    return row, qwen_texts, gemma_texts


def _primary_inputs(
    output: Path, source_plan: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str], list[str]],
    dict[tuple[str, str], list[str]],
    dict[tuple[str, str], list[str]],
]:
    """Validate and load the production Qwen v0495 primary route.

    The historical SOTA's L0 answer is the relation-aware production Qwen
    prediction, not the simple median/majority view reconstructed by
    ``prediction_for_agent``.  Using the latter was the root cause of the
    0.450 end-to-end regression.
    """
    primary_dir = output / "primary_qwen"
    args = argparse.Namespace(
        policy=PRIMARY_POLICY,
        input=str(Path(source_plan["input"]).resolve()),
        output_dir=str(primary_dir),
        dry_run=False,
        skip_inference=True,
        stage="compose",
    )
    submission = PrimarySubmission(args)
    # This verifies every raw/prediction/manifest bundle against the exact
    # frozen command contract without executing a model.
    submission.preflight()
    submission.split_inputs()
    submission.validate_all_bundles()
    manifest = _json(primary_dir / "MANIFEST.json")
    prediction_path = primary_dir / f"submission_{PRIMARY_POLICY}.jsonl"
    if (
        manifest.get("policy") != PRIMARY_POLICY
        # PrimarySubmission is invoked on source_plan["input"] above, so its
        # manifest must bind to that exact official split artifact.  The
        # separately normalized INPUT_ROWS file is used by the heterogeneous
        # route planner, not by the production Qwen runner.
        or manifest.get("input_sha256") != source_plan["input_sha256"]
        or manifest.get("submission_sha256") != sha256(prediction_path)
    ):
        raise ContractError("primary Qwen submission manifest mismatch")
    predictions = {
        _key(row): [str(value) for value in row["ObjectEntities"]]
        for row in read_jsonl(prediction_path)
    }
    if len(predictions) != int(source_plan["rows"]):
        raise ContractError("primary Qwen prediction coverage mismatch")

    raw: dict[tuple[str, str], list[str]] = {}
    for path in (
        primary_dir / "raw_borders.jsonl",
        primary_dir / "raw_fp16.jsonl",
    ):
        for row in read_jsonl(path):
            key = _key(row)
            samples = [str(value) for value in row.get("raw_samples", [])]
            if len(samples) != 10 or key in raw:
                raise ContractError(f"invalid primary Qwen raw row: {key}")
            raw[key] = samples
    # Awards are generated through System-2 rather than the System-1 raw
    # route.  The proof decoder is identity on awards, but their primary set
    # still has to enter the candidate graph.  Represent it as one exact
    # proposal event; it cannot affect a proof edit.
    for key, objects in predictions.items():
        if key[1] == "awardWonBy":
            raw[key] = [
                "ANSWER: " + ("; ".join(objects) if objects else "None")]
    if len(raw) != int(source_plan["rows"]):
        raise ContractError("primary Qwen raw coverage mismatch")

    system2 = {
        _key(row): [str(value) for value in row["ObjectEntities"]]
        for row in read_jsonl(primary_dir / "pred_system2.jsonl")
    }
    return predictions, raw, system2


def _assemble_from_primary(
    output: Path,
    source_plan: Mapping[str, Any],
    primary: Mapping[tuple[str, str], Sequence[str]],
    qwen_raw: Mapping[tuple[str, str], Sequence[str]],
) -> list[dict[str, Any]]:
    """Create the two-family base graph without a duplicate Qwen GPU run."""
    rows = read_jsonl(Path(source_plan["input_rows"]))
    cot_config = load_agent_config(Path(source_plan["cot_agents"]))
    agents = [
        e2e._agent_of(cot_config, e2e.QWEN),
        e2e._agent_of(cot_config, e2e.GEMMA),
    ]
    gemma_job = source_plan["jobs"]["gemma:independent"]
    gemma_rows = e2e._validated_responses(gemma_job)
    gemma_map = {
        (
            str(row["subject"]),
            str(row["relation"]),
            str(row["phase"]),
        ): row
        for row in gemma_rows
    }
    qwen_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        objects = list(primary[key])
        cardinality = "ZERO" if not objects else "ONE" if len(objects) == 1 else "MANY"
        qwen_map[(key[0], key[1], "commit_existence")] = {
            "subject": key[0],
            "relation": key[1],
            "phase": "commit_existence",
            "selected_choice": "NO" if not objects else "YES",
            "choice_probabilities": {"NO" if not objects else "YES": 1.0},
        }
        qwen_map[(key[0], key[1], "commit_cardinality")] = {
            "subject": key[0],
            "relation": key[1],
            "phase": "commit_cardinality",
            "selected_choice": cardinality,
            "choice_probabilities": {cardinality: 1.0},
        }
        qwen_map[(key[0], key[1], "propose")] = {
            "subject": key[0],
            "relation": key[1],
            "phase": "propose",
            "generations": list(qwen_raw[key]),
        }
    return assemble_graphs(
        rows,
        agents,
        {e2e.QWEN: qwen_map, e2e.GEMMA: gemma_map},
    )


def _full_evidence_graph(
    base: Mapping[str, Any],
    *,
    qwen_texts: Sequence[str],
    gemma_texts: Sequence[str],
    ministral_n3: Mapping[str, Any],
    ministral_cot40: Mapping[str, Any],
) -> dict[str, Any]:
    graph = copy.deepcopy(dict(base))
    e2e._attach_supply_route(
        graph, ministral_n3, route_name=e2e.MINISTRAL_N3, samples=3)
    e2e._attach_supply_route(
        graph, ministral_cot40, route_name=e2e.MINISTRAL_COT40, samples=10)
    graph.pop("relational_graph", None)
    graph.pop("relational_graph_schema", None)
    graph = augment_relational_graph(graph)

    _replace_route_events(
        graph,
        route="qwen:self_consistency",
        family=e2e.QWEN,
        records=_qwen_records(graph, qwen_texts),
        raw_texts=qwen_texts,
        provenance="frozen_split_generation",
    )
    _replace_route_events(
        graph,
        route="gemma:independent",
        family=e2e.GEMMA,
        records=[_generic_record(graph, text) for text in gemma_texts],
        raw_texts=gemma_texts,
        provenance="frozen_split_generation",
    )
    cot40_texts = [
        str(value) for value in ministral_cot40["generations"]]
    _replace_route_events(
        graph,
        route=e2e.MINISTRAL_COT40,
        family=e2e.MINISTRAL,
        records=[_generic_record(graph, text) for text in cot40_texts],
        raw_texts=cot40_texts,
        provenance="frozen_split_generation",
    )
    _state_and_relation_edges(graph)
    graph["schema"] = GRAPH_SCHEMA
    graph["contains_labels"] = False
    graph["gold_aware"] = False
    return graph


def _prediction_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    result = {_key(row): list(map(str, row["ObjectEntities"])) for row in rows}
    if len(result) != len(rows):
        raise ContractError("prediction rows contain duplicate keys")
    return result


def _apply_relation_typed_graph_correction(
    graphs: Sequence[Mapping[str, Any]],
    incumbents: Mapping[tuple[str, str], Sequence[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the set proof and its singleton-numeric counterpart.

    The generic proof requires a strict cardinality-compatibility advantage.
    That is meaningful for variable-length entity sets, but impossible when
    both capacity hypotheses are valid non-empty scalar singletons.  Capacity
    therefore uses the separately frozen numeric proof over the same exact
    event-to-component support graph.  Invalid numeric evidence fails closed
    to the preceding incumbent, just like the generic proof.
    """
    predictions: list[dict[str, Any] | None] = [None] * len(graphs)
    decisions: list[dict[str, Any] | None] = [None] * len(graphs)
    set_indices = [
        index for index, graph in enumerate(graphs)
        if str(graph["Relation"]) not in STRICT_NUMERIC_RELATIONS
    ]
    if set_indices:
        set_predictions, set_decisions = proof_decode(
            [graphs[index] for index in set_indices],
            incumbents,
            PRIMARY_ARM,
            fail_closed_invalid_evidence=True,
        )
        for index, prediction, decision in zip(
            set_indices, set_predictions, set_decisions, strict=True,
        ):
            predictions[index] = dict(prediction)
            decisions[index] = dict(decision)

    for index, graph in enumerate(graphs):
        relation = str(graph["Relation"])
        if relation not in STRICT_NUMERIC_RELATIONS:
            continue
        key = _key(graph)
        incumbent = list(map(str, incumbents[key]))
        try:
            objects, detail = strict_numeric_decode(graph, incumbent)
            decision = {
                "SubjectEntity": key[0],
                "Relation": key[1],
                "arm": "strict_singleton_numeric_proof",
                "evidence_invalid_fallback": False,
                **detail,
            }
        except (ContractError, StopIteration, TypeError, ValueError) as exc:
            objects = incumbent
            decision = {
                "SubjectEntity": key[0],
                "Relation": key[1],
                "arm": "strict_singleton_numeric_proof",
                "changed": False,
                "evidence_invalid_fallback": True,
                "fallback_reason": str(exc),
            }
        predictions[index] = {
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": list(map(str, objects)),
        }
        decisions[index] = decision

    if any(value is None for value in predictions + decisions):
        raise ContractError("relation-typed graph correction lost row coverage")
    return (
        [dict(value) for value in predictions if value is not None],
        [dict(value) for value in decisions if value is not None],
    )


def _apply_frozen_stack(
    base_graphs: Sequence[Mapping[str, Any]],
    full_graphs: Sequence[Mapping[str, Any]],
    models: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selector = _json(models["candidate_selector"])
    cardinality_payload = _json(models["cardinality_result"])
    candidate_model = _calibrator(selector["candidate_model"])
    cardinality_model = _cardinality_model(
        cardinality_payload["cardinality_model"])
    l1_rows, l1_details = cardinality_prediction_rows(
        base_graphs,
        candidate_model,
        cardinality_model,
        float(cardinality_payload["guard_margin"]),
    )

    numeric_payload = _json(models["numeric_model"])
    numeric_model = _numeric_model(numeric_payload)
    l1_by = _prediction_map(l1_rows)
    numeric_replacements: dict[tuple[str, str], list[str]] = {}
    for graph in base_graphs:
        relation = str(graph["Relation"])
        if relation not in numeric_model.models:
            continue
        if not bool(numeric_payload["stable_relations"][relation]):
            numeric_replacements[_key(graph)] = l1_by[_key(graph)]
            continue
        objects, _ = numeric_model.decode(
            graph, float(numeric_payload["best_mean_margins"][relation]))
        numeric_replacements[_key(graph)] = objects
    l2_rows = _merge_numeric(l1_rows, numeric_replacements)

    route_payload = _json(models["route_models"])
    chosen = route_payload["chosen_arm"]
    margins = route_payload["selected_margins"]
    serialized = route_payload["models"]
    base_by = {_key(row): row for row in base_graphs}
    full_by = {_key(row): row for row in full_graphs}
    l2_by = _prediction_map(l2_rows)
    route_replacements: dict[tuple[str, str], list[str]] = {}
    for key in base_by:
        relation = key[1]
        arm = chosen.get(relation)
        if arm is None:
            continue
        source = base_by if arm == "base_residual" else full_by
        model = _residual_model(serialized[arm][relation])
        objects, _ = decode_route(
            model,
            source[key],
            l2_by[key],
            arm,
            float(margins[arm][relation]),
        )
        route_replacements[key] = objects
    l3_rows = route_prediction_rows(l2_rows, route_replacements)

    component_models = _json(models["component_models"])
    current = _prediction_map(l3_rows)
    traces: list[dict[str, Any]] = []
    preproof: list[dict[str, Any]] = []
    for graph in full_graphs:
        key = _key(graph)
        objects = list(current[key])
        trace: list[dict[str, Any]] = []
        for name, function in (
            (
                "component_surface_residual",
                lambda value, answer: apply_component_residual(
                    value, answer, component_models),
            ),
            ("area_unanimous_new_component", apply_area_unanimity),
            ("ministral_cot40_two_thirds", apply_cot40_support),
        ):
            before = list(objects)
            objects, detail = function(graph, objects)
            objects = list(map(str, objects))
            trace.append({
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
            "l3_route": current[key],
            "layers": trace,
        })

    incumbents = _prediction_map(preproof)
    predictions, proof_decisions = _apply_relation_typed_graph_correction(
        full_graphs, incumbents)
    # Exact validation Qwen award events were historically unavailable.  The
    # final policy predeclares award as identity on every split, avoiding a
    # different action inventory between development and test.
    for index, graph in enumerate(full_graphs):
        key = _key(graph)
        if str(graph["Relation"]) in IDENTITY_RELATIONS:
            predictions[index] = {
                "SubjectEntity": key[0],
                "Relation": key[1],
                "ObjectEntities": list(incumbents[key]),
            }
            proof_decisions[index]["identity_fallback"] = True
            proof_decisions[index]["changed"] = False
        traces[index]["proof"] = proof_decisions[index]
        traces[index]["prediction"] = list(
            predictions[index]["ObjectEntities"])
        traces[index]["l1_detail"] = l1_details[index]
    return predictions, traces


def build(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    policy, model_paths = _validate_policy(output)
    source_plan = e2e._validate_plan(output)
    primary, qwen_raw, system2 = _primary_inputs(output, source_plan)
    base_rows = _assemble_from_primary(
        output, source_plan, primary, qwen_raw)
    if len(base_rows) != int(source_plan["rows"]):
        raise ContractError("base graph coverage mismatch")

    gemma = _response_map(source_plan, "gemma:independent")
    ministral_n3 = _response_map(source_plan, e2e.MINISTRAL_N3)
    ministral_cot40 = _response_map(source_plan, e2e.MINISTRAL_COT40)

    decision_graphs: list[dict[str, Any]] = []
    full_graphs: list[dict[str, Any]] = []
    for source in base_rows:
        key = _key(source)
        # _prepare_base_row expects the raw proposal response shape.  The
        # production primary route supplies exact N=10 samples directly.
        qwen_response = {"generations": list(qwen_raw[key])}
        base, qwen_texts, gemma_texts = _prepare_base_row(
            source,
            qwen_response,
            gemma[key],
            primary_objects=primary[key],
            system2_objects=system2.get(key, ()),
        )
        full = _full_evidence_graph(
            base,
            qwen_texts=qwen_texts,
            gemma_texts=gemma_texts,
            ministral_n3=ministral_n3[key],
            ministral_cot40=ministral_cot40[key],
        )
        decision_graphs.append(base)
        full_graphs.append(full)

    graph_path = output / "graph/FINAL_EXACT_EVIDENCE_GRAPH.jsonl"
    write_jsonl_atomic(graph_path, full_graphs)
    predictions, decisions = _apply_frozen_stack(
        decision_graphs, full_graphs, model_paths)
    prediction_path = output / "FINAL_PREDICTIONS.jsonl"
    decision_path = output / "FINAL_DECISIONS.jsonl"
    write_jsonl_atomic(prediction_path, predictions)
    write_jsonl_atomic(decision_path, decisions)

    proof_changes = sum(
        bool(row.get("proof", {}).get("changed")) for row in decisions)
    cache_artifacts = {}
    for name in ("CACHE_PLAN.json", "PRIMARY_MERGE.json"):
        path = output / "cache" / name
        if path.is_file():
            record = _json(path)
            if bool(record.get("contains_labels")) or bool(record.get("gold_aware")):
                raise ContractError(f"cache artifact is not blind-safe: {path}")
            cache_artifacts[name] = {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "schema": record.get("schema"),
            }
    manifest = {
        "schema": RESULT_SCHEMA,
        "policy_id": POLICY_ID,
        "split": policy["split"],
        "blind": policy["blind"],
        "contains_labels": False,
        "gold_aware": False,
        "validation_selected_lineage": True,
        "rows": len(predictions),
        "verified_parameter_total": policy["verified_parameter_total"],
        "parameter_cap": policy["parameter_cap"],
        "policy": str((output / "plan/FINAL_POLICY.json").resolve()),
        "policy_sha256": sha256(output / "plan/FINAL_POLICY.json"),
        "graph": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "decisions": str(decision_path),
        "decisions_sha256": sha256(decision_path),
        "proof_changed_rows": proof_changes,
        "capacity_graph_correction": "strict_singleton_numeric_proof",
        "primary_qwen_policy": PRIMARY_POLICY,
        "qwen_system2_available": True,
        "semantic_inference_cache": cache_artifacts,
    }
    _write_json(output / "FINAL_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def score_validation(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    policy, _ = _validate_policy(output)
    if policy.get("blind") or policy.get("split") != "validation":
        raise ContractError("refusing to score a blind/test prediction")
    path = output / "FINAL_PREDICTIONS.jsonl"
    scores = _score(read_jsonl(path), Path(args.gold).resolve())
    result = {
        "schema": "heterogeneous-final-validation-score-v1",
        "development_only": True,
        "validation_selected_lineage": True,
        "predictions_sha256": sha256(path),
        "pooled_macro_f1": scores[POOLED]["macro-f1"],
        "per_relation": scores,
    }
    _write_json(output / "analysis/FINAL_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def package(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    policy, _ = _validate_policy(output)
    manifest = _json(output / "FINAL_MANIFEST.json")
    predictions = output / "FINAL_PREDICTIONS.jsonl"
    if (
        manifest.get("schema") != RESULT_SCHEMA
        or manifest.get("policy_id") != POLICY_ID
        or manifest.get("predictions_sha256") != sha256(predictions)
        or manifest.get("split") != policy.get("split")
    ):
        raise ContractError("final prediction manifest mismatch")
    source_rows = read_jsonl(Path(e2e._validate_plan(output)["input_rows"]))
    prediction_rows = read_jsonl(predictions)
    if [_key(row) for row in source_rows] != [_key(row) for row in prediction_rows]:
        raise ContractError("final predictions do not match split row order")
    if any(not isinstance(row.get("ObjectEntities"), list)
           for row in prediction_rows):
        raise ContractError("prediction ObjectEntities must be lists")

    package_dir = output / "submission"
    package_dir.mkdir(parents=True, exist_ok=True)
    staged = package_dir / "predictions.jsonl"
    staged.write_bytes(predictions.read_bytes())
    archive = package_dir / f"{POLICY_ID}_{policy['split']}.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED,
    ) as handle:
        handle.write(staged, arcname="predictions.jsonl")
    package_manifest = {
        "schema": "heterogeneous-final-submission-package-v1",
        "split": policy["split"],
        "rows": len(prediction_rows),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "member": "predictions.jsonl",
        "predictions_sha256": sha256(staged),
    }
    _write_json(package_dir / "PACKAGE.json", package_manifest)
    print(json.dumps(package_manifest, indent=2, sort_keys=True))
    return 0


def verify_package(args: argparse.Namespace) -> int:
    """Independently reopen and verify the final Codabench archive."""
    output = Path(args.output_dir).resolve()
    policy, _ = _validate_policy(output)
    source_plan = e2e._validate_plan(output)
    if policy.get("split") != "test" or not bool(policy.get("blind")):
        raise ContractError("official test package must be blind and test-only")
    input_path = Path(str(source_plan["input"])).resolve()
    if sha256(input_path) != OFFICIAL_TEST_SHA256:
        raise ContractError("official test input hash mismatch")

    predictions = output / "FINAL_PREDICTIONS.jsonl"
    final_manifest = _json(output / "FINAL_MANIFEST.json")
    package_manifest = _json(output / "submission/PACKAGE.json")
    archive = Path(str(package_manifest.get("archive", ""))).resolve()
    staged = output / "submission/predictions.jsonl"
    source_rows = read_jsonl(Path(source_plan["input_rows"]))
    prediction_rows = read_jsonl(predictions)
    if (
        final_manifest.get("schema") != RESULT_SCHEMA
        or final_manifest.get("policy_id") != POLICY_ID
        or final_manifest.get("split") != "test"
        or not bool(final_manifest.get("blind"))
        or bool(final_manifest.get("contains_labels"))
        or bool(final_manifest.get("gold_aware"))
        or final_manifest.get("predictions_sha256") != sha256(predictions)
    ):
        raise ContractError("final blind-test manifest contract failed")
    for name, record in final_manifest.get(
            "semantic_inference_cache", {}).items():
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file() or record.get("sha256") != sha256(path):
            raise ContractError(f"semantic cache provenance drift: {name}")
    if (
        len(source_rows) != 475
        or len(prediction_rows) != 475
        or [_key(row) for row in source_rows]
            != [_key(row) for row in prediction_rows]
        or any(not isinstance(row.get("ObjectEntities"), list)
               for row in prediction_rows)
    ):
        raise ContractError("official test prediction coverage/order failed")
    if (
        not archive.is_file()
        or not staged.is_file()
        or int(package_manifest.get("rows", -1)) != 475
        or package_manifest.get("archive_sha256") != sha256(archive)
        or package_manifest.get("predictions_sha256") != sha256(staged)
        or staged.read_bytes() != predictions.read_bytes()
    ):
        raise ContractError("submission package manifest/bytes failed")
    with zipfile.ZipFile(archive) as handle:
        if handle.namelist() != ["predictions.jsonl"]:
            raise ContractError("archive must contain one root predictions.jsonl")
        if handle.read("predictions.jsonl") != predictions.read_bytes():
            raise ContractError("archive prediction bytes differ from final artifact")

    result = {
        "verified": True,
        "blind": True,
        "split": "test",
        "rows": 475,
        "input_sha256": OFFICIAL_TEST_SHA256,
        "predictions_sha256": sha256(predictions),
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "member": "predictions.jsonl",
        "policy_id": POLICY_ID,
        "verified_parameter_total": policy["verified_parameter_total"],
    }
    _write_json(output / "submission/VERIFICATION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    plan_path = output / "plan/PLAN.json"
    if plan_path.is_file():
        plan = _json(plan_path)
        print(
            f"split={plan.get('split')} rows={plan.get('rows')} "
            f"params={plan.get('verified_parameter_total')}/"
            f"{plan.get('parameter_cap')}")
        for route in (
            "gemma:independent",
            e2e.MINISTRAL_N3,
            e2e.MINISTRAL_COT40,
        ):
            job = plan.get("jobs", {}).get(route, {})
            path = Path(str(job.get("response_path", "")))
            done = (
                sum(1 for line in path.open() if line.strip())
                if path.is_file() else 0
            )
            total = int(job.get("tasks", 0))
            state = "complete" if total and done == total else "pending"
            print(f"{route:34s} {done:5d}/{total:<5d} ({state})")
    paths = (
        "plan/PLAN.json",
        "plan/FINAL_POLICY.json",
        "primary_qwen/MANIFEST.json",
        "graph/FINAL_EXACT_EVIDENCE_GRAPH.jsonl",
        "FINAL_PREDICTIONS.jsonl",
        "FINAL_MANIFEST.json",
        f"submission/{POLICY_ID}_test.zip",
    )
    for value in paths:
        path = output / value
        print(f"{value}: {'ready' if path.is_file() else 'pending'}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="command", required=True)
    for command, function in (
        ("freeze", freeze),
        ("build", build),
        ("package", package),
        ("verify-package", verify_package),
        ("status", status),
    ):
        current = subparsers.add_parser(command)
        current.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
        current.set_defaults(function=function)
    score_parser = subparsers.add_parser("score-validation")
    score_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    score_parser.add_argument("--gold", default=str(ROOT / "data/val.jsonl"))
    score_parser.set_defaults(function=score_validation)
    return value


def main() -> int:
    arguments = parser().parse_args()
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
