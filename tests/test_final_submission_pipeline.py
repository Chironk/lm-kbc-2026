import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from experiments.heterogeneous_agents.core import ContractError
from experiments.heterogeneous_agents import final_submission_pipeline as final


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_frozen_model_snapshot_is_intact_and_split_independent() -> None:
    manifest, paths = final._snapshot_artifacts()
    assert manifest["verified_parameter_total"] == 30_515_165_024
    assert manifest["parameter_cap"] == 32_000_000_000
    assert set(paths) == set(final.MODEL_ARTIFACTS)
    serialized = json.dumps(manifest, sort_keys=True)
    assert "ObjectEntities" not in serialized
    assert "validation_predictions" not in serialized.lower()


def test_exact_candidate_recovery_counts_each_generation_once() -> None:
    row = {
        "Relation": "companyTradesAtStockExchange",
        "candidates": [],
    }
    final._add_exact_base_candidates(
        row,
        agent_id=final.e2e.QWEN,
        raw_texts=("NASDAQ; NASDAQ", "NYSE; NASDAQ"),
        parser=lambda text, _: [part.strip() for part in text.split(";")],
        selected_objects=("NASDAQ",),
    )
    by_item = {node["item"]: node for node in row["candidates"]}
    assert by_item["NASDAQ"]["proposal_support"][final.e2e.QWEN] == 2
    assert by_item["NYSE"]["proposal_support"][final.e2e.QWEN] == 1
    assert by_item["NASDAQ"]["selected_by"][final.e2e.QWEN] is True
    assert by_item["NYSE"]["selected_by"][final.e2e.QWEN] is False


def test_primary_qwen_adapter_rows_are_visible_to_generic_assembler() -> None:
    source = {
        "SubjectEntity": "Example Island",
        "Relation": "hasArea",
        "ObjectEntities": [],
    }
    qwen_agent = {"id": final.e2e.QWEN}
    gemma_agent = {"id": final.e2e.GEMMA}
    gemma_map = {}
    for phase, selected in (
        ("commit_existence", "YES"),
        ("commit_cardinality", "ONE"),
    ):
        gemma_map[phase] = {
            "subject": source["SubjectEntity"],
            "relation": source["Relation"],
            "phase": phase,
            "selected_choice": selected,
            "choice_probabilities": {selected: 1.0},
        }
    gemma_map["propose"] = {
        "subject": source["SubjectEntity"],
        "relation": source["Relation"],
        "phase": "propose",
        "generations": ["ANSWER: 42"],
    }
    source_plan = {
        "input_rows": "/unused/input.jsonl",
        "cot_agents": "/unused/agents.json",
        "jobs": {"gemma:independent": {}},
    }
    with (
        mock.patch.object(final, "read_jsonl", return_value=[source]),
        mock.patch.object(final, "load_agent_config", return_value={}),
        mock.patch.object(
            final.e2e, "_agent_of",
            side_effect=[qwen_agent, gemma_agent],
        ),
        mock.patch.object(
            final.e2e, "_validated_responses",
            return_value=list(gemma_map.values()),
        ),
    ):
        graphs = final._assemble_from_primary(
            Path("/unused"),
            source_plan,
            {(source["SubjectEntity"], source["Relation"]): ["42"]},
            {(source["SubjectEntity"], source["Relation"]): ["ANSWER: 42"]},
        )
    assert len(graphs) == 1
    assert graphs[0]["commitments"][final.e2e.QWEN] == {
        "existence": "YES",
        "existence_probabilities": {"YES": 1.0},
        "cardinality": "ONE",
        "cardinality_probabilities": {"ONE": 1.0},
    }
    assert graphs[0]["proposal_sample_counts"][final.e2e.QWEN] == 1


def test_prepare_base_row_drops_legacy_parser_only_candidate() -> None:
    source = {
        "SubjectEntity": "Example Corp",
        "Relation": "companyTradesAtStockExchange",
        "ObjectEntities": [],
        "candidates": [{
            "key": "moscow exchange think",
            "item": "Moscow Exchange</think>",
            "proposal_support": {final.e2e.QWEN: 1},
            "proposer_agents": [final.e2e.QWEN],
            "selected_by": {final.e2e.QWEN: False, final.e2e.GEMMA: False},
        }],
        "commitments": {
            final.e2e.QWEN: {"existence": "YES", "cardinality": "ONE"},
            final.e2e.GEMMA: {"existence": "NO", "cardinality": "ZERO"},
        },
        "proposal_sample_counts": {final.e2e.QWEN: 1, final.e2e.GEMMA: 1},
        "proposal_parse_diagnostics": {
            final.e2e.QWEN: {"parsed_nonempty": 1},
            final.e2e.GEMMA: {"explicit_none": 1},
        },
    }
    with (
        mock.patch.object(final, "prediction_for_agent", return_value=[]),
        mock.patch.object(final, "augment_graph", side_effect=lambda row, _: row),
        mock.patch.object(
            final, "augment_relational_graph", side_effect=lambda row: row),
    ):
        row, _, _ = final._prepare_base_row(
            source,
            {"generations": ["ANSWER: Moscow Exchange"]},
            {"generations": ["ANSWER: None"]},
            primary_objects=["Moscow Exchange"],
            system2_objects=[],
        )
    assert [node["item"] for node in row["candidates"]] == ["Moscow Exchange"]


def test_primary_manifest_binds_to_normalized_input_rows() -> None:
    """The primary adapter validates the artifact it actually consumed."""
    source = [{
        "SubjectEntity": "Example Island",
        "Relation": "hasArea",
        "ObjectEntities": [],
    }]
    prediction = [{**source[0], "ObjectEntities": ["42"]}]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        primary = output / "primary_qwen"
        normalized = output / "plan/INPUT_ROWS.jsonl"
        raw_split = output / "raw-test.jsonl"
        prediction_path = primary / f"submission_{final.PRIMARY_POLICY}.jsonl"
        _write_jsonl(normalized, source)
        _write_jsonl(raw_split, source + [{
            "SubjectEntity": "Different bytes",
            "Relation": "hasArea",
            "ObjectEntities": [],
        }])
        _write_jsonl(prediction_path, prediction)
        final._write_json(primary / "MANIFEST.json", {
            "policy": final.PRIMARY_POLICY,
            "input_sha256": final.sha256(normalized),
            "submission_sha256": final.sha256(prediction_path),
        })
        plan = {
            "input": str(raw_split),
            "input_sha256": final.sha256(raw_split),
            "input_rows": str(normalized),
            "input_rows_sha256": final.sha256(normalized),
            "rows": 1,
        }
        submission = mock.MagicMock()
        with (
            mock.patch.object(final, "PrimarySubmission", return_value=submission),
            mock.patch.object(final, "read_jsonl", return_value=prediction),
        ):
            # It advances past the manifest check and fails only because this
            # focused fixture intentionally substitutes an invalid raw row.
            with pytest.raises(ContractError, match="invalid primary Qwen raw row"):
                final._primary_inputs(output, plan)
        submission.validate_all_bundles.assert_called_once_with()


def test_relation_typed_graph_correction_uses_numeric_proof_for_capacity() -> None:
    graphs = [{
        "SubjectEntity": "Example Venue",
        "Relation": "hasCapacity",
        "relational_graph": {
            "components": [{
                "id": "component:0",
                "node_type": "candidate_component",
                "representative": "10000",
            }],
            "nodes": [
                {
                    "id": f"event:{family}",
                    "node_type": "evidence_event",
                    "model_family": family,
                    "status": "candidate_set",
                }
                for family in (
                    "qwen_recall", "gemma_independent",
                    "ministral_independent",
                )
            ],
            "edges": [
                {
                    "source": f"event:{family}",
                    "target": "component:0",
                    "edge_type": "supports",
                }
                for family in (
                    "qwen_recall", "gemma_independent",
                    "ministral_independent",
                )
            ],
        },
    }]
    predictions, decisions = final._apply_relation_typed_graph_correction(
        graphs, {("Example Venue", "hasCapacity"): ["9000"]})
    assert predictions[0]["ObjectEntities"] == ["10000"]
    assert decisions[0]["arm"] == "strict_singleton_numeric_proof"
    assert decisions[0]["changed"] is True


def test_relation_typed_graph_correction_keeps_set_proof_for_strings() -> None:
    graph = {
        "SubjectEntity": "Example Corp",
        "Relation": "companyTradesAtStockExchange",
    }
    expected_prediction = {
        **graph,
        "ObjectEntities": ["NASDAQ"],
    }
    expected_decision = {
        **graph,
        "arm": final.PRIMARY_ARM,
        "changed": True,
    }
    with mock.patch.object(
        final,
        "proof_decode",
        return_value=([expected_prediction], [expected_decision]),
    ) as decoder:
        predictions, decisions = final._apply_relation_typed_graph_correction(
            [graph], {(graph["SubjectEntity"], graph["Relation"]): ["NYSE"]})
    decoder.assert_called_once()
    assert predictions == [expected_prediction]
    assert decisions == [expected_decision]


def test_test_split_scoring_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        with mock.patch.object(
            final, "_validate_policy",
            return_value=({"blind": True, "split": "test"}, {}),
        ):
            with pytest.raises(ContractError, match="refusing to score"):
                final.score_validation(SimpleNamespace(
                    output_dir=str(output), gold="data/val.jsonl"))


def test_package_has_exactly_predictions_jsonl_at_zip_root() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        source = [{
            "SubjectEntity": "Example",
            "Relation": "hasArea",
        }]
        predictions = [{
            "SubjectEntity": "Example",
            "Relation": "hasArea",
            "ObjectEntities": ["42"],
        }]
        _write_jsonl(output / "plan/INPUT_ROWS.jsonl", source)
        _write_jsonl(output / "FINAL_PREDICTIONS.jsonl", predictions)
        final._write_json(output / "FINAL_MANIFEST.json", {
            "schema": final.RESULT_SCHEMA,
            "policy_id": final.POLICY_ID,
            "split": "test",
            "predictions_sha256": final.sha256(
                output / "FINAL_PREDICTIONS.jsonl"),
        })
        policy = {"split": "test"}
        source_plan = {"input_rows": str(output / "plan/INPUT_ROWS.jsonl")}
        with (
            mock.patch.object(final, "_validate_policy", return_value=(policy, {})),
            mock.patch.object(final.e2e, "_validate_plan", return_value=source_plan),
        ):
            final.package(SimpleNamespace(output_dir=str(output)))
        archive = (
            output / "submission" /
            f"{final.POLICY_ID}_test.zip"
        )
        with zipfile.ZipFile(archive) as handle:
            assert handle.namelist() == ["predictions.jsonl"]
            assert handle.read("predictions.jsonl") == (
                output / "FINAL_PREDICTIONS.jsonl").read_bytes()


def test_cli_exposes_frozen_submission_stages() -> None:
    parser = final.parser()
    for command in (
        "freeze", "build", "package", "verify-package",
        "score-validation", "status",
    ):
        args = parser.parse_args([command, "--output-dir", "/tmp/final-test"])
        assert callable(args.function)


def test_final_wrapper_uses_aligned_capacity_prompt_contract() -> None:
    wrapper = (final.HERE / "run_final_submission_pipeline.sh").read_text()
    assert "data/synthetic_cot_capacity_aligned_v2.jsonl" in wrapper
    assert 'QUESTION_CONTRACT="${QUESTION_CONTRACT:-official-v1}"' in wrapper
    assert 'SYNTHETIC_COT="$SYNTHETIC_COT"' in wrapper
    assert 'QUESTION_CONTRACT="$QUESTION_CONTRACT"' in wrapper
