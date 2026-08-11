import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.heterogeneous_agents import historical_sota_test_pipeline as paired


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_archived_04845_submission_is_intact() -> None:
    payload = paired._archived_prediction_bytes()
    assert len([line for line in payload.splitlines() if line]) == 475


def test_historical_prompt_and_portfolio_inputs_are_hash_pinned() -> None:
    assert paired.sha256(paired.DEFAULT_INPUT) == paired.EXPECTED_TEST_SHA256
    assert paired.sha256(paired.DEFAULT_SYNTHETIC) == paired.EXPECTED_SYNTHETIC_SHA256
    assert paired.sha256(paired.DEFAULT_COT_AGENTS) == paired.EXPECTED_COT_CONFIG_SHA256
    assert paired.sha256(paired.DEFAULT_SUPPLY_AGENTS) == paired.EXPECTED_SUPPLY_CONFIG_SHA256


def test_component_cot40_area_uses_unique_seven_of_ten_component() -> None:
    graph = {
        "SubjectEntity": "Example Island",
        "Relation": "hasArea",
        "relational_graph": {
            "nodes": [
                {
                    "node_type": "candidate_component",
                    "representative": "1000",
                    "routes": {
                        paired.e2e.MINISTRAL_COT40: {
                            "distinct_generation_support": 8,
                        },
                    },
                },
                {
                    "node_type": "candidate_component",
                    "representative": "2000",
                    "routes": {
                        paired.e2e.MINISTRAL_COT40: {
                            "distinct_generation_support": 2,
                        },
                    },
                },
            ],
        },
    }
    selected, detail = paired._component_cot40_area(graph, ["900"])
    assert selected == ["1000"]
    assert detail["highest_support"] == 8
    assert detail["applied"] is True


def test_component_cot40_area_fails_closed_on_tied_winners() -> None:
    graph = {
        "SubjectEntity": "Example Island",
        "Relation": "hasArea",
        "relational_graph": {
            "nodes": [
                {
                    "node_type": "candidate_component",
                    "representative": value,
                    "routes": {
                        paired.e2e.MINISTRAL_COT40: {
                            "distinct_generation_support": 7,
                        },
                    },
                }
                for value in ("1000", "2000")
            ],
        },
    }
    selected, detail = paired._component_cot40_area(graph, ["900"])
    assert selected == ["900"]
    assert detail["reason"] == "no_unique_7_of_10_numeric_component"


def test_historical_adapter_does_not_apply_later_parser_inventory_filter() -> None:
    source = {
        "SubjectEntity": "Example Corp",
        "Relation": "companyTradesAtStockExchange",
        "ObjectEntities": [],
        "candidates": [{
            "key": "legacy-only",
            "item": "Legacy-only candidate",
            "proposal_support": {paired.e2e.QWEN: 1},
            "proposer_agents": [paired.e2e.QWEN],
            "selected_by": {paired.e2e.QWEN: False, paired.e2e.GEMMA: False},
        }],
        "commitments": {
            paired.e2e.QWEN: {"existence": "YES", "cardinality": "ONE"},
            paired.e2e.GEMMA: {"existence": "NO", "cardinality": "ZERO"},
        },
        "proposal_sample_counts": {paired.e2e.QWEN: 1, paired.e2e.GEMMA: 1},
        "proposal_parse_diagnostics": {
            paired.e2e.QWEN: {"parsed_nonempty": 1},
            paired.e2e.GEMMA: {"explicit_none": 1},
        },
    }
    with (
        mock.patch.object(paired, "prediction_for_agent", return_value=[]),
        mock.patch.object(paired, "augment_graph", side_effect=lambda row, _: row),
        mock.patch.object(paired, "augment_relational_graph", side_effect=lambda row: row),
    ):
        row, _, _ = paired._prepare_base_row_historical(
            source,
            {"generations": ["ANSWER: NASDAQ"]},
            {"generations": ["ANSWER: None"]},
            primary_objects=["NASDAQ"],
            system2_objects=[],
        )
    assert "Legacy-only candidate" in [node["item"] for node in row["candidates"]]


def test_package_writes_exactly_one_predictions_member() -> None:
    source = [{
        "SubjectEntity": "Example",
        "Relation": "hasArea",
    }]
    prediction = [{**source[0], "ObjectEntities": ["42"]}]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        source_path = output / "plan/INPUT_ROWS.jsonl"
        prediction_path = output / "FINAL_PREDICTIONS.jsonl"
        _write_jsonl(source_path, source)
        _write_jsonl(prediction_path, prediction)
        with (
            mock.patch.object(
                paired,
                "_validate_plan_contract",
                return_value={"input_rows": str(source_path)},
            ),
            mock.patch.object(paired, "EXPECTED_TEST_ROWS", 1),
        ):
            result = paired._package_one(output, prediction_path, "submission.zip")
        with zipfile.ZipFile(result["archive"]) as handle:
            assert handle.namelist() == ["predictions.jsonl"]
            assert handle.read("predictions.jsonl") == prediction_path.read_bytes()


def test_cli_exposes_resumable_cpu_stages() -> None:
    parser = paired.parser()
    for command in ("plan", "build", "package", "status"):
        args = parser.parse_args([command, "--output-dir", "/tmp/paired-sota"])
        assert callable(args.function)
