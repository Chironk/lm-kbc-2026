from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.heterogeneous_agents import rekey_frozen_validation as subject


def test_official_migration_inventory_matches_current_rows() -> None:
    old_rows, current_rows = subject.verify_inputs()
    migrated = [subject.migrated_key(subject.key(row)) for row in old_rows]
    migrated = [value for value in migrated if value is not None]
    assert migrated == [subject.key(row) for row in current_rows]
    assert len(subject.RENAMES) == 20
    assert len(subject.DROPPED) == 3


def test_migration_preserves_predictions_and_drops_only_declared_rows() -> None:
    old_rows, current_rows = subject.verify_inputs()
    source_rows = subject.read_jsonl(subject.SOURCE_ARTIFACTS["strict"])
    migrated = subject.migrate(source_rows, old_rows, current_rows)
    source_by_migrated_key = {
        subject.migrated_key(subject.key(row)): row["ObjectEntities"]
        for row in source_rows
        if subject.migrated_key(subject.key(row)) is not None
    }
    assert len(migrated) == 475
    assert all(
        row["ObjectEntities"] == source_by_migrated_key[subject.key(row)]
        for row in migrated
    )


def test_current_validation_scores_are_exact(tmp_path: Path) -> None:
    result = subject.run(tmp_path)
    safe = result["artifacts"]["safe"]["scores"]["*** All Relations ***"]["macro-f1"]
    strict = result["artifacts"]["strict"]["scores"]["*** All Relations ***"]["macro-f1"]
    assert safe == pytest.approx(0.5259345596620683)
    assert strict == pytest.approx(0.5282278686922194)
    assert result["fresh_model_inference_claimed"] is False
    assert result["gold_used_for_mapping"] is False


def test_result_is_self_consistent(tmp_path: Path) -> None:
    result = subject.run(tmp_path)
    stored = json.loads((tmp_path / "RESULT.json").read_text())
    assert stored == result
    for record in stored["artifacts"].values():
        path = Path(record["prediction"])
        if not path.is_absolute():
            path = subject.ROOT / path
        assert path.is_file()
