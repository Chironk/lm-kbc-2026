#!/usr/bin/env python3
"""Re-key frozen 478-row development predictions to the official 475 rows.

The organizer update in commit ``e079024`` disambiguated twenty subject
strings and removed three unresolved capacity queries.  It did not provide a
new model run.  This utility applies only that published key migration to the
two immutable development prediction artifacts frozen on 2026-08-03.

This is an artifact-compatibility evaluation, not fresh model inference.  The
mapping never examines gold objects; development labels are opened only after
the migrated predictions have been written, in order to score them with the
official evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate as official


ROOT = Path(__file__).resolve().parents[2]
ORGANIZER_UPDATE_COMMIT = "e079024b809568f28cded5537203dfd52f5b04ae"
OLD_VALIDATION = ROOT / "data/archive/validation_478_20260729.jsonl"
CURRENT_VALIDATION = ROOT / "data/val.jsonl"
EVALUATOR = ROOT / "evaluate.py"
SOURCE_DIR = ROOT / "results/heterogeneous/candidates/frozen_20260803"
DEFAULT_OUTPUT = (
    ROOT / "results/heterogeneous/candidates/"
    "frozen_20260811_current_validation"
)

EXPECTED_HASHES = {
    "old_validation": "90e4f2475e7e69caf9316ffd3b2e0bc4fe2cd428a99027f2abf08c9f88c18d02",
    "current_validation": "ba86b53ac38eb4b23b80391b291e5987ff4bbfe79827596fc09751b1bb0ce2be",
    "evaluator": "2d592ae177c7b230922bb959da7a8ee1c4c662bf72a99d4dbd0cf62170ff9e22",
    "safe_source": "6c4d4bb1ed60054cb1b2d9a6aa728a1a5f0422c714bdd8486963cc986ca348ae",
    "strict_source": "61d062f8c10e6262666d0e581197f4db41962125db16351258270bb13a2f0bc7",
}

SOURCE_ARTIFACTS = {
    "safe": SOURCE_DIR / "safe_0_518450_validation.jsonl",
    "strict": SOURCE_DIR / "strict_proof_0_520729_validation.jsonl",
}

# Exact subject-key changes published by the organizer.  Relation names do
# not change.  Keeping these explicit avoids inferring identity from labels.
RENAMES = {
    ("BDO", "companyTradesAtStockExchange"): "BDO Unibank",
    ("Toho", "companyTradesAtStockExchange"): "Toho Co., Ltd.",
    ("Orbis", "companyTradesAtStockExchange"): "Orbis S.A. (hotel group)",
    ("Coco Island", "hasArea"): "Cocos Island, Costa Rica",
    ("Basque Country", "hasArea"): "Basque Country (greater region)",
    ("San Michele", "hasArea"): "Isola di San Michele",
    ("Lake Prespa", "hasArea"): "Great Prespa Lake",
    ("Mainland", "hasArea"): "Mainland, Orkney",
    ("Lough Erne", "hasArea"): "Lower Lough Erne",
    ("Grande Terre", "hasArea"): "Grande Terre (New Caledonia)",
    ("Erich Bethe", "personHasCityOfDeath"): "Erich Bethe (philanthropist)",
    ("Bernard Shaw", "personHasCityOfDeath"): "Bernard Shaw (journalist)",
    ("Sultan Khan", "personHasCityOfDeath"): "Sultan Khan (musician)",
    ("Peter Gregg", "personHasCityOfDeath"): "Peter Gregg (racing driver)",
    ("Terry Harper", "personHasCityOfDeath"): "Terry Harper (ice hockey)",
    ("Thomas Köhler", "personHasCityOfDeath"): "Thomas Köhler (luger)",
    ("James Caan", "personHasCityOfDeath"): "James Caan (actor)",
    ("Peter Diamond", "personHasCityOfDeath"): "Peter Diamond (economist)",
    ("Vitaly Efimov", "personHasCityOfDeath"): "Vitaly Efimov (politician)",
    ("Vladimir Vasiliev", "personHasCityOfDeath"): "Vladimir Vasiliev (dancer)",
}

DROPPED = frozenset({
    ("University Stadium in Georgia", "hasCapacity"),
    ("League Park in Ohio", "hasCapacity"),
    ("Tiger Stadium in Texas", "hasCapacity"),
})


class ContractError(RuntimeError):
    """Raised when an immutable migration input no longer matches."""


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["SubjectEntity"]), str(row["Relation"])


def migrated_key(source_key: tuple[str, str]) -> tuple[str, str] | None:
    if source_key in DROPPED:
        return None
    subject, relation = source_key
    return RENAMES.get(source_key, subject), relation


def verify_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = {
        "old_validation": OLD_VALIDATION,
        "current_validation": CURRENT_VALIDATION,
        "evaluator": EVALUATOR,
        "safe_source": SOURCE_ARTIFACTS["safe"],
        "strict_source": SOURCE_ARTIFACTS["strict"],
    }
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != EXPECTED_HASHES[name]:
            raise ContractError(f"immutable input drift: {name} ({path})")
    old_rows = read_jsonl(OLD_VALIDATION)
    current_rows = read_jsonl(CURRENT_VALIDATION)
    if len(old_rows) != 478 or len(current_rows) != 475:
        raise ContractError("unexpected validation row count")
    expected_keys = [migrated_key(key(row)) for row in old_rows]
    expected_keys = [value for value in expected_keys if value is not None]
    current_keys = [key(row) for row in current_rows]
    if expected_keys != current_keys:
        raise ContractError("organizer key migration no longer matches current validation")
    if len(RENAMES) != 20 or len(DROPPED) != 3:
        raise ContractError("migration inventory drift")
    return old_rows, current_rows


def migrate(
    source_rows: Sequence[Mapping[str, Any]],
    old_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if [key(row) for row in source_rows] != [key(row) for row in old_rows]:
        raise ContractError("source prediction order does not match old validation")
    migrated: list[dict[str, Any]] = []
    for row in source_rows:
        target_key = migrated_key(key(row))
        if target_key is None:
            continue
        migrated.append({
            "SubjectEntity": target_key[0],
            "Relation": target_key[1],
            "ObjectEntities": list(row.get("ObjectEntities", [])),
        })
    if len(migrated) != 475 or [key(row) for row in migrated] != [
        key(row) for row in current_rows
    ]:
        raise ContractError("migrated prediction coverage/order mismatch")
    return migrated


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ))


def display_path(path: Path) -> str:
    """Prefer repository-relative paths while supporting temporary test dirs."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def score(
    predictions: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return official.macro_average_per_relation(
        official.evaluate_per_sr_pair(
            list(predictions), list(gold), official.RELATION_TYPE))


def run(output: Path) -> dict[str, Any]:
    old_rows, current_rows = verify_inputs()
    records: dict[str, Any] = {}
    for name, source_path in SOURCE_ARTIFACTS.items():
        source = read_jsonl(source_path)
        predictions = migrate(source, old_rows, current_rows)
        output_path = output / f"{name}_current_validation_475.jsonl"
        write_jsonl(output_path, predictions)
        scores = score(predictions, current_rows)
        records[name] = {
            "source": display_path(source_path),
            "source_sha256": sha256(source_path),
            "prediction": display_path(output_path),
            "prediction_sha256": sha256(output_path),
            "rows": len(predictions),
            "scores": scores,
        }
    result = {
        "schema": "frozen-validation-artifact-rekey-v1",
        "development_only": True,
        "evaluation_split": "current_validation_475",
        "reproduction_tier": "frozen_prediction_artifact_compatibility_replay",
        "fresh_model_inference_claimed": False,
        "organizer_update_commit": ORGANIZER_UPDATE_COMMIT,
        "old_validation_sha256": EXPECTED_HASHES["old_validation"],
        "current_validation_sha256": EXPECTED_HASHES["current_validation"],
        "evaluator_sha256": EXPECTED_HASHES["evaluator"],
        "renamed_subjects": len(RENAMES),
        "dropped_queries": len(DROPPED),
        "gold_used_for_mapping": False,
        "gold_used_for_scoring": True,
        "artifacts": records,
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output_dir.resolve())
    print(json.dumps({
        name: {
            "macro_f1": record["scores"]["*** All Relations ***"]["macro-f1"],
            "prediction": record["prediction"],
            "sha256": record["prediction_sha256"],
        }
        for name, record in result["artifacts"].items()
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
