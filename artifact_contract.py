"""Fail-closed validation for resumable production inference artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from evaluate import read_jsonl_file
from sample_evidence import validate_recorded_statuses
from validate_artifact import validate_predictions, validate_raw


class ArtifactContractError(RuntimeError):
    """An existing artifact is incomplete, stale, or has false provenance."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(label: str, errors: Iterable[str]) -> None:
    errors = list(errors)
    if errors:
        joined = "\n  - ".join(errors)
        raise ArtifactContractError(f"{label} failed artifact contract:\n  - {joined}")


def _read_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ArtifactContractError(f"missing manifest: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"unreadable manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactContractError(f"manifest is not a JSON object: {path}")
    return value


def _verify_manifest(
    label: str,
    manifest_path: Path,
    expected_fields: Mapping[str, Any],
    hash_fields: Mapping[str, Path],
) -> Dict[str, Any]:
    manifest = _read_manifest(manifest_path)
    errors = []
    for field, expected in expected_fields.items():
        actual = manifest.get(field, "<MISSING>")
        if actual != expected:
            errors.append(f"manifest {field}: expected {expected!r}, got {actual!r}")
    for field, path in hash_fields.items():
        if not path.is_file():
            errors.append(f"missing artifact for {field}: {path}")
            continue
        actual = sha256(path)
        declared = manifest.get(field)
        if declared != actual:
            errors.append(f"{field}: manifest={declared!r}, actual={actual!r}")
    _fail(label, errors)
    return manifest


def _keyed(rows: Sequence[Dict[str, Any]], label: str) -> Dict[tuple, Dict[str, Any]]:
    result: Dict[tuple, Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ArtifactContractError(f"{label} row {index} is not an object")
        key = row.get("SubjectEntity"), row.get("Relation")
        if key in result:
            raise ArtifactContractError(f"duplicate {label} key: {key}")
        result[key] = row
    return result


def validate_system1_bundle(
    *,
    label: str,
    reference_path: Path,
    predictions_path: Path,
    raw_path: Path,
    manifest_path: Path,
    expected_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a System-1 prediction/raw/manifest bundle before reuse."""
    reference = read_jsonl_file(reference_path)
    predictions = read_jsonl_file(predictions_path) if predictions_path.is_file() else []
    raw = read_jsonl_file(raw_path) if raw_path.is_file() else []
    errors = validate_predictions(predictions, reference)
    errors.extend(validate_raw(raw, reference, n_samples=10))

    raw_by_key = _keyed(raw, f"{label} raw")
    for key, row in raw_by_key.items():
        subject = key[0]
        samples = row.get("raw_samples", [])
        if any(not isinstance(sample, str) for sample in samples):
            errors.append(f"{key}: raw_samples must contain only strings")
        statuses = row.get("sample_statuses")
        if not isinstance(statuses, list) or len(statuses) != len(samples):
            errors.append(f"{key}: sample_statuses must match raw_samples")
        else:
            protocol = expected_manifest.get("response_protocol", "legacy-cot")
            errors.extend(
                f"{key}: {error}"
                for error in validate_recorded_statuses(samples, statuses, protocol)
            )
        variants = row.get("prompt_variants")
        if not isinstance(variants, list) or len(variants) != len(samples):
            errors.append(f"{key}: prompt_variants must match raw_samples")
        shot_sets = row.get("shot_subjects_by_sample")
        if not isinstance(shot_sets, list) or len(shot_sets) != len(samples):
            errors.append(f"{key}: shot_subjects_by_sample must match raw_samples")
        else:
            for sample_index, shot_subjects in enumerate(shot_sets):
                if (not isinstance(shot_subjects, list)
                        or any(not isinstance(item, str) for item in shot_subjects)):
                    errors.append(f"{key}: invalid shot list for sample {sample_index}")
                    continue
                if subject in shot_subjects:
                    errors.append(f"{key}: target appears in sample {sample_index} shots")
        if subject in row.get("shot_subjects", []):
            errors.append(f"{key}: target appears in summary shot_subjects")
        if len(errors) >= 50:
            break
    _fail(label, errors)

    return _verify_manifest(
        label,
        manifest_path,
        expected_manifest,
        {
            "input_sha256": reference_path,
            "output_sha256": predictions_path,
            "raw_cache_sha256": raw_path,
        },
    )


def validate_system2_bundle(
    *,
    label: str,
    reference_path: Path,
    predictions_path: Path,
    raw_path: Path,
    manifest_path: Path,
    expected_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a System-2 prediction/raw/manifest bundle before reuse."""
    reference = read_jsonl_file(reference_path)
    predictions = read_jsonl_file(predictions_path) if predictions_path.is_file() else []
    raw = read_jsonl_file(raw_path) if raw_path.is_file() else []
    errors = validate_predictions(predictions, reference)
    reference_keys = set(_keyed(reference, "System-2 reference"))
    raw_keys = set(_keyed(raw, "System-2 raw"))
    if raw_keys != reference_keys:
        errors.append(
            f"raw/reference key mismatch: missing={len(reference_keys - raw_keys)}, "
            f"extra={len(raw_keys - reference_keys)}")
    _fail(label, errors)

    return _verify_manifest(
        label,
        manifest_path,
        expected_manifest,
        {
            "input_sha256": reference_path,
            "config_sha256": Path(expected_manifest["config"]),
            "output_sha256": predictions_path,
            "raw_cache_sha256": raw_path,
        },
    )
