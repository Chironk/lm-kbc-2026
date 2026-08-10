#!/usr/bin/env python3
"""Strict pre-submission validator for prediction and raw-cache artifacts.

The official evaluator intentionally tolerates missing/duplicate/malformed rows
by treating them as empty or overwriting them in a dictionary.  That is useful
for scoring but dangerous for producing a submission.  This validator fails
closed and never modifies evaluate.py.
"""
import argparse
from collections import Counter

from evaluate import RELATION_TYPE, read_jsonl_file

NEVER_EMPTY = {"hasArea", "hasCapacity", "awardWonBy"}


def keyed(rows, label):
    keys = [(r.get("SubjectEntity"), r.get("Relation")) for r in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"{label}: {len(duplicates)} duplicate keys, e.g. {duplicates[:3]}")
    return set(keys)


def validate_predictions(predictions, reference, allow_never_empty=False):
    errors = []
    pred_keys = keyed(predictions, "predictions")
    ref_keys = keyed(reference, "reference")
    missing, extra = ref_keys - pred_keys, pred_keys - ref_keys
    if missing:
        errors.append(f"{len(missing)} missing keys, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} extra keys, e.g. {sorted(extra)[:3]}")

    for i, row in enumerate(predictions):
        rel = row.get("Relation")
        objects = row.get("ObjectEntities")
        if rel not in RELATION_TYPE:
            errors.append(f"row {i}: unknown relation {rel!r}")
        if not isinstance(row.get("SubjectEntity"), str):
            errors.append(f"row {i}: SubjectEntity is not a string")
        if not isinstance(objects, list) or any(not isinstance(x, str) for x in objects):
            errors.append(f"row {i}: ObjectEntities must be a flat list of strings")
        elif rel in NEVER_EMPTY and not objects and not allow_never_empty:
            errors.append(f"row {i}: guaranteed-nonempty relation {rel} predicted empty")
        if len(errors) >= 25:
            break
    return errors


def validate_raw(raw_rows, reference, n_samples, *,
                 require_system1_provenance=False,
                 exclude_target_from_shots=False):
    errors = []
    raw_keys = keyed(raw_rows, "raw cache")
    ref_keys = keyed(reference, "reference")
    if raw_keys != ref_keys:
        errors.append(f"raw/reference key mismatch: missing={len(ref_keys - raw_keys)}, "
                      f"extra={len(raw_keys - ref_keys)}")
    for i, row in enumerate(raw_rows):
        samples = row.get("raw_samples")
        if not isinstance(samples, list) or len(samples) != n_samples:
            errors.append(f"raw row {i}: expected {n_samples} samples, got "
                          f"{len(samples) if isinstance(samples, list) else type(samples).__name__}")
        statuses = row.get("sample_statuses")
        if statuses is not None and len(statuses) != len(samples or []):
            errors.append(f"raw row {i}: sample_statuses length mismatch")
        if require_system1_provenance:
            if not isinstance(statuses, list) or len(statuses) != len(samples or []):
                errors.append(f"raw row {i}: complete sample_statuses required")
            variants = row.get("prompt_variants")
            if not isinstance(variants, list) or len(variants) != len(samples or []):
                errors.append(f"raw row {i}: prompt_variants length mismatch")
            shot_sets = row.get("shot_subjects_by_sample")
            if not isinstance(shot_sets, list) or len(shot_sets) != len(samples or []):
                errors.append(f"raw row {i}: shot_subjects_by_sample length mismatch")
            else:
                target = row.get("SubjectEntity")
                for sample_index, shot_subjects in enumerate(shot_sets):
                    if (not isinstance(shot_subjects, list)
                            or any(not isinstance(item, str) for item in shot_subjects)):
                        errors.append(
                            f"raw row {i}: invalid shot list at sample {sample_index}")
                    elif exclude_target_from_shots and target in shot_subjects:
                        errors.append(
                            f"raw row {i}: target appears in sample {sample_index} shots")
            if (exclude_target_from_shots
                    and row.get("SubjectEntity") in row.get("shot_subjects", [])):
                errors.append(f"raw row {i}: target appears in summary shot_subjects")
        if len(errors) >= 25:
            break
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--reference", required=True,
                    help="Input/reference JSONL defining the exact expected keys")
    ap.add_argument("--raw-cache")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--allow-never-empty", action="store_true",
                    help="Diagnostic only: permit empty area/capacity/award rows")
    ap.add_argument("--require-system1-provenance", action="store_true",
                    help="Require statuses, prompt variants, and per-sample shot lists.")
    ap.add_argument("--exclude-target-from-shots", action="store_true",
                    help="With --require-system1-provenance, reject target-shot leakage.")
    args = ap.parse_args()
    if args.exclude_target_from_shots and not args.require_system1_provenance:
        ap.error("--exclude-target-from-shots requires --require-system1-provenance")

    reference = read_jsonl_file(args.reference)
    errors = validate_predictions(
        read_jsonl_file(args.predictions), reference, args.allow_never_empty)
    if args.raw_cache:
        errors.extend(validate_raw(
            read_jsonl_file(args.raw_cache), reference, args.n_samples,
            require_system1_provenance=args.require_system1_provenance,
            exclude_target_from_shots=args.exclude_target_from_shots))
    if errors:
        print("INVALID artifact:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(f"VALID: {args.predictions} matches {len(reference)} reference keys")
    if args.raw_cache:
        print(f"VALID: {args.raw_cache} has {args.n_samples} samples per key")


if __name__ == "__main__":
    main()
