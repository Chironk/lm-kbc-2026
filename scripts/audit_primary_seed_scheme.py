#!/usr/bin/env python3
"""Verify that a generated primary-Qwen bundle obeys its frozen seed regime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_inference import generation_seed_for
from run_submission import Submission


BASE_SEED = 45


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text().splitlines()
              if line.strip()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"expected JSONL objects: {path}")
    return values


def audit(output: Path) -> dict[str, Any]:
    policy = _json(output / "plan/FINAL_POLICY.json")
    source_plan = _json(output / "plan/PLAN.json")
    scheme = policy.get("primary_seed_scheme")
    if scheme not in {"legacy", "stable-key"}:
        raise ValueError(f"invalid frozen primary seed scheme: {scheme!r}")

    primary = output / "primary_qwen"
    # Reuse the production artifact contracts before auditing individual seeds.
    namespace = argparse.Namespace(
        policy="v0495",
        input=str(Path(source_plan["input"])),
        output_dir=str(primary),
        dry_run=False,
        skip_inference=True,
        stage="compose",
        seed_scheme=scheme,
    )
    submission = Submission(namespace)
    submission.validate_all_bundles()

    checked_rows = 0
    retry_rows = 0
    keys: set[tuple[str, str]] = set()
    for tag in ("borders", "fp16"):
        manifest_path = primary / f"raw_{tag}.jsonl.manifest.json"
        raw_path = primary / f"raw_{tag}.jsonl"
        manifest = _json(manifest_path)
        if manifest.get("seed_scheme") != scheme:
            raise ValueError(
                f"{tag}: manifest seed scheme does not match frozen policy")
        for index, row in enumerate(_rows(raw_path)):
            subject = str(row["SubjectEntity"])
            relation = str(row["Relation"])
            key = (subject, relation)
            if key in keys:
                raise ValueError(f"duplicate primary raw key: {key}")
            keys.add(key)
            actual = int(row["generation_seed"])
            attempts = [
                generation_seed_for(
                    BASE_SEED, index, relation, subject, attempt, scheme)
                for attempt in (0, 1)
            ]
            if actual not in attempts:
                raise ValueError(
                    f"{tag}/{key}: recorded seed {actual} is not an allowed "
                    f"attempt seed {attempts}")
            attempt = attempts.index(actual)
            retry_rows += int(attempt > 0)
            if scheme == "stable-key":
                moved = generation_seed_for(
                    BASE_SEED, index + 1_000_000, relation, subject,
                    attempt, scheme)
                if moved != actual:
                    raise ValueError(
                        f"{tag}/{key}: stable seed changes with row position")
            checked_rows += 1

    report = {
        "schema": "primary-qwen-seed-audit-v1",
        "verified": True,
        "seed_scheme": scheme,
        "primary_rows_checked": checked_rows,
        "retry_rows": retry_rows,
        "row_order_invariant": scheme == "stable-key",
    }
    destination = output / "analysis/PRIMARY_SEED_AUDIT.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.output_dir).resolve()),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
