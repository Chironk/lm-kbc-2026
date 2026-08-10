#!/usr/bin/env python3
"""Fast, model-free integrity check for the paper release."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {"train": 477, "val": 475, "test": 475}
EXPECTED_TEST_SHA256 = (
    "67c31c8388c585634df55500612f522ad42da6735d4c89eb59a9ef5a39f043f1"
)
EXPECTED_PARAMETERS = 30_515_165_024
PARAMETER_CAP = 32_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def fail(message: str) -> None:
    raise SystemExit(f"release verification failed: {message}")


def main() -> int:
    for split, expected in EXPECTED_ROWS.items():
        path = ROOT / "data" / f"{split}.jsonl"
        observed = rows(path)
        if len(observed) != expected:
            fail(f"{path}: expected {expected} rows, found {len(observed)}")
        keys = [(row["SubjectEntity"], row["Relation"]) for row in observed]
        if len(set(keys)) != len(keys):
            fail(f"{path}: duplicate subject-relation key")
    if sha256(ROOT / "data/test.jsonl") != EXPECTED_TEST_SHA256:
        fail("official test split hash mismatch")
    if any(row.get("ObjectEntities") for row in rows(ROOT / "data/test.jsonl")):
        fail("test split unexpectedly contains labels")

    manifest_path = ROOT / "artifacts/frozen/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "heterogeneous-final-artifacts-v1":
        fail("foreign frozen-artifact manifest")
    if int(manifest.get("verified_parameter_total", -1)) != EXPECTED_PARAMETERS:
        fail("manifest parameter total mismatch")
    if int(manifest.get("parameter_cap", -1)) != PARAMETER_CAP:
        fail("manifest parameter cap mismatch")
    for name, record in manifest.get("artifacts", {}).items():
        path = manifest_path.parent / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            fail(f"frozen artifact hash mismatch: {name}")

    for config_name in ("portfolio_cot.json", "portfolio_supply.json"):
        config = json.loads((ROOT / "configs/final" / config_name).read_text())
        total = sum(int(agent["verified_parameter_count"])
                    for agent in config["agents"])
        if total != EXPECTED_PARAMETERS:
            fail(f"{config_name}: parameter total mismatch")
        if int(config["parameter_cap"]) != PARAMETER_CAP:
            fail(f"{config_name}: parameter cap mismatch")
        for agent in config["agents"]:
            if not agent.get("revision"):
                fail(f"{config_name}: unpinned model {agent.get('id')}")

    forbidden = (
        "experiments/heterogeneous_agents/runs/",
        ".env",
        "__pycache__/",
        ".pytest_cache/",
    )
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=True, text=True,
            capture_output=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"cannot inspect tracked files: {exc}")
    bad = [path for path in tracked if any(token in path for token in forbidden)]
    if bad:
        fail(f"generated/private files are tracked: {bad[:5]}")

    # Import only after cheap filesystem checks so dependency errors are clear.
    sys.path.insert(0, str(ROOT))
    from experiments.heterogeneous_agents import final_submission_pipeline
    final_submission_pipeline._snapshot_artifacts()

    print(json.dumps({
        "status": "verified",
        "split_rows": EXPECTED_ROWS,
        "official_test_sha256": EXPECTED_TEST_SHA256,
        "verified_parameter_total": EXPECTED_PARAMETERS,
        "parameter_cap": PARAMETER_CAP,
        "frozen_artifacts": sorted(manifest["artifacts"]),
        "tracked_files": len(tracked),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
