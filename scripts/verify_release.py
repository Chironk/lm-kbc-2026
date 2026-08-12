#!/usr/bin/env python3
"""Fast, model-free integrity check for the paper release."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {"train": 477, "val": 475, "test": 475}
EXPECTED_TEST_SHA256 = (
    "67c31c8388c585634df55500612f522ad42da6735d4c89eb59a9ef5a39f043f1"
)
EXPECTED_PARAMETERS = 30_515_165_024
PARAMETER_CAP = 32_000_000_000
EXPECTED_DEVELOPMENT_SCORE = 0.5184496147269507
EXPECTED_GRAPH_CORRECTED_DEVELOPMENT_SCORE = 0.5207285306041929
EXPECTED_SAFE_DEVELOPMENT_SHA256 = (
    "6c4d4bb1ed60054cb1b2d9a6aa728a1a5f0422c714bdd8486963cc986ca348ae"
)
EXPECTED_GRAPH_CORRECTED_DEVELOPMENT_SHA256 = (
    "61d062f8c10e6262666d0e581197f4db41962125db16351258270bb13a2f0bc7"
)
EXPECTED_OFFICIAL_ARCHIVE_SHA256 = (
    "3f73d01fe5d4b3c9b9cc7e2f5dba8348d0e1fec19fc0ddb797ff2e0f460b11e4"
)
EXPECTED_OFFICIAL_MEMBER_SHA256 = (
    "73621130839b572a7fdfdc2f8a58c4bf3f00beece4be86ff4a7874c96b63bb53"
)
EXPECTED_SYNTHETIC_SHA256 = (
    "72f9974c355dd98eab9d13e61a6b2e120a8e9fcc40e39fb8251b54ab8d01aacb"
)
EXPECTED_CURRENT_VALIDATION_SHA256 = (
    "ba86b53ac38eb4b23b80391b291e5987ff4bbfe79827596fc09751b1bb0ce2be"
)
EXPECTED_CURRENT_SAFE_SHA256 = (
    "a789e2412cb0779722366afddf83fc87ebba422fd4d177207f5f87a74b06e07a"
)
EXPECTED_CURRENT_STRICT_SHA256 = (
    "3656c70185a42173a886b080e46df3ed3a546fbd41342e31de7b2423453b2032"
)
EXPECTED_CURRENT_SAFE_SCORE = 0.5259345596620683
EXPECTED_CURRENT_STRICT_SCORE = 0.5282278686922194


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


def release_files() -> tuple[list[str], str]:
    """Return release paths from Git, or from an unpacked source archive."""
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip()
        if Path(top).resolve() == ROOT.resolve():
            tracked = subprocess.run(
                ["git", "ls-files"], cwd=ROOT, check=True, text=True,
                capture_output=True,
            ).stdout.splitlines()
            return tracked, "git-index"
    except (OSError, subprocess.CalledProcessError):
        pass

    # GitHub source archives do not contain .git. Scanning also makes the
    # verifier useful to artifact reviewers who receive only the source tree.
    files = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    ]
    return sorted(files), "filesystem"


def verify_official_submission() -> dict:
    directory = ROOT / "submissions/official_test"
    archive = directory / "heterogeneous_final_strict_proof_20260803_v1_test.zip"
    manifest_path = archive.with_suffix(".manifest.json")
    if not archive.is_file() or sha256(archive) != EXPECTED_OFFICIAL_ARCHIVE_SHA256:
        fail("official-test submission archive hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("archive_sha256") != EXPECTED_OFFICIAL_ARCHIVE_SHA256
        or manifest.get("member") != "predictions.jsonl"
        or manifest.get("member_sha256") != EXPECTED_OFFICIAL_MEMBER_SHA256
        or int(manifest.get("rows", -1)) != EXPECTED_ROWS["test"]
    ):
        fail("official-test submission manifest mismatch")
    with zipfile.ZipFile(archive) as handle:
        if handle.namelist() != ["predictions.jsonl"]:
            fail("official-test archive must contain only predictions.jsonl")
        payload = handle.read("predictions.jsonl")
    if hashlib.sha256(payload).hexdigest() != EXPECTED_OFFICIAL_MEMBER_SHA256:
        fail("official-test predictions member hash mismatch")
    predictions = [json.loads(line) for line in payload.splitlines() if line]
    expected_keys = [
        (row["SubjectEntity"], row["Relation"])
        for row in rows(ROOT / "data/test.jsonl")
    ]
    prediction_keys = [
        (row["SubjectEntity"], row["Relation"]) for row in predictions
    ]
    if prediction_keys != expected_keys:
        fail("official-test prediction coverage or order mismatch")
    return {
        "archive_sha256": EXPECTED_OFFICIAL_ARCHIVE_SHA256,
        "member_sha256": EXPECTED_OFFICIAL_MEMBER_SHA256,
        "rows": len(predictions),
        "recorded_macro_f1": manifest.get("codabench_macro_f1"),
    }


def verify_development_candidates() -> dict:
    directory = ROOT / "results/heterogeneous/candidates/frozen_20260803"
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    expected = {
        "safe_locked": (
            "safe_0_518450_validation.jsonl",
            EXPECTED_SAFE_DEVELOPMENT_SHA256,
            EXPECTED_DEVELOPMENT_SCORE,
        ),
        "strict_proof": (
            "strict_proof_0_520729_validation.jsonl",
            EXPECTED_GRAPH_CORRECTED_DEVELOPMENT_SHA256,
            EXPECTED_GRAPH_CORRECTED_DEVELOPMENT_SCORE,
        ),
    }
    observed = {}
    for name, (filename, expected_hash, expected_score) in expected.items():
        record = manifest.get("candidates", {}).get(name, {})
        path = directory / filename
        if (
            record.get("file") != filename
            or record.get("sha256") != expected_hash
            or abs(float(record.get("official_macro_f1", -1)) - expected_score)
                > 1e-9
            or not path.is_file()
            or sha256(path) != expected_hash
            or len(rows(path)) != 478
        ):
            fail(f"development candidate mismatch: {name}")
        observed[name] = {
            "predictions_sha256": expected_hash,
            "macro_f1": expected_score,
            "selection_status": record.get("selection_status"),
        }
    return observed


def verify_current_validation_candidates() -> dict:
    """Verify the label-free migration of frozen predictions to 475 rows."""
    directory = (
        ROOT / "results/heterogeneous/candidates/"
        "frozen_20260811_current_validation"
    )
    result = json.loads((directory / "RESULT.json").read_text())
    if (
        result.get("schema") != "frozen-validation-artifact-rekey-v1"
        or result.get("current_validation_sha256")
            != EXPECTED_CURRENT_VALIDATION_SHA256
        or result.get("fresh_model_inference_claimed") is not False
        or result.get("gold_used_for_mapping") is not False
        or int(result.get("dropped_queries", -1)) != 3
        or int(result.get("renamed_subjects", -1)) != 20
    ):
        fail("current-validation compatibility-replay contract mismatch")
    expected = {
        "safe": (
            "safe_current_validation_475.jsonl",
            EXPECTED_CURRENT_SAFE_SHA256,
            EXPECTED_CURRENT_SAFE_SCORE,
        ),
        "strict": (
            "strict_current_validation_475.jsonl",
            EXPECTED_CURRENT_STRICT_SHA256,
            EXPECTED_CURRENT_STRICT_SCORE,
        ),
    }
    gold = rows(ROOT / "data/val.jsonl")
    sys.path.insert(0, str(ROOT))
    from experiments.heterogeneous_agents.assemble_and_audit import score

    observed = {}
    for name, (filename, digest, expected_score) in expected.items():
        record = result.get("artifacts", {}).get(name, {})
        path = directory / filename
        predictions = rows(path)
        actual_score = score(predictions, gold)["*** All Relations ***"]
        recorded_score = record.get("scores", {}).get(
            "*** All Relations ***", {}).get("macro-f1")
        if (
            record.get("prediction_sha256") != digest
            or int(record.get("rows", -1)) != EXPECTED_ROWS["val"]
            or not path.is_file()
            or sha256(path) != digest
            or len(predictions) != EXPECTED_ROWS["val"]
            or abs(float(recorded_score) - expected_score) > 1e-12
            or abs(float(actual_score) - expected_score) > 1e-12
        ):
            fail(f"current-validation candidate mismatch: {name}")
        observed[name] = {
            "predictions_sha256": digest,
            "macro_f1": expected_score,
            "reproduction_tier": result["reproduction_tier"],
        }
    return observed


def verify_synthetic_demonstrations() -> dict:
    path = ROOT / "data/synthetic_cot_faithful.jsonl"
    pool = rows(path)
    per_relation = Counter(str(row.get("Relation")) for row in pool)
    query_counts = Counter(
        (str(row.get("SubjectEntity")), str(row.get("Relation")))
        for row in pool
    )
    expected_relations = {
        "awardWonBy": 84,
        "companyTradesAtStockExchange": 358,
        "countryLandBordersCountry": 278,
        "hasArea": 361,
        "hasCapacity": 343,
        "personHasCityOfDeath": 432,
    }
    if (
        sha256(path) != EXPECTED_SYNTHETIC_SHA256
        or len(pool) != 1856
        or len(query_counts) != 389
        or dict(per_relation) != expected_relations
        or min(query_counts.values()) != 1
        or max(query_counts.values()) != 14
    ):
        fail("SyntheticCoT pool identity or coverage mismatch")
    return {
        "sha256": EXPECTED_SYNTHETIC_SHA256,
        "rows": len(pool),
        "covered_training_queries": len(query_counts),
        "examples_per_covered_query": {
            "minimum": min(query_counts.values()),
            "maximum": max(query_counts.values()),
        },
        "per_relation": dict(sorted(per_relation.items())),
    }


def verify_paper_system_contract() -> dict:
    """Bind prose-facing architecture facts to the executable configuration."""
    path = ROOT / "configs/final/paper_system_contract.json"
    contract = json.loads(path.read_text())
    if (
        contract.get("schema") != "lm-kbc-paper-system-contract-v1"
        or contract.get("system_id")
            != "heterogeneous_final_strict_proof_20260803_v1"
    ):
        fail("paper-system contract identity mismatch")

    sys.path.insert(0, str(ROOT))
    import run_inference
    import run_submission
    from experiments.heterogeneous_agents import end_to_end_pipeline as e2e
    from models.baseline_qwen import BaselineQwenModel

    routes = contract["routes"]
    input_contract = contract["input"]
    synthetic_contract = contract["synthetic_demonstrations"]
    official_contract = contract["official_result"]
    if (
        input_contract.get("path") != "data/test.jsonl"
        or int(input_contract.get("rows", -1)) != EXPECTED_ROWS["test"]
        or input_contract.get("sha256") != EXPECTED_TEST_SHA256
        or synthetic_contract.get("path")
            != "data/synthetic_cot_faithful.jsonl"
        or synthetic_contract.get("sha256") != EXPECTED_SYNTHETIC_SHA256
        or int(synthetic_contract.get("rows", -1)) != 1856
        or int(synthetic_contract.get("covered_training_queries", -1)) != 389
        or official_contract.get("archive")
            != ("submissions/official_test/"
                "heterogeneous_final_strict_proof_20260803_v1_test.zip")
        or official_contract.get("archive_sha256")
            != EXPECTED_OFFICIAL_ARCHIVE_SHA256
        or official_contract.get("member_sha256")
            != EXPECTED_OFFICIAL_MEMBER_SHA256
        or abs(float(official_contract.get("codabench_macro_f1", -1)) - 0.4845)
            > 1e-12
    ):
        fail("paper-system data/result contract drift")

    primary = routes["qwen_primary_system1"]
    if (
        primary.get("policy") != "v0495"
        or primary.get("seed_scheme") != "legacy"
        or int(primary.get("samples", -1)) != int(run_inference.N_CONSISTENCY)
        or float(primary.get("temperature", -1))
            != float(run_inference.TEMPERATURE)
        or int(primary.get("max_new_tokens", {}).get("default", -1))
            != int(run_inference.MAX_NEW_TOKENS)
        or int(primary.get("max_new_tokens", {}).get("awardWonBy", -1))
            != int(run_inference.AWARD_MAX_NEW_TOKENS)
        or Path(run_submission.POOL).resolve()
            != (ROOT / contract["synthetic_demonstrations"]["path"]).resolve()
    ):
        fail("primary Qwen paper-system contract drift")

    auxiliary = routes["qwen_auxiliary"]
    auxiliary_path = ROOT / auxiliary["config"]
    auxiliary_config = yaml.safe_load(auxiliary_path.read_text())
    if (
        sha256(auxiliary_path) != auxiliary.get("config_sha256")
        or int(auxiliary_config.get("judge_n_samples", -1)) != 5
        or float(auxiliary_config.get("judge_sample_temperature", -1)) != 0.8
        or int(auxiliary_config.get("award_max_rounds", -1)) != 4
        or BaselineQwenModel.SINGLE_SHOT_JUDGE_RELATIONS
            != {"personHasCityOfDeath"}
        or BaselineQwenModel.ITERATIVE_RELATIONS != {"awardWonBy"}
    ):
        fail("auxiliary Qwen paper-system contract drift")

    route_by_name = {str(value["route"]): value for value in e2e.ROUTES}
    expected_routes = {
        "gemma:independent": (1, ("commit_existence", "commit_cardinality", "propose")),
        e2e.MINISTRAL_N3: (3, ("propose",)),
        e2e.MINISTRAL_COT40: (10, ("propose",)),
    }
    for name, (samples, phases) in expected_routes.items():
        route = route_by_name.get(name, {})
        if (
            int(route.get("n_proposals", -1)) != samples
            or tuple(route.get("phases", ())) != phases
        ):
            fail(f"heterogeneous route contract drift: {name}")

    portfolio = {
        str(model["id"]): model for model in contract["portfolio"]["models"]
    }
    config = json.loads((ROOT / "configs/final/portfolio_cot.json").read_text())
    configured = {str(agent["id"]): agent for agent in config["agents"]}
    for agent_id, model in portfolio.items():
        agent = configured.get(agent_id, {})
        if any(
            agent.get(field) != model.get(field)
            for field in ("model", "revision", "verified_parameter_count")
        ):
            fail(f"paper portfolio drift: {agent_id}")
    if (
        sum(int(model["verified_parameter_count"]) for model in portfolio.values())
        != int(contract["portfolio"]["verified_parameter_total"])
        or int(contract["portfolio"]["verified_parameter_total"])
            != EXPECTED_PARAMETERS
    ):
        fail("paper portfolio total mismatch")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "system_id": contract["system_id"],
        "qwen_primary_temperature": primary["temperature"],
        "qwen_auxiliary_route_present": True,
        "fresh_inference_byte_identical_claim": contract[
            "reproduction_tiers"]["fresh_inference_byte_identical_to_official_submission"],
    }


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

    tracked, inventory_mode = release_files()
    bad = []
    for raw in tracked:
        path = Path(raw)
        if (
            (raw.startswith("experiments/heterogeneous_agents/runs/")
             and raw != "experiments/heterogeneous_agents/runs/README.md")
            or path.name == ".env"
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
        ):
            bad.append(raw)
    if bad:
        fail(f"generated/private files are tracked: {bad[:5]}")

    official_submission = verify_official_submission()
    development_candidates = verify_development_candidates()
    current_validation_candidates = verify_current_validation_candidates()
    synthetic_demonstrations = verify_synthetic_demonstrations()
    paper_system_contract = verify_paper_system_contract()

    # Import only after cheap filesystem checks so dependency errors are clear.
    sys.path.insert(0, str(ROOT))
    from experiments.heterogeneous_agents import final_submission_pipeline
    from experiments.heterogeneous_agents import historical_sota_test_pipeline
    from experiments.heterogeneous_agents.assemble_and_audit import score
    from experiments.heterogeneous_agents.sota_reproduction import verify_snapshot
    final_submission_pipeline._snapshot_artifacts()
    historical_sota_test_pipeline._archived_prediction_bytes()
    development = verify_snapshot()
    if development["manifest"].get("reported_pooled_macro_f1") != EXPECTED_DEVELOPMENT_SCORE:
        fail("development reproduction score mismatch")
    development_gold = rows(ROOT / "data/archive/validation_478_20260729.jsonl")
    development_dir = ROOT / "results/heterogeneous/candidates/frozen_20260803"
    scored = {
        "safe_locked": score(
            rows(development_dir / "safe_0_518450_validation.jsonl"),
            development_gold,
        )["*** All Relations ***"],
        "strict_proof": score(
            rows(development_dir / "strict_proof_0_520729_validation.jsonl"),
            development_gold,
        )["*** All Relations ***"],
    }
    if (
        abs(scored["safe_locked"] - EXPECTED_DEVELOPMENT_SCORE) > 1e-12
        or abs(
            scored["strict_proof"]
            - EXPECTED_GRAPH_CORRECTED_DEVELOPMENT_SCORE
        ) > 1e-12
    ):
        fail("official evaluator does not reproduce development candidate scores")

    print(json.dumps({
        "status": "release_artifacts_verified",
        "paper_prose_alignment_checked": False,
        "split_rows": EXPECTED_ROWS,
        "official_test_sha256": EXPECTED_TEST_SHA256,
        "verified_parameter_total": EXPECTED_PARAMETERS,
        "parameter_cap": PARAMETER_CAP,
        "frozen_artifacts": sorted(manifest["artifacts"]),
        "development_replay_macro_f1": EXPECTED_DEVELOPMENT_SCORE,
        "development_candidates": development_candidates,
        "current_validation_candidates": current_validation_candidates,
        "synthetic_demonstrations": synthetic_demonstrations,
        "paper_system_contract": paper_system_contract,
        "official_test_submission": official_submission,
        "tracked_files": len(tracked),
        "inventory_mode": inventory_mode,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
