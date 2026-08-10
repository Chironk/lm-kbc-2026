#!/usr/bin/env python3
"""THE production entrypoint: one command from input file to submission file.

Fixes audit P0-2 (76 __main__ scripts, no single entrypoint) and P0-5 (the
v0495/v0501 area rule existed only as manual swaps in experiment scripts).
Every policy is a named, immutable constant; the compositor never performs an
undocumented replacement.

POLICIES (val evidence in each baseline README):
  v0491  0.490647  arch-v2 + capacity recovery; area = strict median (9B fp16)
  v0495  0.494831  v0491 + area strict upward quantile q=0.55 (9B fp16)
  v0501  0.501107  v0495 + area from Qwen2.5-14B (quantile q=0.55) -- PRIMARY

PIPELINE: preflight -> input split -> 3-4 inference runs (9B borders 4bit /
9B fp16 / System-2 incl. FRESH award / 14B area for v0501) -> named-policy
composition -> strict validation (keys, never-empty, shot leakage, sample
counts) -> atomic write -> manifest last.

Usage:
    python3 run_submission.py --policy v0501 --input data/test.jsonl \
        --output-dir submissions/test_v0501 --dry-run     # no GPU, full plan
    python3 run_submission.py --policy v0501 --input data/test.jsonl \
        --output-dir submissions/test_v0501               # the real run
    # Safe stage-by-stage execution (same frozen child commands):
    python3 run_submission.py --policy v0501 --input data/test.jsonl \
        --output-dir submissions/test_v0501 --stage fp16
    python3 run_submission.py --policy v0501 --input data/test.jsonl \
        --output-dir submissions/test_v0501 --stage compose
    # --skip-inference re-composes/validates from existing raws in output-dir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from evaluate import RELATION_TYPE, read_jsonl_file
from artifact_contract import (ArtifactContractError, validate_system1_bundle,
                               validate_system2_bundle)
from validate_artifact import validate_predictions

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STUDENT = "Qwen/Qwen3.5-9B"
STUDENT_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
AREA_MODEL = "Qwen/Qwen2.5-14B-Instruct"
POOL = ROOT / "data/synthetic_cot_faithful.jsonl"
POOL_SHA = "72f9974c355dd98eab9d13e61a6b2e120a8e9fcc40e39fb8251b54ab8d01aacb"
S2_CONFIG = ROOT / "configs/baseline-qwen-3.5-9b.yaml"
PROMPT_TEMPLATES = ROOT / "prompt_templates/question_prompts.csv"
SEED = "45"
NEVER_EMPTY = {"hasArea", "hasCapacity", "awardWonBy"}
STAGE_CHOICES = ("all", "borders", "fp16", "system2", "area14b", "compose")

POLICIES = {
    # name -> {area_source: fp16|area14b, area_agg: median|quantile055}
    "v0491": {"area_source": "fp16", "area_agg": "median",
              "val_f1": 0.490647, "params_b": 9.65},
    "v0495": {"area_source": "fp16", "area_agg": "quantile055",
              "val_f1": 0.494831, "params_b": 9.65},
    "v0501": {"area_source": "area14b", "area_agg": "quantile055",
              "val_f1": 0.501107, "params_b": 24.42},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_jsonl(path: Path, rows: List[Dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class Submission:
    def __init__(self, args):
        self.policy_name = args.policy
        self.policy = POLICIES[args.policy]
        self.input_path = Path(args.input).resolve()
        self.out = Path(args.output_dir).resolve()
        self.dry_run = args.dry_run
        self.skip_inference = args.skip_inference
        self.stage = getattr(args, "stage", "all")
        self.seed_scheme = getattr(args, "seed_scheme", "legacy")
        if self.seed_scheme not in {"legacy", "stable-key"}:
            raise ValueError(f"unsupported seed scheme: {self.seed_scheme!r}")
        self.rows = read_jsonl_file(self.input_path)
        self.area_revision: Optional[str] = None

    @staticmethod
    def gpu_workers(maximum: int) -> int:
        """Use every visible GPU up to a route's verified replica limit."""
        try:
            import torch
            visible = int(torch.cuda.device_count())
        except (ImportError, RuntimeError):
            visible = 0
        # Keep dry-run and command inspection usable on CPU-only hosts.  The
        # inference preflight still rejects a real run without CUDA.
        return max(1, min(int(maximum), visible or 1))

    # ---------- paths ----------
    def p(self, name: str) -> Path:
        return self.out / name

    # ---------- phase 1: preflight ----------
    def preflight(self) -> None:
        relations = {r["Relation"] for r in self.rows}
        missing = set(RELATION_TYPE) - relations
        if missing:
            raise SystemExit(f"input lacks relations: {sorted(missing)}")
        keys = [(r["SubjectEntity"], r["Relation"]) for r in self.rows]
        if len(keys) != len(set(keys)):
            raise SystemExit("duplicate subject-relation keys in input")
        if sha256(POOL) != POOL_SHA:
            raise SystemExit("exemplar pool sha mismatch vs frozen production pool")
        from run_inference import resolve_effective_revision
        if self.policy["area_source"] == "area14b":
            self.area_revision = resolve_effective_revision(AREA_MODEL, None)
        nonempty_gold = sum(1 for r in self.rows if r.get("ObjectEntities"))
        print(f"preflight OK: {len(self.rows)} rows, policy {self.policy_name} "
              f"({self.policy['params_b']}B <= 32B), "
              f"{nonempty_gold} rows with non-empty ObjectEntities "
              f"({'labeled/rehearsal-like input' if nonempty_gold else 'blind-or-all-null input'})")
        if self.area_revision:
            print(f"  area model {AREA_MODEL} pinned @ {self.area_revision}")

    # ---------- phase 2: split ----------
    def split_inputs(self) -> None:
        def rows_for(rels):
            return [{"SubjectEntity": r["SubjectEntity"], "Relation": r["Relation"],
                     "ObjectEntities": []} for r in self.rows if r["Relation"] in rels]
        atomic_write_jsonl(self.p("in_borders.jsonl"),
                           rows_for({"countryLandBordersCountry"}))
        fp16_relations = {"hasCapacity", "companyTradesAtStockExchange",
                          "personHasCityOfDeath"}
        if self.policy["area_source"] == "fp16":
            fp16_relations.add("hasArea")
        atomic_write_jsonl(self.p("in_fp16.jsonl"), rows_for(fp16_relations))
        # audit P0-4: System-2 input MUST include awardWonBy for fresh test
        # generation -- val award artifacts are invalid for other subjects.
        atomic_write_jsonl(self.p("in_system2.jsonl"),
                           rows_for({"companyTradesAtStockExchange",
                                     "personHasCityOfDeath", "awardWonBy"}))
        if self.policy["area_source"] == "area14b":
            atomic_write_jsonl(self.p("in_area.jsonl"), rows_for({"hasArea"}))

    # ---------- phase 3: inference commands (frozen) ----------
    def commands(self) -> Dict[str, List[str]]:
        def s1(input_name, tag, precision, workers, recover, extra=None):
            cmd = [PY, "-u", "run_inference.py",
                   "--input", str(self.p(input_name)),
                   "--output", str(self.p(f"pred_{tag}.jsonl")),
                   "--raw-cache", str(self.p(f"raw_{tag}.jsonl")),
                   "--manifest", str(self.p(f"raw_{tag}.jsonl.manifest.json")),
                   "--synthetic-cot", str(POOL),
                   "--company-soft-abstain", "--exclude-target-from-shots",
                   "--response-protocol", "legacy-cot",
                   "--aggregation-profile", "relation-v1",
                   "--shot-sampling", "legacy", "--prompt-profile", "single",
                   "--temperature-profile", "uniform", "--subject-retries", "1",
                   "--seed", SEED, "--seed-scheme", self.seed_scheme,
                   "--precision", precision, "--num-workers", str(workers),
                   "--max-tile-sub-batch", "10",
                   "--recover-unclosed-relations", recover]
            return cmd + (extra or [])
        cmds = {
            "borders": s1("in_borders.jsonl", "borders", "4bit",
                          self.gpu_workers(4),
                          "countryLandBordersCountry",
                          ["--model-revision", STUDENT_REVISION]),
            "fp16": s1("in_fp16.jsonl", "fp16", "fp16", 1,
                       "companyTradesAtStockExchange,hasCapacity",
                       ["--model-revision", STUDENT_REVISION]),
            "system2": [PY, "run_baseline.py", "-c", str(S2_CONFIG),
                        "-i", str(self.p("in_system2.jsonl")),
                        "-o", str(self.p("pred_system2.jsonl")),
                        "--raw-cache", str(self.p("raw_system2.jsonl")),
                        "--manifest", str(self.p("raw_system2.jsonl.manifest.json")),
                        "--seed", SEED, "--model-revision", STUDENT_REVISION,
                        "-w", "1"],
        }
        if self.policy["area_source"] == "area14b":
            cmds["area14b"] = s1(
                "in_area.jsonl", "area14b", "4bit", self.gpu_workers(2),
                "companyTradesAtStockExchange,hasCapacity",
                ["--model-name", AREA_MODEL,
                 "--model-revision", self.area_revision or "RESOLVE-FIRST"])
        return cmds

    def _bundle_paths(self, name: str) -> Dict[str, Path]:
        if name == "system2":
            return {
                "input": self.p("in_system2.jsonl"),
                "predictions": self.p("pred_system2.jsonl"),
                "raw": self.p("raw_system2.jsonl"),
                "manifest": self.p("raw_system2.jsonl.manifest.json"),
            }
        input_names = {
            "borders": "in_borders.jsonl",
            "fp16": "in_fp16.jsonl",
            "area14b": "in_area.jsonl",
        }
        return {
            "input": self.p(input_names[name]),
            "predictions": self.p(f"pred_{name}.jsonl"),
            "raw": self.p(f"raw_{name}.jsonl"),
            "manifest": self.p(f"raw_{name}.jsonl.manifest.json"),
        }

    def _expected_manifest(self, name: str, cmd: List[str]) -> Dict:
        paths = self._bundle_paths(name)
        n_rows = len(read_jsonl_file(paths["input"]))
        if name == "system2":
            return {
                "argv": cmd[1:],
                "model": STUDENT,
                "model_revision": STUDENT_REVISION,
                "config": str(S2_CONFIG),
                "seed": int(SEED),
                "num_workers": 1,
                "n_rows": n_rows,
            }
        precision = "fp16" if name == "fp16" else "4bit"
        model = AREA_MODEL if name == "area14b" else STUDENT
        revision = self.area_revision if name == "area14b" else STUDENT_REVISION
        recovery = {
            "borders": ["countryLandBordersCountry"],
            "fp16": ["companyTradesAtStockExchange", "hasCapacity"],
            "area14b": ["companyTradesAtStockExchange", "hasCapacity"],
        }[name]
        return {
            "argv": cmd[2:],  # python -u is not part of the child's sys.argv
            "model": model,
            "requested_model_revision": revision,
            "model_revision": revision,
            "precision": precision,
            "n_rows": n_rows,
            "n_completed": n_rows,
            "n_consistency": 10,
            "n_shots": 5,
            "seed": int(SEED),
            "seed_scheme": self.seed_scheme,
            "temperature_profile": "uniform",
            "prompt_profile": "single",
            "exclude_target_from_shots": True,
            "recover_unclosed_relations": recovery,
            "response_protocol": "legacy-cot",
            "aggregation_profile": "relation-v1",
            "shot_sampling": "legacy",
            "synthetic_cot_sha256": POOL_SHA,
            "prompt_templates_sha256": sha256(PROMPT_TEMPLATES),
            "instruction_overrides": ["companyTradesAtStockExchange"],
        }

    def validate_bundle(self, name: str, cmd: List[str]) -> Dict:
        paths = self._bundle_paths(name)
        expected = self._expected_manifest(name, cmd)
        if name == "system2":
            return validate_system2_bundle(
                label=name, reference_path=paths["input"],
                predictions_path=paths["predictions"], raw_path=paths["raw"],
                manifest_path=paths["manifest"], expected_manifest=expected)
        return validate_system1_bundle(
            label=name, reference_path=paths["input"],
            predictions_path=paths["predictions"], raw_path=paths["raw"],
            manifest_path=paths["manifest"], expected_manifest=expected)

    def validate_all_bundles(self) -> Dict[str, Dict]:
        manifests = {}
        for name, cmd in self.commands().items():
            manifests[name] = self.validate_bundle(name, cmd)
        return manifests

    def selected_inference_commands(self) -> Dict[str, List[str]]:
        commands = self.commands()
        if self.stage in {"all", "compose"}:
            return commands
        if self.stage not in commands:
            raise SystemExit(
                f"stage {self.stage!r} is not part of policy {self.policy_name}; "
                f"available GPU stages: {', '.join(commands)}")
        return {self.stage: commands[self.stage]}

    def run_inference_phase(self) -> None:
        for name, cmd in self.selected_inference_commands().items():
            paths = self._bundle_paths(name)
            bundle_files = [paths["predictions"], paths["raw"], paths["manifest"]]
            if any(path.exists() for path in bundle_files):
                try:
                    self.validate_bundle(name, cmd)
                except (ArtifactContractError, KeyError, TypeError, ValueError) as exc:
                    raise SystemExit(
                        f"{name}: refusing to reuse a partial or stale artifact bundle. "
                        f"Use a new output directory or move the invalid bundle aside.\n{exc}")
                print(f"{name}: verified complete artifact bundle; skipping inference")
                continue
            log = self.p(f"{name}.log")
            print(f"[{now()}] running {name} ...")
            with log.open("a") as handle:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=handle,
                                    stderr=subprocess.STDOUT, text=True).returncode
            if rc:
                raise SystemExit(f"{name} inference failed (exit {rc}); see {log}")
            try:
                self.validate_bundle(name, cmd)
            except (ArtifactContractError, KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"{name}: newly generated bundle is invalid:\n{exc}")

    # ---------- phase 4: composition ----------
    def compose(self) -> List[Dict]:
        from architecture_candidate_v2 import build_rows
        from numeric_aggregation import aggregate_quantile
        from run_inference import drop_self_reference, extract_after_think
        border = read_jsonl_file(self.p("raw_borders.jsonl"))
        fp16 = read_jsonl_file(self.p("raw_fp16.jsonl"))
        if self.policy["area_source"] == "area14b":
            # build_rows needs one raw source for every routed relation. Feed it
            # the 14B area rows directly so v0501 does not waste a duplicate
            # 9B-fp16 area generation that is overwritten moments later.
            fp16 += read_jsonl_file(self.p("raw_area14b.jsonl"))
        system2 = read_jsonl_file(self.p("pred_system2.jsonl"))
        rows = build_rows(self.rows, border, fp16, system2, system2)
        if self.policy["area_agg"] == "quantile055":
            source = (read_jsonl_file(self.p("raw_area14b.jsonl"))
                      if self.policy["area_source"] == "area14b" else fp16)
            area_raw = {r["SubjectEntity"]: r["raw_samples"] for r in source
                        if r["Relation"] == "hasArea"}
            for row in rows:
                if row["Relation"] == "hasArea":
                    answers = [extract_after_think(s)
                               for s in area_raw[row["SubjectEntity"]]]
                    row["ObjectEntities"] = drop_self_reference(
                        row["SubjectEntity"], aggregate_quantile(answers, 0.55))
        return rows

    # ---------- phase 5: strict validation ----------
    def validate(self, composed: List[Dict]) -> None:
        strict_errors = validate_predictions(composed, self.rows)
        if strict_errors:
            raise SystemExit("strict composed prediction validation failed:\n  - "
                             + "\n  - ".join(strict_errors))
        want = {(r["SubjectEntity"], r["Relation"]) for r in self.rows}
        got = {(r["SubjectEntity"], r["Relation"]) for r in composed}
        if want != got:
            raise SystemExit(f"composed keys mismatch: missing {len(want-got)}, "
                             f"extra {len(got-want)}")
        empties = [(s, rel) for s, rel in
                   ((r["SubjectEntity"], r["Relation"]) for r in composed
                    if not r["ObjectEntities"]) if rel in NEVER_EMPTY]
        if empties:
            raise SystemExit(f"never-empty relations with empty predictions: "
                             f"{empties[:5]}")
        # shot-leakage check the runbook used to (wrongly) attribute to
        # validate_artifact -- run it here for every System-1 raw cache.
        for tag in ("borders", "fp16", "area14b"):
            raw_path = self.p(f"raw_{tag}.jsonl")
            if not raw_path.exists():
                continue
            for r in read_jsonl_file(raw_path):
                if r["SubjectEntity"] in r.get("shot_subjects", []):
                    raise SystemExit(f"target-shot leakage: {r['SubjectEntity']} "
                                     f"in its own demonstrations ({tag})")
                for sample_index, shot_subjects in enumerate(
                        r.get("shot_subjects_by_sample", [])):
                    if r["SubjectEntity"] in shot_subjects:
                        raise SystemExit(
                            f"target-shot leakage: {r['SubjectEntity']} in sample "
                            f"{sample_index} demonstrations ({tag})")
                n = len(r.get("raw_samples", []))
                if n != 10:
                    raise SystemExit(f"{tag}: {r['SubjectEntity']} has {n} "
                                     "samples (expected 10)")
        print(f"validation OK: {len(composed)} rows, keys exact, never-empty "
              "enforced, zero shot leakage, sample counts complete")

    # ---------- phase 6: atomic output + manifest last ----------
    def finalize(self, composed: List[Dict], source_manifests: Dict[str, Dict]) -> Path:
        final = self.p(f"submission_{self.policy_name}.jsonl")
        atomic_write_jsonl(final, composed)
        manifest = {
            "created_utc": now(),
            "policy": self.policy_name,
            "seed_scheme": self.seed_scheme,
            "policy_detail": self.policy,
            "input": str(self.input_path),
            "input_sha256": sha256(self.input_path),
            "pool_sha256": POOL_SHA,
            "student_model": STUDENT,
            "student_revision": STUDENT_REVISION,
            "area_model": (AREA_MODEL if self.policy["area_source"] == "area14b"
                           else None),
            "area_model_revision": self.area_revision,
            "commands": {k: " ".join(v) for k, v in self.commands().items()},
            "source_manifests": {
                name: {
                    "path": str(self._bundle_paths(name)["manifest"]),
                    "sha256": sha256(self._bundle_paths(name)["manifest"]),
                    "content": source_manifests[name],
                }
                for name in sorted(source_manifests)
            },
            "artifacts": {p.name: sha256(p) for p in sorted(self.out.glob("*.jsonl"))},
            "submission_sha256": sha256(final),
            "test_used_for_tuning": False,
        }
        tmp = self.p("MANIFEST.json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp, self.p("MANIFEST.json"))
        return final

    def run(self) -> int:
        self.out.mkdir(parents=True, exist_ok=True)
        self.preflight()
        self.split_inputs()
        if self.dry_run:
            print("\nDRY RUN -- commands that would execute:")
            if self.stage == "compose":
                print("\n[compose]\n  validate every frozen source bundle, "
                      "compose, strictly validate, and write final artifacts")
            else:
                for name, cmd in self.selected_inference_commands().items():
                    print(f"\n[{name}]\n  " + " ".join(cmd))
            print("\n(no model loaded, no GPU touched)")
            return 0
        if self.stage not in {"all", "compose"}:
            if self.skip_inference:
                raise SystemExit("--skip-inference cannot be combined with a "
                                 "single GPU --stage")
            self.run_inference_phase()
            print(f"\nSTAGE READY: {self.stage} artifact bundle is complete and valid")
            return 0
        if self.stage == "all" and not self.skip_inference:
            self.run_inference_phase()
        try:
            source_manifests = self.validate_all_bundles()
        except (ArtifactContractError, KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"production artifact validation failed:\n{exc}")
        composed = self.compose()
        self.validate(composed)
        final = self.finalize(composed, source_manifests)
        print(f"\nSUBMISSION READY: {final}")
        print(f"  policy {self.policy_name} | manifest {self.p('MANIFEST.json')}")
        return 0


def build_submission_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=sorted(POLICIES), required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--seed-scheme", choices=("legacy", "stable-key"), default="legacy",
        help=("legacy binds stochastic sampling to input row positions; "
              "stable-key binds it to (seed, relation, subject) and is the "
              "required regime for new reproducible experiments"))
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight + split + print commands; no model load")
    ap.add_argument("--skip-inference", action="store_true",
                    help="compose/validate from existing raws in output-dir")
    ap.add_argument(
        "--stage", choices=STAGE_CHOICES, default="all",
        help=("run exactly one frozen GPU stage, compose existing validated "
              "bundles, or run all stages (default)"))
    return ap


def main() -> int:
    return Submission(build_submission_parser().parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
