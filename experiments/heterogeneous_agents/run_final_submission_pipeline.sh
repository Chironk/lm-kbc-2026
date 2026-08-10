#!/usr/bin/env bash
# Frozen final architecture: test rows -> four resumable generation passes ->
# exact typed evidence graph -> frozen decoder -> Codabench-ready zip.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$(command -v python || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "no Python interpreter found; activate the project environment or set PY" >&2
  exit 2
fi
SPLIT="${SPLIT:-test}"
INPUT="${INPUT:-data/test.jsonl}"
OUT="${OUT:-experiments/heterogeneous_agents/runs/final_${SPLIT}_submission_20260809_v3}"
MODULE="experiments.heterogeneous_agents.final_submission_pipeline"
E2E="experiments/heterogeneous_agents/run_end_to_end_pipeline.sh"
STAGE="${1:-status}"
SYNTHETIC_COT="${SYNTHETIC_COT:-data/synthetic_cot_capacity_aligned_v2.jsonl}"
QUESTION_CONTRACT="${QUESTION_CONTRACT:-official-v1}"
PRIMARY_SEED_SCHEME="${PRIMARY_SEED_SCHEME:-legacy}"
OFFICIAL_TEST_SHA256="67c31c8388c585634df55500612f522ad42da6735d4c89eb59a9ef5a39f043f1"

if [[ "$SPLIT" == "validation" && "$INPUT" == "data/test.jsonl" ]]; then
  INPUT="data/val.jsonl"
fi
if [[ "$SPLIT" == "test" && "$INPUT" != "data/test.jsonl" ]]; then
  echo "refusing noncanonical test input: $INPUT" >&2
  exit 2
fi
if [[ "$SPLIT" == "test" ]]; then
  ACTUAL_TEST_SHA256="$(sha256sum "$INPUT" | awk '{print $1}')"
  if [[ "$ACTUAL_TEST_SHA256" != "$OFFICIAL_TEST_SHA256" ]]; then
    echo "refusing stale/nonofficial test input: $INPUT" >&2
    echo "expected $OFFICIAL_TEST_SHA256" >&2
    echo "actual   $ACTUAL_TEST_SHA256" >&2
    exit 2
  fi
fi

delegate() {
  SPLIT="$SPLIT" INPUT="$INPUT" INPUT_TEST="$INPUT" OUT="$OUT" \
    SYNTHETIC_COT="$SYNTHETIC_COT" \
    QUESTION_CONTRACT="$QUESTION_CONTRACT" \
    PY="$PY" "$E2E" "$1"
}

require_policy() {
  if [[ ! -f "$OUT/plan/FINAL_POLICY.json" ]]; then
    echo "missing frozen final policy; run '$0 plan' first" >&2
    exit 2
  fi
}

frozen_primary_seed_scheme() {
  "$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

policy = json.loads((Path(sys.argv[1]) / "plan/FINAL_POLICY.json").read_text())
scheme = policy.get("primary_seed_scheme")
if scheme not in {"legacy", "stable-key"}:
    raise SystemExit(f"invalid frozen primary seed scheme: {scheme!r}")
print(scheme)
PY
}

case "$STAGE" in
  plan)
    delegate plan
    "$PY" -u -m "$MODULE" freeze --output-dir "$OUT" \
      --primary-seed-scheme "$PRIMARY_SEED_SCHEME"
    "$PY" -u run_submission.py --policy v0495 --input "$INPUT" \
      --output-dir "$OUT/primary_qwen" \
      --seed-scheme "$PRIMARY_SEED_SCHEME" --dry-run
    ;;
  preflight|generate-gemma|generate-ministral-n3|generate-ministral-cot40)
    require_policy
    delegate "$STAGE"
    ;;
  smoke)
    require_policy
    delegate smoke-final
    ;;
  generate-primary)
    require_policy
    PRIMARY_SEED_SCHEME="$(frozen_primary_seed_scheme)"
    "$PY" -u run_submission.py --policy v0495 --input "$INPUT" \
      --output-dir "$OUT/primary_qwen" \
      --seed-scheme "$PRIMARY_SEED_SCHEME"
    ;;
  generate)
    require_policy
    "$0" generate-primary
    "$0" generate-gemma
    "$0" generate-ministral-n3
    "$0" generate-ministral-cot40
    ;;
  assemble)
    require_policy
    delegate assemble
    ;;
  decode)
    require_policy
    "$PY" -u -m "$MODULE" build --output-dir "$OUT"
    ;;
  score)
    require_policy
    "$PY" -u -m "$MODULE" score-validation --output-dir "$OUT"
    ;;
  package)
    require_policy
    "$PY" -u -m "$MODULE" package --output-dir "$OUT"
    ;;
  verify-package)
    require_policy
    "$PY" -u -m "$MODULE" verify-package --output-dir "$OUT"
    ;;
  all)
    "$0" plan
    "$0" preflight
    "$0" smoke
    "$0" generate
    "$0" decode
    if [[ "$SPLIT" == "validation" ]]; then
      "$0" score
    fi
    "$0" package
    "$0" verify-package
    ;;
  status)
    "$PY" -u -m "$MODULE" status --output-dir "$OUT"
    ;;
  test)
    "$PY" -m pytest -q tests/test_final_submission_pipeline.py
    ;;
  *)
    echo "usage: $0 {test|plan|preflight|smoke|generate|generate-primary|generate-gemma|generate-ministral-n3|generate-ministral-cot40|decode|score|package|verify-package|all|status}" >&2
    exit 2
    ;;
esac
