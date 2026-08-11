#!/usr/bin/env bash
# Resumable sequential validation run for the paper's model-only ablations.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$(command -v python || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "no Python interpreter found; activate the Python 3.11 environment or set PY" >&2
  exit 2
fi
RUNNER="experiments/heterogeneous_agents/run_end_to_end_pipeline.sh"
OUT="${OUT:-experiments/heterogeneous_agents/runs/standalone_cot_validation_20260810_v1}"
COMMON=(SPLIT=validation INPUT=data/val.jsonl OUT="$OUT" QUESTION_CONTRACT=official-v1 SYNTHETIC_COT=data/synthetic_cot_capacity_aligned_v2.jsonl PY="$PY")

run_stage() {
  echo "[$(date --iso-8601=seconds)] stage=$1"
  env "${COMMON[@]}" bash "$RUNNER" "$1"
}

mkdir -p "$OUT"
exec > >(tee -a "$OUT/overnight.log") 2>&1

run_stage plan
run_stage preflight
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  run_stage smoke
else
  echo "[$(date --iso-8601=seconds)] stage=smoke skipped by SKIP_SMOKE=1"
fi
run_stage generate-qwen
run_stage generate-gemma
run_stage generate-ministral-cot40
run_stage generate-ministral-n3

"$PY" -u -m experiments.heterogeneous_agents.standalone_cot_ablation \
  --output-dir "$OUT" --gold data/val.jsonl

echo "[$(date --iso-8601=seconds)] complete"
