#!/usr/bin/env bash
# New controlled validation regime: regenerate every model route with the
# final architecture while binding primary-Qwen sampling to subject keys.
# This script never opens or packages the official test split.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$(command -v python || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "no Python interpreter found; activate the project environment or set PY" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"

export SPLIT=validation
export INPUT=data/val.jsonl
export PRIMARY_SEED_SCHEME=stable-key
export OUT="${OUT:-experiments/heterogeneous_agents/runs/stable_key_validation_20260810_v1}"
RUNNER="experiments/heterogeneous_agents/run_final_submission_pipeline.sh"
LOG="$OUT/overnight.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

started="$(date -Is)"
echo "[$started] stable-key validation run starting"
echo "python=$PY"
echo "gpus=$CUDA_VISIBLE_DEVICES"
echo "output=$OUT"

on_error() {
  local code=$?
  echo "[$(date -Is)] FAILED exit=$code"
  SPLIT="$SPLIT" INPUT="$INPUT" OUT="$OUT" bash "$RUNNER" status || true
  exit "$code"
}
trap on_error ERR

stage() {
  echo "[$(date -Is)] stage=$1"
  SPLIT="$SPLIT" INPUT="$INPUT" OUT="$OUT" \
    PRIMARY_SEED_SCHEME="$PRIMARY_SEED_SCHEME" \
    bash "$RUNNER" "$1"
}

echo "[$(date -Is)] CPU contracts"
"$PY" -m pytest -q \
  tests/test_submission_commands.py \
  tests/test_final_submission_pipeline.py \
  tests/test_production_contract.py

stage plan

# The smoke pass loads every non-primary checkpoint before the long Qwen pass,
# so an incompatible GPU layout fails early rather than after several hours.
stage smoke
stage generate-primary
"$PY" -u scripts/audit_primary_seed_scheme.py --output-dir "$OUT"
stage generate-gemma
stage generate-ministral-n3
stage generate-ministral-cot40
stage decode
stage score
stage status

echo "[$(date -Is)] COMPLETE"
echo "result=$OUT/analysis/FINAL_RESULT.json"
echo "seed_audit=$OUT/analysis/PRIMARY_SEED_AUDIT.json"
