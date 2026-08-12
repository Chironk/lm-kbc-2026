#!/usr/bin/env bash
# Paired train confirmation of the one-route Ministral area architecture.
# Generates the current two-route control and then removes zero-shot N=3 only
# in the paired CPU replay.  This costs one extra N=3 pass but makes the
# comparison exact: current full system versus proposed one-route system.
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
export SPLIT=train
export INPUT=data/train.jsonl
export INPUT_TRAIN=data/train.jsonl
export PRIMARY_SEED_SCHEME=stable-key
export SYNTHETIC_COT="${SYNTHETIC_COT:-data/synthetic_cot_capacity_aligned_v2.jsonl}"
export QUESTION_CONTRACT=official-v1
export OUT="${OUT:-experiments/heterogeneous_agents/runs/component_cot40_train_confirmation_20260810_v1}"

FINAL_RUNNER="experiments/heterogeneous_agents/run_final_submission_pipeline.sh"
LOG="$OUT/overnight.log"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

on_error() {
  local code=$?
  echo "[$(date -Is)] FAILED exit=$code"
  SPLIT="$SPLIT" INPUT="$INPUT" OUT="$OUT" bash "$FINAL_RUNNER" status || true
  exit "$code"
}
trap on_error ERR

stage() {
  echo "[$(date -Is)] stage=$1"
  SPLIT="$SPLIT" INPUT="$INPUT" OUT="$OUT" \
    PRIMARY_SEED_SCHEME="$PRIMARY_SEED_SCHEME" \
    SYNTHETIC_COT="$SYNTHETIC_COT" \
    QUESTION_CONTRACT="$QUESTION_CONTRACT" \
    bash "$FINAL_RUNNER" "$1"
}

echo "[$(date -Is)] one-route Ministral train confirmation"
echo "python=$PY"
echo "gpus=$CUDA_VISIBLE_DEVICES"
echo "output=$OUT"

"$PY" -m pytest -q \
  tests/test_submission_commands.py \
  tests/test_final_submission_pipeline.py \
  tests/test_production_contract.py

stage plan

stage smoke
stage generate-primary
stage generate-gemma
stage generate-ministral-n3
stage generate-ministral-cot40
stage decode

echo "[$(date -Is)] paired CPU decode and train scoring"
"$PY" -u -m \
  experiments.heterogeneous_agents.analysis.ministral_route1_ablation \
  --source "$OUT" --gold data/train.jsonl

echo "[$(date -Is)] COMPLETE"
echo "result=$OUT/analysis/ministral_route1_ablation/RESULT.json"
