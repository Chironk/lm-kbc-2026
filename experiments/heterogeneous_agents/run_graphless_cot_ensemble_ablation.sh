#!/usr/bin/env bash
# Generate or score graph-free CoT combinations without graph stages.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$(command -v python || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "no Python interpreter found; activate the Python 3.11 environment or set PY" >&2
  exit 2
fi

OUT="${OUT:-experiments/heterogeneous_agents/runs/standalone_cot_validation_20260810_v1}"
MODE="${1:-score}"
RUNNER="experiments/heterogeneous_agents/run_end_to_end_pipeline.sh"

if [[ "$MODE" == "generate" ]]; then
  COMMON=(
    SPLIT=validation
    INPUT=data/val.jsonl
    OUT="$OUT"
    QUESTION_CONTRACT=official-v1
    SYNTHETIC_COT=data/synthetic_cot_capacity_aligned_v2.jsonl
    PY="$PY"
  )
  env "${COMMON[@]}" bash "$RUNNER" plan
  if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    env "${COMMON[@]}" bash "$RUNNER" preflight
  fi
  env "${COMMON[@]}" bash "$RUNNER" generate-qwen
  env "${COMMON[@]}" bash "$RUNNER" generate-gemma
  env "${COMMON[@]}" bash "$RUNNER" generate-ministral-cot40
elif [[ "$MODE" != "score" ]]; then
  echo "usage: $0 {score|generate}" >&2
  exit 2
fi

"$PY" -u -m experiments.heterogeneous_agents.graphless_cot_ensemble_ablation \
  --output-dir "$OUT" \
  --gold data/val.jsonl
