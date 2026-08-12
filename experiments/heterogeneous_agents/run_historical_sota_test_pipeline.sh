#!/usr/bin/env bash
# Reconstruct the official-test 0.4845 architecture and a paired version that
# replaces only Ministral's zero-shot N=3 area route with its CoT-5/N=10
# numeric-component rule. Every generation stage is resumable.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$(command -v python || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "Python was not found. Activate the release environment or set PY=/path/to/python." >&2
  exit 2
fi
OUT="${OUT:-experiments/heterogeneous_agents/runs/historical_sota_single_ministral_test_20260810_v1}"
INPUT="${INPUT:-data/test.jsonl}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"
MODULE="experiments.heterogeneous_agents.historical_sota_test_pipeline"
STAGE="${1:-status}"
LOG="$OUT/run.log"
DEVICES=""
DETECTED_WORKERS=0
WORKERS=0

# GPU discovery is intentionally lazy.  Planning, status, packaging, and the
# unit-test entry point are CPU-only release operations and must remain usable
# on login nodes (or containers) where an nvidia-smi executable is installed
# but cannot communicate with a driver.
initialize_gpu_runtime() {
  if (( DETECTED_WORKERS > 0 )); then
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DEVICES="$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
  fi
  if [[ -z "$DEVICES" ]]; then
    echo "No usable CUDA GPUs were detected. Set CUDA_VISIBLE_DEVICES for a generation stage." >&2
    exit 2
  fi
  IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
  DETECTED_WORKERS="${#DEVICE_LIST[@]}"
  WORKERS="${NUM_WORKERS:-$DETECTED_WORKERS}"
  if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]] || (( WORKERS > DETECTED_WORKERS )); then
    echo "NUM_WORKERS must be between 1 and the $DETECTED_WORKERS visible GPUs." >&2
    exit 2
  fi
}

require_plan() {
  if [[ ! -f "$OUT/plan/HISTORICAL_SOTA_POLICY.json" ]]; then
    echo "missing paired frozen plan; run '$0 plan' first" >&2
    exit 2
  fi
}

job_field() {
  "$PY" - "$OUT" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
p = json.loads((Path(sys.argv[1]) / "plan/PLAN.json").read_text())
print(p["jobs"][sys.argv[2]][sys.argv[3]])
PY
}

generate_route() {
  local route="$1" batch="$2" checkpoint="$3"
  local tasks output agents agent
  initialize_gpu_runtime
  tasks="$(job_field "$route" task_path)"
  output="$(job_field "$route" response_path)"
  agents="$(job_field "$route" agent_config)"
  agent="$(job_field "$route" agent_id)"
  mkdir -p "$(dirname "$output")"
  CUDA_VISIBLE_DEVICES="$DEVICES" \
  PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
  "$PY" -u -m experiments.heterogeneous_agents.run_agent \
    --agent-id "$agent" \
    --tasks "$tasks" \
    --output "$output" \
    --agents "$agents" \
    --precision 4bit \
    --num-workers "$WORKERS" \
    --generation-batch-size 1 \
    --task-batch-size "$batch" \
    --checkpoint-every "$checkpoint" \
    --seed 20260730
}

case "$STAGE" in
  plan)
    "$PY" -u -m "$MODULE" plan \
      --input "$INPUT" --output-dir "$OUT"
    "$PY" -u run_submission.py \
      --policy v0495 --seed-scheme legacy \
      --input "$INPUT" --output-dir "$OUT/primary_qwen" --dry-run
    ;;
  preflight)
    require_plan
    "$PY" -u -m experiments.heterogeneous_agents.preflight \
      --agents configs/final/portfolio_cot.json \
      --online-access --tokenizer-check --minimum-free-gib 30
    ;;
  generate-primary)
    require_plan
    initialize_gpu_runtime
    CUDA_VISIBLE_DEVICES="$DEVICES" \
    PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
    "$PY" -u run_submission.py \
      --policy v0495 --seed-scheme legacy \
      --input "$INPUT" --output-dir "$OUT/primary_qwen"
    ;;
  generate-gemma)
    require_plan
    generate_route "gemma:independent" 2 10 2>&1 | tee -a "$LOG"
    ;;
  generate-ministral-n3)
    require_plan
    generate_route "ministral:self_consistency" 2 10 2>&1 | tee -a "$LOG"
    ;;
  generate-ministral-n10)
    require_plan
    generate_route "ministral:cot5_cap40_n10" 1 10 2>&1 | tee -a "$LOG"
    ;;
  generate)
    "$0" generate-primary
    "$0" generate-gemma
    "$0" generate-ministral-n3
    "$0" generate-ministral-n10
    ;;
  build)
    require_plan
    "$PY" -u -m "$MODULE" build --output-dir "$OUT"
    ;;
  package)
    require_plan
    "$PY" -u -m "$MODULE" package --output-dir "$OUT"
    ;;
  all)
    "$0" plan
    "$0" preflight
    "$0" generate
    "$0" build
    "$0" package
    ;;
  status)
    "$PY" -u -m "$MODULE" status --output-dir "$OUT"
    ;;
  test)
    "$PY" -m pytest -q tests/test_historical_sota_test_pipeline.py
    ;;
  *)
    echo "usage: $0 {plan|preflight|generate|generate-primary|generate-gemma|generate-ministral-n3|generate-ministral-n10|build|package|all|status|test}" >&2
    exit 2
    ;;
esac
