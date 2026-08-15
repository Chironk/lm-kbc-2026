#!/usr/bin/env bash
# Full heterogeneous system on one split: raw rows -> generations -> graph ->
# decode -> score.  Every stage is resumable; re-issuing a completed
# generation stage validates the existing artifact and exits without loading
# a model.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PY="${PY:-$(command -v python || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "no Python interpreter found; activate the project environment or set PY" >&2
  exit 2
fi
SPLIT="${SPLIT:-validation}"
OUT="${OUT:-runs/end_to_end_${SPLIT}_20260730_v1}"
INPUT="${INPUT:-data/val.jsonl}"
SYNTHETIC_COT="${SYNTHETIC_COT:-data/synthetic_cot_faithful.jsonl}"
QUESTION_CONTRACT="${QUESTION_CONTRACT:-legacy}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"
DEVICES="${CUDA_VISIBLE_DEVICES:-}"
MODULE="lm_kbc.end_to_end_pipeline"
STAGE="${1:-status}"
LOG="$OUT/run.log"

# The blind split must never be scored locally.
if [[ "$SPLIT" == "test" ]]; then
  INPUT="${INPUT_TEST:-data/test.jsonl}"
elif [[ "$SPLIT" == "train" ]]; then
  INPUT="${INPUT_TRAIN:-$INPUT}"
fi

require_plan() {
  if [[ ! -f "$OUT/plan/PLAN.json" ]]; then
    echo "missing frozen plan: $OUT/plan/PLAN.json; run '$0 plan' first" >&2
    exit 2
  fi
}

# Route slug -> (agent id, agent config, workers, task batch).  A worker value
# of ``all`` means one independent checkpoint replica per visible GPU.  This
# preserves the verified four-replica 2080-Ti run while using two replicas on
# a dual-GPU host instead of requesting nonexistent workers.
route_agent() {
  case "$1" in
    qwen__self_consistency)          echo "qwen_recall cot 1 4" ;;
    gemma__independent)              echo "gemma_independent cot all 2" ;;
    ministral__self_consistency)     echo "ministral_independent supply all 2" ;;
    ministral__cot5_cap40_n10)       echo "ministral_independent cot all 1" ;;
    *) echo "unknown route: $1" >&2; exit 2 ;;
  esac
}

config_path() {
  "$PY" - "$OUT" "$1" <<'PY'
import json, sys
from pathlib import Path
plan = json.loads((Path(sys.argv[1]) / "plan/PLAN.json").read_text())
print(plan["cot_agents"] if sys.argv[2] == "cot" else plan["supply_agents"])
PY
}

generate() {
  local slug="$1" dir="$2" checkpoint="$3"
  if [[ -z "$DEVICES" ]]; then
    local gpu_count
    gpu_count="$($PY -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "$gpu_count" -lt 1 ]]; then
      echo "no CUDA GPU found; set CUDA_VISIBLE_DEVICES after activating a CUDA environment" >&2
      exit 2
    fi
    DEVICES="$(seq -s, 0 $((gpu_count - 1)))"
  fi
  read -r agent config workers batch <<<"$(route_agent "$slug")"
  local agents_json
  agents_json="$(config_path "$config")"
  if [[ "$workers" == "all" ]]; then
    workers="$(CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -c \
      'import torch; print(torch.cuda.device_count())')"
    if [[ "$workers" -lt 1 ]]; then
      echo "no CUDA GPUs visible through CUDA_VISIBLE_DEVICES=$DEVICES" >&2
      exit 2
    fi
  fi
  echo "[$slug] visible devices=$DEVICES; parallel workers=$workers; task batch=$batch"
  mkdir -p "$OUT/$dir"
  CUDA_VISIBLE_DEVICES="$DEVICES" \
  PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
  "$PY" -u -m lm_kbc.run_agent \
    --agent-id "$agent" \
    --tasks "$OUT/plan/$([[ $dir == smoke_responses ]] && echo smoke || echo tasks)/$slug.jsonl" \
    --output "$OUT/$dir/$slug.jsonl" \
    --agents "$agents_json" \
    --precision 4bit \
    --num-workers "$workers" \
    --generation-batch-size 1 \
    --task-batch-size "$batch" \
    --checkpoint-every "$checkpoint" \
    --seed 20260730
}

ALL_ROUTES=(
  qwen__self_consistency
  gemma__independent
  ministral__self_consistency
  ministral__cot5_cap40_n10
)

# The frozen final-submission decoder uses the separately generated production
# Qwen v0495 artifact, not this generic qwen__self_consistency response.  Keep
# the full smoke stage for the generic end-to-end experiment, but provide a
# final-policy smoke stage that loads only the three routes actually generated
# through this runner.  This avoids loading an unused Qwen checkpoint and is
# especially important on dual 11-GiB GPUs.
FINAL_POLICY_ROUTES=(
  gemma__independent
  ministral__self_consistency
  ministral__cot5_cap40_n10
)

case "$STAGE" in
  plan)
    mkdir -p "$OUT"
    "$PY" -u -m "$MODULE" plan \
      --split "$SPLIT" --input "$INPUT" --output-dir "$OUT" \
      --synthetic-cot "$SYNTHETIC_COT" \
      --question-contract "$QUESTION_CONTRACT"
    ;;
  preflight)
    require_plan
    "$PY" -u -m lm_kbc.preflight \
      --agents "$(config_path cot)" \
      --online-access --tokenizer-check --minimum-free-gib 30
    ;;
  smoke)
    require_plan
    mkdir -p "$OUT"
    for slug in "${ALL_ROUTES[@]}"; do
      echo "[smoke] $slug"
      generate "$slug" smoke_responses 1 2>&1 | tee -a "$LOG"
    done
    ;;
  smoke-final)
    require_plan
    mkdir -p "$OUT"
    for slug in "${FINAL_POLICY_ROUTES[@]}"; do
      echo "[smoke-final] $slug"
      generate "$slug" smoke_responses 1 2>&1 | tee -a "$LOG"
    done
    ;;
  generate)
    require_plan
    mkdir -p "$OUT"
    for slug in "${ALL_ROUTES[@]}"; do
      echo "[generate] $slug"
      generate "$slug" responses 10 2>&1 | tee -a "$LOG"
    done
    ;;
  generate-qwen)      require_plan; generate qwen__self_consistency responses 10 2>&1 | tee -a "$LOG" ;;
  generate-gemma)     require_plan; generate gemma__independent responses 10 2>&1 | tee -a "$LOG" ;;
  generate-ministral-n3)   require_plan; generate ministral__self_consistency responses 10 2>&1 | tee -a "$LOG" ;;
  generate-ministral-cot40) require_plan; generate ministral__cot5_cap40_n10 responses 10 2>&1 | tee -a "$LOG" ;;
  assemble)
    require_plan
    "$PY" -u -m "$MODULE" assemble --output-dir "$OUT"
    ;;
  graph)
    require_plan
    "$PY" -u -m "$MODULE" graph --output-dir "$OUT"
    ;;
  decode)
    require_plan
    "$PY" -u -m "$MODULE" decode --output-dir "$OUT"
    ;;
  score)
    require_plan
    "$PY" -u -m "$MODULE" score --output-dir "$OUT"
    ;;
  all)
    require_plan
    for slug in "${ALL_ROUTES[@]}"; do
      echo "[generate] $slug"
      generate "$slug" responses 10 2>&1 | tee -a "$LOG"
    done
    "$PY" -u -m "$MODULE" assemble --output-dir "$OUT"
    "$PY" -u -m "$MODULE" graph --output-dir "$OUT"
    "$PY" -u -m "$MODULE" decode --output-dir "$OUT"
    "$PY" -u -m "$MODULE" score --output-dir "$OUT"
    ;;
  status)
    "$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
plan_path = output / "plan/PLAN.json"
if not plan_path.exists():
    print(f"plan: missing ({plan_path})")
    raise SystemExit(0)
plan = json.loads(plan_path.read_text())
print(f"split={plan['split']} rows={plan['rows']} "
      f"params={plan['verified_parameter_total']}/{plan['parameter_cap']}")
for name, job in sorted(plan["jobs"].items()):
    path = Path(job["response_path"])
    done = 0
    if path.exists():
        with path.open() as handle:
            done = sum(1 for line in handle if line.strip())
    state = "complete" if done == job["tasks"] else "pending"
    print(f"  {name:34s} {done:5d}/{job['tasks']:<5d} ({state})")
for name in ("graph/BASE_GRAPH.jsonl",):
    print(f"  {name}: {'ready' if (output / name).exists() else 'pending'}")
PY
    ;;
  *)
    echo "usage: $0 {plan|preflight|smoke|smoke-final|generate|generate-qwen|generate-gemma|generate-ministral-n3|generate-ministral-cot40|assemble|graph|decode|score|all|status}" >&2
    exit 2
    ;;
esac
