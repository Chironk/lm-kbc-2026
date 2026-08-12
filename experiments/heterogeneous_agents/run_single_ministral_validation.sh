#!/usr/bin/env bash
# Fresh validation inference for the final one-route Ministral architecture.
# Only the production Qwen route, Gemma, and Ministral SyntheticCoT N=10 are
# generated.  Every stage is resumable and the final ZIP is deterministic.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-$(command -v python || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "activate the project environment or set PY to its Python binary" >&2
  exit 2
fi
if ! "$PY" -c 'import pandas, torch' >/dev/null 2>&1; then
  echo "the selected Python lacks project dependencies: $PY" >&2
  echo "activate the lm-kbc-2026 environment or set PY explicitly" >&2
  exit 2
fi
OUT="${OUT:-experiments/heterogeneous_agents/runs/single_ministral_validation_20260810_v1}"
INPUT="${INPUT:-data/val.jsonl}"
SYNTHETIC_COT="${SYNTHETIC_COT:-data/synthetic_cot_capacity_aligned_v2.jsonl}"
QUESTION_CONTRACT="${QUESTION_CONTRACT:-official-v1}"
DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"
E2E="experiments.heterogeneous_agents.end_to_end_pipeline"
MODULE="experiments.heterogeneous_agents.single_ministral_validation"
STAGE="${1:-status}"
LOG="$OUT/run.log"

require_plan() {
  [[ -f "$OUT/plan/PLAN.json" ]] || {
    echo "missing plan; run '$0 plan' first" >&2
    exit 2
  }
}

gpu_count() {
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -c \
    'import torch; print(torch.cuda.device_count())'
}

job_value() {
  "$PY" - "$OUT" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
plan = json.loads((Path(sys.argv[1]) / "plan/PLAN.json").read_text())
print(plan["jobs"][sys.argv[2]][sys.argv[3]])
PY
}

generate_route() {
  local route="$1" slug="$2" directory="$3" batch="$4" checkpoint="$5"
  local workers tasks output agent agents
  workers="$(gpu_count)"
  [[ "$workers" -ge 1 ]] || { echo "no visible CUDA GPU" >&2; exit 2; }
  tasks="$(job_value "$route" "$([[ "$directory" == smoke_responses ]] && echo smoke_path || echo task_path)")"
  output="$(job_value "$route" "$([[ "$directory" == smoke_responses ]] && echo smoke_response_path || echo response_path)")"
  agent="$(job_value "$route" agent_id)"
  agents="$(job_value "$route" agent_config)"
  mkdir -p "$(dirname "$output")"
  echo "[$route] GPUs=$DEVICES workers=$workers batch=$batch"
  CUDA_VISIBLE_DEVICES="$DEVICES" \
  PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
  "$PY" -u -m experiments.heterogeneous_agents.run_agent \
    --agent-id "$agent" --tasks "$tasks" --output "$output" \
    --agents "$agents" --precision 4bit --num-workers "$workers" \
    --generation-batch-size 1 --task-batch-size "$batch" \
    --checkpoint-every "$checkpoint" --seed 20260730
}

case "$STAGE" in
  plan)
    mkdir -p "$OUT"
    "$PY" -u -m "$E2E" plan \
      --split validation --input "$INPUT" --output-dir "$OUT" \
      --synthetic-cot "$SYNTHETIC_COT" \
      --question-contract "$QUESTION_CONTRACT"
    "$PY" -u -m "$MODULE" freeze --output-dir "$OUT" \
      --primary-seed-scheme stable-key
    "$PY" -u run_submission.py --policy v0495 --input "$INPUT" \
      --output-dir "$OUT/primary_qwen" --seed-scheme stable-key --dry-run
    ;;
  smoke)
    require_plan
    generate_route "gemma:independent" gemma__independent smoke_responses 2 1 \
      2>&1 | tee -a "$LOG"
    generate_route "ministral:cot5_cap40_n10" ministral__cot5_cap40_n10 \
      smoke_responses 1 1 2>&1 | tee -a "$LOG"
    ;;
  generate-primary)
    require_plan
    "$PY" -u run_submission.py --policy v0495 --input "$INPUT" \
      --output-dir "$OUT/primary_qwen" --seed-scheme stable-key \
      2>&1 | tee -a "$LOG"
    ;;
  generate-gemma)
    require_plan
    generate_route "gemma:independent" gemma__independent responses 2 10 \
      2>&1 | tee -a "$LOG"
    ;;
  generate-ministral)
    require_plan
    generate_route "ministral:cot5_cap40_n10" ministral__cot5_cap40_n10 \
      responses 1 10 2>&1 | tee -a "$LOG"
    ;;
  decode)
    require_plan
    "$PY" -u -m "$MODULE" all --output-dir "$OUT" --gold "$INPUT" \
      2>&1 | tee -a "$LOG"
    ;;
  all)
    "$0" plan
    "$0" smoke
    "$0" generate-primary
    "$0" generate-gemma
    "$0" generate-ministral
    "$0" decode
    ;;
  replay)
    require_plan
    "$PY" -u -m "$MODULE" all --output-dir "$OUT" --gold "$INPUT"
    ;;
  status)
    if [[ ! -f "$OUT/plan/PLAN.json" ]]; then
      echo "plan: missing ($OUT/plan/PLAN.json)"
      exit 0
    fi
    "$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
plan = json.loads((out / "plan/PLAN.json").read_text())
print(f"split={plan['split']} rows={plan['rows']}")
for route in ("gemma:independent", "ministral:cot5_cap40_n10"):
    job = plan["jobs"][route]
    path = Path(job["response_path"])
    done = sum(1 for line in path.open() if line.strip()) if path.is_file() else 0
    print(f"{route:34s} {done:5d}/{job['tasks']:<5d}")
for name in (
    "primary_qwen/MANIFEST.json",
    "single_ministral/MANIFEST.json",
    "single_ministral/RESULT.json",
    "single_ministral/VERIFICATION.json",
    "single_ministral/submission/heterogeneous_single_ministral_cot40_component_v1_validation.zip",
):
    print(f"{name}: {'ready' if (out / name).is_file() else 'pending'}")
PY
    ;;
  *)
    echo "usage: $0 {plan|smoke|generate-primary|generate-gemma|generate-ministral|decode|replay|all|status}" >&2
    exit 2
    ;;
esac
