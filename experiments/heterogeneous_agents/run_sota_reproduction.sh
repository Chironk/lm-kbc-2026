#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/hongjing/miniconda3/envs/lm-kbc-2026/bin/python}"
OUT="${OUT:-experiments/heterogeneous_agents/runs/sota_reproduction_20260729_v1}"
PARSER_MODE="${PARSER_MODE:-legacy-20260729}"
STAGE="${1:-status}"
MODULE="experiments.heterogeneous_agents.sota_reproduction"

case "$STAGE" in
  audit|status)
    "$PY" -u -m "$MODULE" "$STAGE" --output-dir "$OUT"
    ;;
  build|decode|verify|all)
    "$PY" -u -m "$MODULE" "$STAGE" \
      --output-dir "$OUT" --parser-mode "$PARSER_MODE"
    ;;
  *)
    echo "usage: $0 {audit|build|decode|verify|all|status}" >&2
    exit 2
    ;;
esac
