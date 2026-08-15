#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PY="${PY:-$(command -v python || true)}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "Python was not found. Activate the release environment or set PY=/path/to/python." >&2
  exit 2
fi
OUT="${OUT:-runs/sota_reproduction_20260729_v1}"
PARSER_MODE="${PARSER_MODE:-legacy-20260729}"
STAGE="${1:-status}"
MODULE="lm_kbc.sota_reproduction"

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
