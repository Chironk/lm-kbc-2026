#!/usr/bin/env bash
# Public entry point for the architecture associated with the official 0.4845
# test submission. The implementation keeps its historical filename so its
# provenance remains explicit; this wrapper provides a stable release name.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
STAGE="${1:-status}"

case "$STAGE" in
  verify-release)
    PY="${PY:-$(command -v python || true)}"
    if [[ -z "$PY" || ! -x "$PY" ]]; then
      echo "Python was not found. Activate the release environment or set PY=/path/to/python." >&2
      exit 2
    fi
    cd "$ROOT"
    exec "$PY" scripts/verify_release.py
    ;;
  *)
    exec "$HERE/internal/run_historical_sota_test_pipeline.sh" "$@"
    ;;
esac
