#!/usr/bin/env bash
# Run both the control plane (:8000) and the dashboard (:3000) together.
# Ctrl-C stops both.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/run-backend.sh" &
BACK=$!
"$HERE/run-frontend.sh" dev &
FRONT=$!

cleanup() { echo; echo "[vllm-studio] stopping…"; kill "$BACK" "$FRONT" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup INT TERM EXIT

echo "[vllm-studio] backend pid=$BACK  frontend pid=$FRONT — Ctrl-C to stop both"
wait -n "$BACK" "$FRONT"
