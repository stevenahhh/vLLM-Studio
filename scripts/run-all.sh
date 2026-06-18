#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_MODE="${1:-dev}"
TMP_DIR="$(mktemp -d -t vllm-studio-run-all.XXXXXX)"

WRAPPERS=()

cleanup() {
  local status=$?
  trap - INT TERM EXIT
  echo
  echo "[run-all] stopping..."

  for pid_file in "$TMP_DIR"/*.pid; do
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  for wrapper in "${WRAPPERS[@]:-}"; do
    if kill -0 "$wrapper" 2>/dev/null; then
      kill "$wrapper" 2>/dev/null || true
    fi
  done

  # Kill vLLM engine and all its worker children
  local vllm_pids all_vllm
  vllm_pids="$(lsof -ti:8001 2>/dev/null || true)"
  if [ -n "$vllm_pids" ]; then
    # collect full process tree (parent + children)
    all_vllm="$vllm_pids"
    for p in $vllm_pids; do
      all_vllm="$all_vllm $(pgrep -P "$p" 2>/dev/null || true)"
    done
    echo "[run-all] killing vLLM engine + workers: $all_vllm"
    kill -9 $all_vllm 2>/dev/null || true
  fi
  # Fallback: kill any remaining VLLM::Worker processes
  pkill -9 -f "VLLM::Worker" 2>/dev/null || true

  wait 2>/dev/null || true
  rm -rf "$TMP_DIR"
  exit "$status"
}

trap cleanup INT TERM EXIT

run_layer() {
  local layer="$1"
  shift

  (
    set +e
    local fifo="$TMP_DIR/$layer.fifo"
    mkfifo "$fifo"

    while IFS= read -r line; do
      printf '[%s] %s\n' "$layer" "$line"
    done < "$fifo" &
    local prefixer=$!

    "$@" > "$fifo" 2>&1 &
    local child=$!
    printf '%s\n' "$child" > "$TMP_DIR/$layer.pid"

    wait "$child"
    local status=$?
    wait "$prefixer" 2>/dev/null || true
    rm -f "$fifo" "$TMP_DIR/$layer.pid"
    exit "$status"
  ) &

  WRAPPERS+=("$!")
}

run_layer backend "$HERE/run-backend.sh"
run_layer frontend "$HERE/run-frontend.sh" "$FRONTEND_MODE"

echo "[run-all] backend + frontend started; Ctrl-C to stop"

set +e
wait -n "${WRAPPERS[@]}"
EXITED=$?
exit "$EXITED"
