#!/usr/bin/env bash
# Start the vLLM Studio dashboard (Next.js on :3000). Talks to the control plane via
# NEXT_PUBLIC_API_BASE. If you open the UI from another machine, point this at the
# control plane's reachable URL, e.g. NEXT_PUBLIC_API_BASE=http://<host>:8000
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../frontend"
export NEXT_PUBLIC_API_BASE="${NEXT_PUBLIC_API_BASE:-http://localhost:8000}"
PORT="${FRONTEND_PORT:-3000}"
MODE="${1:-dev}"   # dev | build | start
echo "[vllm-studio] dashboard → http://localhost:${PORT}  (API_BASE=${NEXT_PUBLIC_API_BASE})"
case "$MODE" in
  build) exec bun run build ;;
  start) exec bun run start -- -p "$PORT" ;;
  *)     exec bun run dev   -- -p "$PORT" ;;
esac
