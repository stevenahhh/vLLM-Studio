#!/usr/bin/env bash
# Start the vLLM Studio control plane (FastAPI on :8000).
# It manages the vLLM engine (:8001) itself — process runner by default,
# or Docker when VLLM_ENGINE_RUNNER=docker. Do not start vLLM by hand.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../backend"
export PYTHONPATH="$PWD"
export VLLM_STUDIO_HOST="${VLLM_STUDIO_HOST:-0.0.0.0}"
export VLLM_STUDIO_PORT="${VLLM_STUDIO_PORT:-8000}"

# --- Model storage: keep models on the FAST root SSD -------------------------
# /mnt/data here reads at ~35 MB/s (vs ~300 MB/s+ on root) → painfully slow model
# loads. Default the HF cache to the root SSD. Everything (downloads, local scan,
# and the selected vLLM runner) follows HF_HOME.
if [ -z "${HF_HOME:-}" ]; then
  export HF_HOME="$HOME/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "[vllm-studio] model cache (HF_HOME) → $HF_HOME"

# --- CUDA toolkit for runtime JIT (FlashInfer/torch) -------------------------
# System nvcc is 11.5, which FlashInfer rejects (needs CUDA >= 12) and crashes
# the engine during KV-cache profiling. Prefer a 12.x toolkit if present.
for _cuda in /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda; do
  if [ -x "$_cuda/bin/nvcc" ]; then
    export CUDA_HOME="${CUDA_HOME:-$_cuda}"
    export CUDACXX="${CUDACXX:-$_cuda/bin/nvcc}"
    case ":$PATH:" in *":$_cuda/bin:"*) ;; *) export PATH="$_cuda/bin:$PATH" ;; esac
    echo "[vllm-studio] CUDA toolkit (JIT) → $CUDA_HOME ($("$_cuda/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1))"
    break
  fi
done
echo "[vllm-studio] control plane → http://${VLLM_STUDIO_HOST}:${VLLM_STUDIO_PORT}  (docs at /docs)"
exec python3 -m uvicorn app.main:app --host "$VLLM_STUDIO_HOST" --port "$VLLM_STUDIO_PORT" "$@"
