#!/usr/bin/env bash
# Download a tiny model for an end-to-end smoke test. Qwen2.5-0.5B-Instruct is ~1 GB
# in fp16 and loads on a single 8 GB RTX 2080 in seconds.
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
echo "[vllm-studio] downloading $MODEL into the HF cache…"
python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1])
print("downloaded to:", p)
PY
