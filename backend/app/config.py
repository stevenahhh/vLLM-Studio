"""Central settings & paths for the vLLM Studio control plane.

Read-only contract module: other modules import these constants. Do not rewrite.
"""
from __future__ import annotations

import os
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2

# --- Ports / processes ---------------------------------------------------------
CONTROL_HOST = os.environ.get("VLLM_STUDIO_HOST", "0.0.0.0")
CONTROL_PORT = int(os.environ.get("VLLM_STUDIO_PORT", "8000"))
# Managed vllm serve subprocess (OpenAI-compatible):
VLLM_HOST = os.environ.get("VLLM_ENGINE_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("VLLM_ENGINE_PORT", "8001"))
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

# --- Paths ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("VLLM_STUDIO_DATA", str(REPO_ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "state.json"
DOWNLOADS_FILE = DATA_DIR / "downloads.json"

# HuggingFace cache location.
# NOTE: /mnt/data is a very slow disk here (~35 MB/s real reads vs ~300 MB/s+ on
# the root SSD) — putting models there made loads take many minutes. Keep the
# cache on the fast root SSD by default. Override with HF_HOME / HF_HUB_CACHE.
def _default_hf_home() -> str:
    return str(Path.home() / ".cache" / "huggingface")


HF_HOME = os.environ.get("HF_HOME") or _default_hf_home()
os.environ.setdefault("HF_HOME", HF_HOME)  # so child vllm processes inherit it
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", str(Path(HF_HOME) / "hub"))
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)


def _setup_cuda_toolkit() -> str:
    """Ensure runtime JIT compilers (FlashInfer / torch cpp_extension) use a
    CUDA **12+** toolkit. The system nvcc here is 11.5, which FlashInfer rejects
    ("CUDA versions below 12 are not supported") — it crashes the vLLM engine
    during KV-cache profiling. A 12.8 toolkit exists at /usr/local/cuda-12.8, so
    point CUDA_HOME/PATH/CUDACXX at it. Child vllm processes inherit os.environ.
    Returns the chosen CUDA home (or "" if none with nvcc was found).
    """
    for cand in ("/usr/local/cuda-12.8", "/usr/local/cuda-12", "/usr/local/cuda"):
        nvcc = Path(cand) / "bin" / "nvcc"
        if nvcc.is_file():
            os.environ.setdefault("CUDA_HOME", cand)
            os.environ.setdefault("CUDACXX", str(nvcc))
            binpath = str(Path(cand) / "bin")
            cur = os.environ.get("PATH", "")
            if binpath not in cur.split(os.pathsep):
                os.environ["PATH"] = binpath + os.pathsep + cur  # 12.x nvcc wins
            return cand
    return ""


CUDA_HOME = _setup_cuda_toolkit()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

# --- Hardware policy (Turing / RTX 2080) --------------------------------------
# Compute capability below 8.0 => no bf16 accel; below 8.9 => no fp8.
# These are computed live in hwinfo.py from the actual GPUs; the defaults below
# are conservative fallbacks used when nvml is unavailable.
DEFAULT_DTYPE = "float16"
DEFAULT_GPU_MEM_UTIL = float(os.environ.get("VLLM_STUDIO_GPU_UTIL", "0.90"))

# --- Estimator constants (see CONTRACT.md §6) ---------------------------------
NON_TORCH_OVERHEAD_PER_GPU = 0.9 * GIB
CUDA_GRAPH_PER_GPU = 0.6 * GIB
ACTIVATION_MIN = 0.25 * GIB
ACTIVATION_FRACTION = 0.12  # of per-gpu weights

# vLLM launch
VLLM_STARTUP_TIMEOUT = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "900"))  # seconds
LOG_TAIL_LINES = 400
VLLM_ENGINE_RUNNER = os.environ.get("VLLM_ENGINE_RUNNER", "process").strip().lower()
VLLM_DOCKER_IMAGE = os.environ.get("VLLM_DOCKER_IMAGE", "vllm/vllm-openai:latest")
VLLM_CONTAINER_NAME = os.environ.get("VLLM_CONTAINER_NAME", f"vllm-studio-engine-{VLLM_PORT}")
VLLM_TURBOQUANT = os.environ.get("VLLM_TURBOQUANT", "").strip().lower() in {"1", "true", "yes", "on"}
