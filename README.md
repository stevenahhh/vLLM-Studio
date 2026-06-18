# vLLM Studio

vLLM Studio is a self-hosted full-stack app for running local LLMs with vLLM.
It combines a FastAPI control plane, a Next.js/shadcn dashboard, model download
tools, GPU telemetry, engine controls, chat, and KV-cache-aware VRAM estimates
before a model is loaded.

```text
Next.js dashboard (:3000)
        |
        | HTTP / SSE
        v
FastAPI control plane (:8000)
        |
        | starts and monitors
        v
vLLM OpenAI-compatible engine (:8001)
```

## Linux Requirements

Target platform: Linux with NVIDIA GPUs. Ubuntu 22.04 or 24.04 is the safest
baseline.

Required:

- NVIDIA driver visible to `nvidia-smi`
- Python 3.10 or 3.11
- Bun 1.1+
- Git
- A CUDA 12.x toolkit if your vLLM/Torch stack needs runtime CUDA compilation
- Enough disk space for Hugging Face model weights

Recommended for production or remote hosts:

- `tmux` or `systemd` to keep the backend and frontend alive
- `git-lfs` if you ever add large non-model assets
- Docker with NVIDIA Container Toolkit if you want the optional Docker vLLM runner

Do not commit model weights, runtime JSON state, local `.env` files, or Hugging
Face tokens. The root `.gitignore` is set up to exclude those by default.

## Fresh Linux Install

Clone the project and enter the root directory:

```bash
git clone <your-repo-url> vllm-studio
cd vllm-studio
```

Create a Python environment for the backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
bun install
cd ..
```

Create local environment files from the committed examples:

```bash
cp .env.example .env.local
cp frontend/.env.example frontend/.env.local
```

Edit the local files for your host. At minimum, set these when opening the UI
from another machine:

```bash
NEXT_PUBLIC_API_BASE=http://<linux-host>:8000
NEXT_ALLOWED_DEV_ORIGINS=<linux-host>,localhost,127.0.0.1
```

For gated or private Hugging Face repositories, export a token in your shell or
store it only in an uncommitted local env file:

```bash
export HF_TOKEN=hf_...
```

## Run Locally

Start the backend control plane:

```bash
scripts/run-backend.sh
```

In a second terminal, start the dashboard:

```bash
scripts/run-frontend.sh dev
```

Open `http://localhost:3000`. The backend API docs are available at
`http://localhost:8000/docs`.

You can also start both layers together:

```bash
scripts/run-all.sh
```

The control plane manages the vLLM engine. Do not start a separate vLLM OpenAI
server by hand on the same engine port unless you also change the app's engine
configuration.

## Production Build

Build the frontend:

```bash
cd frontend
bun run build
cd ..
```

Run the backend and the production frontend in separate supervised processes:

```bash
scripts/run-backend.sh
scripts/run-frontend.sh start
```

Use a process manager such as `systemd`, `supervisord`, or `tmux` for long-lived
remote use. Keep the backend reachable by the browser through
`NEXT_PUBLIC_API_BASE`.

## Model Storage

By default, the backend uses `HF_HOME` if it is set, otherwise
`~/.cache/huggingface`.

For a machine with a larger model disk, set:

```bash
export HF_HOME=/mnt/data/hf-cache
mkdir -p "$HF_HOME"
```

Downloads, local model scans, and the selected vLLM runner all follow `HF_HOME`.
Avoid storing model weights inside the repository.

## Smoke Test

Download a tiny model:

```bash
scripts/pull-smoke-model.sh
```

Then open the dashboard, choose **Load model**, select the downloaded model, and
load it. For API-level testing, use the backend docs at `/docs` or call the
`/api/engine/load` endpoint with the same model settings the UI would send.

## Optional Docker vLLM Runner

The default runner starts vLLM as a host process. To run only the vLLM engine in
Docker while keeping the FastAPI control plane on the host:

```bash
export VLLM_ENGINE_RUNNER=docker
export VLLM_DOCKER_IMAGE=vllm/vllm-openai:latest
scripts/run-backend.sh
```

Docker mode uses NVIDIA GPUs, host IPC, port `8001`, and the configured
Hugging Face cache. If `HF_TOKEN` is set, it is passed by environment name
rather than written into launch logs.

## Optional TurboQuant

TurboQuant support is available for the host process runner only.

Install it into the backend Python environment:

```bash
pip install "turboquant[vllm,triton] @ git+https://github.com/0xSero/turboquant.git"
```

Enable it before starting the backend:

```bash
export VLLM_ENGINE_RUNNER=process
export VLLM_TURBOQUANT=1
scripts/run-backend.sh
```

Default TurboQuant settings:

```bash
VLLM_TURBOQUANT_KEY_BITS=3
VLLM_TURBOQUANT_VALUE_BITS=2
VLLM_TURBOQUANT_BUFFER_SIZE=128
VLLM_TURBOQUANT_INITIAL_LAYERS=4
```

License note: TurboQuant is an optional third-party dependency from
`https://github.com/0xSero/turboquant`. Its upstream repository is licensed
under GPL-3.0. This project does not vendor TurboQuant or install it by default;
if you distribute a bundle or image that includes TurboQuant, review and comply
with its license. See `THIRD_PARTY_NOTICES.md`.

## Features

- Chat with the loaded model through the vLLM OpenAI-compatible API.
- Download Hugging Face models and quantized variants from the dashboard.
- Monitor live GPU memory and aggregate VRAM usage.
- Estimate load-time VRAM, KV cache size, and OOM risk before loading.
- Configure quantization, dtype, context length, tensor parallelism, GPU memory
  utilization, max sequences, KV-cache dtype, and eager mode.
- Persist a load-time system prompt, with an optional custom prompt override.
- Support standard LLM params and diffusion-model params through vLLM
  `extra_body`.

## Hardware Detection

The dashboard hardware page is generated from backend detection, not hard-coded
README text. The backend uses NVML when available and falls back conservatively
when hardware data cannot be detected. Recommendations should be treated as
startup guidance, not as a guarantee that every model will fit.

## VRAM Estimation

The estimator accounts for model weights, KV cache, activations, CUDA graph
overhead, runtime overhead, tensor parallelism, context length, and configured
GPU memory utilization.

Simplified per-GPU shape:

```text
weights_per_gpu   = weights_total / tensor_parallel_size
kv_per_token      = 2 * layers * kv_heads * head_dim * kv_bytes
kv_per_gpu        = kv_per_token * max_model_len * concurrency / tensor_parallel_size
required_per_gpu  = weights_per_gpu + kv_per_gpu + activations + overhead
budget_per_gpu    = gpu_total * gpu_memory_utilization
```

See `CONTRACT.md` for the full API and estimation contract.

## Project Layout

```text
backend/app/      FastAPI control plane
backend/tests/    Backend unit tests
frontend/         Next.js 16 + shadcn dashboard
scripts/          Local run and smoke-test helpers
CONTRACT.md       API and behavior contract
```

Important ports:

- `3000`: Next.js dashboard
- `8000`: FastAPI control plane
- `8001`: managed vLLM engine

## API

All control-plane routes are under `/api`.

Common routes:

- `GET /api/hardware`
- `GET /api/gpu/stats`
- `GET /api/gpu/stream`
- `GET /api/models/downloaded`
- `GET /api/models/search`
- `GET /api/models/variants`
- `GET /api/models/meta`
- `POST /api/estimate`
- `GET|POST /api/downloads`
- `GET|POST /api/engine`
- `POST /api/engine/load`
- `POST /api/engine/unload`
- `GET /api/engine/logs`
- `GET /api/params/schema`
- `GET|PUT /api/settings`
- `POST /api/chat/completions`

Full schema: `http://localhost:8000/docs`.

## Verification

Backend command-building tests:

```bash
PYTHONPATH="$PWD" python -m unittest backend.tests.test_vllm_manager -v
```

Frontend checks:

```bash
cd frontend
bun run typecheck
bun run lint
cd ..
```

GitHub Actions runs the same lightweight backend test and frontend checks without
starting FastAPI, Next.js, or vLLM.

## GitHub Publishing Checklist

This is a full-stack app and should be published as a root-level monorepo.

Before the first root commit:

1. Resolve the existing nested `frontend/.git` directory. If you keep it in
   place, Git will treat `frontend/` as an embedded repository instead of normal
   project files.
2. Keep real runtime state out of Git: `data/*.json`, `.env.local`,
   `frontend/.env.local`, `.omo/`, `.codegraph`, model weights, and caches.
3. Keep the root `LICENSE` and `THIRD_PARTY_NOTICES.md` files. This project is
   MIT licensed; the optional TurboQuant integration is documented separately
   because TurboQuant itself is GPL-3.0 licensed.
4. Run the verification commands above.
5. Inspect the final add set before committing:

```bash
git status --short
git diff --cached --stat
```

Recommended first commits:

```text
chore(repo): prepare publishable monorepo
feat(engine): add vllm studio runtime
docs(readme): document linux installation
ci(github): add backend and frontend checks
```

Do not push until the remote URL, repository visibility, and branch name are
intentional.

## License

vLLM Studio is released under the MIT License. See `LICENSE`.

TurboQuant is an optional, separately installed third-party integration licensed
upstream under GPL-3.0. See `THIRD_PARTY_NOTICES.md`.
