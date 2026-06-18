# vLLM Studio — Build Contract (single source of truth)

This file is the authoritative spec. Every implementation agent MUST read it and the
two contract files (`backend/app/schemas.py`, `frontend/lib/types.ts`) before writing code,
and MUST match the names/signatures/shapes here exactly so independently-written modules
compose.

---
## 0. Hardware reality (this machine)
- **8× NVIDIA GeForce RTX 2080, 8 GB each** (8192 MiB). Driver 570, CUDA 12.8.
- **Compute capability 7.5 (Turing).** Consequences, treated as ground truth everywhere:
  - **No bfloat16** acceleration and **no FP8** (needs sm_89+). vLLM must run **`--dtype float16`**.
  - **No FP8 KV cache**, **no Marlin** kernels. KV cache dtype = fp16 (2 bytes).
  - **AWQ and GPTQ 4-bit DO work** on Turing (good for 8 GB cards). bnb works but slow.
  - Per-GPU only 8 GB ⇒ anything past ~7B-4bit needs **tensor parallelism** across GPUs.
- Attention backend: let vLLM choose; allow env override `VLLM_ATTENTION_BACKEND` (XFORMERS is the safe fallback on sm75).

## 1. Processes & ports
- **Control plane**: FastAPI (this backend) on **:8000**. The brain.
- **Inference**: a managed OpenAI-compatible vLLM engine on **:8001**. By default this is a direct **`vllm serve` subprocess**; set `VLLM_ENGINE_RUNNER=docker` to run the vLLM engine through `docker run` instead. Loading a model = spawn the selected runner; unloading = terminate it (cleanly frees VRAM).
- **Frontend**: Next.js on **:3000**, talks to :8000 directly (CORS open). Base URL from `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`).

## 2. File ownership (one agent per file; write ONLY your file)
Backend (`backend/app/`):
- `config.py`  — DONE (settings/paths). Do not rewrite.
- `schemas.py` — DONE (pydantic contract). Do not rewrite.
- `gpu.py`        — pynvml telemetry.
- `hwinfo.py`     — static capability detection (dtype/quant support).
- `modelmeta.py`  — fetch/parse HF config, param count, family detection.
- `estimator.py`  — VRAM estimate + OOM (THE core math; see §6).
- `params.py`     — model-type-aware editable param schemas (see §7).
- `registry.py`   — downloaded-model scan + HF Hub search + quant variants.
- `downloader.py` — background snapshot_download w/ progress.
- `vllm_manager.py` — spawn/health/kill the selected vLLM runner (`process` or `docker`).
- `chat.py`       — proxy/stream chat to the vllm subprocess.
- `state.py`      — persisted app state (current model, params, system prompt).
- `main.py`       — FastAPI app, CORS, all routes (§5). Imports everything above.

Frontend (`frontend/`):
- `lib/types.ts`  — DONE (TS contract mirror). Do not rewrite.
- `lib/api.ts`    — typed fetch client + SSE helpers.
- `lib/format.ts` — bytes→GB, percent, color thresholds.
- `lib/store.ts`  — zustand store (active config + estimate + live gpu stats).
- `app/layout.tsx` — providers + shell (sidebars). Overwrite the scaffold's.
- `app/page.tsx`   — redirect to `/chat`. Overwrite scaffold.
- `app/chat/page.tsx`      — chat page (uses ChatPanel + sidebars).
- `app/dashboard/page.tsx` — dashboard (GPU grid, models, downloads).
- `components/providers.tsx`  — client providers (SidebarProvider, Tooltip, Toaster, theme).
- `components/app-sidebar.tsx`— LEFT nav sidebar (links + active model status + model picker trigger).
- `components/vram-inspector.tsx` — RIGHT sidebar: live VRAM gauges + estimate breakdown + OOM verdict. Easy-access, collapsible.
- `components/gpu-monitor.tsx`  — per-GPU VRAM gauge list (shared by inspector & dashboard).
- `components/model-picker.tsx` — Dialog, tabs Downloaded / Hugging Face; HF tab: search + per-quant download + OOM precheck.
- `components/param-panel.tsx`  — dynamic editable params from schema (context window, system prompt, sampling/diffusion, engine load args). Drives estimate.
- `components/chat-panel.tsx`   — chat UI (markdown, streaming).
- `components/download-manager.tsx` — active/finished downloads w/ progress.
- `components/oom-badge.tsx`    — reusable OOM verdict badge (ok/tight/oom).

## 3. Conventions
- Python: 3.10, type hints, `from __future__ import annotations` ok. Async FastAPI. Use `pynvml` (import works). HF: `huggingface_hub` (v1.17). Never hard-fail a route on telemetry errors — return best-effort.
- Sizes: **always bytes (int) in the API**; frontend formats to GiB. 1 GiB = 1024³.
- TS: strict. Import shared types from `@/lib/types`. UI from `@/components/ui/*`. Icons from `lucide-react`. `cn` from `@/lib/utils`.
- All list endpoints return `{ items: [...] }` per schemas. All errors: HTTP 4xx/5xx with `{ detail: str }` (FastAPI default ok).

## 4. Model family detection (used by params + estimator)
Family ∈ `"llm" | "diffusion" | "moe"`. Decide from HF config.json:
- `diffusion` if `model_type` or any architecture string contains "diffusion"/"dream"/"llada"/"diffu" (case-insensitive), OR config has keys like `diffusion_steps`/`num_diffusion_timesteps`/`mask_token_id` with a non-causal lm. **diffusiongemma** → diffusion.
- `moe` if config has `num_experts`/`num_local_experts`/`n_routed_experts` > 1.
- else `llm`.

## 5. HTTP API (FastAPI on :8000). All under `/api`.
Request/response bodies are the pydantic models in `schemas.py` (mirror in `types.ts`).
- `GET  /api/health` → `{status:"ok", vllm: EngineStatus}`
- `GET  /api/hardware` → `HardwareInfo` (gpus[], capability flags, recommendations).
- `GET  /api/gpu/stats` → `GpuStats` (per-GPU live). Cheap; frontend polls ~1.5s.
- `GET  /api/gpu/stream` → text/event-stream of `GpuStats` (~1s). (Polling is the fallback.)
- `GET  /api/models/downloaded` → `{items: DownloadedModel[]}`
- `GET  /api/models/search?q=&limit=` → `{items: HFModel[]}` (HF Hub search; include detected quant variants per repo where cheap).
- `GET  /api/models/variants?repo=` → `{items: QuantVariant[]}` (quant options for a repo: distinct quant configs in-repo and/or sibling quant repos; each has repo, quant, revision, est size bytes, files).
- `GET  /api/models/meta?repo=&quant=&revision=` → `ModelMeta` (config-derived; used for estimation BEFORE download).
- `POST /api/estimate` (body `EstimateRequest`) → `VramEstimate` (§6).
- `GET  /api/downloads` → `{items: DownloadJob[]}`
- `POST /api/downloads` (body `DownloadRequest`) → `DownloadJob`
- `DELETE /api/downloads/{id}` → `{ok:true}` (cancel)
- `GET  /api/engine` → `EngineStatus` (loaded repo/quant/args/state/port/logs tail).
- `POST /api/engine/load` (body `LoadRequest`) → `EngineStatus` (spawn vllm serve; returns immediately as state="loading"; frontend polls /api/engine).
- `POST /api/engine/unload` → `EngineStatus`
- `GET  /api/engine/logs` → text/event-stream of log lines.
- `GET  /api/params/schema?repo=&quant=` → `ParamSchema` (§7; family-aware).
- `GET  /api/settings` → `AppSettings`; `PUT /api/settings` (body `AppSettings`) → `AppSettings`.
- `POST /api/chat/completions` (body `ChatRequest`) → text/event-stream of `ChatChunk` (OpenAI-style deltas) when stream=true; else `ChatResponse`.

## 6. VRAM estimate + OOM (estimator.py) — EXACT MATH
Everything in **bytes**. GiB = 1024³. Inputs: `ModelMeta` + `EstimateRequest` (max_model_len, max_num_seqs, kv_concurrency, tensor_parallel_size tp, gpu_memory_utilization util, quant, kv_dtype, enforce_eager) + hardware (per-GPU total bytes, num gpus).

Constants:
- `BYTES = {"fp16":2,"bf16":2,"float16":2,"int8":1,"awq":0.5,"gptq":0.5,"gguf_q4":0.5,"fp8":1,"none":2}` (bytes/param for weights of quant linears).
- `kv_dtype_bytes`: fp16/auto = 2; fp8 = 1 (only if hardware supports; on Turing force 2).
- `NON_TORCH_OVERHEAD_PER_GPU = 0.9 * GiB` (CUDA context + NCCL + driver).
- `CUDA_GRAPH_PER_GPU = 0 if enforce_eager else 0.6 * GiB`.
- `SAFETY = 0.0` (the util factor already creates headroom).

(a) **Weights bytes** `weights_total`:
  1. If real file sizes for the chosen (repo,quant,revision) are known (from HF siblings or local cache), **use the summed size of weight files** (`*.safetensors`,`*.bin`,`*.gguf`). Most accurate. Set `weights_source="files"`.
  2. Else derive param count P from config and multiply by bytes/param:
     - `head_dim = config.head_dim or hidden_size // num_attention_heads`
     - `kv_dim = num_key_value_heads * head_dim`
     - embeddings `= vocab_size * hidden_size`; add another `vocab_size*hidden_size` if `tie_word_embeddings` is False.
     - per layer attn `= hidden*hidden (q) + hidden*kv_dim (k) + hidden*kv_dim (v) + hidden*hidden (o)`.
     - per layer mlp: gated/SwiGLU (gate_proj+up_proj+down_proj) `= 3*hidden*intermediate`; else `2*hidden*intermediate`. (Detect by `hidden_act` containing "silu"/"gelu" with gate, or architecture; default gated for modern models.)
     - MoE: multiply mlp by num_experts (params), but only ~top_k active for compute — for VRAM use FULL expert params (all experts resident).
     - `P = embeddings + num_hidden_layers*(attn+mlp)`.
     - `bytes_per_param`: quant in {awq,gptq,gguf_q4} → use 0.5 for linear weights but keep embeddings+norms at fp16. Approx: `weights_total = (P_linear*0.5) + (P_embed_and_norm*2)`. Simpler accepted approx if separating is hard: `P * 0.6` for 4-bit, `P*1.06` (×1 +scales) for int8, `P*2` for fp16. Document which you used. Set `weights_source="config"`.

(b) **KV cache per token**:
  `kv_per_token = 2 * num_hidden_layers * num_key_value_heads * head_dim * kv_dtype_bytes`
  (2 = K and V). For diffusion family: KV is bounded by `block_length` not max_model_len; if no block_length, treat as full attention over `max_model_len` once (set `kv_concurrency`=1) and flag `approximate=true`.

(c) **KV cache total (the pool the engine must hold)**:
  `kv_total = kv_per_token * max_model_len * kv_concurrency`
  where `kv_concurrency` defaults to 1 (one full-length sequence; the minimum vLLM needs to even start). Expose it so the UI can show multi-sequence scaling.

(d) **Activations / transient** (per GPU, rough):
  `activation = max(0.25*GiB, 0.12 * weights_per_gpu)` (intermediate buffers, logits, sampling).

(e) **Per-GPU split (tensor parallel tp)**: weights and KV shard across tp ranks:
  - `weights_per_gpu = weights_total / tp`
  - `kv_per_gpu      = kv_total / tp`
  - `required_per_gpu = weights_per_gpu + kv_per_gpu + activation + CUDA_GRAPH_PER_GPU + NON_TORCH_OVERHEAD_PER_GPU`
  - `budget_per_gpu   = gpu_total_bytes * util`   (vLLM only uses `util` fraction)
  - `headroom_per_gpu = budget_per_gpu - required_per_gpu`

(f) **OOM verdict** (`status`): compare required vs budget AND vs physical:
  - `"oom"`   if `required_per_gpu > gpu_total_bytes` (won't even fit physically) OR `weights_per_gpu + NON_TORCH_OVERHEAD_PER_GPU > budget_per_gpu` (no room left for any KV → vLLM aborts).
  - `"tight"` if `required_per_gpu > budget_per_gpu` but `<= gpu_total_bytes` (fits physically but exceeds util budget; raise util or risk OOM) OR `headroom_per_gpu < 0.5*GiB`.
  - `"ok"`    otherwise.
  Also: if `tp` > number of GPUs available → `status="oom"`, reason "not enough GPUs".

(g) **max safe context** (solve max_model_len at the util budget, current tp & concurrency):
  `max_ctx = floor( ((budget_per_gpu - weights_per_gpu - activation - CUDA_GRAPH_PER_GPU - NON_TORCH_OVERHEAD_PER_GPU) * tp) / (kv_per_token * kv_concurrency) )`, clamped ≥0 and ≤ config.max_position_embeddings.

Return `VramEstimate` with full breakdown (all the byte fields above), `status`, `reasons: string[]`, `max_safe_context`, `per_gpu` object, `aggregate` totals, `weights_source`, `approximate`.

## 7. Param schema (params.py) — family-aware
`ParamSchema = { family, model_type, groups: ParamGroup[] }`. Each `ParamGroup = {key,label,params: ParamSpec[]}`.
`ParamSpec = {key,label,type:"float"|"int"|"text"|"bool"|"select", min?,max?,step?,default,options?,help?,unit?, affects_vram?:bool, engine_only?:bool}`.
Groups & params:
- **engine** (load-time; affects_vram=true; engine_only=true): `quantization`(select: auto/none/awq/gptq/awq_marlin→disabled on Turing/bnb), `dtype`(select: float16 default; bfloat16 & auto present but flagged unsupported), `tensor_parallel_size`(int 1..8), `gpu_memory_utilization`(float 0.5..0.97 step .01 default .90), `max_model_len`(int; the **context window**; default min(config.max_position_embeddings, 8192)), `max_num_seqs`(int 1..256 default 16), `kv_cache_dtype`(select auto/fp8→disabled), `enforce_eager`(bool), `trust_remote_code`(bool), `kv_concurrency`(int, estimate-only, default 1).
- **sampling** (runtime; family=llm): `temperature`(0..2,.7), `top_p`(0..1,.95), `top_k`(int -1..200, -1), `min_p`(0..1,0), `max_tokens`(int, 512), `repetition_penalty`(.5..2,1.0), `presence_penalty`(-2..2,0), `frequency_penalty`(-2..2,0), `seed`(int optional), `stop`(text list).
- **diffusion** (family=diffusion; replaces sampling): `diffusion_steps`(int 1..512, 64, affects latency), `block_length`(int, 32, affects_vram), `temperature`(0..2,.2), `top_p`(0..1,.95), `alg`(select: "low_confidence"/"random"/"entropy" remasking), `alg_temp`(0..2,0), `max_tokens`(int,256), `denoising_schedule`(select linear/cosine). Mark group help: "Diffusion LMs are non-autoregressive; vLLM support is experimental — params are passed via extra_body."
- **prompt** (runtime): `system_prompt`(text, default "You are a helpful assistant.").
Return sensible defaults pulled from `ModelMeta` (e.g., max_model_len default from config, quantization default from detected quant).

## 8. Frontend store (lib/store.ts, zustand)
State: `{ hardware, gpuStats, engine, settings, activeMeta, paramSchema, paramValues, estimate, ...actions }`.
- `paramValues`: flat `Record<string, any>` of current engine+sampling+prompt values.
- On any engine/affects_vram param change → debounce → `POST /api/estimate` → set `estimate`. The VRAM inspector reads `gpuStats` (live) + `estimate` (predicted) and renders OOM verdict.
- Poll `/api/gpu/stats` every ~1.5s; poll `/api/engine` while state=="loading".

## 9. Layout / UX
- Left sidebar (`app-sidebar`): brand "vLLM Studio", nav (Chat, Dashboard), a "Model" card showing active model + Load button (opens model-picker), quick links to downloads. Collapsible to icons.
- Right sidebar (`vram-inspector`): ALWAYS available (collapsible), shows: aggregate VRAM used/total bar, per-GPU mini gauges (gpu-monitor), then "Predicted" card = estimate breakdown (weights/KV/activation/overhead) with the OOM badge + max-safe-context + headroom; updates live as params change in param-panel.
- Chat page: ChatPanel center, param-panel accessible (e.g., a Sheet or within right area), both sidebars.
- Theme is the preset (green/mono). Use shadcn tokens only; no hardcoded colors. Support dark mode (next-themes already wired).

## 10. Scripts (created at end, not by agents)
- `scripts/run-backend.sh`, `scripts/run-frontend.sh`, `scripts/dev.sh` (both), `backend/requirements.txt`, root `README.md`.
