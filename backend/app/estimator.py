"""VRAM estimation + OOM verdict — the core math (CONTRACT.md §6).

Pure function module: NO network, NO disk, NO torch/vllm/pynvml imports. Given a
`ModelMeta` (config-derived or file-derived weight sizes), an `EstimateRequest`
(engine/runtime knobs) and `HardwareInfo` (per-GPU bytes, gpu count), it returns a
`VramEstimate` with a full byte-level breakdown, an OOM status and the max safe
context window. All sizes are bytes (int); 1 GiB = 1024**3.
"""
from __future__ import annotations

import math

from .config import (
    ACTIVATION_FRACTION,
    ACTIVATION_MIN,
    CUDA_GRAPH_PER_GPU,
    GIB,
    NON_TORCH_OVERHEAD_PER_GPU,
)
from .schemas import (
    EstimateRequest,
    HardwareInfo,
    ModelMeta,
    PerGpuBreakdown,
    VramEstimate,
)

# Bytes-per-param for derived (config-based) weight sizing. The contract's
# "simpler accepted approx" for keeping embeddings/norms at fp16 while quantising
# linear layers: 4-bit ≈ 0.6 B/param, int8 ≈ 1.06 B/param, fp16/none = 2 B/param.
_BYTES_PER_PARAM = {
    "fp16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "none": 2.0,
    "auto": 2.0,
    "int8": 1.06,
    "bitsandbytes": 1.06,
    "bnb": 1.06,
    "awq": 0.6,
    "awq_marlin": 0.6,
    "gptq": 0.6,
    "gptq_marlin": 0.6,
    "gguf_q4": 0.6,
    "gguf": 0.6,
}


def _kv_dtype_bytes(kv_cache_dtype: str) -> int:
    """KV cache element size: 2 bytes (fp16) unless fp8 explicitly requested → 1.

    (On Turing fp8 KV isn't actually usable, but this module is pure: it honours
    the request and lets hwinfo/vllm_manager enforce the hardware policy.)
    """
    return 1 if str(kv_cache_dtype).lower() == "fp8" else 2


def estimate(meta: ModelMeta, req: EstimateRequest, hw: HardwareInfo) -> VramEstimate:
    """Compute the VRAM estimate + OOM verdict per CONTRACT.md §6 (a)-(g)."""
    reasons: list[str] = []
    approximate: bool = False

    # Effective quant: request may override meta.quant (§6 inputs).
    quant = (req.quant or meta.quant or "none").lower()

    tp = max(1, int(req.tensor_parallel_size))
    num_gpus = max(0, int(hw.num_gpus))
    util = float(req.gpu_memory_utilization)
    gpu_total = int(hw.per_gpu_bytes)

    # --- (a) Weights bytes -----------------------------------------------------
    # Prefer real summed weight-file sizes when known (most accurate); otherwise
    # derive from the config-estimated param count via bytes/param for the quant.
    if meta.weight_bytes_known is not None and meta.weight_bytes_known > 0:
        weights_total = int(meta.weight_bytes_known)
        weights_source = "files"
    else:
        bpp = _BYTES_PER_PARAM.get(quant, 2.0)
        weights_total = int(round(int(meta.param_count) * bpp))
        weights_source = "config"
        if meta.param_count <= 0:
            reasons.append("Unknown parameter count; weight estimate may be inaccurate.")

    # --- (b) KV cache per token ------------------------------------------------
    # kv_per_token = 2 (K & V) * layers * kv_heads * head_dim * kv_dtype_bytes.
    kv_dtype_bytes = _kv_dtype_bytes(req.kv_cache_dtype)
    head_dim = meta.head_dim
    if head_dim <= 0 and meta.num_attention_heads > 0 and meta.hidden_size > 0:
        head_dim = meta.hidden_size // meta.num_attention_heads
    kv_heads = meta.num_key_value_heads or meta.num_attention_heads
    kv_per_token = (
        2 * meta.num_hidden_layers * kv_heads * head_dim * kv_dtype_bytes
    )

    # --- (c) KV cache total (pool the engine must hold) ------------------------
    # Diffusion family: KV is bounded by block_length, not max_model_len, and the
    # estimate is flagged approximate (vLLM diffusion support is experimental).
    kv_concurrency = max(1, int(req.kv_concurrency))
    if meta.family == "diffusion":
        approximate = True
        ctx_for_kv = max(1, int(req.block_length))
        reasons.append(
            "Diffusion family: KV bounded by block_length; estimate is approximate."
        )
    else:
        ctx_for_kv = max(1, int(req.max_model_len))
    kv_total = kv_per_token * ctx_for_kv * kv_concurrency

    # --- (e) Per-GPU split across tensor-parallel ranks ------------------------
    # Weights and KV shard evenly across tp ranks.
    weights_per_gpu = weights_total // tp if tp else weights_total
    kv_per_gpu = kv_total // tp if tp else kv_total

    # --- (d) Activations / transient buffers (per GPU) -------------------------
    activation = int(max(ACTIVATION_MIN, ACTIVATION_FRACTION * weights_per_gpu))

    # CUDA graph capture buffers (0 under enforce_eager) and the fixed CUDA/NCCL
    # context + driver overhead.
    cuda_graph = 0 if req.enforce_eager else int(CUDA_GRAPH_PER_GPU)
    overhead = int(NON_TORCH_OVERHEAD_PER_GPU)

    required_per_gpu = weights_per_gpu + kv_per_gpu + activation + cuda_graph + overhead
    budget_per_gpu = int(gpu_total * util)
    headroom_per_gpu = budget_per_gpu - required_per_gpu

    # --- (f) OOM verdict -------------------------------------------------------
    status: str = "ok"

    # Not enough physical GPUs for the requested tensor-parallel size.
    if num_gpus and tp > num_gpus:
        status = "oom"
        reasons.append(
            f"Tensor parallel size {tp} exceeds available GPUs ({num_gpus}): not enough GPUs."
        )
    elif gpu_total <= 0:
        # No hardware telemetry — degrade gracefully, don't crash the request.
        status = "tight"
        reasons.append("Hardware unknown; cannot verify VRAM budget.")
    else:
        # Won't even fit physically, or no room for any KV after weights+overhead.
        if required_per_gpu > gpu_total:
            status = "oom"
            reasons.append(
                "Required per-GPU VRAM exceeds physical capacity; won't fit."
            )
        elif weights_per_gpu + overhead > budget_per_gpu:
            status = "oom"
            reasons.append(
                "Weights + overhead leave no room for KV cache under the util budget; vLLM would abort."
            )
        elif required_per_gpu > budget_per_gpu:
            status = "tight"
            reasons.append(
                "Required VRAM exceeds the gpu_memory_utilization budget; raise util or risk OOM."
            )
        elif headroom_per_gpu < 0.5 * GIB:
            status = "tight"
            reasons.append("Less than 0.5 GiB of headroom under the util budget.")

    if status == "ok":
        reasons.append("Fits within the gpu_memory_utilization budget.")

    # --- (g) Max safe context window ------------------------------------------
    # Solve max_model_len at the util budget for the current tp & concurrency,
    # clamped to >= 0 and to config.max_position_embeddings.
    kv_denom = kv_per_token * kv_concurrency
    if kv_denom > 0:
        free_for_kv = (
            budget_per_gpu - weights_per_gpu - activation - cuda_graph - overhead
        ) * tp
        max_ctx = int(math.floor(free_for_kv / kv_denom))
    else:
        # No attention KV (e.g. unknown dims): fall back to the configured ceiling.
        max_ctx = meta.max_position_embeddings
    if max_ctx < 0:
        max_ctx = 0
    if meta.max_position_embeddings and max_ctx > meta.max_position_embeddings:
        max_ctx = meta.max_position_embeddings

    # --- Aggregates (totals across all participating GPUs) ---------------------
    aggregate_required = required_per_gpu * tp
    aggregate_budget = budget_per_gpu * tp

    per_gpu = PerGpuBreakdown(
        weights_bytes=int(weights_per_gpu),
        kv_bytes=int(kv_per_gpu),
        activation_bytes=int(activation),
        cuda_graph_bytes=int(cuda_graph),
        overhead_bytes=int(overhead),
        required_bytes=int(required_per_gpu),
        budget_bytes=int(budget_per_gpu),
        total_bytes=int(gpu_total),
        headroom_bytes=int(headroom_per_gpu),
    )

    return VramEstimate(
        status=status,  # type: ignore[arg-type]
        reasons=reasons,
        tensor_parallel_size=tp,
        num_gpus_available=num_gpus,
        weights_total_bytes=int(weights_total),
        kv_per_token_bytes=int(kv_per_token),
        kv_total_bytes=int(kv_total),
        activation_bytes=int(activation),
        weights_source=weights_source,  # type: ignore[arg-type]
        approximate=approximate,
        max_safe_context=int(max_ctx),
        per_gpu=per_gpu,
        aggregate_required_bytes=int(aggregate_required),
        aggregate_budget_bytes=int(aggregate_budget),
    )
