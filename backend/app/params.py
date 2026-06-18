"""Model-type-aware editable parameter schemas (CONTRACT.md §7).

Builds a family-aware ``ParamSchema`` describing the engine load-time args,
runtime sampling (or diffusion) params, and prompt params for a given model.
Defaults are seeded from ``ModelMeta``; option availability is gated by the
detected hardware ``Capabilities`` (Turing: no bf16, no fp8, no marlin).

Pure / no IO: only reads the provided ``meta`` and ``caps``. Never raises on a
malformed meta — degrades to safe defaults.
"""
from __future__ import annotations

import math
from typing import Optional

from .config import GIB
from .schemas import (
    Capabilities,
    HardwareInfo,
    ModelMeta,
    ParamGroup,
    ParamSchema,
    ParamSpec,
)

# Fallback context window when the model config does not advertise one.
_DEFAULT_CONTEXT = 8192
_FALLBACK_MAX_CONTEXT = 131072
_UNSUPPORTED_HELP = "unsupported on this GPU"

# Bytes/param used only to size a sensible *default* tensor_parallel_size; mirrors
# estimator._BYTES_PER_PARAM (4-bit ≈ 0.6, int8 ≈ 1.06, fp16 = 2).
_BPP = {
    "fp16": 2.0, "bf16": 2.0, "float16": 2.0, "none": 2.0, "auto": 2.0,
    "int8": 1.06, "bitsandbytes": 1.06, "bnb": 1.06,
    "awq": 0.6, "awq_marlin": 0.6, "gptq": 0.6, "gptq_marlin": 0.6,
    "gguf_q4": 0.6, "gguf": 0.6,
}
# Per-GPU VRAM reserved (besides weights) when picking a default TP: overhead
# (~0.9) + CUDA graphs (~0.6) + a minimal KV/activation budget (~1.1) ≈ 2.6 GiB.
_TP_RESERVE = 2.6 * GIB


def _weights_estimate_bytes(meta: ModelMeta, quant: str) -> int:
    """Rough total weight bytes for TP sizing: real file size if known, else config."""
    known = getattr(meta, "weight_bytes_known", None)
    if known and known > 0:
        return int(known)
    bpp = _BPP.get((quant or "none").lower(), 2.0)
    return int((getattr(meta, "param_count", 0) or 0) * bpp)


def _default_tensor_parallel(
    meta: ModelMeta, quant: str, hardware: Optional[HardwareInfo]
) -> int:
    """Smallest valid TP (1/2/4/8) that fits the weights across the GPUs.

    Returns 1 when hardware is single-GPU/unknown. Prefers a TP that divides the
    attention-head count (vLLM requires this) and leaves room for KV cache.
    """
    if hardware is None:
        return 1
    num_gpus = max(1, int(getattr(hardware, "num_gpus", 1) or 1))
    per_gpu = int(getattr(hardware, "per_gpu_bytes", 0) or 0)
    if num_gpus <= 1 or per_gpu <= 0:
        return 1
    weights = _weights_estimate_bytes(meta, quant)
    if weights <= 0:
        return 1
    usable = max(1.0, per_gpu * 0.90 - _TP_RESERVE)
    needed = math.ceil(weights / usable)
    heads = int(getattr(meta, "num_attention_heads", 0) or 0)
    candidates = [t for t in (1, 2, 4, 8) if t <= num_gpus]
    for t in candidates:
        if t < needed:
            continue
        if heads and heads % t != 0:
            continue
        return t
    # Nothing leaves clean headroom — return the largest head-divisible TP we can.
    best = 1
    for t in candidates:
        if not heads or heads % t == 0:
            best = t
    return best


def _supported(caps: Capabilities) -> list[str]:
    """Return the lower-cased set of quant names supported on this hardware."""
    try:
        return [str(q).lower() for q in (caps.supported_quantization or [])]
    except Exception:  # pragma: no cover - defensive
        return []


def _quant_options(caps: Capabilities, supported: list[str]) -> list[dict]:
    """Build the quantization select options, disabling unsupported entries.

    ``auto`` and ``none`` are always selectable. The remaining quant methods are
    disabled (with an explanatory help) unless present in
    ``caps.supported_quantization``. fp8 is only offered at all when the GPU
    advertises fp8 support.
    """
    options: list[dict] = [
        {"value": "auto", "label": "Auto (detect)"},
        {"value": "none", "label": "None (full precision)"},
    ]

    gated = [
        ("awq", "AWQ (4-bit)"),
        ("gptq", "GPTQ (4-bit)"),
        ("bitsandbytes", "bitsandbytes"),
    ]
    if getattr(caps, "supports_fp8", False):
        gated.append(("fp8", "FP8"))

    for value, label in gated:
        disabled = value not in supported
        opt: dict = {"value": value, "label": label, "disabled": disabled}
        if disabled:
            opt["help"] = _UNSUPPORTED_HELP
        options.append(opt)
    return options


def _quant_default(meta: ModelMeta, supported: list[str]) -> str:
    """Quant default: the detected quant if known & supported, else ``auto``."""
    q = (getattr(meta, "quant", "") or "").lower()
    if q and q not in ("none", "auto"):
        # Use the detected quant only if this hardware can actually run it;
        # otherwise fall back to auto so the UI does not preselect a disabled
        # option.
        if q in supported:
            return q
        return "auto"
    if q == "none":
        return "none"
    return "auto"


def _dtype_options(caps: Capabilities) -> list[dict]:
    """dtype select: float16 default; bfloat16 disabled unless supported."""
    bf16_ok = bool(getattr(caps, "supports_bf16", False))
    bf16_opt: dict = {
        "value": "bfloat16",
        "label": "bfloat16",
        "disabled": not bf16_ok,
    }
    if not bf16_ok:
        bf16_opt["help"] = _UNSUPPORTED_HELP
    return [
        {"value": "float16", "label": "float16"},
        bf16_opt,
        {"value": "auto", "label": "Auto"},
    ]


def _kv_dtype_options(caps: Capabilities) -> list[dict]:
    """kv_cache_dtype select: auto always; fp8 disabled unless supported."""
    fp8_ok = bool(getattr(caps, "supports_fp8", False))
    fp8_opt: dict = {"value": "fp8", "label": "fp8", "disabled": not fp8_ok}
    if not fp8_ok:
        fp8_opt["help"] = _UNSUPPORTED_HELP
    return [
        {"value": "auto", "label": "Auto (fp16)"},
        fp8_opt,
    ]


def _context_bounds(meta: ModelMeta) -> tuple[int, int]:
    """Return (default_max_model_len, max_max_model_len) from meta.

    Default is the model's own max (not capped at _DEFAULT_CONTEXT) so the UI
    starts at the model's advertised context length. Still falls back to
    _DEFAULT_CONTEXT when the model config does not advertise one.
    """
    mpe = int(getattr(meta, "max_position_embeddings", 0) or 0)
    if mpe > 0:
        default = min(mpe, 65535)
        maximum = mpe
    else:
        default = 65535
        maximum = _FALLBACK_MAX_CONTEXT
    if default < 1:
        default = _DEFAULT_CONTEXT
    if maximum < default:
        maximum = default
    return default, maximum


def _engine_group(
    meta: ModelMeta, caps: Capabilities, hardware: Optional[HardwareInfo] = None
) -> ParamGroup:
    supported = _supported(caps)
    ctx_default, ctx_max = _context_bounds(meta)
    quant_default = _quant_default(meta, supported)
    tp_max = max(1, int(getattr(hardware, "num_gpus", 8) or 8)) if hardware else 8
    tp_default = tp_max  # use all available GPUs by default

    params: list[ParamSpec] = [
        ParamSpec(
            key="quantization",
            label="Quantization",
            type="select",
            default=quant_default,
            options=_quant_options(caps, supported),
            help="Weight quantization method. 4-bit AWQ/GPTQ recommended on 8GB cards.",
            affects_vram=True,
            engine_only=True,
        ),
        ParamSpec(
            key="dtype",
            label="Compute dtype",
            type="select",
            default="float16",
            options=_dtype_options(caps),
            help="Activation/compute precision. Turing GPUs require float16.",
            affects_vram=True,
            engine_only=True,
        ),
        ParamSpec(
            key="tensor_parallel_size",
            label="Tensor parallel size",
            type="int",
            default=tp_default,
            min=1,
            max=tp_max,
            step=1,
            help=(
                "Number of GPUs to shard weights & KV cache across. "
                f"Auto-set to {tp_default} for this model on {tp_max} GPU(s)."
            ),
            affects_vram=True,
            engine_only=True,
        ),
        ParamSpec(
            key="gpu_memory_utilization",
            label="GPU memory utilization",
            type="float",
            default=0.85,
            min=0.5,
            max=0.97,
            step=0.01,
            help="Fraction of each GPU's VRAM vLLM may use for the KV cache pool.",
            affects_vram=True,
            engine_only=True,
        ),
        ParamSpec(
            key="max_model_len",
            label="Context window",
            type="int",
            default=ctx_default,
            min=256,
            max=ctx_max,
            step=256,
            unit="tokens",
            help="Maximum sequence length (prompt + generation). Drives KV cache size.",
            affects_vram=True,
            engine_only=True,
        ),
        ParamSpec(
            key="max_num_seqs",
            label="Max concurrent sequences",
            type="int",
            default=16,
            min=1,
            max=256,
            step=1,
            help="Maximum number of sequences batched together.",
            affects_vram=True,
            engine_only=True,
        ),
        ParamSpec(
            key="kv_cache_dtype",
            label="KV cache dtype",
            type="select",
            default="auto",
            options=_kv_dtype_options(caps),
            help="KV cache precision. fp8 halves KV memory but needs sm_89+.",
            affects_vram=True,
            engine_only=True,
        ),
        ParamSpec(
            key="enforce_eager",
            label="Enforce eager",
            type="bool",
            default=False,
            help=(
                "Disable torch.compile + CUDA graphs. Enabling this speeds up "
                "startup and saves ~0.6GB/GPU but significantly slows token "
                "generation. Leave off for production use."
            ),
            affects_vram=True,
            engine_only=True,
        ),
        ParamSpec(
            key="trust_remote_code",
            label="Trust remote code",
            type="bool",
            default=False,
            help="Allow executing custom modeling code from the repo.",
            affects_vram=False,
            engine_only=True,
        ),
        ParamSpec(
            key="kv_concurrency",
            label="KV concurrency (estimate)",
            type="int",
            default=1,
            min=1,
            max=64,
            step=1,
            help="Number of full-length sequences to budget KV for (estimate only).",
            affects_vram=True,
            engine_only=True,
        ),
    ]
    return ParamGroup(key="engine", label="Engine (load-time)", params=params)


def _sampling_group(meta: ModelMeta) -> ParamGroup:
    params: list[ParamSpec] = [
        ParamSpec(
            key="temperature",
            label="Temperature",
            type="float",
            default=0.7,
            min=0.0,
            max=2.0,
            step=0.01,
            help="Sampling randomness. 0 = greedy.",
        ),
        ParamSpec(
            key="top_p",
            label="Top-p",
            type="float",
            default=0.95,
            min=0.0,
            max=1.0,
            step=0.01,
            help="Nucleus sampling cumulative probability.",
        ),
        ParamSpec(
            key="top_k",
            label="Top-k",
            type="int",
            default=-1,
            min=-1,
            max=200,
            step=1,
            help="Restrict to top-k tokens. -1 disables.",
        ),
        ParamSpec(
            key="min_p",
            label="Min-p",
            type="float",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.01,
            help="Minimum token probability relative to the top token.",
        ),
        ParamSpec(
            key="max_tokens",
            label="Max tokens",
            type="int",
            default=512,
            min=1,
            max=131072,
            step=1,
            unit="tokens",
            help="Maximum number of tokens to generate.",
        ),
        ParamSpec(
            key="repetition_penalty",
            label="Repetition penalty",
            type="float",
            default=1.0,
            min=0.5,
            max=2.0,
            step=0.01,
            help="Penalize repeated tokens. 1.0 = off.",
        ),
        ParamSpec(
            key="presence_penalty",
            label="Presence penalty",
            type="float",
            default=0.0,
            min=-2.0,
            max=2.0,
            step=0.01,
            help="Penalize tokens already present.",
        ),
        ParamSpec(
            key="frequency_penalty",
            label="Frequency penalty",
            type="float",
            default=0.0,
            min=-2.0,
            max=2.0,
            step=0.01,
            help="Penalize tokens by frequency.",
        ),
        ParamSpec(
            key="seed",
            label="Seed",
            type="int",
            default=None,
            help="Random seed for reproducible sampling (optional).",
        ),
        ParamSpec(
            key="stop",
            label="Stop sequences",
            type="text",
            default="",
            help="Comma-separated stop strings.",
        ),
    ]
    return ParamGroup(key="sampling", label="Sampling", params=params)


def _diffusion_group(meta: ModelMeta) -> ParamGroup:
    help_note = (
        "Diffusion LMs are non-autoregressive; vLLM support is experimental — "
        "params are passed via extra_body."
    )
    params: list[ParamSpec] = [
        ParamSpec(
            key="diffusion_steps",
            label="Diffusion steps",
            type="int",
            default=64,
            min=1,
            max=512,
            step=1,
            help="Number of denoising steps (affects latency).",
        ),
        ParamSpec(
            key="block_length",
            label="Block length",
            type="int",
            default=32,
            min=1,
            max=2048,
            step=1,
            unit="tokens",
            help="Generation block size; bounds KV cache for diffusion.",
            affects_vram=True,
        ),
        ParamSpec(
            key="temperature",
            label="Temperature",
            type="float",
            default=0.2,
            min=0.0,
            max=2.0,
            step=0.01,
            help="Sampling randomness.",
        ),
        ParamSpec(
            key="top_p",
            label="Top-p",
            type="float",
            default=0.95,
            min=0.0,
            max=1.0,
            step=0.01,
            help="Nucleus sampling cumulative probability.",
        ),
        ParamSpec(
            key="alg",
            label="Remasking algorithm",
            type="select",
            default="low_confidence",
            options=[
                {"value": "low_confidence", "label": "Low confidence"},
                {"value": "random", "label": "Random"},
                {"value": "entropy", "label": "Entropy"},
            ],
            help="Strategy for choosing which tokens to remask each step.",
        ),
        ParamSpec(
            key="alg_temp",
            label="Algorithm temperature",
            type="float",
            default=0.0,
            min=0.0,
            max=2.0,
            step=0.01,
            help="Temperature applied to the remasking algorithm.",
        ),
        ParamSpec(
            key="max_tokens",
            label="Max tokens",
            type="int",
            default=256,
            min=1,
            max=131072,
            step=1,
            unit="tokens",
            help="Maximum number of tokens to generate.",
        ),
        ParamSpec(
            key="denoising_schedule",
            label="Denoising schedule",
            type="select",
            default="linear",
            options=[
                {"value": "linear", "label": "Linear"},
                {"value": "cosine", "label": "Cosine"},
            ],
            help="Noise schedule over the denoising steps.",
        ),
    ]
    return ParamGroup(key="diffusion", label="Diffusion (experimental) — " + help_note, params=params)


def _prompt_group(meta: ModelMeta) -> ParamGroup:
    params = [
        ParamSpec(
            key="system_prompt",
            label="System prompt",
            type="text",
            default="You are a helpful assistant.",
            help="System message prepended to the conversation.",
        ),
    ]
    return ParamGroup(key="prompt", label="Prompt", params=params)


def build_schema(
    meta: ModelMeta,
    caps: Capabilities,
    hardware: Optional[HardwareInfo] = None,
) -> ParamSchema:
    """Build a family-aware editable ``ParamSchema`` for ``meta`` on ``caps``.

    Groups: ``engine`` (load-time, affects_vram + engine_only), then either
    ``sampling`` (family == ``llm``/``moe``) or ``diffusion`` (family ==
    ``diffusion``), then ``prompt``. Defaults are seeded from ``meta`` (and
    ``hardware`` for a multi-GPU-aware tensor_parallel_size default) while option
    availability is gated by ``caps``. Best-effort: never raises.
    """
    family = getattr(meta, "family", "llm") or "llm"
    model_type = getattr(meta, "model_type", "") or ""

    groups: list[ParamGroup] = [_engine_group(meta, caps, hardware)]

    if family == "diffusion":
        groups.append(_diffusion_group(meta))
    else:
        groups.append(_sampling_group(meta))

    groups.append(_prompt_group(meta))

    return ParamSchema(family=family, model_type=model_type, groups=groups)
