"""Static hardware capability detection for vLLM Studio.

Derives dtype/quant support from the GPUs' CUDA compute capability (sm_XY).
On this host (8x RTX 2080, Turing sm_75) that means: no bfloat16, no FP8, no
Marlin — force float16; AWQ/GPTQ 4-bit are fine.

Never raises on telemetry failure: degrades to a conservative Turing-class
fallback so /api/hardware always answers.

Public API (importers rely on these names):
- get_capabilities() -> Capabilities
- get_hardware() -> HardwareInfo
- effective_dtype(requested: str) -> str
- kv_dtype_bytes(kv_cache_dtype: str) -> int
- supports_quant(quant: str) -> bool
"""
from __future__ import annotations

from .config import GIB
from .schemas import Capabilities, GpuInfo, HardwareInfo

# Conservative fallback used when nvml / gpu telemetry is unavailable.
# Matches the documented hardware: RTX 2080, Turing sm_75, 8 GiB each.
_FALLBACK_CC = "7.5"
_FALLBACK_GPU_BYTES = 8 * GIB
_FALLBACK_NUM_GPUS = 1


def _cc_to_float(compute_capability: str) -> float:
    """Parse a compute-capability string like "7.5" -> 7.5.

    Tolerant of odd formats; returns the conservative fallback on failure.
    """
    try:
        s = (compute_capability or "").strip()
        if not s:
            return float(_FALLBACK_CC)
        if "." in s:
            return float(s)
        # e.g. "75" -> 7.5 (major+minor packed) or "8" -> 8.0
        if len(s) >= 2 and s.isdigit():
            return int(s[0]) + int(s[1:]) / 10.0
        return float(s)
    except (ValueError, TypeError, IndexError):
        return float(_FALLBACK_CC)


def _list_gpus() -> list[GpuInfo]:
    """Best-effort fetch of GpuInfo list; never raises."""
    try:
        from . import gpu as _gpu  # local import: gpu may be unavailable

        gpus = _gpu.list_gpus()
        if gpus:
            return list(gpus)
    except Exception:
        pass
    return []


def _min_cc(gpus: list[GpuInfo]) -> float:
    """Lowest compute capability across GPUs (the limiting factor)."""
    if not gpus:
        return float(_FALLBACK_CC)
    ccs = [_cc_to_float(g.compute_capability) for g in gpus]
    return min(ccs) if ccs else float(_FALLBACK_CC)


def get_capabilities() -> Capabilities:
    """Derive dtype/quant capabilities from the lowest-cc GPU present."""
    gpus = _list_gpus()
    cc = _min_cc(gpus)

    supports_bf16 = cc >= 8.0
    supports_fp8 = cc >= 8.9
    supports_marlin = cc >= 8.0

    supported_quantization: list[str] = ["none", "awq", "gptq", "bitsandbytes"]
    if supports_fp8:
        supported_quantization.append("fp8")
    if supports_marlin:
        supported_quantization.extend(["awq_marlin", "gptq_marlin"])

    supported_dtypes: list[str] = ["float16"]
    if supports_bf16:
        supported_dtypes.append("bfloat16")

    notes: list[str] = []
    cc_label = f"sm_{int(round(cc * 10))}" if gpus else "unknown"
    if gpus:
        notes.append(
            f"Detected compute capability {cc:.1f} ({cc_label}); "
            f"limited by the lowest-capability GPU."
        )
    else:
        notes.append(
            "GPU telemetry unavailable; assuming Turing (sm_75) fallback."
        )
    if not supports_bf16:
        notes.append(
            "bfloat16 not accelerated below sm_80 (Turing): forcing float16."
        )
    if not supports_fp8:
        notes.append(
            "FP8 weights and FP8 KV cache require sm_89+: KV cache stays fp16 (2 bytes)."
        )
    if not supports_marlin:
        notes.append(
            "Marlin kernels require sm_80+: awq_marlin/gptq_marlin disabled; "
            "use plain AWQ/GPTQ 4-bit."
        )
    if cc < 8.0:
        notes.append(
            "AWQ and GPTQ 4-bit work on Turing and are recommended for 8 GB cards; "
            "bitsandbytes works but is slow."
        )

    return Capabilities(
        supports_bf16=supports_bf16,
        supports_fp8=supports_fp8,
        supports_marlin=supports_marlin,
        supported_quantization=supported_quantization,
        supported_dtypes=supported_dtypes,
        recommended_dtype="float16",
        notes=notes,
    )


def get_hardware() -> HardwareInfo:
    """Assemble the full HardwareInfo: gpus, totals, capabilities, advice."""
    gpus = _list_gpus()
    caps = get_capabilities()

    if gpus:
        num_gpus = len(gpus)
        totals = [g.total_bytes for g in gpus if g.total_bytes > 0]
        per_gpu_bytes = min(totals) if totals else 0
        total_bytes = sum(g.total_bytes for g in gpus)
    else:
        num_gpus = _FALLBACK_NUM_GPUS
        per_gpu_bytes = _FALLBACK_GPU_BYTES
        total_bytes = _FALLBACK_GPU_BYTES * _FALLBACK_NUM_GPUS

    per_gpu_gib = per_gpu_bytes / GIB if per_gpu_bytes else 0.0

    recommendations: list[str] = []
    if per_gpu_gib and per_gpu_gib <= 12:
        recommendations.append(
            f"Use AWQ/GPTQ 4-bit on {per_gpu_gib:.0f}GB cards to fit larger models."
        )
    else:
        recommendations.append("Use AWQ/GPTQ 4-bit quantization to fit larger models.")
    if num_gpus > 1:
        recommendations.append(
            f"For >7B models set tensor_parallel_size>=2 (up to {num_gpus} GPUs available)."
        )
    else:
        recommendations.append(
            "For >7B models set tensor_parallel_size>=2 (more GPUs required)."
        )
    if not caps.supports_bf16:
        recommendations.append("Always run with --dtype float16 on this hardware.")
    else:
        recommendations.append("float16 is the recommended dtype.")

    return HardwareInfo(
        gpus=gpus,
        num_gpus=num_gpus,
        per_gpu_bytes=per_gpu_bytes,
        total_bytes=total_bytes,
        capabilities=caps,
        recommendations=recommendations,
    )


def effective_dtype(requested: str) -> str:
    """Resolve a requested dtype to one the hardware can actually run.

    "auto" and "bfloat16" collapse to "float16" unless bf16 is supported.
    Anything else is passed through (lower-cased / trimmed).
    """
    req = (requested or "").strip().lower()
    caps = get_capabilities()
    if req in ("", "auto", "bfloat16"):
        if caps.supports_bf16 and req != "":
            # bf16 explicitly requested and supported -> honor it;
            # "auto" / "" still resolve to float16 (our recommended default).
            return "bfloat16" if req == "bfloat16" else "float16"
        return "float16"
    return req


def kv_dtype_bytes(kv_cache_dtype: str) -> int:
    """Bytes per KV element. fp8 variants -> 1 only if hardware supports fp8.

    On Turing this always returns 2 (fp16 KV cache).
    """
    dt = (kv_cache_dtype or "").strip().lower()
    if dt in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        return 1 if get_capabilities().supports_fp8 else 2
    return 2


def supports_quant(quant: str) -> bool:
    """True if the quant method is usable on this hardware.

    "auto"/"none"/"" are always allowed; otherwise membership in the
    hardware's supported_quantization list.
    """
    q = (quant or "").strip().lower()
    if q in ("auto", "none", ""):
        return True
    return q in get_capabilities().supported_quantization
