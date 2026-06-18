"""GPU telemetry via NVML (pynvml).

Public API (imported by hwinfo.py, main.py and others):
- ``list_gpus() -> list[GpuInfo]`` — static per-device info.
- ``get_stats() -> GpuStats`` — live per-device memory/util/temp/power + aggregate.

Everything here is best-effort: NVML or a single device query may fail (driver
mismatch, permissions, no GPU, MIG, etc.). We never raise out of these functions
— on failure we log a warning and return empty lists / zeroed structures so that
routes degrade gracefully rather than 500.

Sizes are bytes (int) per the API contract. ``nvmlInit()`` is performed once and
guarded so repeated calls are cheap.
"""
from __future__ import annotations

import logging
import threading

from .schemas import GpuInfo, GpuLive, GpuStats

logger = logging.getLogger(__name__)

try:  # pynvml is a declared dependency; importing it must not be fatal though.
    import pynvml  # type: ignore
except Exception as exc:  # pragma: no cover - environment without pynvml
    pynvml = None  # type: ignore
    logger.warning("pynvml import failed; GPU telemetry disabled: %s", exc)


# NVML init is process-global; guard it so we only initialize once and tolerate
# repeated concurrent calls from FastAPI request handlers.
_init_lock = threading.Lock()
_init_done = False
_init_ok = False


def _ensure_init() -> bool:
    """Initialize NVML exactly once. Returns True if NVML is usable.

    Best-effort: never raises. Subsequent calls are cheap (no re-init attempt).
    """
    global _init_done, _init_ok
    if pynvml is None:
        return False
    if _init_done:
        return _init_ok
    with _init_lock:
        if _init_done:  # re-check inside the lock
            return _init_ok
        try:
            pynvml.nvmlInit()
            _init_ok = True
        except Exception as exc:  # NVMLError or anything else
            _init_ok = False
            logger.warning("nvmlInit failed; GPU telemetry disabled: %s", exc)
        finally:
            _init_done = True
    return _init_ok


def _decode(value: object) -> str:
    """NVML returns bytes on some bindings, str on others. Normalize to str."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "replace")
        except Exception:
            return str(value)
    return str(value)


def _driver_version() -> str:
    if not _ensure_init():
        return ""
    try:
        return _decode(pynvml.nvmlSystemGetDriverVersion())
    except Exception as exc:  # best-effort
        logger.warning("nvmlSystemGetDriverVersion failed: %s", exc)
        return ""


def _device_count() -> int:
    if not _ensure_init():
        return 0
    try:
        return int(pynvml.nvmlDeviceGetCount())
    except Exception as exc:
        logger.warning("nvmlDeviceGetCount failed: %s", exc)
        return 0


def list_gpus() -> list[GpuInfo]:
    """Static per-GPU info: name, total memory, compute capability, driver.

    Never raises; returns an empty list when NVML is unavailable.
    """
    gpus: list[GpuInfo] = []
    count = _device_count()
    if count <= 0:
        return gpus

    driver = _driver_version()

    for index in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            logger.warning("nvmlDeviceGetHandleByIndex(%d) failed: %s", index, exc)
            continue

        name = ""
        try:
            name = _decode(pynvml.nvmlDeviceGetName(handle))
        except Exception as exc:
            logger.warning("nvmlDeviceGetName(%d) failed: %s", index, exc)

        total_bytes = 0
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_bytes = int(mem.total)
        except Exception as exc:
            logger.warning("nvmlDeviceGetMemoryInfo(%d) failed: %s", index, exc)

        compute_capability = ""
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = f"{int(major)}.{int(minor)}"
        except Exception as exc:
            logger.warning(
                "nvmlDeviceGetCudaComputeCapability(%d) failed: %s", index, exc
            )

        gpus.append(
            GpuInfo(
                index=index,
                name=name,
                total_bytes=total_bytes,
                compute_capability=compute_capability,
                driver_version=driver,
            )
        )

    return gpus


def get_stats() -> GpuStats:
    """Live per-GPU memory/utilization/temperature/power + aggregate totals.

    Cheap enough for ~1s polling. Never raises; on failure returns zeroed
    GpuStats (and per-GPU fields that fail individually fall back to 0).
    """
    live: list[GpuLive] = []
    total_all = 0
    used_all = 0
    free_all = 0

    count = _device_count()
    if count <= 0:
        return GpuStats(gpus=[], total_bytes=0, used_bytes=0, free_bytes=0)

    for index in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            logger.warning("nvmlDeviceGetHandleByIndex(%d) failed: %s", index, exc)
            continue

        name = ""
        try:
            name = _decode(pynvml.nvmlDeviceGetName(handle))
        except Exception as exc:
            logger.warning("nvmlDeviceGetName(%d) failed: %s", index, exc)

        total_bytes = used_bytes = free_bytes = 0
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_bytes = int(mem.total)
            used_bytes = int(mem.used)
            free_bytes = int(mem.free)
        except Exception as exc:
            logger.warning("nvmlDeviceGetMemoryInfo(%d) failed: %s", index, exc)

        utilization = 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization = float(util.gpu)
        except Exception as exc:
            logger.warning("nvmlDeviceGetUtilizationRates(%d) failed: %s", index, exc)

        temperature = 0.0
        try:
            temperature = float(
                pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        except Exception as exc:
            logger.warning("nvmlDeviceGetTemperature(%d) failed: %s", index, exc)

        power_watts = 0.0
        try:
            # nvmlDeviceGetPowerUsage returns milliwatts.
            power_watts = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception as exc:
            logger.warning("nvmlDeviceGetPowerUsage(%d) failed: %s", index, exc)

        live.append(
            GpuLive(
                index=index,
                name=name,
                total_bytes=total_bytes,
                used_bytes=used_bytes,
                free_bytes=free_bytes,
                utilization=utilization,
                temperature=temperature,
                power_watts=power_watts,
            )
        )

        total_all += total_bytes
        used_all += used_bytes
        free_all += free_bytes

    return GpuStats(
        gpus=live,
        total_bytes=total_all,
        used_bytes=used_all,
        free_bytes=free_all,
    )
