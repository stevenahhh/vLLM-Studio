"""vLLM OpenAI server entrypoint with optional TurboQuant hooks enabled.

This module is launched as ``python -m app.turboquant_entrypoint`` by the
process runner. It enables TurboQuant before vLLM imports and initializes the
engine, then delegates to vLLM's standard OpenAI API server module.
"""
from __future__ import annotations

import os
import runpy
from importlib import import_module
from typing import Protocol, cast


class _TurboQuantBackend(Protocol):
    def enable_no_alloc(
        self,
        *,
        key_bits: int = 3,
        value_bits: int = 2,
        buffer_size: int = 128,
        initial_layers_count: int = 4,
    ) -> None: ...


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def main() -> None:
    module = cast(object, import_module("turboquant.vllm_attn_backend"))
    backend = cast(_TurboQuantBackend, module)
    enable_no_alloc = backend.enable_no_alloc

    enable_no_alloc(
        key_bits=_env_int("VLLM_TURBOQUANT_KEY_BITS", 3),
        value_bits=_env_int("VLLM_TURBOQUANT_VALUE_BITS", 2),
        buffer_size=_env_int("VLLM_TURBOQUANT_BUFFER_SIZE", 128),
        initial_layers_count=_env_int("VLLM_TURBOQUANT_INITIAL_LAYERS", 4),
    )
    _ = runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
