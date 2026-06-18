"""Chat proxy: stream / forward chat completions to the managed vllm subprocess.

This module talks to the OpenAI-compatible ``vllm serve`` subprocess over HTTP
(``config.VLLM_BASE_URL``). It never imports torch/vllm; it only speaks HTTP via
aiohttp. Telemetry/network failures degrade gracefully into an error
``ChatChunk`` rather than raising out of a request.

``schemas.py`` (the read-only contract) does not define a chat-delta model, so
``ChatChunk`` (and a convenience ``ChatResponse``) live here and mirror the
frontend contract in ``frontend/lib/types.ts``. Importers should do::

    from app.chat import chat_stream, chat_once, ChatChunk, ChatResponse
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

import aiohttp
from pydantic import BaseModel

from . import config
from .schemas import AppSettings, ChatRequest, EngineStatus

logger = logging.getLogger(__name__)

# Endpoint on the managed vllm subprocess.
_CHAT_URL = config.VLLM_BASE_URL + "/v1/chat/completions"

# How long to wait on the upstream engine before giving up (generous for slow
# first-token latency on long contexts / tensor-parallel setups).
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=600)

# Sampling keys vLLM accepts as top-level OpenAI-style fields. ``stop`` and
# ``seed`` are included; ``None`` values are dropped before sending.
_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "min_p",
    "seed",
    "stop",
)

# Vendor extensions vLLM accepts as top-level fields (not part of the OpenAI
# schema). Everything else from ``extra_body`` is nested under ``extra_body``.
_TOP_LEVEL_VENDOR_KEYS = ("repetition_penalty",)


# --- Delta / response models (mirror frontend/lib/types.ts) -------------------
class ChatChunk(BaseModel):
    """OpenAI-style streamed delta the chat SSE emits."""

    delta: str = ""
    done: bool = False
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    """Aggregated non-streaming chat reply."""

    content: str = ""
    finish_reason: Optional[str] = None


# --- Payload construction -----------------------------------------------------
def _served_model(engine: EngineStatus) -> str:
    return engine.served_model_name or engine.repo or ""


def _build_messages(req: ChatRequest, settings: AppSettings) -> list[dict[str, Any]]:
    sysprompt = req.system_prompt or settings.system_prompt
    messages: list[dict[str, Any]] = []
    if sysprompt:
        messages.append({"role": "system", "content": sysprompt})
    messages.extend(m.model_dump() for m in req.messages)
    return messages


def _build_payload(
    req: ChatRequest,
    settings: AppSettings,
    engine: EngineStatus,
    *,
    stream: bool,
) -> dict[str, Any]:
    """Merge defaults + per-request overrides into a vLLM chat payload."""
    payload: dict[str, Any] = {
        "model": _served_model(engine),
        "messages": _build_messages(req, settings),
        "stream": stream,
    }

    # settings.sampling are the persisted defaults; req.sampling overrides them.
    merged: dict[str, Any] = {}
    merged.update(settings.sampling or {})
    merged.update(req.sampling or {})

    # Known OpenAI-style sampling fields → top-level (drop None / sentinel).
    for key in _SAMPLING_KEYS:
        if key not in merged:
            continue
        value = merged[key]
        if value is None:
            continue
        # top_k == -1 is vLLM's "disabled" sentinel; harmless to send.
        if key == "stop" and isinstance(value, (list, tuple)) and not value:
            continue
        payload[key] = value

    # Known vendor fields vLLM accepts at top level (repetition_penalty, ...).
    extra = dict(req.extra_body or {})
    for key in _TOP_LEVEL_VENDOR_KEYS:
        if key in merged and merged[key] is not None:
            payload[key] = merged[key]
        if key in extra and extra[key] is not None:
            payload[key] = extra.pop(key)

    # Everything else (diffusion / experimental params) goes under extra_body so
    # vLLM forwards it to the sampling params without schema-rejecting the call.
    if extra:
        payload["extra_body"] = extra

    return payload


def _extract_delta(obj: dict[str, Any]) -> tuple[str, Optional[str]]:
    """Pull (content, finish_reason) from a chat.completion.chunk JSON object."""
    content = ""
    finish_reason: Optional[str] = None
    try:
        choices = obj.get("choices") or []
        if choices:
            choice = choices[0] or {}
            finish_reason = choice.get("finish_reason")
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            if not content:
                # Non-streaming bodies put text under "message"; tolerate it.
                message = choice.get("message") or {}
                content = message.get("content") or ""
    except (AttributeError, IndexError, TypeError):
        pass
    return content, finish_reason


# --- Public API ---------------------------------------------------------------
async def chat_stream(
    req: ChatRequest,
    settings: AppSettings,
    engine: EngineStatus,
) -> AsyncIterator[ChatChunk]:
    """Stream chat deltas from the vllm subprocess as ``ChatChunk`` objects.

    Yields incremental ``done=False`` chunks, then a terminal ``done=True``
    chunk. Never raises: on any error a single error chunk is emitted instead.
    """
    if engine.state != "ready":
        yield ChatChunk(delta="[engine not ready]", done=True, finish_reason="error")
        return

    payload = _build_payload(req, settings, engine, stream=True)
    last_finish: Optional[str] = None

    try:
        async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
            async with session.post(_CHAT_URL, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning("vllm chat error %s: %s", resp.status, body[:500])
                    yield ChatChunk(
                        delta=f"[engine error {resp.status}]",
                        done=True,
                        finish_reason="error",
                    )
                    return

                async for raw in resp.content:
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()
                    if not line:
                        continue
                    if line == "[DONE]":
                        yield ChatChunk(delta="", done=True, finish_reason=last_finish)
                        return
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("skipping non-JSON SSE line: %r", line[:200])
                        continue
                    content, finish_reason = _extract_delta(obj)
                    if finish_reason:
                        last_finish = finish_reason
                    if content:
                        yield ChatChunk(delta=content, done=False, finish_reason=None)
    except Exception as exc:  # noqa: BLE001 - never crash a request
        logger.warning("chat_stream failed: %s", exc)
        yield ChatChunk(delta=f"[error: {exc}]", done=True, finish_reason="error")
        return

    # Stream ended without an explicit [DONE] terminator.
    yield ChatChunk(delta="", done=True, finish_reason=last_finish)


async def chat_once(
    req: ChatRequest,
    settings: AppSettings,
    engine: EngineStatus,
) -> str:
    """Non-streaming chat: return the full assistant message as a string.

    Degrades gracefully: returns an ``[...]`` marker string on error rather than
    raising.
    """
    if engine.state != "ready":
        return "[engine not ready]"

    payload = _build_payload(req, settings, engine, stream=False)

    try:
        async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
            async with session.post(_CHAT_URL, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning("vllm chat error %s: %s", resp.status, body[:500])
                    return f"[engine error {resp.status}]"
                data = await resp.json()
    except Exception as exc:  # noqa: BLE001 - never crash a request
        logger.warning("chat_once failed: %s", exc)
        return f"[error: {exc}]"

    parts: list[str] = []
    try:
        for choice in data.get("choices") or []:
            choice = choice or {}
            message = choice.get("message") or {}
            content = message.get("content")
            if content is None:
                # Tolerate streamed-style bodies if the engine returns deltas.
                content = (choice.get("delta") or {}).get("content")
            if content:
                parts.append(content)
    except (AttributeError, TypeError) as exc:
        logger.warning("chat_once parse failed: %s", exc)
        return ""

    return "".join(parts)
