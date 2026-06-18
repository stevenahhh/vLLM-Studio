"""FastAPI control plane for vLLM Studio.

This is the HTTP surface (CONTRACT.md §5). It wires together the peer modules
(gpu / hwinfo / modelmeta / estimator / params / registry / downloader /
vllm_manager / chat / state) and exposes them on :8000 with CORS open so the
Next.js frontend can talk to it directly.

Design notes:
- Never import torch / vllm at module top. We only touch pynvml (via gpu/hwinfo)
  and huggingface_hub (via registry/modelmeta); the engine itself is a
  subprocess owned by ``vllm_manager``.
- Telemetry / network failures must not crash a request — peer modules already
  degrade gracefully; here we additionally wrap blocking/fallible calls and
  translate unexpected failures into HTTPException.
- Blocking HF calls (search / variants / meta) run in handlers declared with
  plain ``def`` so FastAPI executes them in a worker threadpool, keeping the
  event loop free. Streaming endpoints are ``async`` and use StreamingResponse
  with media_type "text/event-stream".
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import chat, config, estimator, gpu, hwinfo, modelmeta, params, registry, state
from .downloader import manager as downloads
from .schemas import (
    AppSettings,
    ChatRequest,
    DownloadJob,
    DownloadRequest,
    EngineStatus,
    EstimateRequest,
    GpuStats,
    HardwareInfo,
    LoadRequest,
    ModelMeta,
    ParamSchema,
    VramEstimate,
)
from .vllm_manager import manager as engine
from .tuner import TunerConfig, get_tuner

logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM Studio", version="1.0.0")

# CORS: the frontend (and dev tools) talk to us cross-origin; open everything.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# How often the GPU SSE stream pushes a fresh sample (seconds).
_GPU_STREAM_INTERVAL = 1.0


def _sse(data: str) -> str:
    """Format a single Server-Sent-Events frame."""
    return f"data: {data}\n\n"


# --- Root banner ---------------------------------------------------------------
@app.get("/")
def root() -> dict:
    """Tiny JSON banner so hitting the root isn't a 404."""
    return {
        "name": "vLLM Studio",
        "service": "control-plane",
        "version": app.version,
        "docs": "/docs",
        "api": "/api",
    }


# --- Health --------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    """Liveness + a snapshot of the managed engine."""
    try:
        status = engine.status()
    except Exception as exc:  # never fail health on engine introspection
        logger.warning("engine status failed in /health: %s", exc)
        status = EngineStatus()
    return {"status": "ok", "vllm": status}


# --- Hardware / GPU ------------------------------------------------------------
@app.get("/api/hardware", response_model=HardwareInfo)
def get_hardware() -> HardwareInfo:
    try:
        return hwinfo.get_hardware()
    except Exception as exc:
        logger.warning("get_hardware failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"hardware probe failed: {exc}")


@app.get("/api/gpu/stats", response_model=GpuStats)
def gpu_stats() -> GpuStats:
    # gpu.get_stats() is best-effort and never raises, but guard anyway.
    try:
        return gpu.get_stats()
    except Exception as exc:
        logger.warning("gpu.get_stats failed: %s", exc)
        return GpuStats()


@app.get("/api/gpu/stream")
async def gpu_stream() -> StreamingResponse:
    """Server-Sent-Events of GpuStats, roughly once per second."""

    async def _gen() -> AsyncIterator[str]:
        while True:
            try:
                stats = await asyncio.to_thread(gpu.get_stats)
            except Exception as exc:  # degrade to an empty sample, keep streaming
                logger.debug("gpu stream sample failed: %s", exc)
                stats = GpuStats()
            yield _sse(json.dumps(stats.model_dump()))
            await asyncio.sleep(_GPU_STREAM_INTERVAL)

    return StreamingResponse(_gen(), media_type="text/event-stream")


# --- Models: registry ----------------------------------------------------------
@app.get("/api/models/downloaded")
def models_downloaded() -> dict:
    try:
        items = registry.list_downloaded()
    except Exception as exc:
        logger.warning("list_downloaded failed: %s", exc)
        items = []
    return {"items": items}


@app.get("/api/models/search")
def models_search(
    q: str = Query("", description="search query"),
    limit: int = Query(30, ge=1, le=100),
) -> dict:
    # Blocking HF Hub call — plain ``def`` handler runs in the threadpool.
    try:
        items = registry.search_hf(q, limit=limit)
    except Exception as exc:
        logger.warning("search_hf failed: %s", exc)
        items = []
    return {"items": items}


@app.delete("/api/models/downloaded")
def models_delete(repo: str = Query(..., description="HF repo id")) -> dict:
    try:
        ok = registry.delete_downloaded(repo)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"delete failed: {exc}")
    if not ok:
        raise HTTPException(status_code=404, detail="model not found in cache")
    return {"ok": True, "repo": repo}


@app.get("/api/models/variants")
def models_variants(repo: str = Query(..., description="HF repo id")) -> dict:
    try:
        items = registry.list_variants(repo)
    except Exception as exc:
        logger.warning("list_variants failed for %s: %s", repo, exc)
        items = []
    return {"items": items}


@app.get("/api/models/meta", response_model=ModelMeta)
def models_meta(
    repo: str = Query(..., description="HF repo id"),
    quant: str = Query("none"),
    revision: str = Query("main"),
    local: bool = Query(False),
) -> ModelMeta:
    # Blocking (config fetch / param-count derivation) — runs in threadpool.
    try:
        return modelmeta.get_meta(repo, quant=quant, revision=revision, local=local)
    except Exception as exc:
        logger.warning("get_meta failed for %s: %s", repo, exc)
        raise HTTPException(status_code=502, detail=f"failed to load model meta: {exc}")


# --- Estimation ----------------------------------------------------------------
@app.post("/api/estimate", response_model=VramEstimate)
def estimate(req: EstimateRequest) -> VramEstimate:
    try:
        hw = hwinfo.get_hardware()
    except Exception as exc:
        logger.warning("hardware probe failed during estimate: %s", exc)
        raise HTTPException(status_code=500, detail=f"hardware probe failed: {exc}")
    try:
        return estimator.estimate(req.meta, req, hw)
    except Exception as exc:
        logger.warning("estimate failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"estimate failed: {exc}")


# --- Downloads -----------------------------------------------------------------
@app.get("/api/downloads")
def downloads_list() -> dict:
    try:
        items = downloads.list_jobs()
    except Exception as exc:
        logger.warning("list_jobs failed: %s", exc)
        items = []
    return {"items": items}


@app.post("/api/downloads", response_model=DownloadJob)
def downloads_start(req: DownloadRequest) -> DownloadJob:
    try:
        return downloads.start(req)
    except Exception as exc:
        logger.warning("download start failed for %s: %s", req.repo, exc)
        raise HTTPException(status_code=400, detail=f"download failed to start: {exc}")


@app.delete("/api/downloads/{job_id}")
def downloads_cancel(job_id: str) -> dict:
    try:
        ok = downloads.cancel(job_id)
    except Exception as exc:
        logger.warning("download cancel failed for %s: %s", job_id, exc)
        raise HTTPException(status_code=400, detail=f"cancel failed: {exc}")
    if not ok:
        raise HTTPException(status_code=404, detail="download job not found or not cancellable")
    return {"ok": True}


@app.delete("/api/downloads/{job_id}/remove")
def downloads_remove(job_id: str) -> dict:
    try:
        ok = downloads.remove(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"remove failed: {exc}")
    if not ok:
        raise HTTPException(status_code=404, detail="job not found or still active")
    return {"ok": True}


# --- Engine --------------------------------------------------------------------
@app.get("/api/engine", response_model=EngineStatus)
def engine_status() -> EngineStatus:
    try:
        return engine.status()
    except Exception as exc:
        logger.warning("engine status failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"engine status failed: {exc}")


@app.post("/api/engine/load", response_model=EngineStatus)
def engine_load(req: LoadRequest) -> EngineStatus:
    # load() is non-blocking (spawns + returns state="loading"); return status.
    try:
        engine.load(req)
        return engine.status()
    except Exception as exc:
        logger.warning("engine load failed for %s: %s", req.repo, exc)
        raise HTTPException(status_code=500, detail=f"engine load failed: {exc}")


@app.post("/api/engine/unload", response_model=EngineStatus)
def engine_unload() -> EngineStatus:
    try:
        return engine.unload()
    except Exception as exc:
        logger.warning("engine unload failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"engine unload failed: {exc}")


@app.get("/api/engine/logs")
async def engine_logs() -> StreamingResponse:
    """Server-Sent-Events of engine log lines (tail first, then live)."""

    async def _gen() -> AsyncIterator[str]:
        try:
            async for line in engine.log_stream():
                yield _sse(json.dumps(line))
        except Exception as exc:  # never crash the stream on a reader hiccup
            logger.debug("engine log stream stopped: %s", exc)
        yield _sse("[DONE]")

    return StreamingResponse(_gen(), media_type="text/event-stream")


# --- Params --------------------------------------------------------------------
@app.get("/api/params/schema", response_model=ParamSchema)
def params_schema(
    repo: str = Query("", description="HF repo id"),
    quant: str = Query("none"),
) -> ParamSchema:
    try:
        meta = modelmeta.get_meta(repo, quant=quant) if repo else ModelMeta(repo="", quant=quant)
    except Exception as exc:
        # Degrade to an empty-config meta so the UI still gets a usable schema.
        logger.warning("get_meta failed for params schema (%s): %s", repo, exc)
        meta = ModelMeta(repo=repo, quant=quant)
    try:
        caps = hwinfo.get_capabilities()
    except Exception as exc:
        logger.warning("get_capabilities failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"capability probe failed: {exc}")
    try:
        hw = hwinfo.get_hardware()
    except Exception as exc:
        logger.warning("get_hardware failed for params schema: %s", exc)
        hw = None
    try:
        return params.build_schema(meta, caps, hw)
    except Exception as exc:
        logger.warning("build_schema failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"param schema build failed: {exc}")


# --- Settings ------------------------------------------------------------------
@app.get("/api/settings", response_model=AppSettings)
def settings_get() -> AppSettings:
    try:
        return state.load_settings()
    except Exception as exc:
        logger.warning("load_settings failed: %s", exc)
        return AppSettings()


@app.put("/api/settings", response_model=AppSettings)
def settings_put(settings: AppSettings) -> AppSettings:
    try:
        return state.save_settings(settings)
    except Exception as exc:
        logger.warning("save_settings failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"failed to save settings: {exc}")


# --- Tuner --------------------------------------------------------------------

@app.post("/api/tuner/start")
async def tuner_start(req: TunerConfig = Body(...)):
    t = get_tuner(engine)
    try:
        return t.start(req).model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/tuner/stop")
async def tuner_stop():
    return get_tuner(engine).stop().model_dump()


@app.get("/api/tuner/status")
async def tuner_status():
    return get_tuner(engine).status().model_dump()


# --- Chat ----------------------------------------------------------------------
@app.post("/api/chat/completions")
async def chat_completions(req: ChatRequest = Body(...)):
    """Chat completions. Streams ChatChunk SSE when ``stream`` is true, else a
    single aggregated ChatResponse JSON object."""
    try:
        settings = state.load_settings()
    except Exception as exc:
        logger.warning("load_settings failed in chat: %s", exc)
        settings = AppSettings()
    status = engine.status()

    if req.stream:

        async def _gen() -> AsyncIterator[str]:
            try:
                async for chunk in chat.chat_stream(req, settings, status):
                    yield _sse(json.dumps(chunk.model_dump()))
            except Exception as exc:  # never crash mid-stream
                logger.warning("chat stream failed: %s", exc)
                err = {"delta": f"[error: {exc}]", "done": True, "finish_reason": "error"}
                yield _sse(json.dumps(err))
            yield _sse("[DONE]")

        return StreamingResponse(_gen(), media_type="text/event-stream")

    # Non-streaming: aggregate into a single ChatResponse-shaped body.
    content = await chat.chat_once(req, settings, status)
    return chat.ChatResponse(content=content)
