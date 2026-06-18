"""Manage the vLLM OpenAI-compatible server as a subprocess.

`VLLMManager` spawns `python -m vllm.entrypoints.openai.api_server` for a
requested model, tails its merged stdout/stderr into a ring buffer, watches the
log for readiness / OOM, and (best-effort) polls the engine `/health` endpoint
to flip state to "ready". `load()` is non-blocking: it returns immediately with
state="loading" and the rest happens on background threads.

Public API (imported by main.py):
    class VLLMManager:
        status() -> EngineStatus
        load(req: LoadRequest) -> EngineStatus      # non-blocking, state="loading"
        unload() -> EngineStatus
        log_lines() -> list[str]
        async log_stream()                          # async generator of new lines
    manager = VLLMManager()   # module singleton

Never imports torch/vllm at module top — only spawns a subprocess and talks
HTTP. Telemetry / network failures degrade gracefully (logged as warnings).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Optional, cast

from . import config
from .schemas import EngineState, EngineStatus, Family, LoadRequest

logger = logging.getLogger(__name__)

# Model types that need TRITON_ATTN because flashinfer doesn't support head_dim=256.
_TRITON_ATTN_MODEL_TYPES = {"qwen3_5", "qwen3-5", "qwen3.5"}

# Substrings (lower-cased) that indicate a fatal error in the engine log.
_ERROR_MARKERS = (
    "cuda out of memory",
    "out of memory",
    "no available memory",
    "torch.cuda.outofmemoryerror",
    "engine core initialization failed",
    "raise enginedeaderror",
    "valueerror: no available memory",
    "the engine core process failed to start",
)
# Substrings that indicate the server finished starting up successfully.
_READY_MARKERS = (
    "application startup complete",
    "uvicorn running",
)

# --- Load-progress parsing ----------------------------------------------------
# vLLM prints recognisable phase markers on stdout. We map them to a 0..1 progress
# and a human phase so the UI can show a live percentage. Progress is monotonic.
_SHARD_RE = re.compile(r"checkpoint shards:\s*(\d+)%")
_DOWNLOAD_RE = re.compile(r"\.(?:safetensors|bin|gguf|pt)[^:]*:\s*(\d+)%")


def _parse_progress(low: str) -> Optional[tuple[float, str]]:
    """Map a (lower-cased) log line to (progress 0..1, phase) or None.

    Pure + testable without spawning anything. Ranges:
      download 0.05–0.12 · weight shards 0.15–0.70 · profiling 0.78 ·
      init 0.74 · capturing CUDA graphs 0.88. (Ready/error handled by caller.)
    """
    m = _SHARD_RE.search(low)
    if m:
        pct = min(100, int(m.group(1)))
        return 0.15 + 0.55 * (pct / 100.0), f"Loading weights ({pct}%)"
    m = _DOWNLOAD_RE.search(low)
    if m:
        pct = min(100, int(m.group(1)))
        return 0.05 + 0.07 * (pct / 100.0), f"Downloading weights ({pct}%)"
    if "capturing" in low and ("cuda graph" in low or "graph" in low):
        return 0.88, "Capturing CUDA graphs"
    if (
        "memory profiling" in low
        or "gpu blocks" in low
        or "available kv cache" in low
        or ("determine" in low and "blocks" in low)
        or ("kv cache" in low and "memory" in low)
    ):
        return 0.78, "Profiling KV cache"
    if "model loading took" in low or "weights loaded" in low:
        return 0.72, "Weights loaded"
    if "init engine" in low or ("initializing" in low and "engine" in low):
        return 0.74, "Initializing engine"
    return None


def _resolve_model_path(repo: str, revision: str = "main") -> str:
    """Return a local snapshot dir if the repo is in the HF cache, else the repo id.

    Best-effort: any failure falls back to returning ``repo`` unchanged so vLLM
    can resolve / download it itself.
    """
    try:
        if not repo:
            return repo
        # A path that already exists on disk (e.g. a manual snapshot) wins.
        if os.path.sep in repo and Path(repo).exists():
            return repo

        cache_root = Path(config.HF_HUB_CACHE)
        folder = "models--" + repo.replace("/", "--")
        model_dir = cache_root / folder
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            return repo

        # Prefer the snapshot pointed to by the requested revision ref.
        ref_file = model_dir / "refs" / (revision or "main")
        if ref_file.is_file():
            try:
                commit = ref_file.read_text().strip()
                cand = snapshots / commit
                if cand.is_dir():
                    return str(cand)
            except OSError:
                pass

        # Otherwise pick the most recently modified snapshot that has a config.
        candidates = [p for p in snapshots.iterdir() if p.is_dir()]
        if not candidates:
            return repo
        with_config = [p for p in candidates if (p / "config.json").exists()]
        pool = with_config or candidates
        pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(pool[0])
    except Exception as exc:  # never fail the load on path resolution
        logger.warning("model path resolution failed for %s: %s", repo, exc)
        return repo


def _detect_family(repo: str, revision: str, quant: str) -> str:
    """Best-effort family detection from a local config.json. Defaults to 'llm'."""
    try:
        from . import modelmeta  # local import: keep module-top import graph light
    except Exception:
        return "llm"

    # Try the cheap config-only path first via a resolved local snapshot.
    try:
        path = _resolve_model_path(repo, revision)
        cfg_path = Path(path) / "config.json"
        if cfg_path.is_file():
            with cfg_path.open("r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            return str(modelmeta.detect_family(cfg, repo))
    except Exception as exc:
        logger.debug("family detection via config failed for %s: %s", repo, exc)

    # Fall back to full meta lookup (may hit the network); stay best-effort.
    try:
        meta = modelmeta.get_meta(repo, quant=quant, revision=revision)
        return str(getattr(meta, "family", "llm") or "llm")
    except Exception as exc:
        logger.debug("family detection via get_meta failed for %s: %s", repo, exc)
        return "llm"


def _effective_dtype(requested: str) -> str:
    """Wrap hwinfo.effective_dtype with a safe fallback."""
    try:
        from . import hwinfo
        return str(hwinfo.effective_dtype(requested))
    except Exception as exc:
        logger.debug("effective_dtype fallback (%s): %s", requested, exc)
        # On Turing we never want bf16/auto; force a safe default.
        if requested in ("", "auto", "bfloat16", "bf16"):
            return config.DEFAULT_DTYPE
        return requested


class VLLMManager:
    """Spawn / health-check / kill the managed vllm serve subprocess."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen[str]] = None
        self._logs: deque[str] = deque(maxlen=config.LOG_TAIL_LINES)
        self._reader: Optional[threading.Thread] = None
        self._poller: Optional[threading.Thread] = None
        # Generation counter: bumped on every load/unload so stale background
        # threads from a previous run know to exit and never touch new state.
        self._gen: int = 0

        self._state: str = "stopped"
        self._error: str = ""
        self._repo: str = ""
        self._revision: str = "main"
        self._quant: str = "none"
        self._family: str = "llm"
        self._served: str = ""
        self._pid: Optional[int] = None
        self._load_request: Optional[LoadRequest] = None
        self._progress: float = 0.0
        self._phase: str = ""
        self._retried: bool = False

    # -- public API -----------------------------------------------------------

    def status(self) -> EngineStatus:
        with self._lock:
            # Reap a process that died without a recognised error marker.
            if self._proc is not None and self._proc.poll() is not None:
                if self._state in ("loading", "ready"):
                    self._state = "error"
                    rc = self._proc.returncode
                    if not self._error:
                        self._error = f"vllm process exited (code {rc})"
                self._pid = None
            return EngineStatus(
                state=cast(EngineState, self._state),
                repo=self._repo,
                revision=self._revision,
                quant=self._quant,
                family=cast(Family, self._family),
                port=config.VLLM_PORT if self._state != "stopped" else 0,
                pid=self._pid,
                load_request=self._load_request,
                error=self._error,
                logs_tail=list(self._logs),
                served_model_name=self._served,
                progress=1.0 if self._state == "ready" else round(self._progress, 3),
                phase=self._phase,
            )

    def _adjust_req_for_retry(self, req: LoadRequest, error_log: str) -> Optional[LoadRequest]:
        """Inspect error log and return a modified LoadRequest for one retry, or None."""
        low = error_log.lower()
        extra = list(req.extra_args or [])

        # OOM during weight loading → try doubling TP (up to num_gpus)
        if "out of memory" in low or "outofmemoryerror" in low:
            try:
                from . import hwinfo
                num_gpus = hwinfo.get_hardware().num_gpus
            except Exception:
                num_gpus = 8
            current_tp = req.tensor_parallel_size or 1
            new_tp = min(current_tp * 2, num_gpus)
            if new_tp > current_tp:
                self._append_log(
                    f"[vllm-studio] auto-retry: OOM detected, increasing TP {current_tp}→{new_tp}"
                )
                return req.model_copy(update={"tensor_parallel_size": new_tp})

        # flashinfer invalid argument (head_dim mismatch) → switch to TRITON_ATTN
        if ("invalid argument" in low or "batchprefillwithpagedkvcache" in low) and \
                "--attention-backend" not in extra:
            self._append_log(
                "[vllm-studio] auto-retry: flashinfer error detected, switching to TRITON_ATTN"
            )
            return req.model_copy(update={"extra_args": extra + ["--attention-backend", "TRITON_ATTN"]})

        return None

    def load(self, req: LoadRequest, _retry: bool = False) -> EngineStatus:  # noqa: C901
        # Tear down any running engine first (releases VRAM).
        self.unload()

        served = (req.repo.split("/")[-1] or req.repo) if req.repo else "model"
        model = _resolve_model_path(req.repo, req.revision)
        family = _detect_family(req.repo, req.revision, req.quant)
        try:
            argv = self._build_argv(req, model, served)
        except ValueError as exc:
            with self._lock:
                self._state = "error"
                self._error = str(exc)
                self._phase = "Error"
                self._repo = req.repo
                self._revision = req.revision
                self._quant = req.quant
                self._family = family
                self._served = served
                self._load_request = req
                self._append_log("[vllm-studio] ERROR " + str(exc))
            return self.status()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Keep HF cache locations explicit so the child sees the same models.
        env.setdefault("HF_HOME", config.HF_HOME)
        env["HF_HUB_CACHE"] = config.HF_HUB_CACHE
        if config.HF_TOKEN:
            env["HF_TOKEN"] = config.HF_TOKEN

        with self._lock:
            self._gen += 1
            gen = self._gen
            self._logs.clear()
            self._state = "loading"
            self._error = ""
            self._progress = 0.03
            self._phase = "Starting engine"
            if not _retry:
                self._retried = False
            self._repo = req.repo
            self._revision = req.revision
            self._quant = req.quant
            self._family = family
            self._served = served
            self._load_request = req

            self._append_log("[vllm-studio] launching: " + " ".join(self._argv_for_log(argv)))
            if config.VLLM_ENGINE_RUNNER == "docker":
                self._remove_container()
            try:
                proc = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                    start_new_session=True,
                )
            except Exception as exc:  # spawn failed outright
                self._state = "error"
                self._error = f"failed to spawn vllm: {exc}"
                self._pid = None
                self._proc = None
                self._append_log("[vllm-studio] ERROR " + self._error)
                return self.status()

            self._proc = proc
            self._pid = proc.pid

            self._reader = threading.Thread(
                target=self._read_loop, args=(proc, gen), daemon=True
            )
            self._reader.start()
            self._poller = threading.Thread(
                target=self._health_loop, args=(gen,), daemon=True
            )
            self._poller.start()

        return self.status()

    def unload(self) -> EngineStatus:
        with self._lock:
            proc = self._proc
            self._gen += 1  # invalidate background threads from the old run
            self._proc = None

        if proc is not None and proc.poll() is None:
            self._terminate(proc)

        with self._lock:
            self._state = "stopped"
            self._error = ""
            self._progress = 0.0
            self._phase = ""
            self._pid = None
            self._load_request = None
            self._repo = ""
            self._revision = "main"
            self._quant = "none"
            self._family = "llm"
            self._served = ""
        return self.status()

    def log_lines(self) -> list[str]:
        with self._lock:
            return list(self._logs)

    async def log_stream(self) -> AsyncIterator[str]:
        """Yield log lines: the current tail first, then new lines as they arrive."""
        with self._lock:
            backlog = list(self._logs)
            idx = len(self._logs)
        for line in backlog:
            yield line

        # Poll the deque for growth. The deque is bounded, so we track the
        # number of lines seen and re-snapshot; if it rotated past us, resync.
        seen = idx
        idle = 0.0
        while True:
            with self._lock:
                running = self._proc is not None and self._proc.poll() is None
                current = list(self._logs)
            total = len(current)
            if total > seen:
                # If the buffer rotated, we may have lost lines; clamp.
                new_count = min(total - seen, len(current))
                for line in current[len(current) - new_count:]:
                    yield line
                seen = total
                idle = 0.0
            else:
                idle += 0.25
            if not running and total <= seen:
                # Process is gone and no new lines pending; stop after a grace.
                if idle >= 1.0:
                    return
            await asyncio.sleep(0.25)

    # -- internals ------------------------------------------------------------

    def _model_type(self, req: LoadRequest) -> str:
        """Best-effort: read model_type from the local config.json."""
        try:
            path = _resolve_model_path(req.repo, req.revision)
            cfg_path = Path(path) / "config.json"
            if cfg_path.is_file():
                import json as _json
                cfg = _json.loads(cfg_path.read_text())
                return str(cfg.get("model_type", "")).lower()
        except Exception:
            pass
        return ""

    def _build_argv(self, req: LoadRequest, model: str, served: str) -> list[str]:
        runner = config.VLLM_ENGINE_RUNNER
        if runner == "process":
            return self._build_process_argv(req, model, served)
        if runner == "docker":
            return self._build_docker_argv(req, model, served)
        raise ValueError(f"unsupported VLLM_ENGINE_RUNNER: {runner}")

    def _build_process_argv(self, req: LoadRequest, model: str, served: str) -> list[str]:
        module = "app.turboquant_entrypoint" if config.VLLM_TURBOQUANT else "vllm.entrypoints.openai.api_server"
        return [
            sys.executable,
            "-m",
            module,
            *self._build_server_args(req, model, served, host=config.VLLM_HOST),
        ]

    def _build_docker_argv(self, req: LoadRequest, model: str, served: str) -> list[str]:
        volumes = self._docker_volumes(model)
        argv: list[str] = [
            "docker",
            "run",
            "--rm",
            "--name",
            config.VLLM_CONTAINER_NAME,
            "--runtime",
            "nvidia",
            "--gpus",
            "all",
            "--ipc=host",
            "-p",
            f"{config.VLLM_PORT}:{config.VLLM_PORT}",
            "-e",
            f"HF_HOME={config.HF_HOME}",
            "-e",
            f"HF_HUB_CACHE={config.HF_HUB_CACHE}",
        ]
        for volume in volumes:
            argv += ["-v", volume]
        if config.HF_TOKEN:
            argv += ["-e", "HF_TOKEN"]
        if os.environ.get("VLLM_ATTENTION_BACKEND"):
            argv += ["-e", f"VLLM_ATTENTION_BACKEND={os.environ['VLLM_ATTENTION_BACKEND']}"]
        argv.append(config.VLLM_DOCKER_IMAGE)
        argv += self._build_server_args(req, model, served, host="0.0.0.0")
        return argv

    def _docker_volumes(self, model: str) -> list[str]:
        mounts: list[tuple[Path, bool]] = [(Path(config.HF_HOME), False)]
        hub = Path(config.HF_HUB_CACHE)
        if not self._path_within(hub, Path(config.HF_HOME)):
            mounts.append((hub, False))

        model_path = Path(model)
        if model_path.is_absolute() and not any(
            self._path_within(model_path, root) for root, _readonly in mounts
        ):
            mounts.append((model_path, True))

        volumes: list[str] = []
        seen: set[str] = set()
        for path, readonly in mounts:
            raw = str(path)
            if raw in seen:
                continue
            seen.add(raw)
            suffix = ":ro" if readonly else ""
            volumes.append(f"{raw}:{raw}{suffix}")
        return volumes

    @staticmethod
    def _path_within(path: Path, root: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            return False

    def _argv_for_log(self, argv: list[str]) -> list[str]:
        redacted: list[str] = []
        for item in argv:
            if "=" in item:
                key, value = item.split("=", 1)
                if value and self._looks_secret_key(key):
                    redacted.append(f"{key}=<redacted>")
                    continue
            redacted.append(item)
        return redacted

    @staticmethod
    def _looks_secret_key(key: str) -> bool:
        normalized = key.upper()
        return any(part in normalized for part in ("TOKEN", "KEY", "SECRET"))

    def _build_server_args(
        self,
        req: LoadRequest,
        model: str,
        served: str,
        *,
        host: str,
    ) -> list[str]:
        argv: list[str] = [
            "--host",
            host,
            "--port",
            str(config.VLLM_PORT),
            "--model",
            model,
            "--served-model-name",
            served,
            "--dtype",
            _effective_dtype(req.dtype),
            "--gpu-memory-utilization",
            str(req.gpu_memory_utilization),
            "--max-model-len",
            str(req.max_model_len),
            "--max-num-seqs",
            str(req.max_num_seqs),
            "--tensor-parallel-size",
            str(req.tensor_parallel_size),
        ]
        if req.quant and req.quant not in ("", "none", "auto"):
            argv += ["--quantization", req.quant]
        if req.kv_cache_dtype and req.kv_cache_dtype != "auto":
            argv += ["--kv-cache-dtype", req.kv_cache_dtype]
        if req.enforce_eager:
            argv += ["--enforce-eager"]
        if req.trust_remote_code:
            argv += ["--trust-remote-code"]
        # Qwen3.5 uses head_dim=256 which flashinfer can't handle on sm75 → TRITON_ATTN.
        model_type = self._model_type(req)
        extra = list(req.extra_args or [])
        if model_type in _TRITON_ATTN_MODEL_TYPES and "--attention-backend" not in extra:
            extra += ["--attention-backend", "TRITON_ATTN"]
            logger.info("auto-adding --attention-backend TRITON_ATTN for model_type=%s", model_type)
        if extra:
            argv += extra
        return argv

    def _remove_container(self) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", config.VLLM_CONTAINER_NAME],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
        except Exception as exc:
            logger.debug("docker container cleanup skipped: %s", exc)

    def _append_log(self, line: str) -> None:
        # Caller may or may not hold the lock; deque.append is thread-safe.
        self._logs.append(line.rstrip("\n"))

    def _read_loop(self, proc: subprocess.Popen[str], gen: int) -> None:
        """Drain the merged stdout/stderr pipe, scanning for ready/error markers."""
        stream = proc.stdout
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, ""):
                if raw == "":
                    break
                line = raw.rstrip("\n")
                self._logs.append(line)
                low = line.lower()
                with self._lock:
                    if gen != self._gen:
                        return  # superseded by a newer load/unload
                    if any(m in low for m in _ERROR_MARKERS):
                        if self._state != "ready":
                            self._state = "error"
                            self._phase = "Error"
                        if not self._error:
                            self._error = line.strip()
                    elif self._state == "loading" and any(
                        m in low for m in _READY_MARKERS
                    ):
                        self._state = "ready"
                        self._error = ""
                        self._progress = 1.0
                        self._phase = "Ready"
                    elif self._state == "loading":
                        parsed = _parse_progress(low)
                        if parsed is not None:
                            p, phase = parsed
                            if p >= self._progress:  # monotonic
                                self._progress = p
                            self._phase = phase
        except Exception as exc:
            logger.debug("log reader stopped: %s", exc)
        finally:
            try:
                stream.close()
            except Exception:
                pass
            rc = proc.poll()
            retry_req: Optional[LoadRequest] = None
            with self._lock:
                if gen != self._gen:
                    return
                if rc is not None and self._state in ("loading", "ready"):
                    self._state = "error"
                    if not self._error:
                        self._error = f"vllm process exited (code {rc})"
                    self._pid = None
                # Auto-retry once on recognisable failures.
                if self._state == "error" and self._load_request is not None and not self._retried:
                    error_log = "\n".join(list(self._logs)[-60:])
                    retry_req = self._adjust_req_for_retry(self._load_request, error_log)
                    if retry_req is not None:
                        self._retried = True  # prevent infinite loop

            if retry_req is not None:
                t = threading.Thread(target=self.load, args=(retry_req, True), daemon=True)
                t.start()

    def _health_loop(self, gen: int) -> None:
        """Poll the engine /health endpoint until ready, dead, or superseded."""
        url = config.VLLM_BASE_URL + "/health"
        deadline = time.monotonic() + float(config.VLLM_STARTUP_TIMEOUT)
        while time.monotonic() < deadline:
            with self._lock:
                if gen != self._gen:
                    return
                state = self._state
                proc = self._proc
            if state == "ready":
                return
            if state == "error":
                return
            if proc is None or proc.poll() is not None:
                # Process gone; let the reader loop set the error state.
                return
            if self._health_ok(url):
                with self._lock:
                    if gen != self._gen:
                        return
                    if self._state == "loading":
                        self._state = "ready"
                        self._error = ""
                        self._progress = 1.0
                        self._phase = "Ready"
                return
            time.sleep(1.0)

    @staticmethod
    def _health_ok(url: str) -> bool:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as exc:
            return 200 <= exc.code < 300
        except Exception:
            return False

    def _terminate(self, proc: subprocess.Popen[str]) -> None:
        """SIGTERM the process group, wait, then SIGKILL if needed."""
        pid = proc.pid
        # Prefer killing the whole session so vLLM worker children die too.
        try:
            pgid = os.getpgid(pid)
        except Exception:
            pgid = None

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            return
        except Exception as exc:
            logger.warning("SIGTERM failed for pid %s: %s", pid, exc)

        try:
            proc.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception as exc:
            logger.debug("wait after SIGTERM failed: %s", exc)

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            return
        except Exception as exc:
            logger.warning("SIGKILL failed for pid %s: %s", pid, exc)
        try:
            proc.wait(timeout=10)
        except Exception as exc:
            logger.debug("wait after SIGKILL failed: %s", exc)
        if config.VLLM_ENGINE_RUNNER == "docker":
            self._remove_container()


# Module singleton — importers use `from .vllm_manager import manager`.
manager = VLLMManager()
