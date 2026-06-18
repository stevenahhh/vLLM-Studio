"""Background model downloader for vLLM Studio.

Wraps ``huggingface_hub.snapshot_download`` in daemon threads and exposes a
thread-safe :class:`DownloadManager` with progress/speed telemetry, persisted
to :data:`config.DOWNLOADS_FILE`.

Public API (importers rely on these names exactly):
    class DownloadManager:
        list_jobs() -> list[DownloadJob]
        start(req: DownloadRequest) -> DownloadJob
        cancel(job_id: str) -> bool
    manager = DownloadManager()  # module singleton

Telemetry / network failures never crash a request: we degrade gracefully and
record warnings on the job instead.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from fnmatch import fnmatch
from typing import Optional

from huggingface_hub import HfApi, snapshot_download

from . import config
from .schemas import DownloadJob, DownloadRequest

# Files we treat as "weights" when no explicit allow_patterns are given but the
# request asks for a specific quant — kept permissive: default snapshot pulls all.
_WATCH_INTERVAL = 1.0  # seconds between progress polls


def _now() -> float:
    return time.time()


def _dir_size(path: str) -> int:
    """Best-effort recursive byte size of a directory; never raises."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for name in files:
                fp = os.path.join(root, name)
                try:
                    # Use lstat so we don't follow symlinks twice (HF cache uses
                    # symlinks into blobs); count the real blob size via stat.
                    st = os.stat(fp)
                    total += st.st_size
                except (OSError, ValueError):
                    continue
    except OSError:
        return total
    return total


class _JobRuntime:
    """Per-job mutable runtime state not part of the serialized model."""

    __slots__ = ("cancel_event", "thread", "target_dir")

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.target_dir: Optional[str] = None


class DownloadManager:
    """Thread-safe manager of background HF snapshot downloads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, DownloadJob] = {}
        self._runtime: dict[str, _JobRuntime] = {}
        self._api = HfApi(token=config.HF_TOKEN)
        self._load()

    # --- persistence ----------------------------------------------------------
    def _load(self) -> None:
        """Load persisted jobs from DOWNLOADS_FILE. Best-effort."""
        path = str(config.DOWNLOADS_FILE)
        try:
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        items = raw.get("items", raw) if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                job = DownloadJob(**item)
            except Exception:
                continue
            # Any job that was mid-flight when we crashed cannot be resumed by a
            # now-dead thread; mark it as error so the UI is honest.
            if job.state in ("queued", "downloading"):
                job.state = "error"
                if not job.error:
                    job.error = "interrupted (control plane restarted)"
                job.updated_at = _now()
            self._jobs[job.id] = job

    def _persist_locked(self) -> None:
        """Write all jobs to DOWNLOADS_FILE. Caller must hold the lock."""
        path = str(config.DOWNLOADS_FILE)
        tmp = f"{path}.tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {"items": [j.model_dump() for j in self._jobs.values()]}
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
        except (OSError, ValueError):
            # Telemetry/persistence failure must never crash a request.
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def _persist(self) -> None:
        with self._lock:
            self._persist_locked()

    # --- helpers --------------------------------------------------------------
    def _compute_total_bytes(self, req: DownloadRequest) -> int:
        """Sum sizes of repo siblings matching allow_patterns. Best-effort -> 0."""
        try:
            info = self._api.model_info(
                req.repo,
                revision=req.revision,
                files_metadata=True,
            )
        except Exception:
            return 0
        siblings = getattr(info, "siblings", None) or []
        patterns = req.allow_patterns
        total = 0
        for sib in siblings:
            name = getattr(sib, "rfilename", None)
            if not name:
                continue
            if patterns and not any(fnmatch(name, p) for p in patterns):
                continue
            size = getattr(sib, "size", None)
            if isinstance(size, int):
                total += size
        return total

    def _update(self, job_id: str, **changes) -> None:
        """Apply field changes to a job, stamp updated_at, persist. Thread-safe."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()
            self._persist_locked()

    # --- worker threads -------------------------------------------------------
    def _run_download(self, job_id: str, req: DownloadRequest) -> None:
        """Body of the download daemon thread."""
        with self._lock:
            rt = self._runtime.get(job_id)
        if rt is None:
            return

        # Watcher thread polls the on-disk size and reports progress.
        watcher_stop = threading.Event()
        watcher = threading.Thread(
            target=self._watch_progress,
            args=(job_id, watcher_stop),
            daemon=True,
            name=f"dl-watch-{job_id[:8]}",
        )

        self._update(job_id, state="downloading")
        watcher.start()
        try:
            snapshot_dir = snapshot_download(
                repo_id=req.repo,
                revision=req.revision,
                allow_patterns=req.allow_patterns,
                cache_dir=config.HF_HUB_CACHE,
                token=config.HF_TOKEN,
            )
        except Exception as exc:  # network, auth, gated, disk, etc.
            watcher_stop.set()
            with self._lock:
                cancelled = rt.cancel_event.is_set()
            if cancelled:
                self._update(job_id, state="cancelled")
            else:
                self._update(job_id, state="error", error=str(exc))
            return
        finally:
            watcher_stop.set()

        with self._lock:
            cancelled = rt.cancel_event.is_set()
            job = self._jobs.get(job_id)
            total = job.total_bytes if job else 0

        if cancelled:
            # snapshot_download cannot be hard-killed; it may have finished
            # anyway. Honour the user's intent and mark cancelled.
            self._update(job_id, state="cancelled", path=snapshot_dir)
            return

        final_size = _dir_size(snapshot_dir)
        if total <= 0:
            total = final_size
        self._update(
            job_id,
            state="completed",
            path=snapshot_dir,
            downloaded_bytes=final_size,
            total_bytes=total,
            progress=1.0,
            speed_bps=0.0,
        )

    def _watch_progress(self, job_id: str, stop: threading.Event) -> None:
        """Poll the target dir size ~every second; update downloaded/progress/speed."""
        with self._lock:
            rt = self._runtime.get(job_id)
            job = self._jobs.get(job_id)
        if rt is None or job is None:
            return

        target = rt.target_dir
        last_bytes = 0
        last_time = _now()

        while not stop.is_set():
            stop.wait(_WATCH_INTERVAL)
            if stop.is_set():
                break
            # If the user cancelled, stop reporting (snapshot keeps running but
            # we no longer advance the bar).
            if rt.cancel_event.is_set():
                break

            cur_bytes = _dir_size(target) if target else 0
            now = _now()
            dt = now - last_time
            speed = 0.0
            if dt > 0 and cur_bytes >= last_bytes:
                speed = (cur_bytes - last_bytes) / dt

            with self._lock:
                cur_job = self._jobs.get(job_id)
                if cur_job is None or cur_job.state not in ("downloading", "queued"):
                    break
                total = cur_job.total_bytes
                progress = 0.0
                if total > 0:
                    progress = min(1.0, max(0.0, cur_bytes / total))
                cur_job.downloaded_bytes = cur_bytes
                cur_job.progress = progress
                cur_job.speed_bps = speed
                cur_job.updated_at = now
                self._persist_locked()

            last_bytes = cur_bytes
            last_time = now

    def _resolve_target_dir(self, repo: str, revision: str) -> str:
        """Best-effort path of the snapshot dir we poll for size while downloading."""
        # HF cache layout: <cache>/models--<org>--<name>/snapshots/<rev or commit>
        safe = "models--" + repo.replace("/", "--")
        return os.path.join(config.HF_HUB_CACHE, safe, "snapshots")

    # --- public API -----------------------------------------------------------
    def start(self, req: DownloadRequest) -> DownloadJob:
        job_id = uuid.uuid4().hex
        now = _now()
        total = self._compute_total_bytes(req)

        job = DownloadJob(
            id=job_id,
            repo=req.repo,
            revision=req.revision,
            quant=req.quant,
            state="queued",
            total_bytes=total,
            downloaded_bytes=0,
            progress=0.0,
            speed_bps=0.0,
            error="",
            path="",
            created_at=now,
            updated_at=now,
        )

        rt = _JobRuntime()
        rt.target_dir = self._resolve_target_dir(req.repo, req.revision)

        with self._lock:
            self._jobs[job_id] = job
            self._runtime[job_id] = rt
            self._persist_locked()

        thread = threading.Thread(
            target=self._run_download,
            args=(job_id, req),
            daemon=True,
            name=f"dl-{job_id[:8]}",
        )
        rt.thread = thread
        thread.start()
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            rt = self._runtime.get(job_id)
            if job is None:
                return False
            if job.state in ("completed", "error", "cancelled"):
                return False
            if rt is not None:
                rt.cancel_event.set()
            job.state = "cancelled"
            job.speed_bps = 0.0
            job.updated_at = _now()
            self._persist_locked()
        return True

    def remove(self, job_id: str) -> bool:
        """Remove a finished job from the list (completed/error/cancelled only)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.state not in ("completed", "error", "cancelled"):
                return False
            del self._jobs[job_id]
            self._runtime.pop(job_id, None)
            self._persist_locked()
        return True

    def list_jobs(self) -> list[DownloadJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)


# Module singleton (importers depend on this exact name).
manager = DownloadManager()
