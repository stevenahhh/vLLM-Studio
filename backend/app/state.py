"""Persisted application state for vLLM Studio.

Stores user-facing app settings (system prompt, sampling/diffusion defaults) as
JSON at ``config.STATE_FILE``. Robust by design: a missing or corrupt state file
degrades to default :class:`AppSettings` rather than raising.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from .schemas import AppSettings

logger = logging.getLogger(__name__)


def load_settings() -> AppSettings:
    """Load persisted :class:`AppSettings` from ``config.STATE_FILE``.

    Returns default settings if the file is missing, unreadable, or contains
    invalid/corrupt JSON. Never raises.
    """
    path = Path(config.STATE_FILE)
    try:
        if not path.exists():
            return AppSettings()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("State file %s is not a JSON object; using defaults.", path)
            return AppSettings()
        return AppSettings(**data)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any failure
        logger.warning("Failed to load settings from %s: %s; using defaults.", path, exc)
        return AppSettings()


def save_settings(s: AppSettings) -> AppSettings:
    """Persist ``s`` to ``config.STATE_FILE`` as JSON and return it.

    Ensures the parent directory exists. Best-effort: write failures are logged
    but do not raise, so a request is never crashed by a telemetry/IO error.
    """
    path = Path(config.STATE_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s.model_dump(), indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any failure
        logger.warning("Failed to save settings to %s: %s", path, exc)
    return s
