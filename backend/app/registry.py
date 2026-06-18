"""Model registry: local cache scan, HF Hub search, and quant-variant discovery.

Public API (imported by main.py and peers):
    list_downloaded() -> list[DownloadedModel]
    search_hf(query: str, limit: int = 30) -> list[HFModel]
    list_variants(repo: str) -> list[QuantVariant]

All functions are best-effort: telemetry/network/gated errors degrade gracefully
to empty or partial results rather than raising. We never import torch/vllm here;
huggingface_hub is a permitted top-level import.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, cast

from huggingface_hub import HfApi

from . import config
from .schemas import DownloadedModel, Family, HFModel, QuantVariant

logger = logging.getLogger("vllm_studio.registry")

# Weight-file extensions that count toward on-disk / repo model size.
_WEIGHT_EXTS = (".safetensors", ".bin", ".gguf", ".pt", ".pth")
# Tag/name tokens that signal a particular quantization scheme.
_QUANT_TOKENS = {
    "awq": "awq",
    "gptq": "gptq",
    "gguf": "gguf_q4",
    "int8": "int8",
    "int4": "gguf_q4",
    "8bit": "int8",
    "4bit": "gguf_q4",
    "bnb": "int8",
    "bitsandbytes": "int8",
    "fp8": "fp8",
}


# --------------------------------------------------------------------------- #
# Lazy peer-module helpers (modelmeta may not be importable in isolation).    #
# --------------------------------------------------------------------------- #
def _detect_quant_from_config(cfg: dict[str, Any], repo: str = "") -> str:
    """Delegate to modelmeta.detect_quant_from_config, degrading to a name heuristic."""
    try:
        from . import modelmeta  # local import: avoid hard dependency at import time

        return modelmeta.detect_quant_from_config(cfg, repo)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("detect_quant_from_config fallback (%s)", exc)
        return _primary_quant_from_name(repo, [])


def _detect_family(cfg: dict[str, Any], repo: str = "") -> str:
    """Delegate to modelmeta.detect_family, degrading to 'llm'."""
    try:
        from . import modelmeta

        return modelmeta.detect_family(cfg, repo)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("detect_family fallback (%s)", exc)
        return "llm"


def _supports_quant(quant: str) -> bool:
    """Delegate to hwinfo.supports_quant, degrading to a Turing-safe allow-list."""
    try:
        from . import hwinfo

        return bool(hwinfo.supports_quant(quant))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("supports_quant fallback (%s)", exc)
        # Turing (sm_75): no fp8/marlin; awq/gptq/gguf/int8/none ok.
        return quant.lower() not in {"fp8", "awq_marlin", "marlin"}


# --------------------------------------------------------------------------- #
# Name/tag based quant detection                                              #
# --------------------------------------------------------------------------- #
def _detect_quant_from_name(repo: str, tags: Optional[list[str]] = None) -> list[str]:
    """Return distinct quant hints derived from a repo id and/or tag list."""
    haystack = (repo or "").lower()
    tag_str = " ".join(t.lower() for t in (tags or []))
    blob = f"{haystack} {tag_str}"
    found: list[str] = []
    for token, quant in _QUANT_TOKENS.items():
        if token in blob and quant not in found:
            found.append(quant)
    return found


def _primary_quant_from_name(repo: str, tags: Optional[list[str]] = None) -> str:
    hints = _detect_quant_from_name(repo, tags)
    return hints[0] if hints else "none"


def _is_weight_file(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(ext) for ext in _WEIGHT_EXTS)


def _sibling_size(sib: Any) -> int:
    """Best-effort byte size of a RepoSibling (plain size or LFS metadata)."""
    sz = getattr(sib, "size", None)
    if not sz:
        lfs = getattr(sib, "lfs", None)
        sz = getattr(lfs, "size", None) if lfs else None
    try:
        return int(sz) if sz else 0
    except (TypeError, ValueError):
        return 0


def _select_weight_files(siblings: Any) -> tuple[list[str], int]:
    """Pick a single canonical weight format and sum its size.

    Repos often ship the same weights twice (e.g. `model.safetensors` AND a
    legacy `consolidated.00.pth`). Summing every weight file double-counts. We
    prefer one format in priority order so the reported size reflects what would
    actually be loaded: safetensors > gguf > bin > pt/pth.
    """
    buckets: dict[str, list[tuple[str, int]]] = {
        "safetensors": [],
        "gguf": [],
        "bin": [],
        "pt": [],
    }
    for sib in siblings or []:
        rfilename = getattr(sib, "rfilename", None)
        if not rfilename or not _is_weight_file(rfilename):
            continue
        low = rfilename.lower()
        size = _sibling_size(sib)
        if low.endswith(".safetensors"):
            buckets["safetensors"].append((rfilename, size))
        elif low.endswith(".gguf"):
            buckets["gguf"].append((rfilename, size))
        elif low.endswith(".bin"):
            buckets["bin"].append((rfilename, size))
        else:  # .pt / .pth
            buckets["pt"].append((rfilename, size))

    for fmt in ("safetensors", "gguf", "bin", "pt"):
        chosen = buckets[fmt]
        if chosen:
            files = [name for name, _ in chosen]
            total = sum(sz for _, sz in chosen)
            return files, total
    return [], 0


def _list_models_sorted(api: HfApi, search: str, limit: int) -> Any:
    """list_models(search, sort=downloads desc) tolerant of API version drift.

    The contract specifies `direction=-1`, but huggingface_hub >=1.x dropped that
    kwarg (sort keys are already descending). `full=True` is requested so that
    `last_modified`/`gated` are populated for the HFModel.updated field. Fall back
    progressively if a given Hub version rejects an argument.
    """
    attempts = (
        {"sort": "downloads", "direction": -1, "full": True},
        {"sort": "downloads", "full": True},
        {"sort": "downloads"},
        {},
    )
    last_exc: Optional[Exception] = None
    for kwargs in attempts:
        try:
            method = getattr(api, "list_models")
            return method(search=search, limit=limit, **kwargs)
        except TypeError as exc:  # unsupported kwarg for this version
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return api.list_models(search=search, limit=limit)


# --------------------------------------------------------------------------- #
# 1. list_downloaded                                                          #
# --------------------------------------------------------------------------- #
def _read_ref(model_dir: Path, ref: str = "main") -> Optional[str]:
    """Return the commit hash a ref points to, if present."""
    try:
        ref_file = model_dir / "refs" / ref
        if ref_file.is_file():
            val = ref_file.read_text().strip()
            return val or None
    except Exception:
        pass
    return None


def _newest_snapshot(model_dir: Path) -> Optional[Path]:
    """Resolve the snapshot to report: prefer refs/main, else newest by mtime."""
    snap_root = model_dir / "snapshots"
    if not snap_root.is_dir():
        return None
    snaps = [p for p in snap_root.iterdir() if p.is_dir()]
    if not snaps:
        return None

    # Prefer the commit referenced by refs/main if it exists on disk.
    head = _read_ref(model_dir, "main")
    if head:
        for s in snaps:
            if s.name == head:
                return s
    # Otherwise newest snapshot directory by modification time.
    try:
        return max(snaps, key=lambda p: p.stat().st_mtime)
    except Exception:
        return snaps[0]


def _snapshot_revision(model_dir: Path, snapshot: Path) -> str:
    """Map a snapshot back to a human revision: 'main' if it is the head, else the hash."""
    head = _read_ref(model_dir, "main")
    if head and snapshot.name == head:
        return "main"
    return snapshot.name


def _resolve_size_bytes(snapshot: Path) -> int:
    """Sum the real byte size of files in a snapshot, resolving symlinks to blobs.

    Each unique resolved blob is counted once (snapshots share blobs via symlink).
    """
    total = 0
    seen: set[str] = set()
    try:
        for root, _dirs, files in os.walk(snapshot, followlinks=False):
            for fname in files:
                fpath = Path(root) / fname
                try:
                    real = os.path.realpath(fpath)
                    if real in seen:
                        continue
                    seen.add(real)
                    total += os.path.getsize(real)
                except OSError:
                    continue
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("size walk failed for %s (%s)", snapshot, exc)
    return total


def _has_weight_files(snapshot: Path) -> bool:
    """True if the snapshot contains at least one real model weight file.

    Browsing a model on the HF tab fetches only its ``config.json`` (via
    get_meta / list_variants), which still creates a ``models--*`` cache dir.
    Those config-only entries are NOT downloaded models, so we require at least
    one weight file (.safetensors/.bin/.gguf/.pt) before listing an entry.
    """
    try:
        for root, _dirs, files in os.walk(snapshot, followlinks=True):
            for fname in files:
                if _is_weight_file(fname):
                    return True
    except Exception:  # pragma: no cover - defensive
        pass
    return False


def _load_config_json(snapshot: Path) -> Optional[dict[str, Any]]:
    cfg_path = snapshot / "config.json"
    try:
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.debug("config.json parse failed for %s (%s)", snapshot, exc)
    return None


def _repo_from_dir_name(dir_name: str) -> Optional[str]:
    """Convert 'models--org--name' (or 'models--name') to 'org/name'."""
    if not dir_name.startswith("models--"):
        return None
    rest = dir_name[len("models--"):]
    parts = rest.split("--")
    if not parts or not parts[0]:
        return None
    return "/".join(parts)


def list_downloaded() -> list[DownloadedModel]:
    """Scan the HF hub cache for downloaded models and summarize each one."""
    out: list[DownloadedModel] = []
    cache_root = Path(config.HF_HUB_CACHE)
    if not cache_root.is_dir():
        return out

    try:
        entries = sorted(cache_root.iterdir(), key=lambda p: p.name)
    except Exception as exc:
        logger.warning("could not list HF cache %s (%s)", cache_root, exc)
        return out

    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            repo = _repo_from_dir_name(entry.name)
            if repo is None:
                continue
            snapshot = _newest_snapshot(entry)
            if snapshot is None:
                continue
            # Skip config-only entries created merely by browsing/metadata fetches.
            if not _has_weight_files(snapshot):
                continue

            revision = _snapshot_revision(entry, snapshot)
            size_bytes = _resolve_size_bytes(snapshot)
            cfg = _load_config_json(snapshot)
            has_config = cfg is not None

            quant = _detect_quant_from_config(cfg or {}, repo)
            family = _detect_family(cfg, repo) if has_config else "llm"
            model_type = ""
            if has_config and isinstance(cfg.get("model_type"), str):
                model_type = cfg["model_type"]

            out.append(
                DownloadedModel(
                    repo=repo,
                    revision=revision,
                    path=str(snapshot),
                    size_bytes=size_bytes,
                    quant=quant or "none",
                    family=cast(Family, family),
                    model_type=model_type,
                    has_config=has_config,
                )
            )
        except Exception as exc:  # never let one bad dir abort the scan
            logger.debug("skipping cache entry %s (%s)", getattr(entry, "name", "?"), exc)
            continue

    return out


# --------------------------------------------------------------------------- #
# 1b. delete_downloaded                                                        #
# --------------------------------------------------------------------------- #
def delete_downloaded(repo: str) -> bool:
    """Delete all cached blobs and snapshots for *repo*. Returns True on success."""
    import shutil

    dir_name = "models--" + repo.replace("/", "--")
    cache_root = Path(config.HF_HUB_CACHE)
    target = cache_root / dir_name
    if not target.is_dir():
        return False
    try:
        shutil.rmtree(target)
        logger.info("deleted model cache: %s", target)
        return True
    except Exception as exc:
        logger.warning("failed to delete %s: %s", target, exc)
        raise


# --------------------------------------------------------------------------- #
# 2. search_hf                                                                 #
# --------------------------------------------------------------------------- #
def _stringify_updated(model: Any) -> str:
    val = getattr(model, "last_modified", None) or getattr(model, "lastModified", None)
    if val is None:
        return ""
    try:
        # datetime -> ISO 8601; anything else -> str().
        return val.isoformat()  # type: ignore[union-attr]
    except Exception:
        return str(val)


def _gated_to_bool(gated: Any) -> bool:
    # HF returns False or a string like "auto"/"manual" when gated.
    if gated is None or gated is False:
        return False
    return True


def search_hf(query: str, limit: int = 30) -> list[HFModel]:
    """Search the HF Hub for models matching `query`, sorted by downloads desc."""
    out: list[HFModel] = []
    try:
        api = HfApi(token=config.HF_TOKEN)
        models = _list_models_sorted(api, query, limit)
    except Exception as exc:
        logger.warning("HF search failed for %r (%s)", query, exc)
        return out

    try:
        for m in models:
            try:
                repo = getattr(m, "id", None) or getattr(m, "modelId", None)
                if not repo:
                    continue
                tags = list(getattr(m, "tags", None) or [])
                out.append(
                    HFModel(
                        repo=repo,
                        downloads=int(getattr(m, "downloads", 0) or 0),
                        likes=int(getattr(m, "likes", 0) or 0),
                        updated=_stringify_updated(m),
                        pipeline_tag=getattr(m, "pipeline_tag", None) or "",
                        tags=tags,
                        detected_quant=_detect_quant_from_name(repo, tags),
                        gated=_gated_to_bool(getattr(m, "gated", None)),
                    )
                )
            except Exception as exc:  # one bad record shouldn't drop the rest
                logger.debug("skip search record (%s)", exc)
                continue
    except Exception as exc:  # iteration itself failed (network)
        logger.warning("HF search iteration failed for %r (%s)", query, exc)

    return out


# --------------------------------------------------------------------------- #
# 3. list_variants                                                            #
# --------------------------------------------------------------------------- #
def _quant_from_repo_config(cfg: Optional[dict[str, Any]], repo: str) -> str:
    """Pick the in-repo quant from quantization_config, else repo-name hint."""
    if cfg:
        qc = cfg.get("quantization_config")
        if isinstance(qc, dict):
            method = qc.get("quant_method") or qc.get("quant_type") or qc.get("method")
            if isinstance(method, str) and method:
                m = method.lower()
                if "awq" in m:
                    return "awq"
                if "gptq" in m:
                    return "gptq"
                if "gguf" in m:
                    return "gguf_q4"
                if "fp8" in m:
                    return "fp8"
                if m in {"bitsandbytes", "bnb"}:
                    return "int8"
                if "int8" in m or "8bit" in m:
                    return "int8"
                if "int4" in m or "4bit" in m:
                    return "gguf_q4"
                return m
        # Some configs carry a top-level torch hint only; fall through.
    return _primary_quant_from_name(repo)


def _fetch_repo_config(api: HfApi, repo: str) -> Optional[dict[str, Any]]:
    """Best-effort fetch of config.json contents for a repo (network)."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo,
            filename="config.json",
            token=config.HF_TOKEN,
        )
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("config.json fetch failed for %s (%s)", repo, exc)
        return None


def _base_name(repo: str) -> str:
    """Strip the org and trailing quant suffixes to get a search base name."""
    name = repo.split("/")[-1] if "/" in repo else repo
    low = name.lower()
    # Drop a trailing quant marker like '-AWQ', '-GPTQ', '-GGUF', '-4bit'.
    for token in ("awq", "gptq", "gguf", "int8", "int4", "4bit", "8bit", "bnb"):
        for sep in ("-", "_", "."):
            suffix = f"{sep}{token}"
            if low.endswith(suffix):
                name = name[: -len(suffix)]
                low = name.lower()
    return name


def _primary_variant(api: HfApi, repo: str) -> Optional[QuantVariant]:
    """Build the in-repo variant from this repo's own weight files."""
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as exc:
        logger.debug("model_info failed for %s (%s)", repo, exc)
        return None

    siblings = getattr(info, "siblings", None) or []
    weight_files, size_total = _select_weight_files(siblings)

    # Determine quant: prefer config.json quantization_config, else repo name.
    cfg: Optional[dict[str, Any]] = None
    try:
        # If model_info exposed config inline, use it; else fetch the file.
        inline = getattr(info, "config", None)
        if isinstance(inline, dict) and inline:
            cfg = inline
    except Exception:
        cfg = None
    if cfg is None:
        cfg = _fetch_repo_config(api, repo)
    quant = _quant_from_repo_config(cfg, repo)

    revision = getattr(info, "sha", None) or "main"
    return QuantVariant(
        repo=repo,
        quant=quant or "none",
        revision=revision,
        size_bytes=size_total,
        file_count=len(weight_files),
        files=weight_files,
        note="in-repo weights",
        supported=_supports_quant(quant or "none"),
    )


def _sibling_variants(
    api: HfApi, repo: str, exclude: set[str]
) -> list[QuantVariant]:
    """Find sibling quant repos (awq/gptq/gguf) by searching the base model name."""
    variants: list[QuantVariant] = []
    base = _base_name(repo)
    if not base:
        return variants
    try:
        candidates = _list_models_sorted(api, base, 30)
    except Exception as exc:
        logger.debug("sibling search failed for %s (%s)", base, exc)
        return variants

    try:
        for m in candidates:
            try:
                cand_repo = getattr(m, "id", None) or getattr(m, "modelId", None)
                if not cand_repo or cand_repo in exclude:
                    continue
                tags = list(getattr(m, "tags", None) or [])
                # Only surface explicit awq/gptq/gguf siblings (per spec). Match
                # the scheme tokens directly so bnb/mlx 4-bit repos are excluded,
                # and prefer the specific scheme (awq/gptq) over generic gguf so
                # e.g. '...-GPTQ-Int4' is labelled gptq (supported) not gguf_q4.
                blob = f"{cand_repo} {' '.join(tags)}".lower()
                quant = None
                if "awq" in blob:
                    quant = "awq"
                elif "gptq" in blob:
                    quant = "gptq"
                elif "gguf" in blob:
                    quant = "gguf_q4"
                if quant is None:
                    continue
                exclude.add(cand_repo)
                variants.append(
                    QuantVariant(
                        repo=cand_repo,
                        quant=quant,
                        revision="main",
                        size_bytes=0,  # best-effort; avoid an info call per sibling
                        file_count=0,
                        files=[],
                        note="sibling quant repo",
                        supported=_supports_quant(quant),
                    )
                )
            except Exception as exc:
                logger.debug("skip sibling candidate (%s)", exc)
                continue
    except Exception as exc:
        logger.debug("sibling iteration failed for %s (%s)", base, exc)

    return variants


def list_variants(repo: str) -> list[QuantVariant]:
    """Return quant variants for a repo: its own weights plus sibling quant repos."""
    out: list[QuantVariant] = []
    if not repo:
        return out

    try:
        api = HfApi(token=config.HF_TOKEN)
    except Exception as exc:
        logger.warning("HfApi init failed (%s)", exc)
        return out

    seen_repos: set[str] = set()

    primary = None
    try:
        primary = _primary_variant(api, repo)
    except Exception as exc:
        logger.debug("primary variant failed for %s (%s)", repo, exc)
    if primary is not None:
        out.append(primary)
        seen_repos.add(repo)

    try:
        out.extend(_sibling_variants(api, repo, seen_repos))
    except Exception as exc:
        logger.debug("sibling variants failed for %s (%s)", repo, exc)

    return out
