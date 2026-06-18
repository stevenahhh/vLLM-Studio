"""Model metadata: fetch/parse HuggingFace config.json, estimate param count,
and detect model family + quantization.

Public API (imported by peers — keep names/signatures EXACT):
- get_meta(repo, quant="none", revision="main", local=False) -> ModelMeta
- detect_family(config, repo="") -> Family
- detect_quant_from_config(config, repo="") -> str

Never hard-fail on network/private/gated errors: fill what is parseable and add
warnings. Only raise if config.json is truly unavailable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

from . import config as app_config
from .schemas import Family, ModelMeta

# Weight-file extensions we sum for weight_bytes_known.
_WEIGHT_EXTS = (".safetensors", ".bin", ".gguf", ".pt")

# Keys kept in config_raw (trimmed to avoid shipping huge nested blobs).
_CONFIG_KEEP_KEYS = (
    "model_type",
    "architectures",
    "hidden_size",
    "n_embd",
    "d_model",
    "num_hidden_layers",
    "n_layer",
    "num_layers",
    "num_attention_heads",
    "n_head",
    "num_heads",
    "num_key_value_heads",
    "num_kv_heads",
    "head_dim",
    "intermediate_size",
    "ffn_dim",
    "n_inner",
    "vocab_size",
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "tie_word_embeddings",
    "torch_dtype",
    "hidden_act",
    "activation_function",
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "quantization_config",
    "diffusion_steps",
    "num_diffusion_timesteps",
    "mask_token_id",
    "block_length",
    "is_encoder_decoder",
    "rope_theta",
    "sliding_window",
    "canvas_length",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _first_int(config: dict[str, Any], *keys: str, default: int = 0) -> int:
    """Return the first key present in config coerced to int, else default."""
    for k in keys:
        v = config.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return default


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def _haystack(config: dict[str, Any], repo: str) -> str:
    """Lowercase searchable string from model_type, architectures, and repo name."""
    parts: list[str] = [str(repo or "")]
    mt = config.get("model_type")
    if mt:
        parts.append(str(mt))
    for a in _as_list(config.get("architectures")):
        parts.append(a)
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# family detection (CONTRACT §4)
# ---------------------------------------------------------------------------
def detect_family(config: dict[str, Any], repo: str = "") -> Family:
    config = config or {}
    hay = _haystack(config, repo)

    # diffusion: model_type / architecture / repo-name string hints.
    diffusion_tokens = ("diffusion", "dream", "llada", "diffu")
    if any(tok in hay for tok in diffusion_tokens):
        return "diffusion"
    # config-key hints (non-causal masked LM with diffusion-style keys).
    has_mask = config.get("mask_token_id") is not None
    has_diffusion_keys = (
        config.get("diffusion_steps") is not None
        or config.get("num_diffusion_timesteps") is not None
    )
    architectures = " ".join(_as_list(config.get("architectures"))).lower()
    is_causal = "causallm" in architectures or "forcausal" in architectures
    if has_diffusion_keys or (has_mask and not is_causal and architectures):
        return "diffusion"

    # moe: experts count > 1.
    n_experts = _first_int(
        config, "num_experts", "num_local_experts", "n_routed_experts", default=0
    )
    if n_experts > 1:
        return "moe"

    return "llm"


# ---------------------------------------------------------------------------
# quant detection
# ---------------------------------------------------------------------------
def detect_quant_from_config(config: dict[str, Any], repo: str = "") -> str:
    """Detect quantization from config.quantization_config and repo-name hints.

    Returns one of: none/awq/gptq/gguf_q4/int8/fp8/bnb.
    """
    config = config or {}

    # 1) Explicit quantization_config block is authoritative.
    qc = config.get("quantization_config")
    if isinstance(qc, dict):
        method = str(qc.get("quant_method", "")).lower()
        bits = qc.get("bits") or qc.get("w_bit") or qc.get("weight_bits")
        if "awq" in method:
            return "awq"
        if "gptq" in method:
            return "gptq"
        if "fp8" in method:
            return "fp8"
        if method in ("bitsandbytes", "bnb"):
            return "bnb"
        if "marlin" in method:
            # marlin packs gptq/awq weights; treat as 4-bit equivalent.
            return "gptq"
        if "gguf" in method:
            return "gguf_q4"
        if method:
            # Unknown explicit method: fall back to bit-width signal.
            try:
                if bits is not None and int(bits) <= 4:
                    return "gptq"
                if bits is not None and int(bits) == 8:
                    return "int8"
            except (TypeError, ValueError):
                pass
            return method

    # 2) Repo-name / tag string hints.
    hay = (repo or "").lower()
    if "awq" in hay:
        return "awq"
    if "gptq" in hay:
        return "gptq"
    if "gguf" in hay:
        return "gguf_q4"
    if "fp8" in hay:
        return "fp8"
    if "bnb" in hay or "bitsandbytes" in hay:
        return "bnb"
    if "8bit" in hay or "8-bit" in hay or "int8" in hay or "w8" in hay:
        return "int8"
    if "4bit" in hay or "4-bit" in hay or "int4" in hay or "w4" in hay:
        return "gptq"

    return "none"


def _quant_subset_exts(quant: str) -> Optional[tuple[str, ...]]:
    """When a repo mixes formats, restrict weight-file matching to the subset
    relevant for the chosen quant. Returns None => no extension filter."""
    q = (quant or "none").lower()
    if q == "gguf_q4":
        return (".gguf",)
    if q in ("none", "auto"):
        # Prefer safetensors weights; exclude gguf clutter from base repos.
        return None
    return None


def _is_weight_file(name: str, quant: str) -> bool:
    name_l = name.lower()
    subset = _quant_subset_exts(quant)
    if subset is not None:
        if not name_l.endswith(subset):
            return False
    else:
        if not name_l.endswith(_WEIGHT_EXTS):
            return False
        # For non-gguf quants, ignore stray gguf files mixed in the repo.
        if name_l.endswith(".gguf") and (quant or "none").lower() != "gguf_q4":
            return False
    # Skip optimizer/training artifacts.
    if "optimizer" in name_l or "training_args" in name_l:
        return False
    return True


# ---------------------------------------------------------------------------
# param count (CONTRACT §6(a)2)
# ---------------------------------------------------------------------------
def _estimate_param_count(
    config: dict[str, Any],
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
    vocab_size: int,
    tie_word_embeddings: bool,
    is_gated: bool,
    family: Family,
) -> int:
    h = hidden_size
    if h <= 0 or num_hidden_layers <= 0:
        return 0

    kv_dim = (num_key_value_heads * head_dim) if (num_key_value_heads and head_dim) else h
    q_dim = (num_attention_heads * head_dim) if (num_attention_heads and head_dim) else h

    # embeddings (+ separate lm_head when not tied)
    embeddings = vocab_size * h
    if not tie_word_embeddings:
        embeddings += vocab_size * h

    # per-layer attention: q + k + v + o projections
    attn = (h * q_dim) + (h * kv_dim) + (h * kv_dim) + (q_dim * h)

    inter = intermediate_size if intermediate_size > 0 else 4 * h

    # per-layer MLP
    if is_gated:
        mlp = 3 * h * inter
    else:
        mlp = 2 * h * inter

    n_experts = _first_int(
        config, "num_experts", "num_local_experts", "n_routed_experts", default=0
    )
    has_routed_experts = n_experts > 1
    if has_routed_experts:
        moe_inter = _first_int(config, "moe_intermediate_size", default=0) or inter
        expert_mlp = (3 * h * moe_inter) if is_gated else (2 * h * moe_inter)
        mlp = expert_mlp * n_experts
        # Some MoE arches keep a dense shared expert alongside routed experts.
        shared_inter = _first_int(config, "shared_expert_intermediate_size", default=0)
        if shared_inter > 0:
            mlp += (3 * h * shared_inter) if is_gated else (2 * h * shared_inter)

    per_layer = attn + mlp
    total = embeddings + num_hidden_layers * per_layer
    return int(total)


def _detect_gated(config: dict[str, Any], architectures: list[str]) -> bool:
    """Detect SwiGLU-style gated MLP. Default True for modern decoder LMs."""
    act = str(
        config.get("hidden_act") or config.get("activation_function") or ""
    ).lower()
    if act:
        if "silu" in act or "swiglu" in act or "swish" in act:
            return True
        if "geglu" in act or "gegelu" in act:
            return True
        # Classic GPT-2 style: plain gelu/gelu_new/relu => non-gated.
        if "gelu" in act or "relu" in act:
            # gelu alone (no gate hint) implies non-gated GPT-style MLP.
            return False
    # Architecture hints for known non-gated families.
    arch = " ".join(architectures).lower()
    if any(x in arch for x in ("gpt2", "gptbigcode", "gptneo", "gptj", "bloom", "opt", "falcon")):
        return False
    # Modern decoder LMs default to gated SwiGLU.
    return True


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------
def _load_local_config(repo: str, revision: str) -> Optional[dict[str, Any]]:
    """Read config.json from the local HF hub snapshot cache, if present."""
    # 1) try_to_load_from_cache resolves the snapshot symlink for us.
    try:
        path = try_to_load_from_cache(
            repo_id=repo, filename="config.json", revision=revision
        )
        if isinstance(path, str) and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass

    # 2) Fall back to scanning the cache dir layout directly.
    try:
        cache_root = Path(app_config.HF_HUB_CACHE)
        folder = "models--" + repo.replace("/", "--")
        snap_dir = cache_root / folder / "snapshots"
        if snap_dir.is_dir():
            # Prefer a snapshot whose name matches the revision; else newest.
            candidates = sorted(snap_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            ordered = [c for c in candidates if c.name == revision] + candidates
            for snap in ordered:
                cfg = snap / "config.json"
                if cfg.is_file():
                    with open(cfg, "r", encoding="utf-8") as fh:
                        return json.load(fh)
    except Exception:
        pass
    return None


def _sum_local_weight_bytes(repo: str, revision: str, quant: str) -> Optional[int]:
    """Sum weight-file sizes from the local snapshot (resolving symlinks)."""
    try:
        cache_root = Path(app_config.HF_HUB_CACHE)
        folder = "models--" + repo.replace("/", "--")
        snap_dir = cache_root / folder / "snapshots"
        if not snap_dir.is_dir():
            return None
        candidates = sorted(snap_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        ordered = [c for c in candidates if c.name == revision] + candidates
        seen: set[str] = set()
        for snap in ordered:
            if not (snap / "config.json").is_file() and not any(snap.iterdir()):
                continue
            total = 0
            found = False
            for f in snap.rglob("*"):
                if not f.is_file():
                    continue
                if not _is_weight_file(f.name, quant):
                    continue
                try:
                    size = f.stat().st_size  # follows symlink into blobs/
                except OSError:
                    continue
                total += size
                found = True
            if found:
                return int(total)
        return None
    except Exception:
        return None


def _sum_remote_weight_bytes(repo: str, revision: str, quant: str) -> Optional[int]:
    """Sum weight-file sizes from HF siblings metadata."""
    try:
        api = HfApi(token=app_config.HF_TOKEN)
        info = api.model_info(repo, revision=revision, files_metadata=True)
        siblings = getattr(info, "siblings", None) or []
        total = 0
        found = False
        for s in siblings:
            name = getattr(s, "rfilename", None)
            size = getattr(s, "size", None)
            if not name:
                continue
            if not _is_weight_file(name, quant):
                continue
            if size is None:
                continue
            total += int(size)
            found = True
        if found:
            return int(total)
        return None
    except Exception:
        return None


def _trim_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _CONFIG_KEEP_KEYS:
        if k in config:
            v = config[k]
            # Keep quantization_config but drop verbose nested module lists.
            if k == "quantization_config" and isinstance(v, dict):
                out[k] = {
                    kk: vv
                    for kk, vv in v.items()
                    if not isinstance(vv, (list, dict)) or kk in ("modules_to_not_convert",)
                }
            else:
                out[k] = v
    return out


def _core_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the config block that owns language-model dimensions."""
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        model_type = str(config.get("model_type", "")).lower()
        if model_type in {"diffusion_gemma"}:
            merged = dict(text_config)
            merged["model_type"] = config.get("model_type", text_config.get("model_type", ""))
            merged["architectures"] = config.get("architectures", text_config.get("architectures", []))
            merged["tie_word_embeddings"] = config.get(
                "tie_word_embeddings", text_config.get("tie_word_embeddings")
            )
            merged["torch_dtype"] = config.get(
                "torch_dtype", config.get("dtype", text_config.get("torch_dtype", ""))
            )
            if "canvas_length" in config:
                merged["block_length"] = config["canvas_length"]
            return merged
    return config


def _meta_from_config(
    *,
    repo: str,
    revision: str,
    quant: str,
    config: dict[str, Any],
    weight_bytes_known: Optional[int],
    warnings: list[str],
) -> ModelMeta:
    core = _core_config(config)

    model_type = str(config.get("model_type", core.get("model_type", "")) or "")
    architectures = _as_list(config.get("architectures") or core.get("architectures"))
    hidden_size = _first_int(core, "hidden_size", "n_embd", "d_model", default=0)
    num_hidden_layers = _first_int(
        core, "num_hidden_layers", "n_layer", "num_layers", default=0
    )
    num_attention_heads = _first_int(
        core, "num_attention_heads", "n_head", "num_heads", default=0
    )
    num_key_value_heads = _first_int(
        core, "num_key_value_heads", "num_kv_heads", default=0
    )
    if num_key_value_heads <= 0:
        num_key_value_heads = num_attention_heads

    head_dim = _first_int(core, "head_dim", default=0)
    if head_dim <= 0 and num_attention_heads > 0 and hidden_size > 0:
        head_dim = hidden_size // num_attention_heads

    intermediate_size = _first_int(
        core, "intermediate_size", "ffn_dim", "n_inner", default=0
    )
    vocab_size = _first_int(core, "vocab_size", default=0)
    max_position_embeddings = _first_int(
        core,
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
        "seq_length",
        default=0,
    )

    tie_raw = core.get("tie_word_embeddings")
    tie_word_embeddings = True if tie_raw is None else bool(tie_raw)

    torch_dtype = str(core.get("torch_dtype") or core.get("dtype") or "")

    family: Family = detect_family(config, repo)

    num_experts = _first_int(
        core, "num_experts", "num_local_experts", "n_routed_experts", default=0
    )

    is_gated = _detect_gated(core, architectures)

    resolved_quant = quant or "none"
    if resolved_quant in ("auto", "none"):
        detected = detect_quant_from_config(config, repo)
        if detected != "none":
            resolved_quant = detected
        elif resolved_quant == "auto":
            resolved_quant = "none"

    param_count = _estimate_param_count(
        core,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        tie_word_embeddings=tie_word_embeddings,
        is_gated=is_gated,
        family=family,
    )

    if hidden_size <= 0 or num_hidden_layers <= 0:
        warnings.append(
            "config.json missing core dims (hidden_size/num_hidden_layers); "
            "param estimate may be unreliable."
        )

    return ModelMeta(
        repo=repo,
        revision=revision,
        quant=resolved_quant,
        family=family,
        model_type=model_type,
        architectures=architectures,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie_word_embeddings,
        torch_dtype=torch_dtype,
        num_experts=num_experts,
        param_count=param_count,
        weight_bytes_known=weight_bytes_known,
        is_gated=is_gated,
        config_raw=_trim_config(core),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# public: get_meta
# ---------------------------------------------------------------------------
def get_meta(
    repo: str,
    quant: str = "none",
    revision: str = "main",
    local: bool = False,
) -> ModelMeta:
    warnings: list[str] = []
    config: Optional[dict[str, Any]] = None

    if local:
        config = _load_local_config(repo, revision)
        if config is None:
            warnings.append("Local snapshot config.json not found; trying remote.")
            local_failed = True
        else:
            local_failed = False
    else:
        local_failed = False

    if config is None:
        try:
            cfg_path = hf_hub_download(
                repo,
                "config.json",
                revision=revision,
                token=app_config.HF_TOKEN,
            )
            with open(cfg_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except Exception as exc:
            # Last-ditch: maybe it's cached locally even if we weren't told so.
            config = _load_local_config(repo, revision)
            if config is None:
                raise RuntimeError(
                    f"config.json unavailable for {repo!r}@{revision}: {exc}"
                ) from exc
            warnings.append(
                "Remote config fetch failed; used cached config "
                f"({type(exc).__name__})."
            )

    if local_failed and not local:
        pass  # already noted above

    config = config or {}

    # --- weight bytes known ------------------------------------------------
    resolved_quant = quant or "none"
    if resolved_quant in ("auto", "none"):
        detected = detect_quant_from_config(config, repo)
        if detected != "none":
            resolved_quant = detected
        elif resolved_quant == "auto":
            resolved_quant = "none"

    weight_bytes_known: Optional[int] = None
    if local:
        weight_bytes_known = _sum_local_weight_bytes(repo, revision, resolved_quant)
        if weight_bytes_known is None:
            warnings.append("Could not sum local weight files; size unknown.")
    else:
        weight_bytes_known = _sum_remote_weight_bytes(repo, revision, resolved_quant)
        if weight_bytes_known is None:
            # Maybe cached locally — try that as a fallback.
            weight_bytes_known = _sum_local_weight_bytes(repo, revision, resolved_quant)
        if weight_bytes_known is None:
            warnings.append(
                "Weight-file sizes unavailable; estimate will use config-derived params."
            )

    return _meta_from_config(
        repo=repo,
        revision=revision,
        quant=resolved_quant,
        config=config,
        weight_bytes_known=weight_bytes_known,
        warnings=warnings,
    )
