"""Pydantic API contract for vLLM Studio. Mirrored by frontend/lib/types.ts.

Read-only contract module. Do not rewrite; other modules import these models.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Family = Literal["llm", "diffusion", "moe"]
OOMStatus = Literal["ok", "tight", "oom"]
EngineState = Literal["stopped", "loading", "ready", "error"]
DownloadState = Literal["queued", "downloading", "completed", "error", "cancelled"]


# --- Hardware / GPU ------------------------------------------------------------
class GpuInfo(BaseModel):
    index: int
    name: str
    total_bytes: int
    compute_capability: str           # e.g. "7.5"
    driver_version: str = ""


class GpuLive(BaseModel):
    index: int
    name: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    utilization: float = 0.0          # 0..100
    temperature: float = 0.0          # C
    power_watts: float = 0.0


class GpuStats(BaseModel):
    gpus: list[GpuLive] = []
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0


class Capabilities(BaseModel):
    supports_bf16: bool
    supports_fp8: bool
    supports_marlin: bool
    supported_quantization: list[str]
    supported_dtypes: list[str]
    recommended_dtype: str
    notes: list[str] = []


class HardwareInfo(BaseModel):
    gpus: list[GpuInfo] = []
    num_gpus: int = 0
    per_gpu_bytes: int = 0
    total_bytes: int = 0
    capabilities: Capabilities
    recommendations: list[str] = []


# --- Model metadata ------------------------------------------------------------
class ModelMeta(BaseModel):
    repo: str
    revision: str = "main"
    quant: str = "none"               # auto/none/awq/gptq/gguf_q4/int8/fp8
    family: Family = "llm"
    model_type: str = ""
    architectures: list[str] = []
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0
    tie_word_embeddings: bool = True
    torch_dtype: str = ""
    num_experts: int = 0
    param_count: int = 0              # estimated total params
    weight_bytes_known: Optional[int] = None  # summed weight-file size if known
    is_gated: bool = True             # SwiGLU-style MLP
    config_raw: dict[str, Any] = {}
    warnings: list[str] = []


# --- Registry / downloads ------------------------------------------------------
class DownloadedModel(BaseModel):
    repo: str
    revision: str = "main"
    path: str
    size_bytes: int = 0
    quant: str = "none"
    family: Family = "llm"
    model_type: str = ""
    has_config: bool = True


class QuantVariant(BaseModel):
    repo: str
    quant: str
    revision: str = "main"
    size_bytes: int = 0
    file_count: int = 0
    files: list[str] = []
    note: str = ""
    supported: bool = True            # supported on this hardware


class HFModel(BaseModel):
    repo: str
    downloads: int = 0
    likes: int = 0
    updated: str = ""
    pipeline_tag: str = ""
    tags: list[str] = []
    detected_quant: list[str] = []    # quant hints from repo name/tags
    gated: bool = False


class DownloadRequest(BaseModel):
    repo: str
    revision: str = "main"
    quant: str = "none"
    allow_patterns: Optional[list[str]] = None


class DownloadJob(BaseModel):
    id: str
    repo: str
    revision: str = "main"
    quant: str = "none"
    state: DownloadState = "queued"
    total_bytes: int = 0
    downloaded_bytes: int = 0
    progress: float = 0.0             # 0..1
    speed_bps: float = 0.0
    error: str = ""
    path: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


# --- Estimation ----------------------------------------------------------------
class EstimateRequest(BaseModel):
    meta: ModelMeta
    quant: Optional[str] = None              # override meta.quant
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: int = 16
    kv_concurrency: int = 1
    kv_cache_dtype: str = "auto"             # auto/fp8
    enforce_eager: bool = False
    block_length: int = 32                    # diffusion only


class PerGpuBreakdown(BaseModel):
    weights_bytes: int
    kv_bytes: int
    activation_bytes: int
    cuda_graph_bytes: int
    overhead_bytes: int
    required_bytes: int
    budget_bytes: int
    total_bytes: int
    headroom_bytes: int


class VramEstimate(BaseModel):
    status: OOMStatus
    reasons: list[str] = []
    tensor_parallel_size: int
    num_gpus_available: int
    weights_total_bytes: int
    kv_per_token_bytes: int
    kv_total_bytes: int
    activation_bytes: int
    weights_source: Literal["files", "config"] = "config"
    approximate: bool = False
    max_safe_context: int = 0
    per_gpu: PerGpuBreakdown
    aggregate_required_bytes: int
    aggregate_budget_bytes: int


# --- Params --------------------------------------------------------------------
class ParamSpec(BaseModel):
    key: str
    label: str
    type: Literal["float", "int", "text", "bool", "select"]
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list[dict[str, Any]]] = None   # [{value,label,disabled?}]
    help: str = ""
    unit: str = ""
    affects_vram: bool = False
    engine_only: bool = False


class ParamGroup(BaseModel):
    key: str
    label: str
    params: list[ParamSpec]


class ParamSchema(BaseModel):
    family: Family
    model_type: str = ""
    groups: list[ParamGroup] = []


# --- Engine --------------------------------------------------------------------
class LoadRequest(BaseModel):
    repo: str
    revision: str = "main"
    quant: str = "none"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: int = 16
    kv_cache_dtype: str = "auto"
    enforce_eager: bool = False
    trust_remote_code: bool = False
    extra_args: list[str] = []


class EngineStatus(BaseModel):
    state: EngineState = "stopped"
    repo: str = ""
    revision: str = "main"
    quant: str = "none"
    family: Family = "llm"
    port: int = 0
    pid: Optional[int] = None
    load_request: Optional[LoadRequest] = None
    error: str = ""
    logs_tail: list[str] = []
    served_model_name: str = ""
    progress: float = 0.0   # 0..1 load progress (parsed from engine logs)
    phase: str = ""         # human-readable load phase


# --- Settings & chat -----------------------------------------------------------
class AppSettings(BaseModel):
    system_prompt: str = "You are a helpful assistant."
    sampling: dict[str, Any] = {}
    diffusion: dict[str, Any] = {}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system_prompt: Optional[str] = None
    stream: bool = True
    sampling: dict[str, Any] = {}     # temperature, top_p, max_tokens, etc.
    extra_body: dict[str, Any] = {}   # diffusion params etc.
