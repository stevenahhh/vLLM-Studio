// API contract mirror of backend/app/schemas.py. Single source of truth on the FE.
// Read-only contract module: import from here, do not rewrite.

export type Family = "llm" | "diffusion" | "moe";
export type OOMStatus = "ok" | "tight" | "oom";
export type EngineState = "stopped" | "loading" | "ready" | "error";
export type DownloadState =
  | "queued"
  | "downloading"
  | "completed"
  | "error"
  | "cancelled";

export interface GpuInfo {
  index: number;
  name: string;
  total_bytes: number;
  compute_capability: string;
  driver_version: string;
}

export interface GpuLive {
  index: number;
  name: string;
  total_bytes: number;
  used_bytes: number;
  free_bytes: number;
  utilization: number; // 0..100
  temperature: number;
  power_watts: number;
}

export interface GpuStats {
  gpus: GpuLive[];
  total_bytes: number;
  used_bytes: number;
  free_bytes: number;
}

export interface Capabilities {
  supports_bf16: boolean;
  supports_fp8: boolean;
  supports_marlin: boolean;
  supported_quantization: string[];
  supported_dtypes: string[];
  recommended_dtype: string;
  notes: string[];
}

export interface HardwareInfo {
  gpus: GpuInfo[];
  num_gpus: number;
  per_gpu_bytes: number;
  total_bytes: number;
  capabilities: Capabilities;
  recommendations: string[];
}

export interface ModelMeta {
  repo: string;
  revision: string;
  quant: string;
  family: Family;
  model_type: string;
  architectures: string[];
  hidden_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  num_key_value_heads: number;
  head_dim: number;
  intermediate_size: number;
  vocab_size: number;
  max_position_embeddings: number;
  tie_word_embeddings: boolean;
  torch_dtype: string;
  num_experts: number;
  param_count: number;
  weight_bytes_known: number | null;
  is_gated: boolean;
  config_raw: Record<string, unknown>;
  warnings: string[];
}

export interface DownloadedModel {
  repo: string;
  revision: string;
  path: string;
  size_bytes: number;
  quant: string;
  family: Family;
  model_type: string;
  has_config: boolean;
}

export interface QuantVariant {
  repo: string;
  quant: string;
  revision: string;
  size_bytes: number;
  file_count: number;
  files: string[];
  note: string;
  supported: boolean;
}

export interface HFModel {
  repo: string;
  downloads: number;
  likes: number;
  updated: string;
  pipeline_tag: string;
  tags: string[];
  detected_quant: string[];
  gated: boolean;
}

export interface DownloadRequest {
  repo: string;
  revision?: string;
  quant?: string;
  allow_patterns?: string[] | null;
}

export interface DownloadJob {
  id: string;
  repo: string;
  revision: string;
  quant: string;
  state: DownloadState;
  total_bytes: number;
  downloaded_bytes: number;
  progress: number; // 0..1
  speed_bps: number;
  error: string;
  path: string;
  created_at: number;
  updated_at: number;
}

export interface EstimateRequest {
  meta: ModelMeta;
  quant?: string | null;
  dtype: string;
  tensor_parallel_size: number;
  gpu_memory_utilization: number;
  max_model_len: number;
  max_num_seqs: number;
  kv_concurrency: number;
  kv_cache_dtype: string;
  enforce_eager: boolean;
  block_length: number;
}

export interface PerGpuBreakdown {
  weights_bytes: number;
  kv_bytes: number;
  activation_bytes: number;
  cuda_graph_bytes: number;
  overhead_bytes: number;
  required_bytes: number;
  budget_bytes: number;
  total_bytes: number;
  headroom_bytes: number;
}

export interface VramEstimate {
  status: OOMStatus;
  reasons: string[];
  tensor_parallel_size: number;
  num_gpus_available: number;
  weights_total_bytes: number;
  kv_per_token_bytes: number;
  kv_total_bytes: number;
  activation_bytes: number;
  weights_source: "files" | "config";
  approximate: boolean;
  max_safe_context: number;
  per_gpu: PerGpuBreakdown;
  aggregate_required_bytes: number;
  aggregate_budget_bytes: number;
}

export interface ParamSpec {
  key: string;
  label: string;
  type: "float" | "int" | "text" | "bool" | "select";
  default: unknown;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  options?: { value: unknown; label: string; disabled?: boolean }[] | null;
  help: string;
  unit: string;
  affects_vram: boolean;
  engine_only: boolean;
}

export interface ParamGroup {
  key: string;
  label: string;
  params: ParamSpec[];
}

export interface ParamSchema {
  family: Family;
  model_type: string;
  groups: ParamGroup[];
}

export interface LoadRequest {
  repo: string;
  revision?: string;
  quant?: string;
  dtype?: string;
  tensor_parallel_size?: number;
  gpu_memory_utilization?: number;
  max_model_len?: number;
  max_num_seqs?: number;
  kv_cache_dtype?: string;
  enforce_eager?: boolean;
  trust_remote_code?: boolean;
  extra_args?: string[];
}

export interface EngineStatus {
  state: EngineState;
  repo: string;
  revision: string;
  quant: string;
  family: Family;
  port: number;
  pid: number | null;
  load_request: LoadRequest | null;
  error: string;
  logs_tail: string[];
  served_model_name: string;
  progress: number; // 0..1 load progress
  phase: string; // human-readable load phase
}

export interface AppSettings {
  system_prompt: string;
  sampling: Record<string, unknown>;
  diffusion: Record<string, unknown>;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  system_prompt?: string | null;
  stream?: boolean;
  sampling?: Record<string, unknown>;
  extra_body?: Record<string, unknown>;
}

// OpenAI-style streamed delta the chat SSE emits.
export interface ChatChunk {
  delta: string;
  done: boolean;
  finish_reason?: string | null;
}
