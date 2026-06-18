// Typed fetch client + SSE helpers for the vLLM Studio backend (FastAPI on :8000).
// All endpoints under /api. Errors throw with the backend `{detail}` message.
import type {
  AppSettings,
  ChatChunk,
  ChatRequest,
  DownloadJob,
  DownloadRequest,
  DownloadedModel,
  EngineStatus,
  EstimateRequest,
  GpuStats,
  HFModel,
  HardwareInfo,
  LoadRequest,
  ModelMeta,
  ParamSchema,
  QuantVariant,
  VramEstimate,
} from "@/lib/types";

function isLocalApiBase(value: string): boolean {
  try {
    const host = new URL(value).hostname;
    return host === "localhost" || host === "127.0.0.1" || host === "::1";
  } catch {
    return false;
  }
}

function resolveApiBase(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE?.trim();
  if (typeof window !== "undefined") {
    const browserHost = window.location.hostname;
    const remoteBrowser =
      browserHost !== "localhost" && browserHost !== "127.0.0.1";
    if (!configured || (remoteBrowser && isLocalApiBase(configured))) {
      return `${window.location.protocol}//${browserHost}:8000`;
    }
  }
  return configured || "http://localhost:8000";
}

export const API_BASE = resolveApiBase();

interface ListResponse<T> {
  items: T[];
}

interface HealthResponse {
  status: string;
  vllm: EngineStatus;
}

/** Build a full URL onto API_BASE, optionally with query params. */
function url(path: string, params?: Record<string, unknown>): string {
  const u = new URL(`${API_BASE}${path}`);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === null || v === "") continue;
      u.searchParams.set(k, String(v));
    }
  }
  return u.toString();
}

/** Extract a useful error message from a non-ok response. */
async function errorMessage(res: Response): Promise<string> {
  let detail = "";
  try {
    const data = await res.json();
    if (data && typeof data === "object" && "detail" in data) {
      const d = (data as { detail: unknown }).detail;
      detail = typeof d === "string" ? d : JSON.stringify(d);
    } else {
      detail = JSON.stringify(data);
    }
  } catch {
    try {
      detail = await res.text();
    } catch {
      detail = "";
    }
  }
  return detail || `${res.status} ${res.statusText}`;
}

/** Typed JSON fetch wrapper; throws on !res.ok with the backend detail. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers:
      init?.body !== undefined
        ? { "Content-Type": "application/json", ...(init?.headers ?? {}) }
        : init?.headers,
    ...init,
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

async function requestUrl<T>(fullUrl: string, init?: RequestInit): Promise<T> {
  const res = await fetch(fullUrl, init);
  if (!res.ok) throw new Error(await errorMessage(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Health / hardware / telemetry
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function getHardware(): Promise<HardwareInfo> {
  return request<HardwareInfo>("/api/hardware");
}

export function getGpuStats(): Promise<GpuStats> {
  return request<GpuStats>("/api/gpu/stats");
}

// ---------------------------------------------------------------------------
// Models / registry / meta
// ---------------------------------------------------------------------------

export async function listDownloaded(): Promise<DownloadedModel[]> {
  const data = await request<ListResponse<DownloadedModel>>(
    "/api/models/downloaded",
  );
  return data.items;
}

export async function deleteDownloaded(repo: string): Promise<void> {
  await requestUrl<{ ok: boolean }>(
    url("/api/models/downloaded", { repo }),
    { method: "DELETE" },
  );
}

export async function searchHF(q: string, limit?: number): Promise<HFModel[]> {
  const data = await requestUrl<ListResponse<HFModel>>(
    url("/api/models/search", { q, limit }),
  );
  return data.items;
}

export async function listVariants(repo: string): Promise<QuantVariant[]> {
  const data = await requestUrl<ListResponse<QuantVariant>>(
    url("/api/models/variants", { repo }),
  );
  return data.items;
}

export function getMeta(
  repo: string,
  quant?: string,
  revision?: string,
): Promise<ModelMeta> {
  return requestUrl<ModelMeta>(
    url("/api/models/meta", { repo, quant, revision }),
  );
}

// ---------------------------------------------------------------------------
// Estimate
// ---------------------------------------------------------------------------

export function estimate(req: EstimateRequest): Promise<VramEstimate> {
  return request<VramEstimate>("/api/estimate", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

export async function listDownloads(): Promise<DownloadJob[]> {
  const data = await request<ListResponse<DownloadJob>>("/api/downloads");
  return data.items;
}

export function startDownload(req: DownloadRequest): Promise<DownloadJob> {
  return request<DownloadJob>("/api/downloads", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function cancelDownload(id: string): Promise<void> {
  await request<{ ok: boolean }>(`/api/downloads/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export async function removeDownload(id: string): Promise<void> {
  await request<{ ok: boolean }>(`/api/downloads/${encodeURIComponent(id)}/remove`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

export function getEngine(): Promise<EngineStatus> {
  return request<EngineStatus>("/api/engine");
}

export function loadModel(req: LoadRequest): Promise<EngineStatus> {
  return request<EngineStatus>("/api/engine/load", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function unloadModel(): Promise<EngineStatus> {
  return request<EngineStatus>("/api/engine/unload", { method: "POST" });
}

// ---------------------------------------------------------------------------
// Params / settings
// ---------------------------------------------------------------------------

export function getParamSchema(
  repo: string,
  quant?: string,
): Promise<ParamSchema> {
  return requestUrl<ParamSchema>(url("/api/params/schema", { repo, quant }));
}

export function getSettings(): Promise<AppSettings> {
  return request<AppSettings>("/api/settings");
}

export function putSettings(s: AppSettings): Promise<AppSettings> {
  return request<AppSettings>("/api/settings", {
    method: "PUT",
    body: JSON.stringify(s),
  });
}

// ---------------------------------------------------------------------------
// SSE streams (EventSource) — return a close function.
// ---------------------------------------------------------------------------

/** Subscribe to live GPU stats via SSE. Returns a close fn. */
export function streamGpuStats(onStats: (s: GpuStats) => void): () => void {
  const es = new EventSource(`${API_BASE}/api/gpu/stream`);
  es.onmessage = (ev) => {
    if (!ev.data) return;
    try {
      onStats(JSON.parse(ev.data) as GpuStats);
    } catch {
      /* ignore malformed frame */
    }
  };
  return () => es.close();
}

/** Subscribe to engine log lines via SSE. Returns a close fn. */
export function streamEngineLogs(onLine: (l: string) => void): () => void {
  const es = new EventSource(`${API_BASE}/api/engine/logs`);
  es.onmessage = (ev) => {
    if (ev.data === undefined || ev.data === null) return;
    onLine(ev.data);
  };
  return () => es.close();
}

// ---------------------------------------------------------------------------
// Chat streaming (POST + ReadableStream SSE parsing)
// ---------------------------------------------------------------------------

/**
 * Stream a chat completion. POSTs ChatRequest (stream forced true), reads the
 * response body, parses "data: {json}" SSE lines into ChatChunk, and invokes
 * onChunk for each. Respects the AbortSignal. Resolves when the stream ends.
 */
export async function streamChat(
  req: ChatRequest,
  onChunk: (c: ChatChunk) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ ...req, stream: true }),
    signal,
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  if (!res.body) throw new Error("No response body for chat stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handleLine = (rawLine: string): boolean => {
    const line = rawLine.trim();
    if (!line || line.startsWith(":")) return false;
    if (!line.startsWith("data:")) return false;
    const payload = line.slice("data:".length).trim();
    if (!payload) return false;
    if (payload === "[DONE]") {
      onChunk({ delta: "", done: true });
      return true;
    }
    try {
      const chunk = JSON.parse(payload) as ChatChunk;
      onChunk(chunk);
      return chunk.done === true;
    } catch {
      return false;
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by newlines; process complete lines.
      let idx: number;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const rawLine = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 1);
        if (handleLine(rawLine)) {
          buffer = "";
          return;
        }
      }
    }
    // Flush any trailing buffered line.
    if (buffer.length) handleLine(buffer);
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* already closed */
    }
  }
}
