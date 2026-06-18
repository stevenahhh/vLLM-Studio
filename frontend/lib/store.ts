// Zustand store: active config + estimate + live GPU stats + engine status.
// Polls GPU stats (~1.5s), polls engine while loading, and debounces estimate
// recomputation when engine/affects_vram params change.
import { create } from "zustand";
import * as api from "@/lib/api";
import type {
  AppSettings,
  EngineStatus,
  EstimateRequest,
  GpuStats,
  HardwareInfo,
  ModelMeta,
  ParamSchema,
  VramEstimate,
} from "@/lib/types";

const GPU_POLL_MS = 1500;
const ENGINE_POLL_MS = 1000; // poll ~1s while loading for a smooth progress %
const ESTIMATE_DEBOUNCE_MS = 300;
const isBrowser = typeof window !== "undefined";

interface StudioState {
  // data
  hardware: HardwareInfo | null;
  gpuStats: GpuStats | null;
  engine: EngineStatus | null;
  settings: AppSettings | null;
  activeMeta: ModelMeta | null;
  paramSchema: ParamSchema | null;
  paramValues: Record<string, unknown>;
  estimate: VramEstimate | null;

  // loading flags
  initialized: boolean;
  loadingHardware: boolean;
  loadingEngine: boolean;
  estimating: boolean;
  error: string | null;

  // actions
  init: () => Promise<void>;
  refreshGpu: () => Promise<void>;
  refreshEngine: () => Promise<void>;
  setParam: (key: string, value: unknown) => void;
  recomputeEstimate: () => Promise<void>;
  selectModel: (meta: ModelMeta, schema: ParamSchema) => void;
  applySettings: (s: AppSettings) => Promise<void>;
}

// Module-scoped handles (not part of reactive state).
let gpuStop: (() => void) | null = null;
let gpuInterval: ReturnType<typeof setInterval> | null = null;
let enginePoll: ReturnType<typeof setInterval> | null = null;
let estimateTimer: ReturnType<typeof setTimeout> | null = null;
let estimateSeq = 0;

/** Seed a flat paramValues record from a schema's defaults. */
function defaultsFromSchema(schema: ParamSchema): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const group of schema.groups) {
    for (const spec of group.params) {
      values[spec.key] = spec.default;
    }
  }
  return values;
}

/** Read a numeric param with a fallback. */
function num(values: Record<string, unknown>, key: string, fallback: number): number {
  const v = values[key];
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

/** Build an EstimateRequest from active meta + current param values. */
function buildEstimateRequest(
  meta: ModelMeta,
  values: Record<string, unknown>,
): EstimateRequest {
  const quant =
    typeof values.quantization === "string" && values.quantization
      ? (values.quantization as string)
      : meta.quant || "none";
  const dtype =
    typeof values.dtype === "string" && values.dtype
      ? (values.dtype as string)
      : "float16";
  const kvDtype =
    typeof values.kv_cache_dtype === "string" && values.kv_cache_dtype
      ? (values.kv_cache_dtype as string)
      : "auto";
  return {
    meta,
    quant,
    dtype,
    tensor_parallel_size: num(values, "tensor_parallel_size", 1),
    gpu_memory_utilization: num(values, "gpu_memory_utilization", 0.9),
    max_model_len: num(
      values,
      "max_model_len",
      Math.min(meta.max_position_embeddings || 65535, 65535),
    ),
    max_num_seqs: num(values, "max_num_seqs", 16),
    kv_concurrency: num(values, "kv_concurrency", 1),
    kv_cache_dtype: kvDtype,
    enforce_eager: Boolean(values.enforce_eager),
    block_length: num(values, "block_length", 32),
  };
}

/** Start GPU polling (browser only). Prefers SSE; falls back to interval. */
function startGpuPolling(set: (s: GpuStats) => void) {
  if (!isBrowser) return;
  stopGpuPolling();
  if (typeof EventSource !== "undefined") {
    try {
      gpuStop = api.streamGpuStats((s) => set(s));
      return;
    } catch {
      gpuStop = null;
    }
  }
  const tick = async () => {
    try {
      set(await api.getGpuStats());
    } catch {
      /* best-effort telemetry */
    }
  };
  void tick();
  gpuInterval = setInterval(tick, GPU_POLL_MS);
}

function stopGpuPolling() {
  if (gpuStop) {
    gpuStop();
    gpuStop = null;
  }
  if (gpuInterval) {
    clearInterval(gpuInterval);
    gpuInterval = null;
  }
}

export const useStudio = create<StudioState>()((set, get) => {
  const startEnginePoll = () => {
    if (!isBrowser || enginePoll) return;
    enginePoll = setInterval(() => {
      void get().refreshEngine();
    }, ENGINE_POLL_MS);
  };
  const stopEnginePoll = () => {
    if (enginePoll) {
      clearInterval(enginePoll);
      enginePoll = null;
    }
  };

  return {
    hardware: null,
    gpuStats: null,
    engine: null,
    settings: null,
    activeMeta: null,
    paramSchema: null,
    paramValues: {},
    estimate: null,

    initialized: false,
    loadingHardware: false,
    loadingEngine: false,
    estimating: false,
    error: null,

    async init() {
      if (get().initialized) return;
      set({ initialized: true, loadingHardware: true, error: null });
      try {
        const [hardware, settings, engine] = await Promise.all([
          api.getHardware(),
          api.getSettings(),
          api.getEngine(),
        ]);
        set({ hardware, settings, engine, loadingHardware: false });
        if (engine.state === "loading") startEnginePoll();
      } catch (e) {
        set({
          loadingHardware: false,
          error: e instanceof Error ? e.message : String(e),
        });
      }
      if (isBrowser) {
        startGpuPolling((s) => set({ gpuStats: s }));
      }
    },

    async refreshGpu() {
      try {
        set({ gpuStats: await api.getGpuStats() });
      } catch {
        /* best-effort */
      }
    },

    async refreshEngine() {
      try {
        const engine = await api.getEngine();
        set({ engine });
        if (engine.state !== "loading") stopEnginePoll();
        else startEnginePoll();
      } catch (e) {
        set({ error: e instanceof Error ? e.message : String(e) });
      }
    },

    setParam(key, value) {
      const spec = get()
        .paramSchema?.groups.flatMap((g) => g.params)
        .find((p) => p.key === key);
      set({ paramValues: { ...get().paramValues, [key]: value } });
      // Only re-estimate for params that move VRAM (engine + affects_vram).
      const affectsVram = spec
        ? spec.affects_vram || spec.engine_only
        : true;
      if (!affectsVram) return;
      if (estimateTimer) clearTimeout(estimateTimer);
      estimateTimer = setTimeout(() => {
        void get().recomputeEstimate();
      }, ESTIMATE_DEBOUNCE_MS);
    },

    async recomputeEstimate() {
      const { activeMeta, paramValues } = get();
      if (!activeMeta) return;
      const req = buildEstimateRequest(activeMeta, paramValues);
      const seq = ++estimateSeq;
      set({ estimating: true });
      try {
        const estimate = await api.estimate(req);
        if (seq !== estimateSeq) return; // stale; a newer request superseded it
        set({ estimate, estimating: false });
      } catch (e) {
        if (seq !== estimateSeq) return;
        set({
          estimating: false,
          error: e instanceof Error ? e.message : String(e),
        });
      }
    },

    selectModel(meta, schema) {
      set({
        activeMeta: meta,
        paramSchema: schema,
        paramValues: defaultsFromSchema(schema),
        estimate: null,
      });
      void get().recomputeEstimate();
    },

    async applySettings(s) {
      const prev = get().settings;
      set({ settings: s });
      try {
        const saved = await api.putSettings(s);
        set({ settings: saved });
      } catch (e) {
        set({
          settings: prev,
          error: e instanceof Error ? e.message : String(e),
        });
      }
    },
  };
});
