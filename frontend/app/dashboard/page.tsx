"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import {
  Cpu,
  Gpu,
  HardDrive,
  Loader2,
  Power,
  RefreshCw,
  Rocket,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { toast } from "sonner";

import * as api from "@/lib/api";
import { useStudio } from "@/lib/store";
import { fmtCount, fmtGiB } from "@/lib/format";
import { cn } from "@/lib/utils";
import type {
  Capabilities,
  DownloadedModel,
  EngineState,
} from "@/lib/types";

import { GpuMonitor } from "@/components/gpu-monitor";
import { DownloadManager } from "@/components/download-manager";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

// ---------------------------------------------------------------------------
// Engine state -> color dot + label
// ---------------------------------------------------------------------------

const ENGINE_DOT: Record<EngineState, string> = {
  stopped: "bg-muted-foreground",
  loading: "bg-yellow-500 dark:bg-yellow-400",
  ready: "bg-primary",
  error: "bg-destructive",
};

const ENGINE_LABEL: Record<EngineState, string> = {
  stopped: "Stopped",
  loading: "Loading",
  ready: "Ready",
  error: "Error",
};

// ---------------------------------------------------------------------------
// Capability chips derived from hardware capabilities
// ---------------------------------------------------------------------------

interface Chip {
  label: string;
  variant: "default" | "secondary" | "destructive" | "outline";
}

function capabilityChips(caps: Capabilities): Chip[] {
  const chips: Chip[] = [];
  chips.push(
    caps.supports_bf16
      ? { label: "bf16", variant: "secondary" }
      : { label: "no bf16", variant: "destructive" },
  );
  chips.push(
    caps.supports_fp8
      ? { label: "fp8", variant: "secondary" }
      : { label: "no fp8", variant: "destructive" },
  );
  const quants = (caps.supported_quantization ?? [])
    .map((q) => q.toLowerCase())
    .filter((q) => q !== "none");
  if (quants.includes("awq") || quants.includes("gptq")) {
    chips.push({ label: "AWQ/GPTQ ok", variant: "default" });
  }
  if (caps.supports_marlin) {
    chips.push({ label: "Marlin", variant: "secondary" });
  }
  for (const dt of caps.supported_dtypes ?? []) {
    chips.push({
      label: dt === caps.recommended_dtype ? `${dt} (rec)` : dt,
      variant: dt === caps.recommended_dtype ? "default" : "outline",
    });
  }
  return chips;
}

// ---------------------------------------------------------------------------
// Hardware card
// ---------------------------------------------------------------------------

function HardwareCard() {
  const hardware = useStudio((s) => s.hardware);
  const loading = useStudio((s) => s.loadingHardware);

  if (loading && !hardware) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="size-4 text-muted-foreground" />
            Hardware
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-5 w-2/3" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-16 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (!hardware) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="size-4 text-muted-foreground" />
            Hardware
          </CardTitle>
          <CardDescription>Hardware info unavailable.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const gpuName = hardware.gpus[0]?.name ?? "Unknown GPU";
  const cc = hardware.gpus[0]?.compute_capability;
  const chips = capabilityChips(hardware.capabilities);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cpu className="size-4 text-muted-foreground" />
          Hardware
        </CardTitle>
        <CardDescription>
          {hardware.num_gpus}× {gpuName}
          {cc ? ` · sm_${cc.replace(".", "")}` : ""}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          <dt className="text-muted-foreground">GPUs</dt>
          <dd className="text-right font-medium tabular-nums">
            {hardware.num_gpus}
          </dd>
          <dt className="text-muted-foreground">Per-GPU VRAM</dt>
          <dd className="text-right font-medium tabular-nums">
            {fmtGiB(hardware.per_gpu_bytes)}
          </dd>
          <dt className="text-muted-foreground">Total VRAM</dt>
          <dd className="text-right font-medium tabular-nums">
            {fmtGiB(hardware.total_bytes)}
          </dd>
        </dl>

        <div className="flex flex-wrap gap-1.5">
          {chips.map((chip, i) => (
            <Badge key={`${chip.label}-${i}`} variant={chip.variant}>
              {chip.label}
            </Badge>
          ))}
        </div>

        {hardware.capabilities.notes?.length ? (
          <ul className="space-y-1 text-xs text-muted-foreground">
            {hardware.capabilities.notes.map((note, i) => (
              <li key={i} className="flex gap-1.5">
                <span aria-hidden className="text-muted-foreground/60">
                  ·
                </span>
                <span>{note}</span>
              </li>
            ))}
          </ul>
        ) : null}

        {hardware.recommendations?.length ? (
          <>
            <Separator />
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-foreground">
                Recommendations
              </p>
              <ul className="space-y-1 text-xs text-muted-foreground">
                {hardware.recommendations.map((rec, i) => (
                  <li key={i} className="flex gap-1.5">
                    <Rocket
                      aria-hidden
                      className="mt-0.5 size-3 shrink-0 text-primary"
                    />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Live GPUs card (full width)
// ---------------------------------------------------------------------------

function LiveGpusCard() {
  const gpuStats = useStudio((s) => s.gpuStats);
  const estimate = useStudio((s) => s.estimate);

  return (
    <Card className="md:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Gpu className="size-4 text-muted-foreground" />
          Live GPUs
        </CardTitle>
        <CardDescription>
          {gpuStats
            ? `${fmtGiB(gpuStats.used_bytes)} / ${fmtGiB(
                gpuStats.total_bytes,
              )} VRAM in use`
            : "Waiting for telemetry…"}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {gpuStats && gpuStats.gpus.length ? (
          <GpuMonitor gpus={gpuStats.gpus} breakdown={estimate?.per_gpu ?? null} />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Engine card
// ---------------------------------------------------------------------------

function EngineCard() {
  const engine = useStudio((s) => s.engine);
  const refreshEngine = useStudio((s) => s.refreshEngine);
  const [unloading, setUnloading] = React.useState(false);

  const state: EngineState = engine?.state ?? "stopped";
  const isActive = state === "ready" || state === "loading";

  const onUnload = async () => {
    setUnloading(true);
    try {
      await api.unloadModel();
      await refreshEngine();
      toast.success("Engine unloaded", {
        description: "VRAM has been freed.",
      });
    } catch (e) {
      toast.error("Failed to unload", {
        description: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setUnloading(false);
    }
  };

  const load = engine?.load_request;
  const loadArgs: Array<{ label: string; value: string }> = [];
  if (load) {
    if (load.dtype) loadArgs.push({ label: "dtype", value: load.dtype });
    if (load.tensor_parallel_size)
      loadArgs.push({
        label: "tp",
        value: String(load.tensor_parallel_size),
      });
    if (typeof load.gpu_memory_utilization === "number")
      loadArgs.push({
        label: "util",
        value: load.gpu_memory_utilization.toFixed(2),
      });
    if (load.max_model_len)
      loadArgs.push({
        label: "ctx",
        value: fmtCount(load.max_model_len),
      });
    if (load.max_num_seqs)
      loadArgs.push({ label: "seqs", value: String(load.max_num_seqs) });
    if (load.kv_cache_dtype)
      loadArgs.push({ label: "kv", value: load.kv_cache_dtype });
    if (load.enforce_eager) loadArgs.push({ label: "eager", value: "on" });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span
            className={cn(
              "inline-block size-2.5 rounded-full",
              ENGINE_DOT[state],
              state === "loading" && "animate-pulse",
            )}
            aria-hidden
          />
          Engine
        </CardTitle>
        <CardDescription>{ENGINE_LABEL[state]}</CardDescription>
        <CardAction>
          {isActive ? (
            <Button
              variant="destructive"
              size="sm"
              onClick={onUnload}
              disabled={unloading}
            >
              {unloading ? (
                <Loader2 className="animate-spin" />
              ) : (
                <Power />
              )}
              Unload
            </Button>
          ) : null}
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-3">
        {engine && engine.repo ? (
          <>
            <div className="min-w-0">
              <p className="truncate font-medium text-foreground">
                {engine.repo}
              </p>
              <p className="text-xs text-muted-foreground">
                {engine.quant && engine.quant !== "none"
                  ? engine.quant
                  : "unquantized"}
                {engine.revision && engine.revision !== "main"
                  ? ` · ${engine.revision}`
                  : ""}
                {engine.family ? ` · ${engine.family}` : ""}
              </p>
            </div>

            {loadArgs.length ? (
              <div className="flex flex-wrap gap-1.5">
                {loadArgs.map((a) => (
                  <Badge key={a.label} variant="outline">
                    <span className="text-muted-foreground">{a.label}</span>
                    <span className="font-medium">{a.value}</span>
                  </Badge>
                ))}
              </div>
            ) : null}

            {state === "error" && engine.error ? (
              <p className="flex items-start gap-1.5 text-xs text-destructive">
                <TriangleAlert className="mt-0.5 size-3 shrink-0" />
                <span className="break-words">{engine.error}</span>
              </p>
            ) : null}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            No model loaded. Pick a model to spawn the vLLM engine.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Downloads card
// ---------------------------------------------------------------------------

function DownloadsCard() {
  return (
    <Card className="md:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HardDrive className="size-4 text-muted-foreground" />
          Downloads
        </CardTitle>
        <CardDescription>Active and finished downloads.</CardDescription>
      </CardHeader>
      <CardContent>
        <DownloadManager />
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Downloaded models card (table + Load action)
// ---------------------------------------------------------------------------

function DownloadedModelsCard() {
  const router = useRouter();
  const selectModel = useStudio((s) => s.selectModel);
  const [models, setModels] = React.useState<DownloadedModel[] | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [refreshing, setRefreshing] = React.useState(false);
  const [loadingRepo, setLoadingRepo] = React.useState<string | null>(null);
  const [deletingRepo, setDeletingRepo] = React.useState<string | null>(null);

  const refresh = React.useCallback(async () => {
    setRefreshing(true);
    try {
      const items = await api.listDownloaded();
      setModels(items);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRefreshing(false);
    }
  }, []);

  React.useEffect(() => {
    queueMicrotask(() => void refresh());
  }, [refresh]);

  const onDelete = async (m: DownloadedModel) => {
    setDeletingRepo(m.repo);
    try {
      await api.deleteDownloaded(m.repo);
      toast.success("Model deleted", { description: m.repo });
      await refresh();
    } catch (e) {
      toast.error("Failed to delete model", {
        description: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setDeletingRepo(null);
    }
  };

  const onLoad = async (m: DownloadedModel) => {
    const key = `${m.repo}@${m.revision}`;
    setLoadingRepo(key);
    try {
      const [meta, schema] = await Promise.all([
        api.getMeta(m.repo, m.quant, m.revision),
        api.getParamSchema(m.repo, m.quant),
      ]);
      selectModel(meta, schema);
      toast.success("Model selected", {
        description: `${m.repo} — tune params, then Load in the engine.`,
      });
      router.push("/chat");
    } catch (e) {
      toast.error("Failed to open model", {
        description: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setLoadingRepo(null);
    }
  };

  return (
    <Card className="md:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HardDrive className="size-4 text-muted-foreground" />
          Downloaded models
        </CardTitle>
        <CardDescription>
          {models ? `${models.length} model(s) in local cache` : "Loading…"}
        </CardDescription>
        <CardAction>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => void refresh()}
            disabled={refreshing}
            aria-label="Refresh downloaded models"
          >
            <RefreshCw className={cn(refreshing && "animate-spin")} />
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {error ? (
          <p className="flex items-center gap-1.5 text-sm text-destructive">
            <TriangleAlert className="size-4" />
            {error}
          </p>
        ) : models === null ? (
          <div className="space-y-2">
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
          </div>
        ) : models.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No models downloaded yet. Search Hugging Face to fetch one.
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Repo</TableHead>
                <TableHead>Quant</TableHead>
                <TableHead>Family</TableHead>
                <TableHead className="text-right">Size</TableHead>
                <TableHead className="text-right">Action</TableHead>
                <TableHead className="w-8" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {models.map((m) => {
                const key = `${m.repo}@${m.revision}`;
                const busy = loadingRepo === key;
                const deleting = deletingRepo === m.repo;
                return (
                  <TableRow key={key}>
                    <TableCell className="max-w-[18rem]">
                      <span className="block truncate font-medium text-foreground">
                        {m.repo}
                      </span>
                      {m.revision && m.revision !== "main" ? (
                        <span className="block truncate text-xs text-muted-foreground">
                          {m.revision}
                        </span>
                      ) : null}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          m.quant && m.quant !== "none"
                            ? "secondary"
                            : "outline"
                        }
                      >
                        {m.quant && m.quant !== "none" ? m.quant : "fp16"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {m.family}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {fmtGiB(m.size_bytes)}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => void onLoad(m)}
                        disabled={busy || deleting || !m.has_config}
                        title={
                          m.has_config
                            ? undefined
                            : "Missing config.json — cannot load"
                        }
                      >
                        {busy ? (
                          <Loader2 className="animate-spin" />
                        ) : (
                          <Rocket />
                        )}
                        Load
                      </Button>
                    </TableCell>
                    <TableCell>
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        className="text-muted-foreground hover:text-destructive"
                        title="Delete model from cache"
                        disabled={deleting || busy}
                        onClick={() => void onDelete(m)}
                      >
                        {deleting ? (
                          <Loader2 className="animate-spin" />
                        ) : (
                          <Trash2 className="size-3.5" />
                        )}
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const init = useStudio((s) => s.init);

  React.useEffect(() => {
    void init();
  }, [init]);

  return (
    <div className="h-full min-h-0 flex-1 space-y-6 overflow-y-auto p-6">
      <div className="space-y-0.5">
        <h1 className="font-heading text-2xl font-semibold text-foreground">
          Dashboard
        </h1>
        <p className="text-sm text-muted-foreground">
          Hardware, live GPU telemetry, engine status, and local models.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <HardwareCard />
        <EngineCard />
        <DownloadsCard />
        <LiveGpusCard />
        <DownloadedModelsCard />
      </div>
    </div>
  );
}
