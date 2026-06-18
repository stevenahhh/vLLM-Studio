"use client";

import * as React from "react";
import {
  Activity,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Cpu,
  Loader2,
  Play,
  Square,
  Trophy,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import { API_BASE } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types (mirrors backend TunerConfig / TrialResult / TunerStatus)
// ---------------------------------------------------------------------------

interface TunerConfig {
  repo: string;
  revision: string;
  quant: string;
  dtype: string;
  tensor_parallel_size: number;
  extra_args: string[];
  throughput_weight: number;
  latency_weight: number;
  memory_weight: number;
  gpu_memory_utilization_min: number;
  gpu_memory_utilization_max: number;
  max_num_seqs_min: number;
  max_num_seqs_max: number;
  max_num_batched_tokens_min: number;
  max_num_batched_tokens_max: number;
  max_model_len: number;
  n_trials: number;
  timeout_minutes: number;
  concurrent_requests: number;
  requests_per_trial: number;
  max_tokens: number;
  prompts: string[];
}

interface TrialResult {
  trial_id: number;
  state: "completed" | "failed" | "pruned";
  params: Record<string, number>;
  throughput: number;
  avg_latency_ms: number;
  p99_latency_ms: number;
  memory_util: number;
  score: number;
  error: string;
}

interface TunerStatus {
  state: "idle" | "running" | "done" | "stopped" | "error";
  current_trial: number;
  total_trials: number;
  trials: TrialResult[];
  best_params: Record<string, number> | null;
  best_score: number;
  elapsed_seconds: number;
  error: string;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function apiStart(cfg: TunerConfig): Promise<TunerStatus> {
  const r = await fetch(`${API_BASE}/api/tuner/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error((e as { detail?: string }).detail ?? r.statusText);
  }
  return r.json();
}

async function apiStop(): Promise<TunerStatus> {
  const r = await fetch(`${API_BASE}/api/tuner/stop`, { method: "POST" });
  return r.json();
}

async function apiStatus(): Promise<TunerStatus> {
  const r = await fetch(`${API_BASE}/api/tuner/status`);
  return r.json();
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function StatBadge({ label, value, unit = "" }: { label: string; value: string | number; unit?: string }) {
  return (
    <div className="flex flex-col gap-0.5 rounded-lg border border-border bg-muted/40 px-3 py-2 text-center">
      <span className="text-[0.7rem] text-muted-foreground">{label}</span>
      <span className="text-sm font-semibold tabular-nums">
        {value}
        {unit && <span className="ml-0.5 text-xs font-normal text-muted-foreground">{unit}</span>}
      </span>
    </div>
  );
}

function TrialRow({ trial, rank }: { trial: TrialResult; rank?: number }) {
  const [open, setOpen] = React.useState(false);
  const isOk = trial.state === "completed";

  return (
    <div className="flex flex-col rounded-md border border-border text-xs">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 px-3 py-2 text-left hover:bg-muted/40"
      >
        {isOk ? (
          <CheckCircle className="size-3.5 shrink-0 text-green-500" />
        ) : (
          <XCircle className="size-3.5 shrink-0 text-destructive" />
        )}
        <span className="w-12 shrink-0 tabular-nums text-muted-foreground">
          #{trial.trial_id}
        </span>
        {rank === 0 && (
          <Trophy className="size-3 shrink-0 text-amber-400" />
        )}
        <span className="flex-1 tabular-nums">
          {isOk ? (
            <>
              <span className="text-green-600 dark:text-green-400">
                {trial.throughput.toFixed(2)} req/s
              </span>
              {" · "}
              <span>{trial.avg_latency_ms.toFixed(0)} ms avg</span>
            </>
          ) : (
            <span className="text-destructive">{trial.state}</span>
          )}
        </span>
        <span className="shrink-0 font-medium">score {trial.score.toFixed(3)}</span>
        {open ? (
          <ChevronUp className="size-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronDown className="size-3 shrink-0 text-muted-foreground" />
        )}
      </button>
      {open && (
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 border-t border-border px-3 py-2 tabular-nums sm:grid-cols-3">
          {Object.entries(trial.params).map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2">
              <span className="text-muted-foreground">{k}</span>
              <span className="font-medium">{typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : v}</span>
            </div>
          ))}
          <div className="flex justify-between gap-2">
            <span className="text-muted-foreground">p99 latency</span>
            <span className="font-medium">{trial.p99_latency_ms.toFixed(0)} ms</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-muted-foreground">mem util</span>
            <span className="font-medium">{(trial.memory_util * 100).toFixed(1)}%</span>
          </div>
          {trial.error && (
            <div className="col-span-full text-destructive">{trial.error}</div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Weight slider row (3 weights that sum to 100)
// ---------------------------------------------------------------------------

function WeightRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 text-xs">{label}</span>
      <Slider
        min={0}
        max={100}
        step={5}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        className="flex-1"
      />
      <span className="w-8 text-right text-xs tabular-nums">{value}%</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

const DEFAULT_CFG: TunerConfig = {
  repo: "",
  revision: "main",
  quant: "none",
  dtype: "float16",
  tensor_parallel_size: 8,
  extra_args: [],
  throughput_weight: 60,
  latency_weight: 30,
  memory_weight: 10,
  gpu_memory_utilization_min: 0.70,
  gpu_memory_utilization_max: 0.95,
  max_num_seqs_min: 4,
  max_num_seqs_max: 128,
  max_num_batched_tokens_min: 2048,
  max_num_batched_tokens_max: 32768,
  max_model_len: 8192,
  n_trials: 20,
  timeout_minutes: 120,
  concurrent_requests: 10,
  requests_per_trial: 50,
  max_tokens: 256,
  prompts: [],
};

export default function TunerPage() {
  const [cfg, setCfg] = React.useState<TunerConfig>(DEFAULT_CFG);
  const [status, setStatus] = React.useState<TunerStatus | null>(null);
  const [loading, setLoading] = React.useState(false);
  const pollRef = React.useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll while running
  React.useEffect(() => {
    const fetchStatus = async () => {
      try {
        const s = await apiStatus();
        setStatus(s);
        if (s.state !== "running") stopPoll();
      } catch { /* ignore */ }
    };

    fetchStatus();
  }, []);

  function startPoll() {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const s = await apiStatus();
        setStatus(s);
        if (s.state !== "running") stopPoll();
      } catch { /* ignore */ }
    }, 3000);
  }

  function stopPoll() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  React.useEffect(() => () => stopPoll(), []);

  function setWeight(key: "throughput_weight" | "latency_weight" | "memory_weight", val: number) {
    setCfg((c) => ({ ...c, [key]: val }));
  }

  const weightSum = cfg.throughput_weight + cfg.latency_weight + cfg.memory_weight;

  async function handleStart() {
    if (!cfg.repo.trim()) {
      toast.error("Model repo required", { description: "Enter e.g. Jackrong/Qwopus3.5-9B-Coder" });
      return;
    }
    if (weightSum !== 100) {
      toast.error("Objective weights must sum to 100", { description: `Current sum: ${weightSum}` });
      return;
    }
    setLoading(true);
    try {
      const s = await apiStart(cfg);
      setStatus(s);
      startPoll();
      toast.success("Tuning study started", { description: `${cfg.n_trials} trials · ${cfg.timeout_minutes} min timeout` });
    } catch (e) {
      toast.error("Failed to start tuner", { description: String(e) });
    } finally {
      setLoading(false);
    }
  }

  async function handleStop() {
    stopPoll();
    const s = await apiStop();
    setStatus(s);
    toast.info("Tuner stopped");
  }

  const isRunning = status?.state === "running";
  const sortedTrials = [...(status?.trials ?? [])].sort((a, b) => b.score - a.score);
  const progress = status
    ? Math.round((status.current_trial / Math.max(1, status.total_trials)) * 100)
    : 0;

  return (
    <div className="container mx-auto max-w-4xl space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Activity className="size-6 text-primary" />
        <div>
          <h1 className="text-2xl font-bold">vLLM Tuner</h1>
          <p className="text-sm text-muted-foreground">
            Bayesian optimisation (Optuna) for throughput · latency · memory
          </p>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Config */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Configuration</CardTitle>
            <CardDescription>Model and search-space settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Model */}
            <div className="space-y-1.5">
              <Label className="text-xs">Model repo</Label>
              <Input
                value={cfg.repo}
                onChange={(e) => setCfg((c) => ({ ...c, repo: e.target.value }))}
                placeholder="Jackrong/Qwopus3.5-9B-Coder"
                className="h-8 text-sm"
                disabled={isRunning}
              />
            </div>

            {/* TP + max_model_len */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Tensor parallel</Label>
                <Input
                  type="number"
                  min={1}
                  max={8}
                  value={cfg.tensor_parallel_size}
                  onChange={(e) => setCfg((c) => ({ ...c, tensor_parallel_size: Number(e.target.value) }))}
                  className="h-8 text-sm"
                  disabled={isRunning}
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Max model len</Label>
                <Input
                  type="number"
                  min={256}
                  step={256}
                  value={cfg.max_model_len}
                  onChange={(e) => setCfg((c) => ({ ...c, max_model_len: Number(e.target.value) }))}
                  className="h-8 text-sm"
                  disabled={isRunning}
                />
              </div>
            </div>

            {/* Trials + timeout */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Trials</Label>
                <Input
                  type="number"
                  min={1}
                  value={cfg.n_trials}
                  onChange={(e) => setCfg((c) => ({ ...c, n_trials: Number(e.target.value) }))}
                  className="h-8 text-sm"
                  disabled={isRunning}
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Timeout (min)</Label>
                <Input
                  type="number"
                  min={5}
                  value={cfg.timeout_minutes}
                  onChange={(e) => setCfg((c) => ({ ...c, timeout_minutes: Number(e.target.value) }))}
                  className="h-8 text-sm"
                  disabled={isRunning}
                />
              </div>
            </div>

            {/* Benchmark */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Concurrent requests</Label>
                <Input
                  type="number"
                  min={1}
                  value={cfg.concurrent_requests}
                  onChange={(e) => setCfg((c) => ({ ...c, concurrent_requests: Number(e.target.value) }))}
                  className="h-8 text-sm"
                  disabled={isRunning}
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Requests / trial</Label>
                <Input
                  type="number"
                  min={5}
                  value={cfg.requests_per_trial}
                  onChange={(e) => setCfg((c) => ({ ...c, requests_per_trial: Number(e.target.value) }))}
                  className="h-8 text-sm"
                  disabled={isRunning}
                />
              </div>
            </div>

            {/* Objectives */}
            <div className="space-y-2">
              <Label className="text-xs">
                Objective weights{" "}
                <span className={cn("font-medium tabular-nums", weightSum !== 100 && "text-destructive")}>
                  (sum: {weightSum}/100)
                </span>
              </Label>
              <WeightRow label="Throughput" value={cfg.throughput_weight} onChange={(v) => setWeight("throughput_weight", v)} />
              <WeightRow label="Latency" value={cfg.latency_weight} onChange={(v) => setWeight("latency_weight", v)} />
              <WeightRow label="Memory" value={cfg.memory_weight} onChange={(v) => setWeight("memory_weight", v)} />
            </div>

            {/* Search space */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Search space</Label>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="space-y-1">
                  <span className="text-muted-foreground">GPU mem util</span>
                  <div className="flex gap-1">
                    <Input type="number" step={0.01} min={0.5} max={1} value={cfg.gpu_memory_utilization_min}
                      onChange={(e) => setCfg((c) => ({ ...c, gpu_memory_utilization_min: Number(e.target.value) }))}
                      className="h-7 text-xs" disabled={isRunning} />
                    <span className="self-center text-muted-foreground">–</span>
                    <Input type="number" step={0.01} min={0.5} max={1} value={cfg.gpu_memory_utilization_max}
                      onChange={(e) => setCfg((c) => ({ ...c, gpu_memory_utilization_max: Number(e.target.value) }))}
                      className="h-7 text-xs" disabled={isRunning} />
                  </div>
                </div>
                <div className="space-y-1">
                  <span className="text-muted-foreground">Max num seqs</span>
                  <div className="flex gap-1">
                    <Input type="number" min={1} value={cfg.max_num_seqs_min}
                      onChange={(e) => setCfg((c) => ({ ...c, max_num_seqs_min: Number(e.target.value) }))}
                      className="h-7 text-xs" disabled={isRunning} />
                    <span className="self-center text-muted-foreground">–</span>
                    <Input type="number" min={1} value={cfg.max_num_seqs_max}
                      onChange={(e) => setCfg((c) => ({ ...c, max_num_seqs_max: Number(e.target.value) }))}
                      className="h-7 text-xs" disabled={isRunning} />
                  </div>
                </div>
              </div>
            </div>

            {/* Start / Stop */}
            <div className="flex gap-2 pt-1">
              {isRunning ? (
                <Button variant="destructive" size="sm" onClick={handleStop} className="flex-1 gap-1.5">
                  <Square className="size-3.5" /> Stop
                </Button>
              ) : (
                <Button size="sm" onClick={handleStart} disabled={loading} className="flex-1 gap-1.5">
                  {loading ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
                  {loading ? "Starting…" : "Start tuning"}
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Status */}
        <div className="flex flex-col gap-4">
          {/* Progress */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Cpu className="size-4" />
                Study progress
                {isRunning && <Badge variant="secondary" className="ml-auto animate-pulse">Running</Badge>}
                {status?.state === "done" && <Badge className="ml-auto bg-green-500/20 text-green-600">Done</Badge>}
                {status?.state === "stopped" && <Badge variant="secondary" className="ml-auto">Stopped</Badge>}
                {status?.state === "error" && <Badge variant="destructive" className="ml-auto">Error</Badge>}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {status ? (
                <>
                  <Progress value={progress} className="h-2" />
                  <div className="grid grid-cols-3 gap-2">
                    <StatBadge label="Trial" value={`${status.current_trial}/${status.total_trials}`} />
                    <StatBadge label="Elapsed" value={Math.round(status.elapsed_seconds / 60)} unit="min" />
                    <StatBadge label="Best score" value={status.best_score.toFixed(3)} />
                  </div>
                  {status.error && (
                    <p className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">{status.error}</p>
                  )}
                </>
              ) : (
                <p className="text-xs text-muted-foreground">No study running yet.</p>
              )}
            </CardContent>
          </Card>

          {/* Best params */}
          {status?.best_params && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Trophy className="size-4 text-amber-400" />
                  Best parameters
                </CardTitle>
                <CardDescription>score {status.best_score.toFixed(3)}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 gap-1 text-xs">
                  {Object.entries(status.best_params).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2 rounded-sm px-2 py-1 odd:bg-muted/30">
                      <span className="text-muted-foreground">{k}</span>
                      <span className="font-mono font-medium">
                        {typeof v === "number" && !Number.isInteger(v) ? v.toFixed(4) : v}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Trial results */}
      {sortedTrials.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              Trial results{" "}
              <span className="text-sm font-normal text-muted-foreground">
                ({sortedTrials.length} completed, sorted by score)
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5 max-h-[28rem] overflow-y-auto pr-1">
            {sortedTrials.map((t, i) => (
              <TrialRow key={t.trial_id} trial={t} rank={i} />
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
