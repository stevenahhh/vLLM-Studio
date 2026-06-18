"use client";

import * as React from "react";
import {
  ArrowLeft,
  Cpu,
  Download,
  Gauge,
  HardDrive,
  Heart,
  Loader2,
  Lock,
  Search,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { OomBadge } from "@/components/oom-badge";
import { StackedVramBar, VRAM_COLORS } from "@/components/vram-bar";
import * as api from "@/lib/api";
import { FIXED_SYSTEM_PROMPT } from "@/lib/fixed-system-prompt";
import {
  fmtCount,
  fmtGiB,
  statusColor,
  suggestTensorParallel,
} from "@/lib/format";
import { useStudio } from "@/lib/store";
import { cn } from "@/lib/utils";
import type {
  DownloadedModel,
  HFModel,
  LoadRequest,
  ParamSpec,
  QuantVariant,
  VramEstimate,
} from "@/lib/types";

interface ModelPickerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function ModelPicker({ open, onOpenChange }: ModelPickerProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[85vh] w-full max-w-3xl flex-col gap-4 overflow-hidden sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Select a model</DialogTitle>
          <DialogDescription>
            Load a downloaded model or browse Hugging Face. VRAM fit is checked
            before you load.
          </DialogDescription>
        </DialogHeader>

        <Tabs
          defaultValue="downloaded"
          className="flex min-h-0 flex-1 flex-col"
        >
          <TabsList className="w-full">
            <TabsTrigger value="downloaded">
              <HardDrive aria-hidden />
              Downloaded
            </TabsTrigger>
            <TabsTrigger value="huggingface">
              <Search aria-hidden />
              Hugging Face
            </TabsTrigger>
          </TabsList>

          <TabsContent value="downloaded" className="mt-1 min-h-0 flex-1">
            <DownloadedTab open={open} onLoaded={() => onOpenChange(false)} />
          </TabsContent>

          <TabsContent value="huggingface" className="mt-1 min-h-0 flex-1">
            <HuggingFaceTab open={open} />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}

function DownloadedTab({
  open,
  onLoaded,
}: {
  open: boolean;
  onLoaded: () => void;
}) {
  const [items, setItems] = React.useState<DownloadedModel[] | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [selected, setSelected] = React.useState<DownloadedModel | null>(null);
  const [deleting, setDeleting] = React.useState<Set<string>>(new Set());

  function refresh() {
    setLoading(true);
    api.listDownloaded()
      .then((data) => setItems(data))
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }

  React.useEffect(() => {
    if (!open) {
      queueMicrotask(() => setSelected(null));
      return;
    }
    let cancelled = false;
    queueMicrotask(() => setLoading(true));
    api
      .listDownloaded()
      .then((data) => {
        if (!cancelled) setItems(data);
      })
      .catch((e) => {
        if (!cancelled) {
          setItems([]);
          toast.error("Failed to list downloaded models", {
            description: errMsg(e),
          });
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  async function handleDelete(repo: string) {
    setDeleting((prev) => new Set(prev).add(repo));
    try {
      await api.deleteDownloaded(repo);
      toast.success("Model deleted", { description: repo });
      refresh();
    } catch (e) {
      toast.error("Failed to delete model", { description: errMsg(e) });
    } finally {
      setDeleting((prev) => {
        const next = new Set(prev);
        next.delete(repo);
        return next;
      });
    }
  }

  if (selected) {
    return (
      <ConfigureLoad
        model={selected}
        onBack={() => setSelected(null)}
        onLoaded={onLoaded}
      />
    );
  }

  if (loading || items === null) {
    return <RowSkeletons rows={4} />;
  }

  if (items.length === 0) {
    return (
      <EmptyState
        icon={<HardDrive aria-hidden className="size-6" />}
        title="No downloaded models"
        hint="Switch to the Hugging Face tab to find and download a model."
      />
    );
  }

  return (
    <ScrollArea className="h-[22rem] pr-3">
      <div className="flex flex-col gap-2">
        {items.map((m) => (
          <div
            key={`${m.repo}@${m.revision}:${m.quant}`}
            className="flex items-center gap-3 rounded-2xl border border-border bg-card/40 px-3 py-2.5"
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate font-medium">{m.repo}</span>
                <Badge variant="secondary">{m.quant || "none"}</Badge>
              </div>
              <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                <span>{fmtGiB(m.size_bytes)}</span>
                <span aria-hidden>·</span>
                <span className="capitalize">{m.family}</span>
                {m.model_type ? (
                  <>
                    <span aria-hidden>·</span>
                    <span>{m.model_type}</span>
                  </>
                ) : null}
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              <Button size="sm" onClick={() => setSelected(m)}>
                <SlidersHorizontal aria-hidden />
                Configure &amp; Load
              </Button>
              <Button
                size="icon-sm"
                variant="ghost"
                className="text-muted-foreground hover:text-destructive"
                title="Delete model from cache"
                disabled={deleting.has(m.repo)}
                onClick={() => handleDelete(m.repo)}
              >
                {deleting.has(m.repo) ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Trash2 className="size-3.5" />
                )}
              </Button>
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}

const EXTRA_FLAG_OPTIONS: {
  id: string;
  flag: string;
  label: string;
  tooltip: string;
  defaultOn?: boolean;
}[] = [
  {
    id: "enable-prefix-caching",
    flag: "--enable-prefix-caching",
    label: "Prefix caching ⚡",
    tooltip:
      "KV 캐시 블록을 공유 프리픽스(시스템 프롬프트, few-shot 예시)에 재사용. 같은 시스템 프롬프트로 반복 요청 시 TTFT를 크게 줄여 줌. 추론 속도 향상에 권장.",
    defaultOn: true,
  },
  {
    id: "triton-attn",
    flag: "--attention-backend TRITON_ATTN",
    label: "TRITON_ATTN 백엔드",
    tooltip:
      "어텐션 커널을 FlashInfer에서 Triton으로 교체. head_dim=256 모델(예: Qwen3.5)에서 FlashInfer가 'invalid argument'로 뻗을 때 필수. FlashInfer 대비 처리량이 약간 낮아짐.",
  },
  {
    id: "disable-custom-allreduce",
    flag: "--disable-custom-all-reduce",
    label: "NCCL all-reduce 사용",
    tooltip:
      "vLLM의 커스텀 all-reduce 커널 대신 NCCL 사용. NVLink 없는 PCIe 멀티-GPU 환경에서 커스텀 커널이 데드락이나 오류를 낼 때 켜면 안정적. 약간의 통신 오버헤드 증가.",
  },
  {
    id: "trust-remote",
    flag: "--trust-remote-code",
    label: "Trust remote code",
    tooltip:
      "모델 로딩 시 repo의 커스텀 Python 코드 실행 허용(토크나이저, 모델 클래스). 일부 커뮤니티 모델에 필수. 신뢰하는 repo에만 사용.",
  },
];

function ExtraArgsCheckboxes({
  flags,
  onChange,
}: {
  flags: Set<string>;
  onChange: (s: Set<string>) => void;
}) {
  function toggle(flag: string) {
    const next = new Set(flags);
    if (next.has(flag)) next.delete(flag);
    else next.add(flag);
    onChange(next);
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-muted-foreground">
          Extra options
        </span>
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          {EXTRA_FLAG_OPTIONS.map((opt) => (
            <div key={opt.id} className="flex items-center gap-2">
              <Checkbox
                id={opt.id}
                checked={flags.has(opt.flag)}
                onCheckedChange={() => toggle(opt.flag)}
              />
              <Tooltip>
                <TooltipTrigger asChild>
                  <label
                    htmlFor={opt.id}
                    className="cursor-pointer select-none text-xs leading-none underline decoration-dotted decoration-muted-foreground/50 underline-offset-2"
                  >
                    {opt.label}
                  </label>
                </TooltipTrigger>
                <TooltipContent
                  side="top"
                  className="max-w-[260px] text-xs leading-relaxed"
                >
                  {opt.tooltip}
                </TooltipContent>
              </Tooltip>
            </div>
          ))}
        </div>
      </div>
    </TooltipProvider>
  );
}

function ConfigureLoad({
  model,
  onBack,
  onLoaded,
}: {
  model: DownloadedModel;
  onBack: () => void;
  onLoaded: () => void;
}) {
  const selectModel = useStudio((s) => s.selectModel);
  const setParam = useStudio((s) => s.setParam);
  const recomputeEstimate = useStudio((s) => s.recomputeEstimate);
  const refreshEngine = useStudio((s) => s.refreshEngine);
  const paramValues = useStudio((s) => s.paramValues);
  const paramSchema = useStudio((s) => s.paramSchema);
  const estimate = useStudio((s) => s.estimate);
  const estimating = useStudio((s) => s.estimating);
  const settings = useStudio((s) => s.settings);

  const [prepping, setPrepping] = React.useState(true);
  const [prepError, setPrepError] = React.useState<string | null>(null);
  const [loadingModel, setLoadingModel] = React.useState(false);
  const [useCustomSystemPrompt, setUseCustomSystemPrompt] = React.useState(false);
  const [customSystemPrompt, setCustomSystemPrompt] = React.useState("");
  const [extraFlags, setExtraFlags] = React.useState<Set<string>>(
    () => new Set(EXTRA_FLAG_OPTIONS.filter((o) => o.defaultOn).map((o) => o.flag)),
  );

  React.useEffect(() => {
    let cancelled = false;
    queueMicrotask(() => {
      setPrepping(true);
      setPrepError(null);
    });
    Promise.all([
      api.getMeta(model.repo, model.quant, model.revision),
      api.getParamSchema(model.repo, model.quant),
    ])
      .then(([meta, schema]) => {
        if (cancelled) return;
        selectModel(meta, schema);
        void recomputeEstimate();
      })
      .catch((e) => {
        if (cancelled) return;
        setPrepError(errMsg(e));
        toast.error("Failed to load model metadata", {
          description: errMsg(e),
        });
      })
      .finally(() => {
        if (!cancelled) setPrepping(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model.repo, model.quant, model.revision]);

  const engineSpecs = React.useMemo(() => {
    const map = new Map<string, ParamSpec>();
    const group = paramSchema?.groups.find((g) => g.key === "engine");
    if (group) {
      for (const p of group.params) map.set(p.key, p);
    }
    return map;
  }, [paramSchema]);

  async function handleLoad() {
    setLoadingModel(true);
    try {
      const req: LoadRequest = {
        repo: model.repo,
        revision: model.revision,
        quant:
          (paramValues.quantization as string | undefined) ||
          model.quant ||
          undefined,
        dtype: (paramValues.dtype as string | undefined) || undefined,
        tensor_parallel_size: numOr(paramValues.tensor_parallel_size, 1),
        gpu_memory_utilization: numOr(
          paramValues.gpu_memory_utilization,
          0.9,
        ),
        max_model_len: numOr(paramValues.max_model_len, 8192),
        max_num_seqs: numOr(paramValues.max_num_seqs, 16),
        kv_cache_dtype:
          (paramValues.kv_cache_dtype as string | undefined) || undefined,
        enforce_eager: Boolean(paramValues.enforce_eager),
        trust_remote_code: Boolean(paramValues.trust_remote_code),
        extra_args: Array.from(extraFlags).flatMap((f) => f.split(" ")),
      };
      const trimmedCustomSystemPrompt = customSystemPrompt.trim();
      const promptToSave =
        useCustomSystemPrompt && trimmedCustomSystemPrompt
          ? trimmedCustomSystemPrompt
          : FIXED_SYSTEM_PROMPT;
      const currentSettings = settings ?? (await api.getSettings());
      const savedSettings = await api.putSettings({
        ...currentSettings,
        system_prompt: promptToSave,
      });
      useStudio.setState({ settings: savedSettings });
      await api.loadModel(req);
      void refreshEngine();
      toast.success("Loading model", {
        description: `${model.repo} is starting — watch live progress in the Inspector.`,
      });
      onLoaded();
    } catch (e) {
      toast.error("Failed to load model", { description: errMsg(e) });
    } finally {
      setLoadingModel(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" onClick={onBack}>
          <ArrowLeft aria-hidden />
          <span className="sr-only">Back</span>
        </Button>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate font-medium">{model.repo}</span>
            <Badge variant="secondary">{model.quant || "none"}</Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            {fmtGiB(model.size_bytes)} · {model.family}
          </p>
        </div>
      </div>

      {prepping ? (
        <div className="flex flex-col gap-3">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
      ) : prepError ? (
        <EmptyState
          icon={<Cpu aria-hidden className="size-6" />}
          title="Couldn't read model config"
          hint={prepError}
        />
      ) : (
        <>
          <PrecheckCard estimate={estimate} estimating={estimating} />

          <ScrollArea className="h-[12.5rem] pr-3">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <EngineControl
                  spec={engineSpecs.get("max_model_len")}
                  label="Context window"
                  fallbackKey="max_model_len"
                  value={paramValues.max_model_len}
                  onChange={(v) => setParam("max_model_len", v)}
                  kind="int"
                />
                <div className="flex gap-1">
                  {([32768, 65536, 131072, 262144] as const).map((k) => (
                    <button
                      key={k}
                      type="button"
                      onClick={() => setParam("max_model_len", k)}
                      className={cn(
                        "flex-1 rounded border px-1 py-0.5 text-[10px] font-medium transition-colors",
                        paramValues.max_model_len === k
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-border bg-muted/40 text-muted-foreground hover:border-primary/50 hover:text-foreground",
                      )}
                    >
                      {k === 32768 ? "32K" : k === 65536 ? "64K" : k === 131072 ? "128K" : "256K"}
                    </button>
                  ))}
                </div>
              </div>
              <EngineControl
                spec={engineSpecs.get("tensor_parallel_size")}
                label="Tensor parallel size"
                fallbackKey="tensor_parallel_size"
                value={paramValues.tensor_parallel_size}
                onChange={(v) => setParam("tensor_parallel_size", v)}
                kind="int"
              />
              <GpuUtilControl
                value={numOr(paramValues.gpu_memory_utilization, 0.9)}
                onChange={(v) => setParam("gpu_memory_utilization", v)}
                spec={engineSpecs.get("gpu_memory_utilization")}
              />
              <EngineControl
                spec={engineSpecs.get("quantization")}
                label="Quantization"
                fallbackKey="quantization"
                value={paramValues.quantization}
                onChange={(v) => setParam("quantization", v)}
                kind="select"
              />
            </div>
          </ScrollArea>

          <ExtraArgsCheckboxes flags={extraFlags} onChange={setExtraFlags} />

          <div className="flex flex-col gap-3 rounded-xl border border-border bg-muted/30 p-3">
            <div className="flex items-start gap-2">
              <Checkbox
                id="custom-system-prompt-on-load"
                checked={useCustomSystemPrompt}
                onCheckedChange={(checked) =>
                  setUseCustomSystemPrompt(checked === true)
                }
              />
              <label
                htmlFor="custom-system-prompt-on-load"
                className="grid cursor-pointer gap-1 text-xs leading-relaxed"
              >
                <span className="font-medium">Custom System Prompt</span>
                <span className="text-muted-foreground">
                  Off by default: Load saves the fixed long system prompt before
                  starting the model. Turn on to save a one-off custom prompt instead.
                </span>
              </label>
            </div>
            {useCustomSystemPrompt ? (
              <div className="grid gap-1.5 pl-6">
                <Label htmlFor="custom-system-prompt-text" className="text-xs">
                  System prompt for this load
                </Label>
                <Textarea
                  id="custom-system-prompt-text"
                  value={customSystemPrompt}
                  onChange={(e) => setCustomSystemPrompt(e.target.value)}
                  rows={4}
                  placeholder="Leave blank to use the fixed long prompt."
                  className="min-h-24"
                />
                <p className="text-xs text-muted-foreground">
                  Blank or whitespace-only text falls back to the fixed long prompt.
                </p>
              </div>
            ) : null}
          </div>
        </>
      )}

      <div className="flex items-center justify-end gap-2 border-t border-border pt-3">
        <Button variant="outline" onClick={onBack} disabled={loadingModel}>
          Back
        </Button>
        <Button
          onClick={handleLoad}
          disabled={prepping || !!prepError || loadingModel}
        >
          {loadingModel ? (
            <Loader2 aria-hidden className="animate-spin" />
          ) : (
            <Cpu aria-hidden />
          )}
          Load
        </Button>
      </div>
    </div>
  );
}

function HuggingFaceTab({ open }: { open: boolean }) {
  const [query, setQuery] = React.useState("");
  const [results, setResults] = React.useState<HFModel[] | null>(null);
  const [searching, setSearching] = React.useState(false);
  const [selectedRepo, setSelectedRepo] = React.useState<HFModel | null>(null);

  React.useEffect(() => {
    if (!open) {
      queueMicrotask(() => setSelectedRepo(null));
    }
  }, [open]);

  React.useEffect(() => {
    const q = query.trim();
    if (!q) {
      queueMicrotask(() => {
        setResults(null);
        setSearching(false);
      });
      return;
    }
    queueMicrotask(() => setSearching(true));
    let cancelled = false;
    const t = setTimeout(() => {
      api
        .searchHF(q, 25)
        .then((data) => {
          if (!cancelled) setResults(data);
        })
        .catch((e) => {
          if (!cancelled) {
            setResults([]);
            toast.error("Search failed", { description: errMsg(e) });
          }
        })
        .finally(() => {
          if (!cancelled) setSearching(false);
        });
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [query]);

  if (selectedRepo) {
    return (
      <VariantList
        model={selectedRepo}
        onBack={() => setSelectedRepo(null)}
      />
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="relative">
        <Search
          aria-hidden
          className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search Hugging Face (e.g. Qwen2.5-7B-Instruct-AWQ)"
          className="pl-8"
        />
      </div>

      {searching ? (
        <RowSkeletons rows={5} />
      ) : results === null ? (
        <EmptyState
          icon={<Search aria-hidden className="size-6" />}
          title="Search for a model"
          hint="Type a model name, open a result, then click Download on a quantization variant. Gated repos require HF_TOKEN on the backend."
        />
      ) : results.length === 0 ? (
        <EmptyState
          icon={<Search aria-hidden className="size-6" />}
          title="No results"
          hint="Try a different query."
        />
      ) : (
        <ScrollArea className="h-[19rem] pr-3">
          <div className="flex flex-col gap-2">
            {results.map((m) => (
              <button
                key={m.repo}
                type="button"
                onClick={() => setSelectedRepo(m)}
                className="flex flex-col gap-1.5 rounded-2xl border border-border bg-card/40 px-3 py-2.5 text-left transition-colors hover:bg-muted/50"
              >
                <div className="flex items-center gap-2">
                  <span className="truncate font-medium">{m.repo}</span>
                  {m.gated ? (
                    <Badge variant="outline">
                      <Lock aria-hidden />
                      gated
                    </Badge>
                  ) : null}
                </div>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                  <span className="inline-flex items-center gap-1">
                    <Download aria-hidden className="size-3" />
                    {fmtCount(m.downloads)}
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <Heart aria-hidden className="size-3" />
                    {fmtCount(m.likes)}
                  </span>
                  {m.pipeline_tag ? (
                    <span className="truncate">{m.pipeline_tag}</span>
                  ) : null}
                </div>
                {m.tags && m.tags.length > 0 ? (
                  <div className="flex flex-wrap gap-1">
                    {m.tags.slice(0, 5).map((tag) => (
                      <Badge key={tag} variant="secondary" className="font-normal">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                ) : null}
              </button>
            ))}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}

function VariantList({
  model,
  onBack,
}: {
  model: HFModel;
  onBack: () => void;
}) {
  const [variants, setVariants] = React.useState<QuantVariant[] | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [downloading, setDownloading] = React.useState<Record<string, boolean>>(
    {},
  );
  const [checking, setChecking] = React.useState<Record<string, boolean>>({});
  const [fits, setFits] = React.useState<Record<string, VramEstimate>>({});
  const hardware = useStudio((s) => s.hardware);

  React.useEffect(() => {
    let cancelled = false;
    queueMicrotask(() => setLoading(true));
    api
      .listVariants(model.repo)
      .then((data) => {
        if (!cancelled) setVariants(data);
      })
      .catch((e) => {
        if (!cancelled) {
          setVariants([]);
          toast.error("Failed to list variants", { description: errMsg(e) });
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [model.repo]);

  function keyOf(v: QuantVariant): string {
    return `${v.repo}@${v.revision}:${v.quant}`;
  }

  async function handleDownload(v: QuantVariant) {
    const k = keyOf(v);
    setDownloading((d) => ({ ...d, [k]: true }));
    try {
      await api.startDownload({
        repo: v.repo,
        quant: v.quant,
        revision: v.revision,
      });
      toast.success("Download started", {
        description: `${v.repo} (${v.quant || "none"}) — track progress in the Download Manager.`,
      });
    } catch (e) {
      toast.error("Failed to start download", { description: errMsg(e) });
    } finally {
      setDownloading((d) => ({ ...d, [k]: false }));
    }
  }

  async function handleCheckFit(v: QuantVariant) {
    const k = keyOf(v);
    setChecking((c) => ({ ...c, [k]: true }));
    try {
      const meta = await api.getMeta(v.repo, v.quant, v.revision);
      const weights = meta.weight_bytes_known || v.size_bytes || 0;
      const tp = suggestTensorParallel(
        weights,
        hardware?.per_gpu_bytes ?? 0,
        hardware?.num_gpus ?? 1,
      );
      const est = await api.estimate({
        meta,
        quant: v.quant || meta.quant || "none",
        dtype: hardware?.capabilities.recommended_dtype || "float16",
        tensor_parallel_size: tp,
        gpu_memory_utilization: 0.9,
        max_model_len: Math.min(meta.max_position_embeddings || 65535, 65535),
        max_num_seqs: 16,
        kv_concurrency: 1,
        kv_cache_dtype: "auto",
        enforce_eager: false,
        block_length: 32,
      });
      setFits((f) => ({ ...f, [k]: est }));
    } catch (e) {
      toast.error("Fit check failed", { description: errMsg(e) });
    } finally {
      setChecking((c) => ({ ...c, [k]: false }));
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" onClick={onBack}>
          <ArrowLeft aria-hidden />
          <span className="sr-only">Back to search</span>
        </Button>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate font-medium">{model.repo}</span>
            {model.gated ? (
              <Badge variant="outline">
                <Lock aria-hidden />
                gated
              </Badge>
            ) : null}
          </div>
          <p className="text-xs text-muted-foreground">
            Pick a variant, optionally check fit, then download it to the local cache.
            Gated models need HF_TOKEN before starting the backend.
          </p>
        </div>
      </div>

      {loading || variants === null ? (
        <RowSkeletons rows={4} />
      ) : variants.length === 0 ? (
        <EmptyState
          icon={<HardDrive aria-hidden className="size-6" />}
          title="No variants detected"
          hint="This repo has no distinct quantization variants we could enumerate."
        />
      ) : (
        <ScrollArea className="h-[19rem] pr-3">
          <div className="flex flex-col gap-2">
            {variants.map((v) => {
              const k = keyOf(v);
              const est = fits[k];
              return (
                <div
                  key={k}
                  className="flex flex-col gap-2 rounded-2xl border border-border bg-card/40 px-3 py-2.5"
                >
                  <div className="flex items-center gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary">{v.quant || "none"}</Badge>
                        <span className="text-sm">{fmtGiB(v.size_bytes)}</span>
                        {!v.supported ? (
                          <Badge variant="destructive">unsupported</Badge>
                        ) : null}
                      </div>
                      {v.note ? (
                        <p className="mt-0.5 truncate text-xs text-muted-foreground">
                          {v.note}
                        </p>
                      ) : null}
                    </div>
                    <div className="flex shrink-0 items-center gap-1.5">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleCheckFit(v)}
                        disabled={checking[k]}
                      >
                        {checking[k] ? (
                          <Loader2 aria-hidden className="animate-spin" />
                        ) : (
                          <Gauge aria-hidden />
                        )}
                        Check fit
                      </Button>
                      <Button
                        size="sm"
                        onClick={() => handleDownload(v)}
                        disabled={downloading[k] || !v.supported}
                      >
                        {downloading[k] ? (
                          <Loader2 aria-hidden className="animate-spin" />
                        ) : (
                          <Download aria-hidden />
                        )}
                        Download
                      </Button>
                    </div>
                  </div>
                  {est ? (
                    <div className="flex flex-wrap items-center gap-2 border-t border-border pt-2 text-xs text-muted-foreground">
                      <OomBadge status={est.status} />
                      <span>
                        ≈ {fmtGiB(est.per_gpu.required_bytes)} / GPU ×{" "}
                        {est.tensor_parallel_size} GPU
                        {est.tensor_parallel_size > 1 ? "s" : ""}
                      </span>
                      <span aria-hidden>·</span>
                      <span>
                        max ctx {est.max_safe_context.toLocaleString()}
                      </span>
                      {est.reasons.length > 0 ? (
                        <span className="truncate text-foreground/70">
                          {est.reasons[0]}
                        </span>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}

function PrecheckCard({
  estimate,
  estimating,
}: {
  estimate: VramEstimate | null;
  estimating: boolean;
}) {
  const perGpu = estimate?.per_gpu ?? null;

  return (
    <div className="rounded-2xl border border-border bg-card/40 p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Gauge aria-hidden className="size-4 text-muted-foreground" />
          VRAM precheck
        </div>
        <div className="flex items-center gap-2">
          {estimating ? (
            <Loader2 aria-hidden className="size-4 animate-spin text-muted-foreground" />
          ) : null}
          {estimate ? (
            <OomBadge status={estimate.status} />
          ) : (
            <Badge variant="secondary">pending</Badge>
          )}
        </div>
      </div>
      {estimate && perGpu ? (
        <div className="mt-3 flex flex-col gap-3">
          <div className="grid grid-cols-1 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-2">
            <Metric
              label="Max safe context"
              value={`${fmtCount(estimate.max_safe_context)} tok`}
              valueClassName={statusColor(estimate.status)}
            />
            <Metric
              label="Per-GPU headroom"
              value={fmtGiB(perGpu.headroom_bytes)}
              valueClassName={
                perGpu.headroom_bytes < 0 ? "text-destructive" : undefined
              }
            />
            <Metric
              label="Tensor parallel"
              value={`${estimate.tensor_parallel_size} GPU${
                estimate.tensor_parallel_size > 1 ? "s" : ""
              } of ${estimate.num_gpus_available}`}
            />
            <Metric
              label="Per GPU req / budget"
              value={`${fmtGiB(perGpu.required_bytes)} / ${fmtGiB(
                perGpu.budget_bytes,
              )}`}
            />
            {estimate.tensor_parallel_size > 1 ? (
              <Metric
                label={`Total (${estimate.tensor_parallel_size} GPUs)`}
                value={`${fmtGiB(estimate.aggregate_required_bytes)} / ${fmtGiB(
                  estimate.aggregate_budget_bytes,
                )}`}
              />
            ) : null}
          </div>

          <Separator />

          <div className="flex flex-col gap-2">
            <p className="text-xs font-medium text-muted-foreground">
              Per-GPU VRAM (predicted, of {fmtGiB(perGpu.total_bytes)})
            </p>
            <StackedVramBar
              total={perGpu.total_bytes}
              budgetBytes={perGpu.budget_bytes}
              segments={[
                {
                  key: "weights",
                  label: "Weights",
                  bytes: perGpu.weights_bytes,
                  className: VRAM_COLORS.weights,
                },
                {
                  key: "kv",
                  label: "KV cache",
                  bytes: perGpu.kv_bytes,
                  className: VRAM_COLORS.kv,
                },
                {
                  key: "activation",
                  label: "Activation",
                  bytes: perGpu.activation_bytes,
                  className: VRAM_COLORS.activation,
                },
                {
                  key: "overhead",
                  label: "Overhead",
                  bytes: perGpu.overhead_bytes + perGpu.cuda_graph_bytes,
                  className: VRAM_COLORS.overhead,
                },
              ]}
            />
          </div>
        </div>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">
          {estimating
            ? "Computing prediction…"
            : "Adjust the engine controls below to estimate VRAM usage."}
        </p>
      )}
      {estimate && estimate.reasons.length > 0 ? (
        <ul className="mt-2 space-y-0.5 text-xs text-foreground/70">
          {estimate.reasons.slice(0, 3).map((r, i) => (
            <li key={i}>· {r}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function Metric({
  label,
  value,
  valueClassName,
}: {
  label: string;
  value: string;
  valueClassName?: string;
}) {
  return (
    <div className="flex flex-col">
      <span className="text-[0.65rem] uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <span className={cn("font-medium text-foreground", valueClassName)}>
        {value}
      </span>
    </div>
  );
}

type EngineSpec =
  | {
      key: string;
      label: string;
      type: "float" | "int" | "text" | "bool" | "select";
      min?: number | null;
      max?: number | null;
      step?: number | null;
      options?: { value: unknown; label: string; disabled?: boolean }[] | null;
      help?: string;
      unit?: string;
    }
  | undefined;

function EngineControl({
  spec,
  label,
  fallbackKey,
  value,
  onChange,
  kind,
}: {
  spec: EngineSpec;
  label: string;
  fallbackKey: string;
  value: unknown;
  onChange: (v: unknown) => void;
  kind: "int" | "float" | "select";
}) {
  const id = `engine-${fallbackKey}`;
  const resolvedLabel = spec?.label || label;

  if (kind === "select") {
    const options = spec?.options ?? [
      { value: "auto", label: "auto" },
      { value: "none", label: "none" },
      { value: "awq", label: "awq" },
      { value: "gptq", label: "gptq" },
    ];
    const current = value === undefined || value === null ? "" : String(value);
    return (
      <div className="flex flex-col gap-1.5">
        <Label htmlFor={id}>{resolvedLabel}</Label>
        <Select value={current} onValueChange={(v) => onChange(v)}>
          <SelectTrigger id={id} className="w-full">
            <SelectValue placeholder="Select" />
          </SelectTrigger>
          <SelectContent>
            {options.map((opt) => (
              <SelectItem
                key={String(opt.value)}
                value={String(opt.value)}
                disabled={opt.disabled}
              >
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={id}>
        {resolvedLabel}
        {spec?.unit ? (
          <span className="text-xs font-normal text-muted-foreground">
            {spec.unit}
          </span>
        ) : null}
      </Label>
      <Input
        id={id}
        type="number"
        inputMode="numeric"
        value={value === undefined || value === null ? "" : String(value)}
        min={spec?.min ?? undefined}
        max={spec?.max ?? undefined}
        step={spec?.step ?? (kind === "int" ? 1 : 0.01)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") {
            onChange("");
            return;
          }
          const n = kind === "int" ? parseInt(raw, 10) : parseFloat(raw);
          onChange(Number.isFinite(n) ? n : raw);
        }}
      />
    </div>
  );
}

function GpuUtilControl({
  value,
  onChange,
  spec,
}: {
  value: number;
  onChange: (v: number) => void;
  spec: EngineSpec;
}) {
  const min = spec?.min ?? 0.5;
  const max = spec?.max ?? 0.97;
  const step = spec?.step ?? 0.01;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <Label htmlFor="engine-gpu-util">
          {spec?.label || "GPU memory utilization"}
        </Label>
        <span className="text-xs tabular-nums text-muted-foreground">
          {Math.round(value * 100)}%
        </span>
      </div>
      <Slider
        id="engine-gpu-util"
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(vals) => onChange(vals[0] ?? value)}
        className="mt-2"
      />
    </div>
  );
}

function RowSkeletons({ rows }: { rows: number }) {
  return (
    <div className="flex flex-col gap-2">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-16 w-full" />
      ))}
    </div>
  );
}

function EmptyState({
  icon,
  title,
  hint,
  className,
}: {
  icon: React.ReactNode;
  title: string;
  hint?: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex h-[18rem] flex-col items-center justify-center gap-2 text-center",
        className,
      )}
    >
      <div className="flex size-12 items-center justify-center rounded-2xl bg-muted text-muted-foreground">
        {icon}
      </div>
      <p className="text-sm font-medium">{title}</p>
      {hint ? (
        <p className="max-w-sm text-xs text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  );
}

function numOr(v: unknown, fallback: number): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export default ModelPicker;
