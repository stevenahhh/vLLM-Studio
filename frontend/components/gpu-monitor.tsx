"use client";

import * as React from "react";
import { ChevronDown, Thermometer, Zap } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { StackedVramBar, VRAM_COLORS } from "@/components/vram-bar";
import { cn } from "@/lib/utils";
import { fmtGiB, fmtPct } from "@/lib/format";
import type { GpuLive, PerGpuBreakdown } from "@/lib/types";

/** Trim vendor noise so "NVIDIA GeForce RTX 2080" -> "RTX 2080". */
function shortName(name: string): string {
  return (
    name
      ?.replace(/NVIDIA/gi, "")
      .replace(/GeForce/gi, "")
      .replace(/\s+/g, " ")
      .trim() || "GPU"
  );
}

/** Tailwind text color (shadcn tokens / amber) for a used fraction. */
function fractionTextColor(frac: number): string {
  if (frac > 0.9) return "text-destructive";
  if (frac > 0.75) return "text-amber-500";
  return "text-foreground";
}

/** Progress indicator color for a used fraction. */
function fractionBarColor(frac: number): string {
  if (frac > 0.9) return "[&>[data-slot=progress-indicator]]:bg-destructive";
  if (frac > 0.75) return "[&>[data-slot=progress-indicator]]:bg-amber-500";
  return "";
}

function GpuRow({
  gpu,
  compact,
  breakdown,
}: {
  gpu: GpuLive;
  compact?: boolean;
  /** Predicted per-GPU composition (from the VRAM estimate), if a model is
   *  selected/loaded — shown in the expanded detail since live telemetry only
   *  reports total used (it cannot itself decompose weights/kv/etc). */
  breakdown?: PerGpuBreakdown | null;
}) {
  const [open, setOpen] = React.useState(false);
  const total = gpu.total_bytes || 0;
  const used = gpu.used_bytes || 0;
  const free = gpu.free_bytes || Math.max(0, total - used);
  const frac = total > 0 ? used / total : 0;
  const pct = frac * 100;

  return (
    <div className="flex flex-col gap-1.5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full flex-col gap-1 rounded-md text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-expanded={open}
      >
        <div className="flex items-center justify-between gap-2">
          <span
            className={cn(
              "flex min-w-0 items-center gap-1.5 font-medium text-foreground",
              compact ? "text-xs" : "text-sm",
            )}
          >
            <ChevronDown
              className={cn(
                "shrink-0 text-muted-foreground transition-transform",
                compact ? "size-3" : "size-3.5",
                open && "rotate-180",
              )}
              aria-hidden
            />
            <span className="truncate">
              GPU {gpu.index} - {shortName(gpu.name)}
            </span>
          </span>
          <Badge
            variant="secondary"
            className={cn("shrink-0 tabular-nums", compact && "h-4 text-[10px]")}
          >
            {fmtPct(gpu.utilization)} util
          </Badge>
        </div>

        <Progress
          value={pct}
          className={cn(compact ? "h-1.5" : "h-2", fractionBarColor(frac))}
        />

        <div
          className={cn(
            "flex items-center justify-between tabular-nums",
            compact ? "text-[10px]" : "text-xs",
            "text-muted-foreground",
          )}
        >
          <span className={cn("font-medium", fractionTextColor(frac))}>
            {fmtGiB(used)} / {fmtGiB(total)}
          </span>
          <span className={fractionTextColor(frac)}>({fmtPct(pct)})</span>
        </div>
      </button>

      {open ? (
        <div className="flex flex-col gap-2 rounded-md border border-border bg-muted/30 p-2.5 text-xs">
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 tabular-nums">
            <span className="text-muted-foreground">Used</span>
            <span className="text-right font-medium">{fmtGiB(used)}</span>
            <span className="text-muted-foreground">Free</span>
            <span className="text-right font-medium">{fmtGiB(free)}</span>
            <span className="flex items-center gap-1 text-muted-foreground">
              <Thermometer className="size-3" /> Temp
            </span>
            <span className="text-right font-medium">
              {Math.round(gpu.temperature)}°C
            </span>
            <span className="flex items-center gap-1 text-muted-foreground">
              <Zap className="size-3" /> Power
            </span>
            <span className="text-right font-medium">
              {Math.round(gpu.power_watts)} W
            </span>
          </div>

          {breakdown ? (
            <div className="flex flex-col gap-1.5 border-t border-border pt-2">
              <span className="text-muted-foreground">
                Predicted composition when loaded
              </span>
              <StackedVramBar
                total={breakdown.total_bytes || total}
                budgetBytes={breakdown.budget_bytes}
                showScale={false}
                height="h-3"
                segments={[
                  {
                    key: "weights",
                    label: "Weights",
                    bytes: breakdown.weights_bytes,
                    className: VRAM_COLORS.weights,
                  },
                  {
                    key: "kv",
                    label: "KV",
                    bytes: breakdown.kv_bytes,
                    className: VRAM_COLORS.kv,
                  },
                  {
                    key: "activation",
                    label: "Act",
                    bytes: breakdown.activation_bytes,
                    className: VRAM_COLORS.activation,
                  },
                  {
                    key: "overhead",
                    label: "Overhead",
                    bytes: breakdown.overhead_bytes + breakdown.cuda_graph_bytes,
                    className: VRAM_COLORS.overhead,
                  },
                ]}
              />
            </div>
          ) : (
            <p className="border-t border-border pt-2 text-muted-foreground">
              Live telemetry reports total usage only. Select a model to see the
              predicted weights / KV / activation / overhead split.
            </p>
          )}
        </div>
      ) : null}
    </div>
  );
}

export function GpuMonitor({
  gpus,
  compact,
  breakdown,
}: {
  gpus: GpuLive[];
  compact?: boolean;
  breakdown?: PerGpuBreakdown | null;
}) {
  if (!gpus || gpus.length === 0) {
    return (
      <p
        className={cn(
          "text-muted-foreground",
          compact ? "text-[10px]" : "text-xs",
        )}
      >
        No GPUs detected.
      </p>
    );
  }

  return (
    <div className={cn("flex flex-col", compact ? "gap-2.5" : "gap-4")}>
      {gpus.map((gpu) => (
        <GpuRow
          key={gpu.index}
          gpu={gpu}
          compact={compact}
          breakdown={breakdown}
        />
      ))}
    </div>
  );
}
