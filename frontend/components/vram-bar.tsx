"use client";

import * as React from "react";

import { fmtGiB } from "@/lib/format";
import { cn } from "@/lib/utils";

export interface VramSegment {
  key: string;
  label: string;
  bytes: number;
  /** A Tailwind background color utility, e.g. "bg-primary". */
  className: string;
}

/** Distinct colors for the VRAM composition (weights/kv/activation/overhead). */
export const VRAM_COLORS = {
  weights: "bg-primary",
  kv: "bg-sky-500",
  activation: "bg-amber-500",
  overhead: "bg-rose-500",
  used: "bg-primary",
} as const;

/**
 * A single horizontal bar that stacks `segments` (each a different color) over a
 * track whose full width represents `total` bytes (0 → total = per-GPU max VRAM).
 * Optionally draws a vertical marker at `budgetBytes` (the util budget line).
 */
export function StackedVramBar({
  segments,
  total,
  budgetBytes,
  height = "h-3.5",
  showLegend = true,
  showScale = true,
  className,
}: {
  segments: VramSegment[];
  total: number;
  budgetBytes?: number;
  height?: string;
  showLegend?: boolean;
  showScale?: boolean;
  className?: string;
}) {
  const safeTotal = total > 0 ? total : 1;
  const used = segments.reduce((s, x) => s + Math.max(0, x.bytes), 0);
  const budgetPct =
    budgetBytes && budgetBytes > 0
      ? Math.min(100, (budgetBytes / safeTotal) * 100)
      : null;

  return (
    <div className={cn("flex w-full flex-col gap-1.5", className)}>
      <div
        className={cn(
          "relative flex w-full overflow-hidden rounded-full bg-muted",
          height,
        )}
      >
        {segments.map((seg) => {
          const pct = Math.max(0, Math.min(100, (seg.bytes / safeTotal) * 100));
          if (pct <= 0) return null;
          return (
            <div
              key={seg.key}
              className={cn("h-full", seg.className)}
              style={{ width: `${pct}%` }}
              title={`${seg.label}: ${fmtGiB(seg.bytes)}`}
            />
          );
        })}
        {budgetPct !== null ? (
          <div
            className="absolute inset-y-0 z-10 w-0.5 bg-foreground/80"
            style={{ left: `calc(${budgetPct}% - 1px)` }}
            title={`Budget: ${fmtGiB(budgetBytes as number)}`}
          />
        ) : null}
      </div>

      {showScale ? (
        <div className="flex items-center justify-between text-[0.7rem] text-muted-foreground tabular-nums">
          <span>0 GiB</span>
          <span>
            {fmtGiB(used)} / {fmtGiB(safeTotal)}
          </span>
        </div>
      ) : null}

      {showLegend ? (
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {segments.map((seg) => (
            <span
              key={seg.key}
              className="flex items-center gap-1.5 text-[0.7rem]"
            >
              <span className={cn("size-2.5 rounded-[3px]", seg.className)} />
              <span className="text-muted-foreground">{seg.label}</span>
              <span className="font-medium text-foreground tabular-nums">
                {fmtGiB(seg.bytes)}
              </span>
            </span>
          ))}
          {budgetPct !== null ? (
            <span className="flex items-center gap-1.5 text-[0.7rem]">
              <span className="h-2.5 w-0.5 bg-foreground/80" />
              <span className="text-muted-foreground">budget</span>
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export default StackedVramBar;
