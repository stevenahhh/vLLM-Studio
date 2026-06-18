"use client";

import * as React from "react";
import {
  PanelRightClose,
  PanelRightOpen,
  Gauge,
  Activity,
  Loader2,
} from "lucide-react";

import { GpuMonitor } from "@/components/gpu-monitor";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { useStudio } from "@/lib/store";
import { fmtGiB, fmtPct, statusColor } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { PerGpuBreakdown } from "@/lib/types";

export function VramInspector({ className }: { className?: string }) {
  const [collapsed, setCollapsed] = React.useState(false);

  const gpuStats = useStudio((s) => s.gpuStats);
  const estimate = useStudio((s) => s.estimate);
  const engine = useStudio((s) => s.engine);

  if (collapsed) {
    return (
      <aside
        className={cn(
          "sticky top-0 flex h-svh w-12 shrink-0 flex-col items-center gap-3 border-l bg-background py-3",
          className,
        )}
      >
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Expand inspector"
          onClick={() => setCollapsed(false)}
        >
          <PanelRightOpen />
        </Button>
        <Separator className="w-6" />
        <div className="flex flex-1 items-center">
          <span className="rotate-180 text-xs font-medium tracking-widest text-muted-foreground [writing-mode:vertical-rl]">
            VRAM
          </span>
        </div>
        {estimate ? (
          <Gauge className={cn("size-4", statusColor(estimate.status))} />
        ) : null}
      </aside>
    );
  }

  const total = gpuStats?.total_bytes ?? 0;
  const used = gpuStats?.used_bytes ?? 0;
  const usedPct = total > 0 ? Math.min(100, (used / total) * 100) : 0;

  const perGpu: PerGpuBreakdown | null = estimate?.per_gpu ?? null;

  return (
    <aside
      className={cn(
        "sticky top-0 flex h-svh min-h-0 w-[22rem] shrink-0 flex-col overflow-y-auto border-l bg-background",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <Gauge className="size-4 text-primary" />
          <h2 className="font-heading text-sm font-medium">Inspector</h2>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Collapse inspector"
          onClick={() => setCollapsed(true)}
        >
          <PanelRightClose />
        </Button>
      </div>

      <div className="flex flex-col gap-4 p-4">
        {engine && engine.state === "loading" ? (
          <Card size="sm" className="border-primary/40">
            <CardHeader>
              <CardTitle className="flex items-center justify-between gap-2 text-sm">
                <span className="flex items-center gap-2">
                  <Loader2 className="size-4 animate-spin text-primary" />
                  Loading model
                </span>
                <span className="font-medium tabular-nums text-primary">
                  {Math.round((engine.progress ?? 0) * 100)}%
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2">
              <Progress value={(engine.progress ?? 0) * 100} />
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="text-muted-foreground">
                  {engine.phase || "Starting engine"}
                </span>
                {engine.repo ? (
                  <span className="truncate font-medium text-foreground">
                    {engine.repo}
                  </span>
                ) : null}
              </div>
            </CardContent>
          </Card>
        ) : null}

        <Card size="sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Activity className="size-4 text-primary" />
              Live VRAM
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="text-muted-foreground">Aggregate</span>
                <span className="font-medium tabular-nums">
                  {fmtGiB(used)} / {fmtGiB(total)}{" "}
                  <span className="text-muted-foreground">
                    ({fmtPct(usedPct)})
                  </span>
                </span>
              </div>
              <Progress value={usedPct} />
            </div>
            {gpuStats && gpuStats.gpus.length > 0 ? (
              <GpuMonitor gpus={gpuStats.gpus} compact breakdown={perGpu} />
            ) : (
              <p className="text-xs text-muted-foreground">
                Waiting for GPU telemetry…
              </p>
            )}
          </CardContent>
        </Card>

        {engine && engine.repo ? (
          <p className="px-1 text-xs text-muted-foreground">
            Engine:{" "}
            <span className="font-medium text-foreground">{engine.repo}</span>{" "}
            <span className="capitalize">({engine.state})</span>
          </p>
        ) : null}
      </div>
    </aside>
  );
}

export default VramInspector;
