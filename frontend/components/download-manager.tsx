"use client";

import * as React from "react";
import useSWR from "swr";
import { Download, Loader2, X } from "lucide-react";

import * as api from "@/lib/api";
import { fmtGiB } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { DownloadJob, DownloadState } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const ACTIVE_STATES: DownloadState[] = ["queued", "downloading"];

const STATE_LABEL: Record<DownloadState, string> = {
  queued: "Queued",
  downloading: "Downloading",
  completed: "Completed",
  error: "Error",
  cancelled: "Cancelled",
};

/** shadcn-token classes per download state (no hardcoded hex). */
function stateBadgeClass(state: DownloadState): string {
  switch (state) {
    case "downloading":
      return "bg-primary/15 text-primary";
    case "completed":
      return "bg-primary text-primary-foreground";
    case "error":
      return "bg-destructive/15 text-destructive";
    case "cancelled":
      return "bg-muted text-muted-foreground";
    case "queued":
      return "bg-amber-500/15 text-amber-600 dark:text-amber-400";
    default:
      return "bg-muted text-muted-foreground";
  }
}

function isActive(state: DownloadState): boolean {
  return ACTIVE_STATES.includes(state);
}

interface DownloadManagerProps {
  className?: string;
}

export function DownloadManager({ className }: DownloadManagerProps) {
  const { data, mutate } = useSWR<DownloadJob[]>(
    "downloads",
    () => api.listDownloads(),
    { refreshInterval: 1500, revalidateOnFocus: false }
  );

  const [busy, setBusy] = React.useState<Set<string>>(new Set());

  const jobs = data ?? [];

  const setBusyFor = (id: string, on: boolean) =>
    setBusy((prev) => {
      const next = new Set(prev);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });

  const onCancel = React.useCallback(
    async (id: string) => {
      setBusyFor(id, true);
      try {
        await api.cancelDownload(id);
        await mutate();
      } catch {
        // best-effort
      } finally {
        setBusyFor(id, false);
      }
    },
    [mutate]
  );

  const onRemove = React.useCallback(
    async (id: string) => {
      setBusyFor(id, true);
      try {
        await api.removeDownload(id);
        await mutate();
      } catch {
        // best-effort
      } finally {
        setBusyFor(id, false);
      }
    },
    [mutate]
  );

  if (jobs.length === 0) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed py-8 text-center",
          className
        )}
      >
        <Download className="size-5 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">No downloads</p>
      </div>
    );
  }

  return (
    <div className={cn("w-full", className)}>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Model</TableHead>
            <TableHead>State</TableHead>
            <TableHead className="w-[34%]">Progress</TableHead>
            <TableHead className="text-right">Speed</TableHead>
            <TableHead className="w-8" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {jobs.map((job) => {
            const active = isActive(job.state);
            const pct = Math.max(0, Math.min(100, (job.progress || 0) * 100));
            const isBusy = busy.has(job.id);
            return (
              <TableRow key={job.id}>
                <TableCell className="max-w-[14rem] align-top">
                  <div className="flex flex-col gap-1">
                    <span
                      className="truncate font-medium text-foreground"
                      title={job.repo}
                    >
                      {job.repo}
                    </span>
                    {job.quant && job.quant !== "none" ? (
                      <Badge
                        variant="outline"
                        className="w-fit font-mono text-[10px]"
                      >
                        {job.quant}
                      </Badge>
                    ) : null}
                    {job.state === "error" && job.error ? (
                      <span
                        className="truncate text-xs text-destructive"
                        title={job.error}
                      >
                        {job.error}
                      </span>
                    ) : null}
                  </div>
                </TableCell>
                <TableCell className="align-top">
                  <Badge
                    className={cn("border-transparent", stateBadgeClass(job.state))}
                  >
                    {job.state === "downloading" ? (
                      <Loader2 className="animate-spin" />
                    ) : null}
                    {STATE_LABEL[job.state] ?? job.state}
                  </Badge>
                </TableCell>
                <TableCell className="align-top">
                  <div className="flex flex-col gap-1.5">
                    <Progress value={pct} className="h-1.5" />
                    <div className="flex items-center justify-between text-xs text-muted-foreground tabular-nums">
                      <span>
                        {fmtGiB(job.downloaded_bytes)} /{" "}
                        {fmtGiB(job.total_bytes)}
                      </span>
                      <span>{Math.round(pct)}%</span>
                    </div>
                  </div>
                </TableCell>
                <TableCell className="text-right align-top text-xs text-muted-foreground tabular-nums">
                  {active && job.speed_bps > 0
                    ? `${fmtGiB(job.speed_bps)}/s`
                    : "—"}
                </TableCell>
                <TableCell className="align-top">
                  {active ? (
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label="Cancel download"
                      title="Cancel download"
                      disabled={isBusy}
                      onClick={() => onCancel(job.id)}
                    >
                      {isBusy ? <Loader2 className="animate-spin" /> : <X />}
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label="Remove from list"
                      title="Remove from list"
                      disabled={isBusy}
                      onClick={() => onRemove(job.id)}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      {isBusy ? <Loader2 className="animate-spin" /> : <X />}
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

export default DownloadManager;
