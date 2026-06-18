// Pure formatting helpers. Sizes from the API are always bytes (int).
// 1 GiB = 1024^3. Frontend formats to GiB for display.
import type { OOMStatus } from "@/lib/types";

const GIB = 1024 ** 3;

/** Convert a byte count to a GiB number (not rounded). */
export function bytesToGiB(n: number): number {
  if (!Number.isFinite(n)) return 0;
  return n / GIB;
}

/** Format a byte count as a GiB string, e.g. "5.4 GiB". */
export function fmtGiB(n: number, digits = 1): string {
  return `${bytesToGiB(n).toFixed(digits)} GiB`;
}

/** Format a fraction (0..1) or a percent (0..100) heuristically as a percent.
 *  Values <= 1 are treated as fractions; values > 1 are treated as already-percent. */
export function fmtPct(n: number): string {
  if (!Number.isFinite(n)) return "0%";
  const pct = Math.abs(n) <= 1 ? n * 100 : n;
  return `${pct.toFixed(0)}%`;
}

/** Compact count formatting: 1.2K / 3.4M / 1.1B. */
export function fmtCount(n: number): string {
  if (!Number.isFinite(n)) return "0";
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return `${Math.round(n)}`;
}

/** Suggest the smallest tensor_parallel_size (1/2/4/8) that fits `weightsBytes`
 *  across the available GPUs, leaving ~2.6 GiB/GPU for KV + overhead. Mirrors the
 *  backend default so the HF "Check fit" estimate isn't stuck at TP1. */
export function suggestTensorParallel(
  weightsBytes: number,
  perGpuBytes: number,
  numGpus: number,
  util = 0.9,
): number {
  if (!perGpuBytes || numGpus <= 1 || weightsBytes <= 0) return 1;
  const reserve = 2.6 * GIB;
  const usable = Math.max(1, perGpuBytes * util - reserve);
  const needed = Math.ceil(weightsBytes / usable);
  const candidates = [1, 2, 4, 8].filter((t) => t <= numGpus);
  for (const t of candidates) {
    if (t >= needed) return t;
  }
  return candidates.length ? candidates[candidates.length - 1] : 1;
}

/** Tailwind text-color classes (shadcn tokens) for an OOM verdict. */
export function statusColor(status: OOMStatus): string {
  switch (status) {
    case "ok":
      return "text-primary";
    case "tight":
      return "text-yellow-500 dark:text-yellow-400";
    case "oom":
      return "text-destructive";
    default:
      return "text-muted-foreground";
  }
}
