import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { OOMStatus } from "@/lib/types";

export function OomBadge({
  status,
  className,
}: {
  status: OOMStatus;
  className?: string;
}) {
  if (status === "tight") {
    return (
      <Badge
        variant="outline"
        className={cn(
          "border-amber-500/50 bg-amber-500/10 text-amber-500",
          className
        )}
      >
        <AlertTriangle aria-hidden />
        TIGHT
      </Badge>
    );
  }

  if (status === "oom") {
    return (
      <Badge variant="destructive" className={className}>
        <XCircle aria-hidden />
        OOM
      </Badge>
    );
  }

  return (
    <Badge variant="default" className={className}>
      <CheckCircle2 aria-hidden />
      OK
    </Badge>
  );
}
