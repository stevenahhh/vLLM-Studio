"use client";

// Dynamic, editable parameter panel. Renders the family-aware ParamSchema from the
// store and writes every change back through useStudio.setParam (which debounces and
// recomputes the VRAM estimate for engine/affects_vram params).
import * as React from "react";
import { Info, Sparkles, Cpu, Sliders, Wand2, MessageSquare } from "lucide-react";

import type { ParamGroup, ParamSpec } from "@/lib/types";
import { useStudio } from "@/lib/store";
import { cn } from "@/lib/utils";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const GROUP_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  engine: Cpu,
  sampling: Sliders,
  diffusion: Wand2,
  prompt: MessageSquare,
};

/** Coerce an unknown param value to a finite number using the spec's default. */
function asNumber(value: unknown, spec: ParamSpec): number {
  const n = typeof value === "number" ? value : Number(value);
  if (Number.isFinite(n)) return n;
  const d = Number(spec.default);
  return Number.isFinite(d) ? d : 0;
}

/** A small muted help affordance: an Info icon with the help text in a tooltip. */
function HelpTip({ help }: { help: string }) {
  if (!help) return null;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label="Parameter help"
          className="inline-flex items-center text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none"
        >
          <Info className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent>{help}</TooltipContent>
    </Tooltip>
  );
}

function VramBadge() {
  return (
    <Badge variant="secondary" className="h-4 px-1.5 text-[10px] uppercase">
      VRAM
    </Badge>
  );
}

/** Numeric (float/int) control: Label + value + Slider synced to a small Input. */
function NumberField({
  spec,
  value,
  onChange,
  prominent,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: number) => void;
  prominent?: boolean;
}) {
  const isInt = spec.type === "int";
  const min = typeof spec.min === "number" ? spec.min : 0;
  const max = typeof spec.max === "number" ? spec.max : 100;
  const step = typeof spec.step === "number" ? spec.step : isInt ? 1 : 0.01;
  const current = asNumber(value, spec);

  // Local text state so partial/empty edits in the Input don't fight the store.
  const [text, setText] = React.useState<string>(String(current));
  React.useEffect(() => {
    queueMicrotask(() => setText(String(current)));
  }, [current]);

  const commit = (raw: string) => {
    let n = isInt ? parseInt(raw, 10) : parseFloat(raw);
    if (!Number.isFinite(n)) {
      setText(String(current));
      return;
    }
    n = Math.min(max, Math.max(min, n));
    onChange(n);
  };

  const hasSlider =
    typeof spec.min === "number" && typeof spec.max === "number";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <Label
          htmlFor={`param-${spec.key}`}
          className={cn(prominent && "text-base font-semibold")}
        >
          {prominent && <Sparkles className="size-4 text-primary" />}
          <span>{spec.label}</span>
          {spec.affects_vram && <VramBadge />}
          <HelpTip help={spec.help} />
        </Label>
        <div className="flex items-center gap-1.5">
          <Input
            id={`param-${spec.key}`}
            type="number"
            inputMode={isInt ? "numeric" : "decimal"}
            min={min}
            max={max}
            step={step}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onBlur={(e) => commit(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                commit((e.target as HTMLInputElement).value);
              }
            }}
            className={cn(
              "h-7 w-20 text-right tabular-nums",
              prominent && "w-24 font-semibold",
            )}
          />
          {spec.unit && (
            <span className="w-8 shrink-0 text-xs text-muted-foreground">
              {spec.unit}
            </span>
          )}
        </div>
      </div>
      {hasSlider && (
        <Slider
          min={min}
          max={max}
          step={step}
          value={[current]}
          onValueChange={(vals) => onChange(vals[0])}
          aria-label={spec.label}
          className={cn(prominent && "py-1")}
        />
      )}
    </div>
  );
}

/** text type: multiline Textarea for prompts, single-line Input otherwise. */
function TextField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: string) => void;
}) {
  const multiline = spec.key === "system_prompt";
  const str = typeof value === "string" ? value : value == null ? "" : String(value);
  return (
    <div className="space-y-2">
      <Label htmlFor={`param-${spec.key}`}>
        <span>{spec.label}</span>
        {spec.affects_vram && <VramBadge />}
        <HelpTip help={spec.help} />
      </Label>
      {multiline ? (
        <Textarea
          id={`param-${spec.key}`}
          value={str}
          onChange={(e) => onChange(e.target.value)}
          rows={4}
          placeholder={spec.help || undefined}
        />
      ) : (
        <Input
          id={`param-${spec.key}`}
          value={str}
          onChange={(e) => onChange(e.target.value)}
          placeholder={spec.help || undefined}
        />
      )}
    </div>
  );
}

/** bool type: a Switch on a labelled row. */
function BoolField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <Label htmlFor={`param-${spec.key}`} className="flex-wrap">
        <span>{spec.label}</span>
        {spec.affects_vram && <VramBadge />}
        <HelpTip help={spec.help} />
      </Label>
      <Switch
        id={`param-${spec.key}`}
        checked={Boolean(value)}
        onCheckedChange={onChange}
      />
    </div>
  );
}

/** select type: radix Select keyed on stringified option values (round-tripped). */
function SelectField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const options = spec.options ?? [];
  const current = value == null ? "" : String(value);

  return (
    <div className="space-y-2">
      <Label htmlFor={`param-${spec.key}`}>
        <span>{spec.label}</span>
        {spec.affects_vram && <VramBadge />}
        <HelpTip help={spec.help} />
      </Label>
      <Select
        value={current}
        onValueChange={(v) => {
          const match = options.find((o) => String(o.value) === v);
          onChange(match ? match.value : v);
        }}
      >
        <SelectTrigger id={`param-${spec.key}`} className="w-full">
          <SelectValue placeholder="Select…" />
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

function ParamField({ spec }: { spec: ParamSpec }) {
  const value = useStudio((s) => s.paramValues[spec.key]);
  const setParam = useStudio((s) => s.setParam);
  const onChange = (v: unknown) => setParam(spec.key, v);

  // The context window is the most consequential knob: make it prominent.
  const prominent = spec.key === "max_model_len";

  switch (spec.type) {
    case "float":
    case "int":
      return (
        <NumberField
          spec={spec}
          value={value}
          onChange={onChange}
          prominent={prominent}
        />
      );
    case "text":
      return <TextField spec={spec} value={value} onChange={onChange} />;
    case "bool":
      return <BoolField spec={spec} value={value} onChange={onChange} />;
    case "select":
      return <SelectField spec={spec} value={value} onChange={onChange} />;
    default:
      return null;
  }
}

function GroupCard({ group }: { group: ParamGroup }) {
  const Icon = GROUP_ICONS[group.key] ?? Sliders;
  const prominent = group.params.some((p) => p.key === "max_model_len");
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Icon className="size-4 text-muted-foreground" />
          {group.label}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {group.params.map((spec) => (
          <div
            key={spec.key}
            className={cn(
              prominent &&
                spec.key === "max_model_len" &&
                "rounded-2xl bg-primary/5 p-3 ring-1 ring-primary/20",
            )}
          >
            <ParamField spec={spec} />
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function ParamPanel({ className }: { className?: string }) {
  const schema = useStudio((s) => s.paramSchema);

  if (!schema) {
    return (
      <div
        className={cn(
          "flex h-full min-h-32 items-center justify-center p-6 text-center text-sm text-muted-foreground",
          className,
        )}
      >
        Load or select a model to edit parameters
      </div>
    );
  }

  const isDiffusion = schema.family === "diffusion";

  return (
    <ScrollArea className={cn("h-full", className)}>
      <div className="space-y-4 p-1">
        {isDiffusion && (
          <div className="flex items-start gap-2 rounded-2xl border bg-muted/40 p-3 text-xs text-muted-foreground">
            <Sparkles className="mt-0.5 size-3.5 shrink-0 text-primary" />
            <span>
              Diffusion LM ({schema.model_type}): non-autoregressive generation.
              vLLM support is experimental — sampling params are passed via
              extra_body.
            </span>
          </div>
        )}
        {schema.groups.map((group) => (
          <GroupCard key={group.key} group={group} />
        ))}
      </div>
    </ScrollArea>
  );
}

export default ParamPanel;
