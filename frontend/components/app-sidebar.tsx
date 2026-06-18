"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import {
  Activity,
  Boxes,
  Cpu,
  Download,
  LayoutDashboard,
  MessagesSquare,
  Moon,
  Sun,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { useStudio } from "@/lib/store";
import type { EngineState } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { ModelPicker } from "@/components/model-picker";
import { DownloadManager } from "@/components/download-manager";

const NAV_ITEMS = [
  { href: "/chat", label: "Chat", icon: MessagesSquare },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/tuner", label: "Tuner", icon: Activity },
] as const;

/** Map an engine state to a status dot color + descriptive label. */
function engineStatus(state: EngineState | undefined): {
  dot: string;
  label: string;
  pulse: boolean;
} {
  switch (state) {
    case "ready":
      return { dot: "bg-primary", label: "Ready", pulse: false };
    case "loading":
      return { dot: "bg-amber-500", label: "Loading", pulse: true };
    case "error":
      return { dot: "bg-destructive", label: "Error", pulse: false };
    default:
      return { dot: "bg-muted-foreground", label: "Stopped", pulse: false };
  }
}

export function AppSidebar() {
  const pathname = usePathname();

  const engine = useStudio((s) => s.engine);
  const hardware = useStudio((s) => s.hardware);

  const [pickerOpen, setPickerOpen] = React.useState(false);
  const [downloadsOpen, setDownloadsOpen] = React.useState(false);

  const status = engineStatus(engine?.state);
  const modelLabel = engine?.repo || "No model loaded";
  const numGpus = hardware?.num_gpus ?? hardware?.gpus.length ?? 0;

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center justify-between gap-2">
          <Link
            href="/chat"
            className="flex min-w-0 items-center gap-2 px-1 font-heading text-sm font-semibold text-foreground"
          >
            <Boxes className="size-5 shrink-0 text-primary" />
            <span className="truncate group-data-[collapsible=icon]:hidden">
              vLLM Studio
            </span>
          </Link>
          <SidebarTrigger className="group-data-[collapsible=icon]:hidden" />
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigate</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV_ITEMS.map((item) => {
                const isActive =
                  pathname === item.href ||
                  pathname.startsWith(`${item.href}/`);
                const Icon = item.icon;
                return (
                  <SidebarMenuItem key={item.href}>
                    <SidebarMenuButton
                      asChild
                      isActive={isActive}
                      tooltip={item.label}
                    >
                      <Link href={item.href}>
                        <Icon />
                        <span>{item.label}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Model</SidebarGroupLabel>
          <SidebarGroupContent className="flex flex-col gap-2">
            <div
              className="flex items-center gap-2 rounded-xl border border-sidebar-border bg-sidebar-accent/40 px-2.5 py-2 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-0"
              title={`${status.label}: ${modelLabel}`}
            >
              <span className="relative flex size-2 shrink-0 items-center justify-center">
                {status.pulse && (
                  <span
                    className={cn(
                      "absolute inline-flex size-2 animate-ping rounded-full opacity-75",
                      status.dot,
                    )}
                  />
                )}
                <span
                  className={cn(
                    "relative inline-flex size-2 rounded-full",
                    status.dot,
                  )}
                />
              </span>
              <div className="flex min-w-0 flex-col group-data-[collapsible=icon]:hidden">
                <span className="truncate text-sm font-medium text-sidebar-foreground">
                  {modelLabel}
                </span>
                <span className="truncate text-xs text-muted-foreground">
                  {status.label}
                  {engine?.quant && engine.quant !== "none"
                    ? ` · ${engine.quant}`
                    : ""}
                </span>
              </div>
            </div>

            <Button
              size="sm"
              className="w-full group-data-[collapsible=icon]:hidden"
              onClick={() => setPickerOpen(true)}
            >
              <Boxes />
              Load model
            </Button>
            <Button
              size="icon-sm"
              className="mx-auto hidden group-data-[collapsible=icon]:flex"
              onClick={() => setPickerOpen(true)}
              title="Load model"
            >
              <Boxes />
              <span className="sr-only">Load model</span>
            </Button>

            <Button
              size="sm"
              variant="ghost"
              className="w-full justify-start group-data-[collapsible=icon]:hidden"
              onClick={() => setDownloadsOpen(true)}
            >
              <Download />
              Downloads
            </Button>
            <Button
              size="icon-sm"
              variant="ghost"
              className="mx-auto hidden group-data-[collapsible=icon]:flex"
              onClick={() => setDownloadsOpen(true)}
              title="Downloads"
            >
              <Download />
              <span className="sr-only">Downloads</span>
            </Button>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <div className="flex items-center justify-between gap-2 group-data-[collapsible=icon]:flex-col">
          <ThemeToggle />
          <div
            className="flex items-center gap-1.5 px-1 text-xs text-muted-foreground group-data-[collapsible=icon]:px-0"
            title={`${numGpus} GPU${numGpus === 1 ? "" : "s"} detected`}
          >
            <Cpu className="size-3.5 shrink-0" />
            <span className="group-data-[collapsible=icon]:hidden">
              {numGpus} {numGpus === 1 ? "GPU" : "GPUs"}
            </span>
            <span className="hidden tabular-nums group-data-[collapsible=icon]:inline">
              {numGpus}
            </span>
          </div>
        </div>
      </SidebarFooter>

      <ModelPicker open={pickerOpen} onOpenChange={setPickerOpen} />

      <Sheet open={downloadsOpen} onOpenChange={setDownloadsOpen}>
        <SheetContent
          side="right"
          className="w-full gap-0 sm:max-w-md"
        >
          <SheetHeader>
            <SheetTitle>Downloads</SheetTitle>
            <SheetDescription>
              Active and finished model downloads.
            </SheetDescription>
          </SheetHeader>
          <div className="min-h-0 flex-1 overflow-auto px-6 pb-6">
            <DownloadManager />
          </div>
        </SheetContent>
      </Sheet>
    </Sidebar>
  );
}

/** next-themes Sun/Moon toggle. Avoids hydration mismatch by gating on mount. */
function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const isDark = resolvedTheme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      title="Toggle theme"
    >
      {mounted && isDark ? <Sun /> : <Moon />}
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}

export default AppSidebar;
