"use client"

import * as React from "react"
import { ThemeProvider } from "next-themes"

import { SidebarProvider } from "@/components/ui/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Toaster } from "@/components/ui/sonner"
import { useStudio } from "@/lib/store"

export function Providers({ children }: { children: React.ReactNode }) {
  React.useEffect(() => {
    useStudio.getState().init()
  }, [])

  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <TooltipProvider>
        <SidebarProvider>{children}</SidebarProvider>
      </TooltipProvider>
      <Toaster />
    </ThemeProvider>
  )
}
