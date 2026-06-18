import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"

import "./globals.css"
import { cn } from "@/lib/utils"
import { Providers } from "@/components/providers"
import { AppSidebar } from "@/components/app-sidebar"
import { VramInspector } from "@/components/vram-inspector"
import { SidebarInset } from "@/components/ui/sidebar"

const geistHeading = Geist({subsets:['latin'],variable:'--font-heading'});

const fontSans = Geist({
  subsets: ["latin"],
  variable: "--font-sans",
})

const geistMono = Geist_Mono({subsets:['latin'],variable:'--font-mono'})

export const metadata: Metadata = {
  title: "vLLM Studio",
  description: "Local vLLM control plane: model picker, VRAM estimation, and chat.",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn(
        "h-svh overflow-hidden antialiased",
        fontSans.variable,
        "font-mono",
        geistMono.variable,
        geistHeading.variable,
      )}
    >
      <body className="h-svh overflow-hidden">
        <Providers>
          <AppSidebar />
          <SidebarInset>{children}</SidebarInset>
          <VramInspector />
        </Providers>
      </body>
    </html>
  )
}
