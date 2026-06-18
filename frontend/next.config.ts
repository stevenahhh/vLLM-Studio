import type { NextConfig } from "next"

const allowedDevOrigins = (process.env.NEXT_ALLOWED_DEV_ORIGINS ?? "100.78.6.88")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean)

const nextConfig: NextConfig = {
  allowedDevOrigins,
}

export default nextConfig
