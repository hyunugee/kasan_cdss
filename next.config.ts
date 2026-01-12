import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  // Ensure the python service files (including vendored libs) are included in the standalone build
  outputFileTracingIncludes: {
    '/api/predict': ['./tacrolimus-service/**/*', './tacrolimus-service/libs/**/*'],
  },
};

export default nextConfig;
