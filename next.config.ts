import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
    // Ensure the python service files are included in the standalone build
    // @ts-ignore
    outputFileTracingIncludes: {
      '/api/predict': ['./tacrolimus-service/**/*'],
    },
  },
};

export default nextConfig;
