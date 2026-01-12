import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  // Ensure the converted ONNX model and necessary assets are included
  outputFileTracingIncludes: {
    '/api/predict': [
      './tacrolimus-service/**/*.onnx',
      './src/app/analytics/page.module.css'
    ],
  },
};

export default nextConfig;
