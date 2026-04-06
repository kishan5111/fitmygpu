import { DECIMAL_GB } from "@/lib/constants";
import type { Dtype } from "@/lib/types";

export function bytesToGb(bytes: number): number {
  return bytes / DECIMAL_GB;
}

export function formatGb(bytes: number): string {
  return `${bytesToGb(bytes).toFixed(1)} GB`;
}

export function formatBandwidth(bandwidthGbps?: number): string {
  if (!bandwidthGbps) {
    return "Not modeled";
  }

  return `${bandwidthGbps.toLocaleString()} GB/s`;
}

export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString();
}

export function formatParamCount(params: number): string {
  if (params >= 1_000_000_000) {
    return `${stripTrailingZero((params / 1_000_000_000).toFixed(1))}B`;
  }

  if (params >= 1_000_000) {
    return `${stripTrailingZero((params / 1_000_000).toFixed(1))}M`;
  }

  return formatInteger(params);
}

export function formatBytesPerParam(bytes: number): string {
  return stripTrailingZero(bytes.toFixed(2));
}

export function formatDtype(dtype: Dtype): string {
  switch (dtype) {
    case "fp32":
      return "FP32";
    case "fp16":
      return "FP16";
    case "bf16":
      return "BF16";
    case "fp8":
      return "FP8";
    case "int8":
      return "INT8";
    case "int4":
      return "4-bit";
  }
}

export function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

function stripTrailingZero(value: string): string {
  return value.replace(/\.0$/, "");
}
