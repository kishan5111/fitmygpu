import type { GpuSpec } from "@/lib/types";

export const gpus: GpuSpec[] = [
  {
    id: "rtx-3090-24gb",
    displayName: "RTX 3090 24GB",
    vramGb: 24,
    memoryBandwidthGbps: 936,
    classType: "Consumer",
  },
  {
    id: "rtx-4090-24gb",
    displayName: "RTX 4090 24GB",
    vramGb: 24,
    memoryBandwidthGbps: 1008,
    classType: "Consumer",
  },
  {
    id: "a100-40gb",
    displayName: "A100 40GB",
    vramGb: 40,
    memoryBandwidthGbps: 1555,
    classType: "Datacenter",
  },
  {
    id: "a100-80gb",
    displayName: "A100 80GB",
    vramGb: 80,
    memoryBandwidthGbps: 2039,
    classType: "Datacenter",
  },
  {
    id: "h100-80gb",
    displayName: "H100 80GB",
    vramGb: 80,
    memoryBandwidthGbps: 3350,
    classType: "Datacenter",
  },
  {
    id: "custom",
    displayName: "Custom VRAM",
    vramGb: 24,
    classType: "Custom",
  },
];
