import type { KvCacheDtype, RuntimeId, RuntimeSpec } from "@/lib/types";

export const runtimeOptions: RuntimeSpec[] = [
  { id: "vllm", label: "vLLM" },
  { id: "transformers", label: "Transformers" },
];

export const kvCacheDtypeOptions: Array<{
  value: KvCacheDtype;
  label: string;
}> = [
  { value: "bf16", label: "BF16" },
  { value: "fp8", label: "FP8" },
];

export const ALL_RUNTIMES = runtimeOptions.map((runtime) => runtime.id);

const runtimeMap = new Map(runtimeOptions.map((runtime) => [runtime.id, runtime]));

export function getRuntimeSpec(runtimeId: RuntimeId): RuntimeSpec {
  return runtimeMap.get(runtimeId) ?? runtimeOptions[0];
}

export function runtimeSupportsKvCacheDtype(runtimeId: RuntimeId): boolean {
  return runtimeId === "vllm";
}
