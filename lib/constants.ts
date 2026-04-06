import { gpus } from "@/data/gpus";
import { models } from "@/data/models";
import type {
  Dtype,
  EstimateInput,
  KvCacheDtype,
  Mode,
  RuntimeId,
  TrainingType,
} from "@/lib/types";

export const DECIMAL_GB = 1_000_000_000;
export const BINARY_GIB = 1024 * 1024 * 1024;
export const LORA_RANK = 16;
export const QLORA_WEIGHT_BYTES = 0.55;
export const TRAINING_ENABLED = false;
export const TRANSFORMERS_BASELINE_CONTEXT_LENGTH = 4096;
export const TRANSFORMERS_BASELINE_BATCH_SIZE = 1;

export const modeOptions: Array<{ value: Mode; label: string }> = [
  { value: "inference", label: "Inference" },
];

export const trainingTypeOptions: Array<{
  value: TrainingType;
  label: string;
}> = [
  { value: "sft", label: "SFT" },
  { value: "lora", label: "LoRA" },
  { value: "qlora", label: "QLoRA" },
  { value: "grpo", label: "GRPO (beta)" },
];

export const dtypeOptions: Array<{ value: Dtype; label: string }> = [
  { value: "fp32", label: "FP32" },
  { value: "fp16", label: "FP16" },
  { value: "bf16", label: "BF16" },
  { value: "fp8", label: "FP8" },
  { value: "int8", label: "INT8" },
  { value: "int4", label: "4-bit" },
];

export const defaultRuntimeId: RuntimeId = "vllm";
export const DEFAULT_KV_CACHE_DTYPE: KvCacheDtype = "bf16";

export const DEFAULT_INPUT: EstimateInput = {
  mode: "inference",
  trainingType: "sft",
  runtimeId: defaultRuntimeId,
  kvCacheDtype: DEFAULT_KV_CACHE_DTYPE,
  modelId: models[0].id,
  dtype: "fp16",
  inferenceProfileId: models[0].inferenceProfiles[0]?.id ?? "",
  gpuId: gpus[1].id,
  customVramGb: 24,
  contextLength: TRANSFORMERS_BASELINE_CONTEXT_LENGTH,
  batchSize: 1,
  gradientCheckpointing: false,
  sequencePacking: false,
};
