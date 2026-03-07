import { gpus } from "@/data/gpus";
import { models } from "@/data/models";
import type { Dtype, EstimateInput, Mode, TrainingType } from "@/lib/types";

export const DECIMAL_GB = 1_000_000_000;
export const LORA_RANK = 16;
export const QLORA_WEIGHT_BYTES = 0.55;

export const modeOptions: Array<{ value: Mode; label: string }> = [
  { value: "inference", label: "Inference" },
  { value: "training", label: "Training" },
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

export const DEFAULT_INPUT: EstimateInput = {
  mode: "inference",
  trainingType: "sft",
  modelId: models[0].id,
  dtype: "fp16",
  gpuId: gpus[1].id,
  customVramGb: 24,
  contextLength: 4096,
  batchSize: 1,
  gradientCheckpointing: false,
  sequencePacking: false,
};
