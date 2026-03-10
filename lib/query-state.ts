import { gpus } from "@/data/gpus";
import { models } from "@/data/models";
import { DEFAULT_INPUT } from "@/lib/constants";
import type {
  Dtype,
  EstimateInput,
  Mode,
  TrainingType,
} from "@/lib/types";

type SearchParamsLike =
  | URLSearchParams
  | {
      get(name: string): string | null;
      keys(): IterableIterator<string>;
    }
  | Record<string, string | string[] | undefined>;

type SearchParamsRecord = Record<string, string | string[] | undefined>;

const modeValues = new Set<Mode>(["inference", "training"]);
const trainingTypeValues = new Set<TrainingType>([
  "sft",
  "lora",
  "qlora",
  "grpo",
]);
const dtypeValues = new Set<Dtype>(["fp32", "fp16", "bf16", "fp8", "int8", "int4"]);
const modelIds = new Set(models.map((model) => model.id));
const gpuIds = new Set(gpus.map((gpu) => gpu.id));

export function parseSearchParams(searchParams: SearchParamsLike): EstimateInput {
  const read = (key: string): string | undefined => {
    if (searchParams instanceof URLSearchParams) {
      return searchParams.get(key) ?? undefined;
    }

    if (hasGetMethod(searchParams)) {
      return searchParams.get(key) ?? undefined;
    }

    const value = searchParams[key];
    return Array.isArray(value) ? value[0] : value;
  };

  const mode = read("mode");
  const trainingType = read("train");
  const dtype = read("dtype");
  const inferenceProfileId = read("profile");
  const modelId = read("model");
  const gpuId = read("gpu");
  const customVramGb = parseNumber(read("vram"), DEFAULT_INPUT.customVramGb);
  const contextLength = parseNumber(read("ctx"), DEFAULT_INPUT.contextLength);
  const batchSize = parseNumber(read("batch"), DEFAULT_INPUT.batchSize);

  return {
    mode: modeValues.has(mode as Mode) ? (mode as Mode) : DEFAULT_INPUT.mode,
    trainingType: trainingTypeValues.has(trainingType as TrainingType)
      ? (trainingType as TrainingType)
      : DEFAULT_INPUT.trainingType,
    dtype: dtypeValues.has(dtype as Dtype) ? (dtype as Dtype) : DEFAULT_INPUT.dtype,
    inferenceProfileId: inferenceProfileId ?? "",
    modelId: modelIds.has(modelId ?? "") ? (modelId as string) : DEFAULT_INPUT.modelId,
    gpuId: gpuIds.has(gpuId ?? "") ? (gpuId as string) : DEFAULT_INPUT.gpuId,
    customVramGb,
    contextLength,
    batchSize,
    gradientCheckpointing: read("gc") === "1",
    sequencePacking: read("pack") === "1",
  };
}

export function normalizeEstimateInput(input: EstimateInput): EstimateInput {
  return {
    ...input,
    contextLength: clampInteger(input.contextLength, 256, 256_000),
    batchSize: clampInteger(input.batchSize, 1, 64),
    customVramGb: clampFloat(input.customVramGb, 1, 512),
  };
}

export function serializeEstimateInput(input: EstimateInput): URLSearchParams {
  const normalized = normalizeEstimateInput(input);
  const params = new URLSearchParams();
  params.set("mode", normalized.mode);
  params.set("train", normalized.trainingType);
  params.set("model", normalized.modelId);
  params.set("dtype", normalized.dtype);
  params.set("profile", normalized.inferenceProfileId);
  params.set("gpu", normalized.gpuId);
  params.set("vram", normalized.customVramGb.toString());
  params.set("ctx", normalized.contextLength.toString());
  params.set("batch", normalized.batchSize.toString());
  params.set("gc", normalized.gradientCheckpointing ? "1" : "0");
  params.set("pack", normalized.sequencePacking ? "1" : "0");
  return params;
}

export function hasMeaningfulSearchParams(searchParams: SearchParamsLike): boolean {
  if (searchParams instanceof URLSearchParams) {
    return Array.from(searchParams.keys()).length > 0;
  }

  if (hasGetMethod(searchParams)) {
    return Array.from(searchParams.keys()).length > 0;
  }

  return Object.keys(searchParams).length > 0;
}

function hasGetMethod(
  searchParams: SearchParamsLike,
): searchParams is Exclude<SearchParamsLike, SearchParamsRecord> {
  return typeof (searchParams as { get?: unknown }).get === "function";
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clampInteger(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.round(value)));
}

function clampFloat(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
