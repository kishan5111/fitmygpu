import { models } from "@/data/models";
import {
  DEFAULT_KV_CACHE_DTYPE,
  QLORA_WEIGHT_BYTES,
  TRANSFORMERS_BASELINE_BATCH_SIZE,
  TRANSFORMERS_BASELINE_CONTEXT_LENGTH,
  TRAINING_ENABLED,
} from "@/lib/constants";
import { ALL_RUNTIMES, runtimeSupportsKvCacheDtype } from "@/lib/runtime";
import type {
  Dtype,
  EstimateInput,
  InferenceProfile,
  Mode,
  ModelSpec,
  RuntimeId,
} from "@/lib/types";

const modelMap = new Map(models.map((model) => [model.id, model]));

const proxyProfiles: InferenceProfile[] = [
  {
    id: "proxy-fp32",
    label: "Proxy FP32 estimate",
    effectiveDtype: "fp32",
    official: false,
    confidence: "proxy",
    note: "Fallback dense-style FP32 estimate when no official FP32 deployment checkpoint is selected.",
    sourceUrl: "",
    weightMode: "direct",
    weightBytes: 4,
    supportedRuntimes: ALL_RUNTIMES,
  },
  {
    id: "proxy-fp8",
    label: "Proxy FP8 estimate",
    effectiveDtype: "fp8",
    official: false,
    confidence: "proxy",
    note: "Fallback FP8-style estimate. Real deployed FP8 checkpoints can carry extra metadata or packing overhead.",
    sourceUrl: "",
    weightMode: "direct",
    weightBytes: 1,
    supportedRuntimes: ALL_RUNTIMES,
  },
  {
    id: "proxy-int8",
    label: "Proxy INT8 estimate",
    effectiveDtype: "int8",
    official: false,
    confidence: "proxy",
    note: "Fallback INT8 estimate. Real INT8 checkpoints can land above or below this depending on scales, zeros, and packing.",
    sourceUrl: "",
    weightMode: "direct",
    weightBytes: 1,
    supportedRuntimes: ALL_RUNTIMES,
  },
  {
    id: "proxy-int4",
    label: "Proxy 4-bit estimate",
    effectiveDtype: "int4",
    official: false,
    confidence: "proxy",
    note: "Fallback 4-bit estimate using the generic estimator bytes-per-parameter assumption rather than an official checkpoint size.",
    sourceUrl: "",
    weightMode: "direct",
    weightBytes: QLORA_WEIGHT_BYTES,
    supportedRuntimes: ALL_RUNTIMES,
  },
];

export function getModelSpec(modelId: string): ModelSpec {
  return modelMap.get(modelId) ?? models[0];
}

export function getInferenceProfiles(modelOrId: ModelSpec | string): InferenceProfile[] {
  const model = typeof modelOrId === "string" ? getModelSpec(modelOrId) : modelOrId;
  return [...model.inferenceProfiles, ...proxyProfiles];
}

export function getCompatibleInferenceProfiles(
  modelOrId: ModelSpec | string,
  runtimeId: RuntimeId,
): InferenceProfile[] {
  return getInferenceProfiles(modelOrId).filter((profile) =>
    profileSupportsRuntime(profile, runtimeId),
  );
}

export function getInferenceProfile(
  modelOrId: ModelSpec | string,
  inferenceProfileId?: string,
  fallbackDtype?: Dtype,
  runtimeId?: RuntimeId,
): InferenceProfile {
  const model = typeof modelOrId === "string" ? getModelSpec(modelOrId) : modelOrId;
  const available =
    runtimeId === undefined
      ? getInferenceProfiles(model)
      : getCompatibleInferenceProfiles(model, runtimeId);

  if (inferenceProfileId) {
    const exact = available.find((profile) => profile.id === inferenceProfileId);
    if (exact) {
      return exact;
    }
  }

  return (
    pickInferenceProfileForDtype(fallbackDtype ?? "bf16", available) ??
    getInferenceProfiles(model)[0]
  );
}

export function getCompatibleInferenceProfile(
  modelOrId: ModelSpec | string,
  runtimeId: RuntimeId,
  inferenceProfileId?: string,
  fallbackDtype?: Dtype,
): InferenceProfile | null {
  const model = typeof modelOrId === "string" ? getModelSpec(modelOrId) : modelOrId;
  const available = getCompatibleInferenceProfiles(model, runtimeId);

  if (available.length === 0) {
    return null;
  }

  if (inferenceProfileId) {
    const exact = available.find((profile) => profile.id === inferenceProfileId);
    if (exact) {
      return exact;
    }
  }

  return pickInferenceProfileForDtype(fallbackDtype ?? "bf16", available) ?? available[0];
}

export function hasCompatibleInferenceProfile(
  modelOrId: ModelSpec | string,
  runtimeId: RuntimeId,
): boolean {
  return getCompatibleInferenceProfiles(modelOrId, runtimeId).length > 0;
}

export function getInferenceProfileSourceUrl(
  modelOrId: ModelSpec | string,
  inferenceProfileId: string | undefined,
  runtimeId?: RuntimeId,
  fallbackDtype?: Dtype,
): string {
  const model = typeof modelOrId === "string" ? getModelSpec(modelOrId) : modelOrId;
  const profile = getInferenceProfile(
    model,
    inferenceProfileId,
    fallbackDtype,
    runtimeId,
  );

  return profile.sourceUrl || model.sourceUrl;
}

export function applyModelConstraints(input: EstimateInput): EstimateInput {
  const model = getModelSpec(input.modelId);
  const supportedModes = getSupportedModes(model);
  const nextMode = supportedModes.includes(input.mode) ? input.mode : supportedModes[0];
  const nextDtype = model.fixedDtype ?? input.dtype;
  const nextInferenceProfile = getCompatibleInferenceProfile(
    model,
    input.runtimeId,
    input.inferenceProfileId,
    nextDtype,
  );

  return {
    ...input,
    mode: nextMode,
    contextLength:
      nextMode === "inference" && input.runtimeId === "transformers"
        ? TRANSFORMERS_BASELINE_CONTEXT_LENGTH
        : input.contextLength,
    batchSize:
      nextMode === "inference" && input.runtimeId === "transformers"
        ? TRANSFORMERS_BASELINE_BATCH_SIZE
        : input.batchSize,
    gpuCount:
      nextMode === "inference" && input.runtimeId === "transformers"
        ? 1
        : input.gpuCount,
    kvCacheDtype:
      nextMode === "inference" && runtimeSupportsKvCacheDtype(input.runtimeId)
        ? input.kvCacheDtype
        : DEFAULT_KV_CACHE_DTYPE,
    dtype:
      nextMode === "inference" && nextInferenceProfile
        ? nextInferenceProfile.effectiveDtype
        : nextDtype,
    inferenceProfileId:
      nextMode === "inference" ? nextInferenceProfile?.id ?? "" : input.inferenceProfileId,
  };
}

export function modelSupportsMode(model: ModelSpec, mode: Mode): boolean {
  return getSupportedModes(model).includes(mode);
}

function getSupportedModes(model: ModelSpec): Mode[] {
  const supportedModes = model.supportedModes ?? ["inference", "training"];

  if (!TRAINING_ENABLED) {
    return ["inference"];
  }

  return supportedModes;
}

function pickInferenceProfileForDtype(
  dtype: Dtype,
  availableProfiles: InferenceProfile[],
): InferenceProfile | undefined {
  if (dtype === "fp32") {
    return availableProfiles.find((profile) => profile.id === "proxy-fp32");
  }

  if (dtype === "fp8") {
    return (
      availableProfiles.find((profile) => profile.effectiveDtype === "fp8") ??
      availableProfiles.find((profile) => profile.id === "proxy-fp8")
    );
  }

  if (dtype === "int8") {
    return availableProfiles.find((profile) => profile.id === "proxy-int8");
  }

  if (dtype === "int4") {
    return (
      availableProfiles.find((profile) => profile.effectiveDtype === "int4") ??
      availableProfiles.find((profile) => profile.id === "proxy-int4")
    );
  }

  return (
    availableProfiles.find(
      (profile) => profile.effectiveDtype === "bf16" || profile.effectiveDtype === "fp16",
    ) ?? availableProfiles[0]
  );
}

function profileSupportsRuntime(profile: InferenceProfile, runtimeId: RuntimeId): boolean {
  return (profile.supportedRuntimes ?? ALL_RUNTIMES).includes(runtimeId);
}
