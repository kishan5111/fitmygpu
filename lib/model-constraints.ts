import { models } from "@/data/models";
import {
  DEFAULT_KV_CACHE_DTYPE,
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

export function getModelSpec(modelId: string): ModelSpec {
  return modelMap.get(modelId) ?? models[0];
}

export function getInferenceProfiles(modelOrId: ModelSpec | string): InferenceProfile[] {
  const model = typeof modelOrId === "string" ? getModelSpec(modelOrId) : modelOrId;
  return model.inferenceProfiles;
}

export function getCompatibleInferenceProfiles(
  modelOrId: ModelSpec | string,
  runtimeId: RuntimeId,
): InferenceProfile[] {
  return getInferenceProfiles(modelOrId).filter((profile) =>
    profileSupportsRuntime(profile, runtimeId),
  );
}

export function getAllowedLoadDtypes(profile: InferenceProfile): Dtype[] {
  if (!canEstimateAlternateLoadDtype(profile)) {
    return [profile.effectiveDtype];
  }

  return ["fp32", "fp16", "bf16", "fp8", "int8", "int4"];
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
    model.inferenceProfiles[0]
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
        ? clampLoadDtype(nextInferenceProfile, nextDtype)
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
    return availableProfiles.find((profile) => profile.effectiveDtype === "fp32");
  }

  if (dtype === "fp8") {
    return availableProfiles.find((profile) => profile.effectiveDtype === "fp8");
  }

  if (dtype === "int8") {
    return availableProfiles.find((profile) => profile.effectiveDtype === "int8");
  }

  if (dtype === "int4") {
    return availableProfiles.find((profile) => profile.effectiveDtype === "int4");
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

function clampLoadDtype(profile: InferenceProfile, dtype: Dtype): Dtype {
  return getAllowedLoadDtypes(profile).includes(dtype) ? dtype : profile.effectiveDtype;
}

function canEstimateAlternateLoadDtype(profile: InferenceProfile): boolean {
  if (profile.loadDtypeMode === "profile_only") {
    return false;
  }

  if (profile.loadDtypeMode === "estimate_from_load_dtype") {
    return true;
  }

  if (!profile.official || profile.weightMode !== "direct") {
    return false;
  }

  if (
    profile.effectiveDtype !== "fp32" &&
    profile.effectiveDtype !== "fp16" &&
    profile.effectiveDtype !== "bf16"
  ) {
    return false;
  }

  if (profile.weightBytes === undefined) {
    return true;
  }

  const expectedBytes =
    profile.effectiveDtype === "fp32" ? 4 : 2;

  return Math.abs(profile.weightBytes - expectedBytes) <= 0.2;
}
