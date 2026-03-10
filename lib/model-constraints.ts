import { models } from "@/data/models";
import { QLORA_WEIGHT_BYTES } from "@/lib/constants";
import type { Dtype, EstimateInput, InferenceProfile, Mode, ModelSpec } from "@/lib/types";

const modelMap = new Map(models.map((model) => [model.id, model]));

const proxyProfiles: InferenceProfile[] = [
  {
    id: "proxy-fp32",
    label: "Proxy FP32 estimate",
    effectiveDtype: "fp32",
    official: false,
    note: "Fallback dense-style FP32 estimate when no official FP32 deployment checkpoint is selected.",
    sourceUrl: "",
    weightMode: "direct",
    weightBytes: 4,
  },
  {
    id: "proxy-fp8",
    label: "Proxy FP8 estimate",
    effectiveDtype: "fp8",
    official: false,
    note: "Fallback FP8-style estimate. Real deployed FP8 checkpoints can carry extra metadata or packing overhead.",
    sourceUrl: "",
    weightMode: "direct",
    weightBytes: 1,
  },
  {
    id: "proxy-int8",
    label: "Proxy INT8 estimate",
    effectiveDtype: "int8",
    official: false,
    note: "Fallback INT8 estimate. Real INT8 checkpoints can land above or below this depending on scales, zeros, and packing.",
    sourceUrl: "",
    weightMode: "direct",
    weightBytes: 1,
  },
  {
    id: "proxy-int4",
    label: "Proxy 4-bit estimate",
    effectiveDtype: "int4",
    official: false,
    note: "Fallback 4-bit estimate using the generic v0 bytes-per-parameter assumption rather than an official checkpoint size.",
    sourceUrl: "",
    weightMode: "direct",
    weightBytes: QLORA_WEIGHT_BYTES,
  },
];

export function getModelSpec(modelId: string): ModelSpec {
  return modelMap.get(modelId) ?? models[0];
}

export function getInferenceProfiles(modelOrId: ModelSpec | string): InferenceProfile[] {
  const model = typeof modelOrId === "string" ? getModelSpec(modelOrId) : modelOrId;
  return [...model.inferenceProfiles, ...proxyProfiles];
}

export function getInferenceProfile(
  modelOrId: ModelSpec | string,
  inferenceProfileId?: string,
  fallbackDtype?: Dtype,
): InferenceProfile {
  const model = typeof modelOrId === "string" ? getModelSpec(modelOrId) : modelOrId;
  const available = getInferenceProfiles(model);

  if (inferenceProfileId) {
    const exact = available.find((profile) => profile.id === inferenceProfileId);
    if (exact) {
      return exact;
    }
  }

  return pickInferenceProfileForDtype(model, fallbackDtype ?? "bf16");
}

export function applyModelConstraints(input: EstimateInput): EstimateInput {
  const model = getModelSpec(input.modelId);
  const supportedModes = model.supportedModes ?? ["inference", "training"];
  const nextMode = supportedModes.includes(input.mode) ? input.mode : supportedModes[0];
  const nextDtype = model.fixedDtype ?? input.dtype;
  const nextInferenceProfile = getInferenceProfile(
    model,
    input.inferenceProfileId,
    nextDtype,
  );

  return {
    ...input,
    mode: nextMode,
    dtype: nextMode === "inference" ? nextInferenceProfile.effectiveDtype : nextDtype,
    inferenceProfileId: nextInferenceProfile.id,
  };
}

export function modelSupportsMode(model: ModelSpec, mode: Mode): boolean {
  return (model.supportedModes ?? ["inference", "training"]).includes(mode);
}

function pickInferenceProfileForDtype(model: ModelSpec, dtype: Dtype): InferenceProfile {
  if (dtype === "fp32") {
    return proxyProfiles[0];
  }

  if (dtype === "fp8") {
    return (
      model.inferenceProfiles.find((profile) => profile.effectiveDtype === "fp8") ??
      proxyProfiles[1]
    );
  }

  if (dtype === "int8") {
    return proxyProfiles[2];
  }

  if (dtype === "int4") {
    return (
      model.inferenceProfiles.find((profile) => profile.effectiveDtype === "int4") ??
      proxyProfiles[3]
    );
  }

  return (
    model.inferenceProfiles.find(
      (profile) => profile.effectiveDtype === "bf16" || profile.effectiveDtype === "fp16",
    ) ?? model.inferenceProfiles[0]
  );
}
