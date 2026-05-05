import { formatInteger, formatParamCount } from "@/lib/format";
import { ALL_RUNTIMES, getRuntimeSpec } from "@/lib/runtime";
import type { InferenceProfile, ModelSpec, ProfileConfidence } from "@/lib/types";

export function formatModelAtGlance(model: ModelSpec) {
  const kvHeads = model.numKvHeads ?? model.numAttentionHeads;
  const parameterLine = model.activeParams
    ? `${formatParamCount(model.totalParams)} total • ${formatParamCount(model.activeParams)} active`
    : `${formatParamCount(model.totalParams)} dense`;

  return `${parameterLine} • ${formatInteger(model.contextLength)} context • ${kvHeads} KV heads`;
}

export function getKvBearingLayers(model: ModelSpec) {
  if (model.cacheStrategy === "hybrid_attention") {
    return model.attentionLayerCount ?? model.numLayers;
  }

  if (model.cacheStrategy === "alternating_window_attention") {
    return (
      (model.denseAttentionLayerCount ?? Math.ceil(model.numLayers / 2)) +
      (model.slidingWindowAttentionLayerCount ?? Math.floor(model.numLayers / 2))
    );
  }

  return model.numLayers;
}

export function getInferenceProfileRuntimeLabels(profile: InferenceProfile) {
  return (profile.supportedRuntimes ?? ALL_RUNTIMES).map(
    (runtimeId) => getRuntimeSpec(runtimeId).label,
  );
}

export function getProfileConfidence(profile: InferenceProfile): ProfileConfidence {
  if (profile.confidence) {
    return profile.confidence;
  }

  return profile.official ? "verified" : "proxy";
}

export function getProfileConfidenceLabel(profile: InferenceProfile) {
  switch (getProfileConfidence(profile)) {
    case "verified":
      return "Verified";
    case "estimated":
      return "Estimated";
    case "proxy":
      return "Proxy";
  }
}

export function isMultimodalModel(model: ModelSpec) {
  return model.modality === "multimodal";
}
