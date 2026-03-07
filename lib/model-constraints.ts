import { models } from "@/data/models";
import type { EstimateInput, Mode, ModelSpec } from "@/lib/types";

const modelMap = new Map(models.map((model) => [model.id, model]));

export function getModelSpec(modelId: string): ModelSpec {
  return modelMap.get(modelId) ?? models[0];
}

export function applyModelConstraints(input: EstimateInput): EstimateInput {
  const model = getModelSpec(input.modelId);
  const supportedModes = model.supportedModes ?? ["inference", "training"];
  const nextMode = supportedModes.includes(input.mode) ? input.mode : supportedModes[0];
  const nextDtype = model.fixedDtype ?? input.dtype;

  return {
    ...input,
    mode: nextMode,
    dtype: nextDtype,
  };
}

export function modelSupportsMode(model: ModelSpec, mode: Mode): boolean {
  return (model.supportedModes ?? ["inference", "training"]).includes(mode);
}
