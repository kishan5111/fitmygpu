import { gpus } from "@/data/gpus";
import {
  DECIMAL_GB,
  LORA_RANK,
  QLORA_WEIGHT_BYTES,
} from "@/lib/constants";
import {
  formatBytesPerParam,
  formatDtype,
  formatGb,
  formatInteger,
  formatParamCount,
} from "@/lib/format";
import { applyModelConstraints, getModelSpec } from "@/lib/model-constraints";
import { normalizeEstimateInput } from "@/lib/query-state";
import type {
  BreakdownItem,
  Dtype,
  EstimateInput,
  EstimateResult,
  GpuSpec,
  MathLine,
  ModelSpec,
  TrainingType,
} from "@/lib/types";

const gpuMap = new Map(gpus.map((gpu) => [gpu.id, gpu]));

type Strategy =
  | {
      mode: "inference";
      effectiveDtype: Dtype;
      effectiveTrainingType: TrainingType;
      calculationProfile: string;
      weightBytes: number;
      kvBytes: number;
      warnings: string[];
      proxyReason?: string;
    }
  | {
      mode: "training";
      effectiveDtype: Dtype;
      effectiveTrainingType: TrainingType;
      calculationProfile: string;
      baseWeightBytes: number;
      computeBytes: number;
      warnings: string[];
      proxyReason?: string;
    };

export function estimateVram(input: EstimateInput): EstimateResult {
  const normalized = applyModelConstraints(normalizeEstimateInput(input));
  const model = getModelSpec(normalized.modelId);
  const gpu = getGpu(normalized.gpuId, normalized.customVramGb);
  const warnings: string[] = [];
  const notes: string[] = [];
  const effectiveContextLength = Math.min(normalized.contextLength, model.contextLength);

  if (normalized.contextLength > model.contextLength) {
    warnings.push(
      `Context clipped to ${formatInteger(model.contextLength)} tokens, the model's advertised window.`,
    );
  }

  if (model.isMoe && model.activeParams) {
    notes.push(
      `${model.displayName} is an MoE model: resident VRAM tracks ${formatParamCount(
        model.totalParams,
      )} total params, while per-token compute is closer to ${formatParamCount(
        model.activeParams,
      )} active params.`,
    );
  }

  const strategy = resolveStrategy(normalized);
  warnings.push(...strategy.warnings);

  if (strategy.proxyReason) {
    notes.push(strategy.proxyReason);
  }

  if (model.fixedDtype) {
    notes.push(
      `${model.displayName} is treated as a fixed ${formatDtype(
        model.fixedDtype,
      )} checkpoint in v0.`,
    );
  }

  if ((model.supportedModes ?? ["inference", "training"]).length === 1) {
    notes.push(`${model.displayName} is inference-only in v0.`);
  }

  if (strategy.mode === "inference") {
    return finalizeResult(
      normalized,
      model,
      gpu,
      effectiveContextLength,
      warnings,
      notes,
      buildInferenceEstimate(
        normalized,
        model,
        gpu,
        effectiveContextLength,
        strategy,
      ),
    );
  }

  return finalizeResult(
    normalized,
    model,
    gpu,
    effectiveContextLength,
    warnings,
    notes,
    buildTrainingEstimate(
      normalized,
      model,
      gpu,
      effectiveContextLength,
      strategy,
    ),
  );
}

function buildInferenceEstimate(
  input: EstimateInput,
  model: ModelSpec,
  gpu: GpuSpec,
  effectiveContextLength: number,
  strategy: Extract<Strategy, { mode: "inference" }>,
) {
  const kvHeads = model.numKvHeads ?? model.numAttentionHeads;
  const headDim = model.hiddenSize / model.numAttentionHeads;
  const weightsBytes = model.totalParams * strategy.weightBytes;
  const kvCacheBytes =
    input.batchSize *
    effectiveContextLength *
    model.numLayers *
    2 *
    kvHeads *
    headDim *
    strategy.kvBytes;
  const subtotal = weightsBytes + kvCacheBytes;
  const overheadBytes = Math.max(1.5 * DECIMAL_GB, subtotal * 0.1);
  const totalBytes = subtotal + overheadBytes;

  const breakdown: BreakdownItem[] = [
    {
      key: "weights",
      label: "Weights",
      bytes: weightsBytes,
      note: `${formatParamCount(model.totalParams)} resident parameters at ${formatBytesPerParam(
        strategy.weightBytes,
      )} bytes each.`,
    },
    {
      key: "kvCache",
      label: "KV cache",
      bytes: kvCacheBytes,
      note: `Batch ${input.batchSize}, context ${formatInteger(
        effectiveContextLength,
      )}, ${model.numLayers} layers, ${kvHeads} KV heads.`,
    },
    {
      key: "overhead",
      label: "Runtime / safety overhead",
      bytes: overheadBytes,
      note: "Conservative buffer for allocator fragmentation, kernels, and runtime scratch space.",
    },
  ];

  const math: MathLine[] = [
    {
      label: "Weights",
      symbolic: "parameter count × bytes per weight",
      substituted: `${formatParamCount(model.totalParams)} × ${formatBytesPerParam(
        strategy.weightBytes,
      )} = ${formatGb(weightsBytes)}`,
      note: `${formatDtype(strategy.effectiveDtype)} controls how compact the resident weights are.`,
    },
    {
      label: "KV cache",
      symbolic:
        "batch × context × layers × 2 × KV heads × head dim × 2 bytes",
      substituted: `${input.batchSize} × ${formatInteger(
        effectiveContextLength,
      )} × ${model.numLayers} × 2 × ${kvHeads} × ${headDim} × ${
        strategy.kvBytes
      } = ${formatGb(kvCacheBytes)}`,
      note: "Longer context, deeper models, and larger batches all expand the cache linearly.",
    },
    {
      label: "Overhead",
      symbolic: "max(1.5 GB, 10% of weights + KV cache)",
      substituted: `max(1.5 GB, 10% of ${formatGb(subtotal)}) = ${formatGb(
        overheadBytes,
      )}`,
      note: "This leaves room for runtime buffers instead of claiming an unrealistically exact fit.",
    },
  ];

  return {
    weightsBytes,
    kvCacheBytes,
    activationsBytes: 0,
    gradientsBytes: 0,
    optimizerBytes: 0,
    grpoExtraBytes: 0,
    overheadBytes,
    totalBytes,
    breakdown,
    math,
    effectiveDtype: strategy.effectiveDtype,
    effectiveTrainingType: strategy.effectiveTrainingType,
    calculationProfile: strategy.calculationProfile,
    proxyReason: strategy.proxyReason,
  };
}

function buildTrainingEstimate(
  input: EstimateInput,
  model: ModelSpec,
  gpu: GpuSpec,
  effectiveContextLength: number,
  strategy: Extract<Strategy, { mode: "training" }>,
) {
  const adapterParams = 2 * LORA_RANK * model.hiddenSize * model.numLayers * 4;
  const activationFactor = input.gradientCheckpointing ? 6 : 12;
  const packingMultiplier = input.sequencePacking ? 0.85 : 1;
  const baseActivationBytes =
    input.batchSize *
    effectiveContextLength *
    model.hiddenSize *
    model.numLayers *
    activationFactor *
    strategy.computeBytes;
  const activationsBytes = baseActivationBytes * packingMultiplier;

  let weightsBytes = 0;
  let gradientsBytes = 0;
  let optimizerBytes = 0;
  let grpoExtraBytes = 0;

  if (strategy.effectiveTrainingType === "sft") {
    weightsBytes = model.totalParams * strategy.baseWeightBytes;
    gradientsBytes = model.totalParams * strategy.computeBytes;
    optimizerBytes = model.totalParams * 8;
  } else {
    const adapterWeightBytes = adapterParams * strategy.computeBytes;
    weightsBytes = model.totalParams * strategy.baseWeightBytes + adapterWeightBytes;
    gradientsBytes = adapterParams * strategy.computeBytes;
    optimizerBytes = adapterParams * 8;
  }

  if (strategy.effectiveTrainingType === "grpo") {
    grpoExtraBytes =
      input.batchSize *
      effectiveContextLength *
      model.hiddenSize *
      model.numLayers *
      1.75 *
      2;
  }

  const subtotal =
    weightsBytes +
    gradientsBytes +
    optimizerBytes +
    activationsBytes +
    grpoExtraBytes;
  const overheadBytes = Math.max(2 * DECIMAL_GB, subtotal * 0.1);
  const totalBytes = subtotal + overheadBytes;

  const breakdown: BreakdownItem[] = [
    {
      key: "weights",
      label: "Weights",
      bytes: weightsBytes,
      note:
        strategy.effectiveTrainingType === "sft"
          ? `${formatParamCount(model.totalParams)} trainable parameters remain resident during SFT.`
          : `Frozen base weights plus roughly ${formatParamCount(
              adapterParams,
            )} LoRA adapter parameters.`,
    },
    {
      key: "activations",
      label: "Activations",
      bytes: activationsBytes,
      note: input.gradientCheckpointing
        ? "Checkpointing halves the activation factor from 12 to 6 in this estimator."
        : "Activation memory scales with batch, context, hidden size, and layers.",
    },
    {
      key: "gradients",
      label: "Gradients",
      bytes: gradientsBytes,
      note:
        strategy.effectiveTrainingType === "sft"
          ? "Full fine-tuning keeps gradients for every trainable weight."
          : "Adapter-only training stores gradients just for the LoRA weights.",
    },
    {
      key: "optimizer",
      label: "Optimizer states",
      bytes: optimizerBytes,
      note: "AdamW-style fp32 optimizer state uses 8 bytes per trainable parameter.",
    },
  ];

  if (grpoExtraBytes > 0) {
    breakdown.push({
      key: "grpoExtra",
      label: "GRPO rollout memory",
      bytes: grpoExtraBytes,
      note: "Beta rollout buffer proxy for extra sampled tokens, completions, and reward-side bookkeeping.",
    });
  }

  breakdown.push({
    key: "overhead",
    label: "Runtime / safety overhead",
    bytes: overheadBytes,
    note: "Conservative headroom for kernels, dataloader staging, and allocator fragmentation.",
  });

  const math: MathLine[] = [
    {
      label: "Weights",
      symbolic:
        strategy.effectiveTrainingType === "sft"
          ? "parameter count × bytes per train weight"
          : "base weights + adapter params × compute bytes",
      substituted:
        strategy.effectiveTrainingType === "sft"
          ? `${formatParamCount(model.totalParams)} × ${formatBytesPerParam(
              strategy.baseWeightBytes,
            )} = ${formatGb(weightsBytes)}`
          : `${formatParamCount(model.totalParams)} × ${formatBytesPerParam(
              strategy.baseWeightBytes,
            )} + ${formatParamCount(adapterParams)} × ${formatBytesPerParam(
              strategy.computeBytes,
            )} = ${formatGb(weightsBytes)}`,
      note:
        strategy.effectiveTrainingType === "qlora" ||
        strategy.effectiveTrainingType === "grpo"
          ? "QLoRA-style runs keep the frozen base in 4-bit form and train only adapters."
          : "Dense SFT keeps the whole model trainable; LoRA keeps only adapters trainable.",
    },
    {
      label: "Activations",
      symbolic:
        "batch × context × hidden × layers × activation factor × bytes",
      substituted: `${input.batchSize} × ${formatInteger(
        effectiveContextLength,
      )} × ${formatInteger(model.hiddenSize)} × ${model.numLayers} × ${activationFactor} × ${formatBytesPerParam(
        strategy.computeBytes,
      )}${input.sequencePacking ? " × 0.85" : ""} = ${formatGb(activationsBytes)}`,
      note: input.sequencePacking
        ? "Sequence packing trims the activation estimate by 15% in this v0 approximation."
        : "Checkpointing and packing are the fastest levers for shrinking activation memory.",
    },
    {
      label: "Gradients",
      symbolic:
        strategy.effectiveTrainingType === "sft"
          ? "trainable params × compute bytes"
          : "adapter params × compute bytes",
      substituted:
        strategy.effectiveTrainingType === "sft"
          ? `${formatParamCount(model.totalParams)} × ${formatBytesPerParam(
              strategy.computeBytes,
            )} = ${formatGb(gradientsBytes)}`
          : `${formatParamCount(adapterParams)} × ${formatBytesPerParam(
              strategy.computeBytes,
            )} = ${formatGb(gradientsBytes)}`,
      note: "Training memory jumps because gradients need their own full tensor storage.",
    },
    {
      label: "Optimizer states",
      symbolic: "trainable params × 8 bytes",
      substituted:
        strategy.effectiveTrainingType === "sft"
          ? `${formatParamCount(model.totalParams)} × 8 = ${formatGb(optimizerBytes)}`
          : `${formatParamCount(adapterParams)} × 8 = ${formatGb(optimizerBytes)}`,
      note: "The conservative baseline assumes AdamW-style fp32 moment buffers.",
    },
    {
      label: "Overhead",
      symbolic: "max(2 GB, 10% of subtotal)",
      substituted: `max(2 GB, 10% of ${formatGb(subtotal)}) = ${formatGb(
        overheadBytes,
      )}`,
      note: "This keeps the fit verdict conservative instead of matching an idealized lab setup.",
    },
  ];

  if (grpoExtraBytes > 0) {
    math.splice(4, 0, {
      label: "GRPO rollout",
      symbolic: "batch × context × hidden × layers × 1.75 × 2 bytes",
      substituted: `${input.batchSize} × ${formatInteger(
        effectiveContextLength,
      )} × ${formatInteger(model.hiddenSize)} × ${model.numLayers} × 1.75 × 2 = ${formatGb(
        grpoExtraBytes,
      )}`,
      note: "This beta term stands in for extra sampled tokens and rollout-side buffers.",
    });
  }

  return {
    weightsBytes,
    kvCacheBytes: 0,
    activationsBytes,
    gradientsBytes,
    optimizerBytes,
    grpoExtraBytes,
    overheadBytes,
    totalBytes,
    breakdown,
    math,
    effectiveDtype: strategy.effectiveDtype,
    effectiveTrainingType: strategy.effectiveTrainingType,
    calculationProfile: strategy.calculationProfile,
    proxyReason: strategy.proxyReason,
  };
}

function finalizeResult(
  input: EstimateInput,
  model: ModelSpec,
  gpu: GpuSpec,
  effectiveContextLength: number,
  warnings: string[],
  notes: string[],
  partial: Omit<
    EstimateResult,
    | "input"
    | "model"
    | "gpu"
    | "fits"
    | "gpuBytes"
    | "headroomBytes"
    | "deficitBytes"
    | "warnings"
    | "notes"
    | "tips"
    | "effectiveContextLength"
  >,
): EstimateResult {
  const gpuBytes = gpu.vramGb * DECIMAL_GB;
  const fits = partial.totalBytes <= gpuBytes;
  const headroomBytes = fits ? gpuBytes - partial.totalBytes : 0;
  const deficitBytes = fits ? 0 : partial.totalBytes - gpuBytes;

  return {
    ...partial,
    input,
    model,
    gpu,
    fits,
    gpuBytes,
    headroomBytes,
    deficitBytes,
    warnings,
    notes,
    tips: generateTips({
      model,
      gpu,
      input,
      totalBytes: partial.totalBytes,
      headroomBytes,
      deficitBytes,
      fits,
      activationsBytes: partial.activationsBytes,
      kvCacheBytes: partial.kvCacheBytes,
    }),
    effectiveContextLength,
  };
}

function generateTips(args: {
  model: ModelSpec;
  gpu: GpuSpec;
  input: EstimateInput;
  totalBytes: number;
  headroomBytes: number;
  deficitBytes: number;
  fits: boolean;
  activationsBytes: number;
  kvCacheBytes: number;
}): string[] {
  const tips: string[] = [];

  if (!args.fits && args.input.dtype !== "int4") {
    tips.push("Try 4-bit weights to cut the resident model footprint before touching anything else.");
  }

  if (!args.fits && args.input.contextLength > 2048 && args.kvCacheBytes > 0) {
    tips.push("Reduce context length if KV cache is the fastest-growing term in the estimate.");
  }

  if (!args.fits && args.input.batchSize > 1) {
    tips.push("Lower batch size to shrink KV cache and activation memory linearly.");
  }

  if (
    args.input.mode === "training" &&
    args.input.trainingType === "sft" &&
    !args.fits
  ) {
    tips.push("Use LoRA or QLoRA instead of full SFT if you want a realistic path onto a single card.");
  }

  if (args.input.mode === "training" && !args.input.gradientCheckpointing) {
    tips.push("Enable gradient checkpointing to roughly halve the activation factor in this estimator.");
  }

  if (args.input.mode === "training" && !args.input.sequencePacking) {
    tips.push("Sequence packing is a good next lever when activations dominate the training footprint.");
  }

  if (args.fits && args.headroomBytes < 2 * DECIMAL_GB) {
    tips.push("This technically fits, but the headroom is thin. Leave extra space for kernels and runtime buffers.");
  }

  if (args.input.trainingType === "grpo") {
    tips.push("GRPO memory is highly rollout-dependent; fewer completions or shorter rollouts can swing the result sharply.");
  }

  if (tips.length === 0) {
    tips.push(
      `Keep some spare VRAM on ${args.gpu.displayName} for runtime overhead instead of targeting a zero-margin fit.`,
    );
  }

  return Array.from(new Set(tips)).slice(0, 4);
}

function resolveStrategy(input: EstimateInput): Strategy {
  if (input.mode === "inference") {
    return {
      mode: "inference",
      effectiveDtype: input.dtype,
      effectiveTrainingType: input.trainingType,
      calculationProfile: `${formatDtype(input.dtype)} inference`,
      weightBytes: getInferenceWeightBytes(input.dtype),
      kvBytes: 2,
      warnings: [],
    };
  }

  if (input.trainingType === "sft") {
    if (input.dtype === "int8" || input.dtype === "int4" || input.dtype === "fp8") {
      return {
        mode: "training",
        effectiveDtype: "bf16",
        effectiveTrainingType: "sft",
        calculationProfile: "BF16 SFT proxy",
        baseWeightBytes: 2,
        computeBytes: 2,
        warnings: [
          `${formatDtype(
            input.dtype,
          )} full fine-tuning is not a clean v0 baseline. Using a BF16 SFT proxy instead.`,
        ],
        proxyReason:
          "Full SFT usually keeps trainable weights, gradients, and optimizer state in 16-bit or 32-bit form even if loading starts from a quantized checkpoint.",
      };
    }

    const bytes = input.dtype === "fp32" ? 4 : 2;
    return {
      mode: "training",
      effectiveDtype: input.dtype,
      effectiveTrainingType: "sft",
      calculationProfile: `${formatDtype(input.dtype)} SFT`,
      baseWeightBytes: bytes,
      computeBytes: bytes,
      warnings: [],
    };
  }

  if (input.trainingType === "lora") {
    if (input.dtype === "int4") {
      return {
        mode: "training",
        effectiveDtype: "int4",
        effectiveTrainingType: "qlora",
        calculationProfile: "QLoRA proxy",
        baseWeightBytes: QLORA_WEIGHT_BYTES,
        computeBytes: 2,
        warnings: [
          "4-bit LoRA behaves closer to QLoRA in practice. Using the QLoRA proxy for this estimate.",
        ],
        proxyReason:
          "In v0, 4-bit adapter training is modeled as QLoRA because the frozen base typically sits in 4-bit form while adapters, grads, and activations stay in 16-bit compute.",
      };
    }

    return {
      mode: "training",
      effectiveDtype: input.dtype === "int8" ? "int8" : input.dtype,
      effectiveTrainingType: "lora",
      calculationProfile: `${formatDtype(input.dtype)} LoRA`,
      baseWeightBytes:
        input.dtype === "fp32" ? 4 : input.dtype === "int8" || input.dtype === "fp8" ? 1 : 2,
      computeBytes: input.dtype === "fp32" ? 4 : 2,
      warnings: [],
    };
  }

  if (input.trainingType === "qlora") {
    return {
      mode: "training",
      effectiveDtype: "int4",
      effectiveTrainingType: "qlora",
      calculationProfile: "QLoRA",
      baseWeightBytes: QLORA_WEIGHT_BYTES,
      computeBytes: 2,
      warnings:
        input.dtype === "int4"
          ? []
          : [
              `${formatDtype(
                input.dtype,
              )} is being proxied to the standard 4-bit QLoRA setup for the estimate.`,
            ],
      proxyReason:
        input.dtype === "int4"
          ? undefined
          : "QLoRA is modeled as a 4-bit frozen base with 16-bit adapter math, regardless of the originally selected dtype.",
    };
  }

  return {
    mode: "training",
    effectiveDtype: "int4",
    effectiveTrainingType: "grpo",
    calculationProfile: "GRPO beta",
    baseWeightBytes: QLORA_WEIGHT_BYTES,
    computeBytes: 2,
    warnings:
      input.dtype === "int4"
        ? ["GRPO is a beta estimate. Real memory depends heavily on rollout length and completion count."]
        : [
            `${formatDtype(
              input.dtype,
            )} is being proxied to a QLoRA-style GRPO baseline. Real memory depends heavily on rollout shape.`,
          ],
    proxyReason:
      "GRPO is modeled as QLoRA-style adapter training plus extra rollout memory so the estimate stays conservative and explainable in v0.",
  };
}

function getInferenceWeightBytes(dtype: Dtype): number {
  switch (dtype) {
    case "fp32":
      return 4;
    case "fp16":
    case "bf16":
      return 2;
    case "fp8":
      return 1;
    case "int8":
      return 1;
    case "int4":
      return QLORA_WEIGHT_BYTES;
  }
}

function getGpu(gpuId: string, customVramGb: number): GpuSpec {
  const gpu = gpuMap.get(gpuId) ?? gpus[1];

  if (gpu.id !== "custom") {
    return gpu;
  }

  return {
    ...gpu,
    vramGb: customVramGb,
    displayName: `Custom GPU ${customVramGb}GB`,
  };
}
