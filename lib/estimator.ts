import { gpus } from "@/data/gpus";
import {
  BINARY_GIB,
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
import {
  applyModelConstraints,
  getCompatibleInferenceProfile,
  getModelSpec,
  hasCompatibleInferenceProfile,
} from "@/lib/model-constraints";
import { normalizeEstimateInput } from "@/lib/query-state";
import {
  getRuntimeSpec,
  runtimeSupportsKvCacheDtype,
  VLLM_GPU_UTILIZATION,
} from "@/lib/runtime";
import type {
  BreakdownItem,
  CacheStrategy,
  Dtype,
  EstimateInput,
  EstimateResult,
  GpuSpec,
  InferenceProfile,
  KvCacheDtype,
  MathLine,
  ModelSpec,
  RuntimeId,
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
      weightNote?: string;
      effectiveInferenceProfileId: string;
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

  if (!canEstimateInput(normalized)) {
    throw new Error(
      `${model.displayName} does not have a compatible inference profile for ${getRuntimeSpec(
        normalized.runtimeId,
      ).label}.`,
    );
  }

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

  if (getModelModality(model) === "multimodal") {
    notes.push(
      "This is a text-only estimate. Resident vision and projector weights stay counted in the selected checkpoint footprint, but image and video token memory is excluded in v1.",
    );
  }

  if (getModelCacheStrategy(model) === "hybrid_attention") {
    notes.push(
      `KV cache is modeled only on the ${getKvCacheLayerCount(
        model,
      )} attention-bearing layers in this hybrid stack, not on all ${model.numLayers} layers.`,
    );
    if (getLinearStateLayerCount(model) > 0) {
      notes.push(
        "The linear-attention layers add a static recurrent state and short-convolution buffers. That term is modeled separately and does not scale with context length.",
      );
    }
  }

  if (getModelCacheStrategy(model) === "alternating_window_attention") {
    notes.push(
      `This model alternates ${getDenseAttentionLayerCount(model)} dense attention layers with ${getSlidingWindowAttentionLayerCount(
        model,
      )} sliding-window layers. The sliding-window layers only retain a ${formatInteger(
        getSlidingWindowContextLength(model),
      )}-token cache.`,
    );
  }

  if (normalized.mode === "inference" && runtimeSupportsKvCacheDtype(normalized.runtimeId)) {
    if (normalized.kvCacheDtype === "fp8") {
      notes.push(
        "KV cache dtype is set to FP8 for this runtime estimate. That halves the KV cache term versus BF16, but real deployments can need cache scaling metadata and may trade away some accuracy headroom.",
      );
    } else if (normalized.runtimeId === "vllm") {
      notes.push(
        "This vLLM estimate uses a BF16-equivalent KV cache baseline. In practice the CLI often leaves KV cache dtype on auto unless you explicitly force FP8.",
      );
    }
  }

  if (normalized.mode === "inference" && normalized.runtimeId === "transformers") {
    notes.push(
      "Transformers is modeled as a fixed single-request baseline in v1: 4K context and one active sequence. Switch to vLLM to plan serving context and concurrency.",
    );
  }

  const strategy = resolveStrategy(normalized, model);
  warnings.push(...strategy.warnings);

  if (strategy.proxyReason) {
    notes.push(strategy.proxyReason);
  }

  if (strategy.mode === "inference" && strategy.weightNote) {
    notes.push(strategy.weightNote);
  }

  if (strategy.mode === "training") {
    notes.push(
      "Training estimates assume a single-GPU mixed-precision AdamW setup with fp32 optimizer states, fp32 master weights for trainable params when compute is below fp32, and no CPU offload or ZeRO/FSDP sharding.",
    );

    if (strategy.effectiveTrainingType !== "sft") {
      notes.push(
        "LoRA-style training assumes rank 16 adapters on the q/k/v/o projections in every layer.",
      );
    }
  }

  if (model.fixedDtype) {
    notes.push(
      `${model.displayName} is treated as a fixed ${formatDtype(
        model.fixedDtype,
      )} checkpoint in the current release.`,
    );
  }

  if ((model.supportedModes ?? ["inference", "training"]).length === 1) {
    notes.push(`${model.displayName} is inference-only in the current release.`);
  }

  if (strategy.mode === "inference") {
    const inferenceEstimate = buildInferenceEstimate(
      normalized,
      model,
      gpu,
      effectiveContextLength,
      strategy,
    );

    return finalizeResult(
      normalized,
      model,
      gpu,
      effectiveContextLength,
      warnings,
      notes,
      {
        ...inferenceEstimate,
        maxConcurrencyAtContext:
          normalized.runtimeId === "transformers"
            ? undefined
            : estimateMaxConcurrencyAtContext(
                normalized,
                model,
                gpu,
                effectiveContextLength,
                strategy,
              ),
      },
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
  const headDim = getAttentionHeadDim(model);
  const kvLayerCount = getKvCacheLayerCount(model);
  const kvTokenFactor = getKvCacheTokenFactor(model, effectiveContextLength);
  const linearLayerCount = getLinearStateLayerCount(model);
  const linearStatePerLayerBytes = getLinearStateBytesPerLayer(model);
  const kvCacheBytesPerElement = getKvCacheBytesPerElement(input.kvCacheDtype);
  const weightsBytes = model.totalParams * strategy.weightBytes;
  const kvCacheBytes =
    input.batchSize *
    kvTokenFactor *
    2 *
    kvHeads *
    headDim *
    kvCacheBytesPerElement;
  const linearStateBytes =
    input.batchSize * linearLayerCount * linearStatePerLayerBytes;
  const subtotal = weightsBytes + kvCacheBytes + linearStateBytes;
  const overheadBytes = Math.max(1.5 * DECIMAL_GB, subtotal * 0.1);
  const totalBytes = subtotal + overheadBytes;

  const breakdown: BreakdownItem[] = [
    {
      key: "weights",
      label: "Weights",
      bytes: weightsBytes,
      note: `${formatParamCount(model.totalParams)} resident parameters at ${formatBytesPerParam(
        strategy.weightBytes,
      )} bytes each.${strategy.weightNote ? " Calibrated from the official checkpoint profile." : ""}`,
    },
    {
      key: "kvCache",
      label: "KV cache",
      bytes: kvCacheBytes,
      note: `${
        input.runtimeId === "transformers" ? "Single request" : `Concurrency ${input.batchSize}`
      }, context ${formatInteger(effectiveContextLength)}, ${describeKvCacheLayout(
        model,
        kvLayerCount,
        effectiveContextLength,
      )}, ${kvHeads} KV heads, ${formatDtype(
        input.kvCacheDtype,
      )} cache storage.`,
    },
  ];

  if (linearStateBytes > 0) {
    breakdown.push({
      key: "linearState",
      label: "Linear attention state",
      bytes: linearStateBytes,
      note: `${
        input.runtimeId === "transformers" ? "Single request" : `Concurrency ${input.batchSize}`
      }, ${linearLayerCount} linear-attention layers, static recurrent state, and short-convolution buffers. This term stays flat as context grows.`,
    });
  }

  breakdown.push({
    key: "overhead",
    label: "Runtime / safety overhead",
    bytes: overheadBytes,
    note: "Conservative buffer for allocator fragmentation, kernels, and runtime scratch space.",
  });

  const math: MathLine[] = [
    {
      label: "Weights",
      symbolic: "parameter count × bytes per weight",
      substituted: `${formatParamCount(model.totalParams)} × ${formatBytesPerParam(
        strategy.weightBytes,
      )} = ${formatGb(weightsBytes)}`,
      note:
        strategy.weightNote ??
        `${formatDtype(strategy.effectiveDtype)} controls how compact the resident weights are.`,
    },
    {
      label: "KV cache",
      symbolic:
        "batch × effective KV tokens across attention layers × 2 × KV heads × head dim × bytes per KV element",
      substituted: `${input.batchSize} × ${formatInteger(
        kvTokenFactor,
      )} × 2 × ${kvHeads} × ${headDim} × ${
        kvCacheBytesPerElement
      } = ${formatGb(kvCacheBytes)}`,
      note:
        getModelCacheStrategy(model) === "hybrid_attention"
          ? `Only the attention-bearing layers contribute KV cache in this hybrid stack, and ${formatDtype(
              input.kvCacheDtype,
            )} controls the bytes per stored KV element.`
          : getModelCacheStrategy(model) === "alternating_window_attention"
            ? `Dense attention layers keep the full ${formatInteger(
                effectiveContextLength,
              )}-token cache, while sliding-window layers only keep ${formatInteger(
                Math.min(effectiveContextLength, getSlidingWindowContextLength(model)),
              )} tokens. ${formatDtype(input.kvCacheDtype)} controls the bytes per stored KV element.`
          : `Longer context, deeper models, and larger batches all expand the cache linearly. ${formatDtype(
              input.kvCacheDtype,
            )} controls the bytes per stored KV element.`,
    },
  ];

  if (linearStateBytes > 0) {
    math.push({
      label: "Linear state",
      symbolic:
        "batch × linear layers × (recurrent state + short-conv buffers) × state bytes",
      substituted: `${input.batchSize} × ${linearLayerCount} × ${formatInteger(
        linearStatePerLayerBytes / getLinearStateBytesPerElement(model),
      )} × ${formatBytesPerParam(getLinearStateBytesPerElement(model))} = ${formatGb(
        linearStateBytes,
      )}`,
      note:
        "Hybrid Qwen3.5 layers keep a static recurrent state plus q/k/v short-convolution buffers. The published configs keep that state in float32, so it is modeled separately from the BF16 weight dtype.",
    });
  }

  math.push({
    label: "Overhead",
    symbolic: "max(1.5 GB, 10% of weights + KV cache + linear state)",
    substituted: `max(1.5 GB, 10% of ${formatGb(subtotal)}) = ${formatGb(
      overheadBytes,
    )}`,
    note: "This leaves room for runtime buffers instead of claiming an unrealistically exact fit.",
  });

  return {
    weightsBytes,
    masterWeightsBytes: 0,
    kvCacheBytes,
    linearStateBytes,
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
    effectiveInferenceProfileId: strategy.effectiveInferenceProfileId,
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
  let masterWeightsBytes = 0;
  let gradientsBytes = 0;
  let optimizerBytes = 0;
  let grpoExtraBytes = 0;

  if (strategy.effectiveTrainingType === "sft") {
    weightsBytes = model.totalParams * strategy.baseWeightBytes;
    masterWeightsBytes = strategy.computeBytes < 4 ? model.totalParams * 4 : 0;
    gradientsBytes = model.totalParams * strategy.computeBytes;
    optimizerBytes = model.totalParams * 8;
  } else {
    const adapterWeightBytes = adapterParams * strategy.computeBytes;
    weightsBytes = model.totalParams * strategy.baseWeightBytes + adapterWeightBytes;
    masterWeightsBytes = strategy.computeBytes < 4 ? adapterParams * 4 : 0;
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
    masterWeightsBytes +
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
          ? `${formatParamCount(model.totalParams)} trainable parameters remain resident during SFT in the selected compute format.`
          : `Frozen base weights plus roughly ${formatParamCount(
              adapterParams,
            )} LoRA adapter parameters.`,
    },
    {
      key: "masterWeights",
      label: "Master weights",
      bytes: masterWeightsBytes,
      note:
        masterWeightsBytes > 0
          ? strategy.effectiveTrainingType === "sft"
            ? "Conservative mixed-precision baseline: fp32 master weights for all trainable parameters."
            : "Conservative mixed-precision baseline: fp32 master weights for the trainable adapters."
          : "No separate fp32 master weights are assumed when training directly in fp32.",
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
      label: "Master weights",
      symbolic:
        strategy.effectiveTrainingType === "sft"
          ? "trainable params × 4 bytes"
          : "adapter params × 4 bytes",
      substituted:
        masterWeightsBytes > 0
          ? strategy.effectiveTrainingType === "sft"
            ? `${formatParamCount(model.totalParams)} × 4 = ${formatGb(masterWeightsBytes)}`
            : `${formatParamCount(adapterParams)} × 4 = ${formatGb(masterWeightsBytes)}`
          : `0 = ${formatGb(masterWeightsBytes)}`,
      note:
        masterWeightsBytes > 0
          ? "This conservative training path keeps an fp32 master copy for every trainable parameter."
          : "No separate master copy is needed when the trainable parameters already live in fp32.",
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
        ? "Sequence packing trims the activation estimate by 15% in this current approximation."
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
    masterWeightsBytes,
    kvCacheBytes: 0,
    linearStateBytes: 0,
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
    effectiveInferenceProfileId: undefined,
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
    | "requiredGpuBytes"
    | "fitMetricLabel"
    | "warnings"
    | "notes"
    | "runtime"
    | "runtimeNotes"
    | "tips"
    | "effectiveContextLength"
  >,
): EstimateResult {
  const gpuBytes = getGpuCapacityBytes(gpu);
  const runtime = getRuntimeSpec(input.runtimeId);
  const runtimeNotes = buildRuntimeNotes(runtime.id);
  const { requiredGpuBytes, fitMetricLabel } = resolveRequiredGpuBytes(runtime.id, partial);
  const fits = requiredGpuBytes <= gpuBytes;
  const headroomBytes = fits ? gpuBytes - requiredGpuBytes : 0;
  const deficitBytes = fits ? 0 : requiredGpuBytes - gpuBytes;

  return {
    ...partial,
    input,
    model,
    gpu,
    runtime,
    runtimeNotes,
    fits,
    gpuBytes,
    headroomBytes,
    deficitBytes,
    requiredGpuBytes,
    fitMetricLabel,
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
    if (args.input.runtimeId !== "transformers") {
    tips.push("Reduce context length if KV cache is the fastest-growing term in the estimate.");
    }
  }

  if (!args.fits && args.input.batchSize > 1) {
    if (args.input.runtimeId !== "transformers") {
      tips.push("Lower concurrent requests to shrink KV cache and runtime memory linearly.");
    }
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

function resolveStrategy(input: EstimateInput, model: ModelSpec): Strategy {
  if (input.mode === "inference") {
    const profile = getCompatibleInferenceProfile(
      model,
      input.runtimeId,
      input.inferenceProfileId,
      input.dtype,
    );
    const warnings: string[] = [];
    let proxyReason: string | undefined;
    if (!profile) {
      throw new Error(
        `${model.displayName} does not have a compatible inference profile for ${getRuntimeSpec(
          input.runtimeId,
        ).label}.`,
      );
    }

    let weightNote = profile.note;

    if (!profile.official) {
      warnings.push(
        `${profile.label} is a proxy estimate, not an official ${model.displayName} checkpoint profile.`,
      );
      proxyReason = `${model.displayName} does not ship this exact checkpoint profile. The estimate is using a generic ${profile.label.toLowerCase()} fallback instead of an official release artifact.`;
    } else if (profile.weightMode === "calibrated") {
      weightNote = `${profile.note} The estimator back-solves an effective resident bytes-per-parameter value using a reference batch of ${profile.targetBatchSize ?? 1} and context of ${formatInteger(
        profile.targetContextLength ?? 0,
      )} tokens, then adds KV cache and runtime overhead for your actual batch and context settings.`;
    }

    return {
      mode: "inference",
      effectiveDtype: profile.effectiveDtype,
      effectiveTrainingType: input.trainingType,
      effectiveInferenceProfileId: profile.id,
      calculationProfile: profile.label,
      weightBytes:
        profile.weightMode === "calibrated"
          ? solveOfficialInferenceWeightBytes(model, profile)
          : profile.weightBytes ?? getInferenceWeightBytes(profile.effectiveDtype),
      kvBytes: getKvCacheBytesPerElement(input.kvCacheDtype),
      warnings,
      weightNote,
      proxyReason,
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
          )} full fine-tuning is not a clean current-release baseline. Using a BF16 SFT proxy instead.`,
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
          "In the current release, 4-bit adapter training is modeled as QLoRA because the frozen base typically sits in 4-bit form while adapters, grads, and activations stay in 16-bit compute.",
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
      "GRPO is modeled as QLoRA-style adapter training plus extra rollout memory so the estimate stays conservative and explainable in the current release.",
  };
}

function solveOfficialInferenceWeightBytes(
  model: ModelSpec,
  profile: InferenceProfile,
) {
  if (profile.targetMemoryGb === undefined) {
    throw new Error(`Calibrated profile ${profile.id} is missing targetMemoryGb.`);
  }

  const targetTotalBytes = profile.targetMemoryGb * DECIMAL_GB;
  const targetBatchSize = profile.targetBatchSize ?? 1;
  const targetContextLength = profile.targetContextLength ?? 0;
  const kvHeads = model.numKvHeads ?? model.numAttentionHeads;
  const headDim = getAttentionHeadDim(model);
  const kvTokenFactor = getKvCacheTokenFactor(model, targetContextLength);
  const kvBytes =
    targetBatchSize *
    kvTokenFactor *
    2 *
    kvHeads *
    headDim *
    2;
  const percentOverheadWeightBytes =
    (targetTotalBytes / 1.1 - kvBytes) / model.totalParams;
  const subtotalWithPercentOverhead =
    model.totalParams * percentOverheadWeightBytes + kvBytes;

  if (subtotalWithPercentOverhead * 0.1 >= 1.5 * DECIMAL_GB) {
    return percentOverheadWeightBytes;
  }

  return (targetTotalBytes - kvBytes - 1.5 * DECIMAL_GB) / model.totalParams;
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

function getGpuCapacityBytes(gpu: GpuSpec): number {
  return gpu.vramGb * BINARY_GIB;
}

export function canEstimateInput(input: EstimateInput): boolean {
  if (input.mode !== "inference") {
    return true;
  }

  return hasCompatibleInferenceProfile(input.modelId, input.runtimeId);
}

function resolveRequiredGpuBytes(
  runtimeId: RuntimeId,
  partial: Pick<
    EstimateResult,
    | "weightsBytes"
    | "kvCacheBytes"
    | "linearStateBytes"
    | "overheadBytes"
    | "totalBytes"
    | "activationsBytes"
    | "gradientsBytes"
    | "optimizerBytes"
    | "masterWeightsBytes"
    | "grpoExtraBytes"
  >,
) {
  if (runtimeId === "vllm") {
    return {
      requiredGpuBytes: partial.totalBytes / VLLM_GPU_UTILIZATION,
      fitMetricLabel: "Required GPU VRAM (0.9 budget)",
    };
  }

  return {
    requiredGpuBytes: partial.totalBytes,
    fitMetricLabel: "Required GPU VRAM",
  };
}

function estimateMaxConcurrencyAtContext(
  input: EstimateInput,
  model: ModelSpec,
  gpu: GpuSpec,
  effectiveContextLength: number,
  strategy: Extract<Strategy, { mode: "inference" }>,
): number {
  const gpuBytes = getGpuCapacityBytes(gpu);
  const requiredForBatchSize = (batchSize: number) =>
    resolveRequiredGpuBytes(
      input.runtimeId,
      buildInferenceEstimate(
        { ...input, batchSize },
        model,
        gpu,
        effectiveContextLength,
        strategy,
      ),
    ).requiredGpuBytes;

  if (requiredForBatchSize(1) > gpuBytes) {
    return 0;
  }

  let low = 1;
  let high = Math.max(2, input.batchSize);

  while (high < 4096 && requiredForBatchSize(high) <= gpuBytes) {
    low = high;
    high = Math.min(4096, high * 2);
  }

  if (requiredForBatchSize(high) <= gpuBytes) {
    return high;
  }

  while (low + 1 < high) {
    const mid = Math.floor((low + high) / 2);

    if (requiredForBatchSize(mid) <= gpuBytes) {
      low = mid;
    } else {
      high = mid;
    }
  }

  return low;
}

function buildRuntimeNotes(runtimeId: RuntimeId): string[] {
  switch (runtimeId) {
    case "transformers":
      return [
        "Transformers is treated as a single-request baseline in v1, so the calculator fixes it to 4K context and one active sequence instead of exposing serving controls.",
      ];
    case "vllm":
      return [
        "vLLM converts the core estimate into nominal card VRAM by dividing by the default --gpu-memory-utilization=0.9 executor budget.",
        "vLLM also supports a lower-precision KV cache path; this calculator models BF16 and FP8 cache storage explicitly.",
        "The app also derives a max concurrent-sequence estimate at the selected context length.",
        "That concurrency estimate assumes all active sequences are simultaneously resident at the selected full context, which is more conservative than scheduler caps like --max-num-seqs or per-iteration token limits.",
      ];
  }
}

function getKvCacheLayerCount(model: ModelSpec): number {
  if (getModelCacheStrategy(model) === "hybrid_attention") {
    return model.attentionLayerCount ?? model.numLayers;
  }

  if (getModelCacheStrategy(model) === "alternating_window_attention") {
    return getDenseAttentionLayerCount(model) + getSlidingWindowAttentionLayerCount(model);
  }

  return model.numLayers;
}

function getModelCacheStrategy(model: ModelSpec): CacheStrategy {
  return model.cacheStrategy ?? "standard_gqa";
}

function getModelModality(model: ModelSpec) {
  return model.modality ?? "text";
}

function getAttentionHeadDim(model: ModelSpec): number {
  return model.attentionHeadDim ?? model.hiddenSize / model.numAttentionHeads;
}

function getKvCacheBytesPerElement(kvCacheDtype: KvCacheDtype): number {
  return kvCacheDtype === "fp8" ? 1 : 2;
}

function getLinearStateLayerCount(model: ModelSpec): number {
  return Math.max(0, model.numLayers - getKvCacheLayerCount(model));
}

function getKvCacheTokenFactor(model: ModelSpec, effectiveContextLength: number): number {
  if (getModelCacheStrategy(model) === "alternating_window_attention") {
    return (
      getDenseAttentionLayerCount(model) * effectiveContextLength +
      getSlidingWindowAttentionLayerCount(model) *
        Math.min(effectiveContextLength, getSlidingWindowContextLength(model))
    );
  }

  return getKvCacheLayerCount(model) * effectiveContextLength;
}

function getLinearStateBytesPerLayer(model: ModelSpec): number {
  if (getModelCacheStrategy(model) !== "hybrid_attention") {
    return 0;
  }

  const keyHeads = model.linearNumKeyHeads;
  const valueHeads = model.linearNumValueHeads;
  const keyHeadDim = model.linearKeyHeadDim;
  const valueHeadDim = model.linearValueHeadDim;

  if (!keyHeads || !valueHeads || !keyHeadDim || !valueHeadDim) {
    return 0;
  }

  const keyProjectionDim = keyHeads * keyHeadDim;
  const valueProjectionDim = valueHeads * valueHeadDim;
  const recurrentStateElements =
    keyProjectionDim * Math.ceil(valueProjectionDim / keyHeads);
  const convKernelDim = model.linearConvKernelDim ?? 0;
  const convStateElements =
    convKernelDim * (keyProjectionDim * 2 + valueProjectionDim);

  return (
    (recurrentStateElements + convStateElements) *
    getLinearStateBytesPerElement(model)
  );
}

function getLinearStateBytesPerElement(model: ModelSpec): number {
  return model.linearStateBytesPerElement ?? 2;
}

function getDenseAttentionLayerCount(model: ModelSpec): number {
  return model.denseAttentionLayerCount ?? Math.ceil(model.numLayers / 2);
}

function getSlidingWindowAttentionLayerCount(model: ModelSpec): number {
  return model.slidingWindowAttentionLayerCount ?? Math.floor(model.numLayers / 2);
}

function getSlidingWindowContextLength(model: ModelSpec): number {
  return model.slidingWindowContextLength ?? 128;
}

function describeKvCacheLayout(
  model: ModelSpec,
  kvLayerCount: number,
  effectiveContextLength: number,
) {
  if (getModelCacheStrategy(model) === "alternating_window_attention") {
    return `${getDenseAttentionLayerCount(model)} dense layers @ ${formatInteger(
      effectiveContextLength,
    )} tokens + ${getSlidingWindowAttentionLayerCount(model)} sliding-window layers @ ${formatInteger(
      Math.min(effectiveContextLength, getSlidingWindowContextLength(model)),
    )} tokens`;
  }

  return `${kvLayerCount} KV-bearing layers`;
}
