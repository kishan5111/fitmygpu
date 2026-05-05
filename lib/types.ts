export type Mode = "inference" | "training";

export type TrainingType = "sft" | "lora" | "qlora" | "grpo";

export type Dtype = "fp32" | "fp16" | "bf16" | "fp8" | "int8" | "int4";

export type RuntimeId = "transformers" | "vllm";

export type KvCacheDtype = "bf16" | "fp8";

export type Modality = "text" | "multimodal";

export type ProfileConfidence = "verified" | "estimated" | "proxy";

export type CacheStrategy =
  | "standard_gqa"
  | "hybrid_attention"
  | "alternating_window_attention";

export interface RuntimeSpec {
  id: RuntimeId;
  label: string;
}

export interface InferenceProfile {
  id: string;
  label: string;
  effectiveDtype: Dtype;
  official: boolean;
  confidence?: ProfileConfidence;
  note: string;
  sourceUrl: string;
  weightMode: "direct" | "calibrated";
  weightBytes?: number;
  targetMemoryGb?: number;
  targetBatchSize?: number;
  targetContextLength?: number;
  supportedRuntimes?: RuntimeId[];
}

export interface ModelSpec {
  id: string;
  displayName: string;
  family: string;
  organization: string;
  architectureType: string;
  isMoe: boolean;
  totalParams: number;
  activeParams?: number;
  numLayers: number;
  hiddenSize: number;
  numAttentionHeads: number;
  numKvHeads?: number;
  attentionHeadDim?: number;
  linearNumKeyHeads?: number;
  linearNumValueHeads?: number;
  linearKeyHeadDim?: number;
  linearValueHeadDim?: number;
  linearConvKernelDim?: number;
  linearStateBytesPerElement?: number;
  denseAttentionLayerCount?: number;
  slidingWindowAttentionLayerCount?: number;
  slidingWindowContextLength?: number;
  contextLength: number;
  modality?: Modality;
  cacheStrategy?: CacheStrategy;
  attentionLayerCount?: number;
  vocabSize: number;
  license: string;
  sourceUrl: string;
  shortDescription: string;
  researchHighlight: string;
  memoryNote: string;
  inferenceProfiles: InferenceProfile[];
  fixedDtype?: Dtype;
  supportedModes?: Mode[];
}

export interface GpuSpec {
  id: string;
  displayName: string;
  vramGb: number;
  memoryBandwidthGbps?: number;
  classType: string;
}

export interface EstimateInput {
  mode: Mode;
  trainingType: TrainingType;
  runtimeId: RuntimeId;
  kvCacheDtype: KvCacheDtype;
  vllmGpuUtilization: number;
  modelId: string;
  dtype: Dtype;
  inferenceProfileId: string;
  gpuId: string;
  gpuCount: number;
  customVramGb: number;
  contextLength: number;
  batchSize: number;
  gradientCheckpointing: boolean;
  sequencePacking: boolean;
}

export interface BreakdownItem {
  key:
    | "weights"
    | "masterWeights"
    | "kvCache"
    | "linearState"
    | "activations"
    | "gradients"
    | "optimizer"
    | "grpoExtra"
    | "overhead";
  label: string;
  bytes: number;
  note: string;
}

export interface MathLine {
  label: string;
  symbolic: string;
  substituted: string;
  note: string;
}

export interface EstimateResult {
  input: EstimateInput;
  model: ModelSpec;
  gpu: GpuSpec;
  runtime: RuntimeSpec;
  runtimeNotes: string[];
  fits: boolean;
  totalBytes: number;
  gpuBytes: number;
  headroomBytes: number;
  deficitBytes: number;
  requiredGpuBytes: number;
  fitMetricLabel: string;
  weightsBytes: number;
  masterWeightsBytes: number;
  kvCacheBytes: number;
  linearStateBytes: number;
  activationsBytes: number;
  gradientsBytes: number;
  optimizerBytes: number;
  grpoExtraBytes: number;
  overheadBytes: number;
  breakdown: BreakdownItem[];
  math: MathLine[];
  warnings: string[];
  notes: string[];
  effectiveContextLength: number;
  maxConcurrencyAtContext?: number;
  effectiveDtype: Dtype;
  effectiveTrainingType: TrainingType;
  effectiveInferenceProfileId?: string;
  calculationProfile: string;
  proxyReason?: string;
}
