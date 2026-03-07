export type Mode = "inference" | "training";

export type TrainingType = "sft" | "lora" | "qlora" | "grpo";

export type Dtype = "fp32" | "fp16" | "bf16" | "fp8" | "int8" | "int4";

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
  contextLength: number;
  vocabSize: number;
  license: string;
  sourceUrl: string;
  shortDescription: string;
  researchHighlight: string;
  memoryNote: string;
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
  modelId: string;
  dtype: Dtype;
  gpuId: string;
  customVramGb: number;
  contextLength: number;
  batchSize: number;
  gradientCheckpointing: boolean;
  sequencePacking: boolean;
}

export interface BreakdownItem {
  key:
    | "weights"
    | "kvCache"
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
  fits: boolean;
  totalBytes: number;
  gpuBytes: number;
  headroomBytes: number;
  deficitBytes: number;
  weightsBytes: number;
  kvCacheBytes: number;
  activationsBytes: number;
  gradientsBytes: number;
  optimizerBytes: number;
  grpoExtraBytes: number;
  overheadBytes: number;
  breakdown: BreakdownItem[];
  math: MathLine[];
  warnings: string[];
  notes: string[];
  tips: string[];
  effectiveContextLength: number;
  effectiveDtype: Dtype;
  effectiveTrainingType: TrainingType;
  calculationProfile: string;
  proxyReason?: string;
}
