import { models } from "@/data/models";
import { TRAINING_ENABLED } from "@/lib/constants";
import { estimateVram } from "@/lib/estimator";
import type { EstimateInput } from "@/lib/types";

function buildInput(overrides: Partial<EstimateInput> = {}): EstimateInput {
  return {
    mode: "inference",
    trainingType: "sft",
    runtimeId: "transformers",
    kvCacheDtype: "bf16",
    vllmGpuUtilization: 0.9,
    modelId: "llama-3.1-8b",
    dtype: "fp16",
    inferenceProfileId: "",
    gpuId: "rtx-4090-24gb",
    gpuCount: 1,
    customVramGb: 24,
    contextLength: 4096,
    batchSize: 1,
    gradientCheckpointing: false,
    sequencePacking: false,
    ...overrides,
  };
}

function summarize(result: ReturnType<typeof estimateVram>) {
  return {
    runtime: result.runtime.label,
    profile: result.calculationProfile,
    fits: result.fits,
    maxConcurrency: result.maxConcurrencyAtContext ?? null,
    totalGb: Number((result.totalBytes / 1_000_000_000).toFixed(1)),
    requiredGb: Number((result.requiredGpuBytes / 1_000_000_000).toFixed(1)),
    weightsGb: Number((result.weightsBytes / 1_000_000_000).toFixed(1)),
    masterGb: Number((result.masterWeightsBytes / 1_000_000_000).toFixed(1)),
    kvGb: Number((result.kvCacheBytes / 1_000_000_000).toFixed(1)),
    linearStateGb: Number((result.linearStateBytes / 1_000_000_000).toFixed(3)),
    activationsGb: Number((result.activationsBytes / 1_000_000_000).toFixed(1)),
    gradientsGb: Number((result.gradientsBytes / 1_000_000_000).toFixed(1)),
    optimizerGb: Number((result.optimizerBytes / 1_000_000_000).toFixed(1)),
    warnings: result.warnings,
  };
}

describe("estimateVram", () => {
  it("increases monotonically with context and batch during inference", () => {
    const base = estimateVram(buildInput({ runtimeId: "vllm" }));
    const longerContext = estimateVram(
      buildInput({ runtimeId: "vllm", contextLength: 8192 }),
    );
    const biggerBatch = estimateVram(buildInput({ runtimeId: "vllm", batchSize: 2 }));

    expect(longerContext.totalBytes).toBeGreaterThan(base.totalBytes);
    expect(biggerBatch.totalBytes).toBeGreaterThan(base.totalBytes);
  });

  it("pins transformers to the fixed single-request baseline", () => {
    const result = estimateVram(
      buildInput({
        runtimeId: "transformers",
        contextLength: 32768,
        batchSize: 8,
      }),
    );

    expect(result.input.contextLength).toBe(4096);
    expect(result.input.batchSize).toBe(1);
    expect(result.input.gpuCount).toBe(1);
    expect(result.maxConcurrencyAtContext).toBeUndefined();
    expect(result.runtimeNotes.join(" ")).toMatch(/single-request baseline|4K context/i);
  });

  it("forces training requests back to inference while training is disabled", () => {
    const result = estimateVram(
      buildInput({
        mode: "training",
        trainingType: "qlora",
        dtype: "int4",
      }),
    );

    expect(TRAINING_ENABLED).toBe(false);
    expect(result.input.mode).toBe("inference");
    expect(result.kvCacheBytes).toBeGreaterThan(0);
    expect(result.activationsBytes).toBe(0);
  });

  it("uses total params rather than active params for MoE resident weights", () => {
    const mixtral = models.find((model) => model.id === "mixtral-8x7b");
    expect(mixtral).toBeDefined();

    const result = estimateVram(
      buildInput({
        modelId: "mixtral-8x7b",
      }),
    );

    expect(result.weightsBytes).toBeCloseTo(mixtral!.totalParams * 2, -1);
    expect(result.weightsBytes).toBeGreaterThan((mixtral!.activeParams ?? 0) * 2);
  });

  it("uses the native GPT-OSS checkpoint profile for fp16/bf16-style inference", () => {
    const gptOss20B = estimateVram(
      buildInput({
        modelId: "gpt-oss-20b",
        dtype: "fp16",
        gpuId: "rtx-4090-24gb",
      }),
    );
    const gptOss120B = estimateVram(
      buildInput({
        modelId: "gpt-oss-120b",
        dtype: "bf16",
        gpuId: "h100-80gb",
      }),
    );

    expect(gptOss20B.calculationProfile).toBe("Mixed MXFP4 + BF16 checkpoint");
    expect(gptOss20B.fits).toBe(true);
    expect(gptOss20B.weightsBytes / 1_000_000_000).toBeLessThan(16);

    expect(gptOss120B.calculationProfile).toBe("Mixed MXFP4 + BF16 checkpoint");
    expect(gptOss120B.fits).toBe(true);
    expect(gptOss120B.weightsBytes / 1_000_000_000).toBeLessThan(80);
  });

  it("falls back to the shipped GPT-OSS checkpoint instead of inventing a proxy", () => {
    const result = estimateVram(
      buildInput({
        modelId: "gpt-oss-20b",
        dtype: "int4",
      }),
    );

    expect(result.calculationProfile).toBe("Mixed MXFP4 + BF16 checkpoint");
    expect(result.effectiveDtype).toBe("bf16");
    expect(result.warnings).toEqual([]);
  });

  it("lets a standard float checkpoint use an alternate load dtype estimate", () => {
    const bf16 = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "qwen-2.5-7b",
        dtype: "bf16",
      }),
    );
    const int4 = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "qwen-2.5-7b",
        inferenceProfileId: "official-bf16",
        dtype: "int4",
      }),
    );

    expect(int4.effectiveDtype).toBe("int4");
    expect(int4.calculationProfile).toContain("loaded as 4-bit");
    expect(int4.weightsBytes).toBeLessThan(bf16.weightsBytes);
    expect(int4.notes.join(" ")).toMatch(/load dtype|resident weight memory/i);
  });

  it("keeps mixed official checkpoints pinned to their published artifact dtype behavior", () => {
    const result = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "gpt-oss-120b",
        dtype: "int4",
      }),
    );

    expect(result.calculationProfile).toBe("Mixed MXFP4 + BF16 checkpoint");
    expect(result.effectiveDtype).toBe("bf16");
    expect(result.weightsBytes / 1_000_000_000).toBeCloseTo(65.3, 1);
  });

  it("uses a reduced executor budget for vllm", () => {
    const transformersResult = estimateVram(
      buildInput({
        runtimeId: "transformers",
        modelId: "qwen-2.5-7b",
        gpuId: "custom",
        customVramGb: 17.5,
      }),
    );
    const vllmResult = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "qwen-2.5-7b",
        gpuId: "custom",
        customVramGb: 17.5,
      }),
    );

    expect(transformersResult.totalBytes).toBe(vllmResult.totalBytes);
    expect(transformersResult.requiredGpuBytes).toBe(transformersResult.totalBytes);
    expect(vllmResult.requiredGpuBytes).toBeCloseTo(vllmResult.totalBytes / 0.9, -1);
    expect(vllmResult.requiredGpuBytes).toBeGreaterThan(transformersResult.requiredGpuBytes);
    expect(transformersResult.fits).toBe(true);
    expect(vllmResult.fits).toBe(false);
    expect(vllmResult.runtimeNotes.join(" ")).toMatch(/0.9|90%/i);
  });

  it("uses the selected vllm gpu memory utilization", () => {
    const defaultBudget = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "qwen-2.5-7b",
      }),
    );
    const largerBudget = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "qwen-2.5-7b",
        vllmGpuUtilization: 0.95,
      }),
    );

    expect(defaultBudget.requiredGpuBytes).toBeCloseTo(defaultBudget.totalBytes / 0.9, -1);
    expect(largerBudget.requiredGpuBytes).toBeCloseTo(largerBudget.totalBytes / 0.95, -1);
    expect(largerBudget.requiredGpuBytes).toBeLessThan(defaultBudget.requiredGpuBytes);
    expect(largerBudget.fitMetricLabel).toContain("0.95");
  });

  it("uses aggregate capacity for multi-gpu vllm fits", () => {
    const singleGpu = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "gpt-oss-120b",
        gpuId: "rtx-4090-24gb",
        contextLength: 8192,
        batchSize: 1,
      }),
    );
    const fourGpu = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "gpt-oss-120b",
        gpuId: "rtx-4090-24gb",
        gpuCount: 4,
        contextLength: 8192,
        batchSize: 1,
      }),
    );

    expect(fourGpu.totalBytes).toBe(singleGpu.totalBytes);
    expect(fourGpu.requiredGpuBytes).toBe(singleGpu.requiredGpuBytes);
    expect(fourGpu.gpuBytes).toBe(singleGpu.gpuBytes * 4);
    expect(singleGpu.fits).toBe(false);
    expect(fourGpu.fits).toBe(true);
    expect(fourGpu.runtimeNotes.join(" ")).toMatch(/aggregate tensor-parallel/i);
  });

  it("treats nominal GPU VRAM labels as binary frame-buffer sizes", () => {
    const result = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "gpt-oss-120b",
        kvCacheDtype: "bf16",
        gpuId: "h100-80gb",
        contextLength: 8192,
      }),
    );

    expect(result.gpuBytes).toBe(80 * 1024 * 1024 * 1024);
    expect(result.requiredGpuBytes).toBeLessThan(result.gpuBytes);
    expect(result.fits).toBe(true);
  });

  it("lets FP8 KV cache reduce runtime memory for vllm", () => {
    const vllmBf16 = estimateVram(
      buildInput({
        runtimeId: "vllm",
        kvCacheDtype: "bf16",
        modelId: "qwen-2.5-32b",
        inferenceProfileId: "official-gptq-int4",
        dtype: "int4",
        contextLength: 32768,
        batchSize: 2,
        gpuId: "h100-80gb",
      }),
    );
    const vllmFp8 = estimateVram(
      buildInput({
        runtimeId: "vllm",
        kvCacheDtype: "fp8",
        modelId: "qwen-2.5-32b",
        inferenceProfileId: "official-gptq-int4",
        dtype: "int4",
        contextLength: 32768,
        batchSize: 2,
        gpuId: "h100-80gb",
      }),
    );

    expect(vllmFp8.weightsBytes).toBe(vllmBf16.weightsBytes);
    expect(vllmFp8.kvCacheBytes).toBeCloseTo(vllmBf16.kvCacheBytes / 2, -1);
    expect(vllmFp8.totalBytes).toBeLessThan(vllmBf16.totalBytes);
    expect(vllmFp8.requiredGpuBytes).toBeLessThan(vllmBf16.requiredGpuBytes);
    expect(vllmFp8.notes.join(" ")).toMatch(/FP8|scaling/i);
    expect(vllmBf16.maxConcurrencyAtContext).toBeGreaterThanOrEqual(1);
  });

  it("uses attention-bearing layers for Qwen3.5 KV cache math", () => {
    const result = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "qwen-3.5-4b",
        dtype: "bf16",
        contextLength: 4096,
      }),
    );
    const longerContext = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "qwen-3.5-4b",
        dtype: "bf16",
        contextLength: 8192,
      }),
    );
    const biggerBatch = estimateVram(
      buildInput({
        runtimeId: "vllm",
        modelId: "qwen-3.5-4b",
        dtype: "bf16",
        batchSize: 2,
      }),
    );
    const hybridExpectedKv =
      1 * 4096 * 8 * 2 * 4 * 256 * 2;
    const denseEquivalentKv =
      1 * 4096 * 32 * 2 * 4 * 256 * 2;

    expect(result.model.cacheStrategy).toBe("hybrid_attention");
    expect(result.model.attentionLayerCount).toBe(8);
    expect(result.linearStateBytes).toBeGreaterThan(0);
    expect(result.kvCacheBytes).toBe(hybridExpectedKv);
    expect(result.kvCacheBytes).toBeLessThan(denseEquivalentKv);
    expect(longerContext.linearStateBytes).toBe(result.linearStateBytes);
    expect(biggerBatch.linearStateBytes).toBe(result.linearStateBytes * 2);
    expect(longerContext.totalBytes).toBeGreaterThan(result.totalBytes);
    expect(biggerBatch.totalBytes).toBeGreaterThan(result.totalBytes);
  });

  it("treats multimodal checkpoints as text-only estimates while keeping resident weights", () => {
    const qwen = models.find((model) => model.id === "qwen-3.5-9b");
    expect(qwen).toBeDefined();
    const officialProfile = qwen!.inferenceProfiles.find((profile) => profile.id === "official-bf16");
    expect(officialProfile).toBeDefined();

    const result = estimateVram(
      buildInput({
        modelId: "qwen-3.5-9b",
        dtype: "bf16",
      }),
    );

    expect(result.model.modality).toBe("multimodal");
    expect(result.weightsBytes).toBe(
      qwen!.totalParams * (officialProfile?.weightBytes ?? 0),
    );
    expect(result.linearStateBytes).toBeGreaterThan(0);
    expect(result.notes.join(" ")).toMatch(/text-only|vision|projector/i);
  });

  it("models GPT-OSS alternating sliding-window attention instead of charging every layer at full context", () => {
    const result = estimateVram(
      buildInput({
        runtimeId: "vllm",
        kvCacheDtype: "fp8",
        modelId: "gpt-oss-120b",
        contextLength: 65536,
      }),
    );
    const alternatingExpectedKv =
      (18 * 65536 + 18 * 128) * 2 * 8 * 64 * 1;
    const denseEquivalentKv =
      36 * 65536 * 2 * 8 * 64 * 1;

    expect(result.model.cacheStrategy).toBe("alternating_window_attention");
    expect(result.kvCacheBytes).toBe(alternatingExpectedKv);
    expect(result.kvCacheBytes).toBeLessThan(denseEquivalentKv);
    expect(result.notes.join(" ")).toMatch(/sliding-window|128-token cache/i);
  });

  it("matches the canonical inference scenarios snapshot", () => {
    const scenarios = {
      inference7B24Gb: summarize(
        estimateVram(
          buildInput({
            modelId: "qwen-2.5-7b",
            dtype: "fp16",
            gpuId: "rtx-4090-24gb",
          }),
        ),
      ),
      inference70B4bit80Gb: summarize(
        estimateVram(
          buildInput({
            modelId: "llama-3.1-70b",
            dtype: "int4",
            gpuId: "a100-80gb",
          }),
        ),
      ),
      mixtral24Gb: summarize(
        estimateVram(
          buildInput({
            modelId: "mixtral-8x7b",
            dtype: "fp16",
          }),
        ),
      ),
      qwen35Transformers: summarize(
        estimateVram(
          buildInput({
            runtimeId: "transformers",
            modelId: "qwen-3.5-4b",
            dtype: "bf16",
          }),
        ),
      ),
      qwen35Vllm: summarize(
        estimateVram(
          buildInput({
            runtimeId: "vllm",
            kvCacheDtype: "bf16",
            modelId: "qwen-3.5-4b",
            dtype: "bf16",
          }),
        ),
      ),
      nemotronVllm: summarize(
        estimateVram(
          buildInput({
            runtimeId: "vllm",
            kvCacheDtype: "bf16",
            modelId: "openreasoning-nemotron-14b",
            dtype: "bf16",
            gpuId: "a100-40gb",
          }),
        ),
      ),
    };

    expect(scenarios).toMatchInlineSnapshot(`
      {
        "inference70B4bit80Gb": {
          "activationsGb": 0,
          "fits": true,
          "gradientsGb": 0,
          "kvGb": 1.3,
          "linearStateGb": 0,
          "masterGb": 0,
          "maxConcurrency": null,
          "optimizerGb": 0,
          "profile": "Official BF16 checkpoint loaded as 4-bit",
          "requiredGb": 44.2,
          "runtime": "Transformers",
          "totalGb": 44.2,
          "warnings": [],
          "weightsGb": 38.8,
        },
        "inference7B24Gb": {
          "activationsGb": 0,
          "fits": true,
          "gradientsGb": 0,
          "kvGb": 0.2,
          "linearStateGb": 0,
          "masterGb": 0,
          "maxConcurrency": null,
          "optimizerGb": 0,
          "profile": "Official BF16 checkpoint",
          "requiredGb": 17,
          "runtime": "Transformers",
          "totalGb": 17,
          "warnings": [],
          "weightsGb": 15.2,
        },
        "mixtral24Gb": {
          "activationsGb": 0,
          "fits": false,
          "gradientsGb": 0,
          "kvGb": 0.5,
          "linearStateGb": 0,
          "masterGb": 0,
          "maxConcurrency": null,
          "optimizerGb": 0,
          "profile": "Official BF16 checkpoint",
          "requiredGb": 103.3,
          "runtime": "Transformers",
          "totalGb": 103.3,
          "warnings": [],
          "weightsGb": 93.4,
        },
        "nemotronVllm": {
          "activationsGb": 0,
          "fits": true,
          "gradientsGb": 0,
          "kvGb": 0.8,
          "linearStateGb": 0,
          "masterGb": 0,
          "maxConcurrency": 6,
          "optimizerGb": 0,
          "profile": "Official BF16 checkpoint",
          "requiredGb": 37.1,
          "runtime": "vLLM",
          "totalGb": 33.4,
          "warnings": [],
          "weightsGb": 29.5,
        },
        "qwen35Transformers": {
          "activationsGb": 0,
          "fits": true,
          "gradientsGb": 0,
          "kvGb": 0.1,
          "linearStateGb": 0.053,
          "masterGb": 0,
          "maxConcurrency": null,
          "optimizerGb": 0,
          "profile": "Official BF16 checkpoint",
          "requiredGb": 6.3,
          "runtime": "Transformers",
          "totalGb": 6.3,
          "warnings": [],
          "weightsGb": 4.7,
        },
        "qwen35Vllm": {
          "activationsGb": 0,
          "fits": true,
          "gradientsGb": 0,
          "kvGb": 0.1,
          "linearStateGb": 0.053,
          "masterGb": 0,
          "maxConcurrency": 87,
          "optimizerGb": 0,
          "profile": "Official BF16 checkpoint",
          "requiredGb": 7.1,
          "runtime": "vLLM",
          "totalGb": 6.3,
          "warnings": [],
          "weightsGb": 4.7,
        },
      }
    `);
  });

  it("clips excessive context length while preserving inference behavior", () => {
    expect(
      summarize(
        estimateVram(
          buildInput({
            runtimeId: "vllm",
            modelId: "gpt-oss-120b",
            contextLength: 256000,
            gpuId: "h100-80gb",
          }),
        ),
      ),
    ).toMatchInlineSnapshot(`
      {
        "activationsGb": 0,
        "fits": true,
        "gradientsGb": 0,
        "kvGb": 4.7,
        "linearStateGb": 0,
        "masterGb": 0,
        "maxConcurrency": 1,
        "optimizerGb": 0,
        "profile": "Mixed MXFP4 + BF16 checkpoint",
        "requiredGb": 85.6,
        "runtime": "vLLM",
        "totalGb": 77,
        "warnings": [
          "Context clipped to 128,000 tokens, the model's advertised window.",
        ],
        "weightsGb": 65.3,
      }
    `);
  });
});
