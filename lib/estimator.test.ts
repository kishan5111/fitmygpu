import { models } from "@/data/models";
import { estimateVram } from "@/lib/estimator";
import type { EstimateInput } from "@/lib/types";

function buildInput(overrides: Partial<EstimateInput> = {}): EstimateInput {
  return {
    mode: "inference",
    trainingType: "sft",
    modelId: "llama-3.1-8b",
    dtype: "fp16",
    inferenceProfileId: "",
    gpuId: "rtx-4090-24gb",
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
    profile: result.calculationProfile,
    fits: result.fits,
    totalGb: Number((result.totalBytes / 1_000_000_000).toFixed(1)),
    weightsGb: Number((result.weightsBytes / 1_000_000_000).toFixed(1)),
    kvGb: Number((result.kvCacheBytes / 1_000_000_000).toFixed(1)),
    activationsGb: Number((result.activationsBytes / 1_000_000_000).toFixed(1)),
    gradientsGb: Number((result.gradientsBytes / 1_000_000_000).toFixed(1)),
    optimizerGb: Number((result.optimizerBytes / 1_000_000_000).toFixed(1)),
    warnings: result.warnings,
  };
}

describe("estimateVram", () => {
  it("increases monotonically with context and batch during inference", () => {
    const base = estimateVram(buildInput());
    const longerContext = estimateVram(buildInput({ contextLength: 8192 }));
    const biggerBatch = estimateVram(buildInput({ batchSize: 2 }));

    expect(longerContext.totalBytes).toBeGreaterThan(base.totalBytes);
    expect(biggerBatch.totalBytes).toBeGreaterThan(base.totalBytes);
  });

  it("keeps activation memory aligned across SFT, LoRA, and QLoRA", () => {
    const sft = estimateVram(
      buildInput({
        mode: "training",
        trainingType: "sft",
        dtype: "bf16",
      }),
    );
    const lora = estimateVram(
      buildInput({
        mode: "training",
        trainingType: "lora",
        dtype: "bf16",
      }),
    );
    const qlora = estimateVram(
      buildInput({
        mode: "training",
        trainingType: "qlora",
        dtype: "int4",
      }),
    );

    expect(qlora.weightsBytes).toBeLessThan(lora.weightsBytes);
    expect(lora.activationsBytes).toBeCloseTo(sft.activationsBytes, -1);
    expect(qlora.activationsBytes).toBeCloseTo(sft.activationsBytes, -1);
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

    expect(gptOss20B.calculationProfile).toBe("Official mixed checkpoint");
    expect(gptOss20B.fits).toBe(true);
    expect(gptOss20B.weightsBytes / 1_000_000_000).toBeLessThan(16);

    expect(gptOss120B.calculationProfile).toBe("Official mixed checkpoint");
    expect(gptOss120B.fits).toBe(true);
    expect(gptOss120B.weightsBytes / 1_000_000_000).toBeLessThan(80);
  });

  it("warns when a GPT-OSS inference run is treated as a requantized proxy", () => {
    const proxy = estimateVram(
      buildInput({
        modelId: "gpt-oss-20b",
        dtype: "int4",
      }),
    );

    expect(proxy.calculationProfile).toBe("Proxy 4-bit estimate");
    expect(proxy.warnings.join(" ")).toMatch(/proxy estimate|official/i);
  });

  it("proxies unsupported training + quantization combinations with warnings", () => {
    const sftInt4 = estimateVram(
      buildInput({
        mode: "training",
        trainingType: "sft",
        dtype: "int4",
      }),
    );
    const loraInt4 = estimateVram(
      buildInput({
        mode: "training",
        trainingType: "lora",
        dtype: "int4",
      }),
    );

    expect(sftInt4.calculationProfile).toBe("BF16 SFT proxy");
    expect(sftInt4.warnings.join(" ")).toMatch(/BF16 SFT proxy|BF16/);
    expect(loraInt4.calculationProfile).toBe("QLoRA proxy");
    expect(loraInt4.warnings.join(" ")).toMatch(/QLoRA proxy|QLoRA/);
  });

  it("matches the canonical scenarios snapshot", () => {
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
      sft7BBf1624Gb: summarize(
        estimateVram(
          buildInput({
            mode: "training",
            trainingType: "sft",
            modelId: "qwen-2.5-7b",
            dtype: "bf16",
          }),
        ),
      ),
      qlora7B24Gb: summarize(
        estimateVram(
          buildInput({
            mode: "training",
            trainingType: "qlora",
            modelId: "qwen-2.5-7b",
            dtype: "int4",
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
    };

    expect(scenarios).toMatchInlineSnapshot(`
      {
        "inference70B4bit80Gb": {
          "activationsGb": 0,
          "fits": true,
          "gradientsGb": 0,
          "kvGb": 1.3,
          "optimizerGb": 0,
          "profile": "Proxy 4-bit estimate",
          "totalGb": 44.2,
          "warnings": [
            "Proxy 4-bit estimate is a proxy estimate, not an official Llama 3.1 70B checkpoint profile.",
          ],
          "weightsGb": 38.8,
        },
        "inference7B24Gb": {
          "activationsGb": 0,
          "fits": true,
          "gradientsGb": 0,
          "kvGb": 0.2,
          "optimizerGb": 0,
          "profile": "Official BF16 checkpoint",
          "totalGb": 17,
          "warnings": [],
          "weightsGb": 15.2,
        },
        "mixtral24Gb": {
          "activationsGb": 0,
          "fits": false,
          "gradientsGb": 0,
          "kvGb": 0.5,
          "optimizerGb": 0,
          "profile": "Official BF16 checkpoint",
          "totalGb": 103.3,
          "warnings": [],
          "weightsGb": 93.4,
        },
        "qlora7B24Gb": {
          "activationsGb": 9.9,
          "fits": true,
          "gradientsGb": 0,
          "kvGb": 0,
          "optimizerGb": 0.1,
          "profile": "QLoRA",
          "totalGb": 16.2,
          "warnings": [],
          "weightsGb": 4.2,
        },
        "sft7BBf1624Gb": {
          "activationsGb": 9.9,
          "fits": false,
          "gradientsGb": 15.2,
          "kvGb": 0,
          "optimizerGb": 60.9,
          "profile": "BF16 SFT",
          "totalGb": 111.3,
          "warnings": [],
          "weightsGb": 15.2,
        },
      }
    `);
  });
});
