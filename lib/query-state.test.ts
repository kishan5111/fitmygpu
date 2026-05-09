import { DEFAULT_INPUT } from "@/lib/constants";
import { parseSearchParams, serializeEstimateInput } from "@/lib/query-state";

describe("query state", () => {
  it("round-trips the runtime selection", () => {
    const params = serializeEstimateInput({
      ...DEFAULT_INPUT,
      runtimeId: "vllm",
      kvCacheDtype: "fp8",
      vllmGpuUtilization: 0.95,
      gpuCount: 4,
      modelId: "qwen-3.5-4b",
      inferenceProfileId: "official-bf16",
      dtype: "bf16",
    });

    expect(params.get("rt")).toBe("vllm");
    expect(params.get("kvd")).toBe("fp8");
    expect(params.get("vutil")).toBe("0.95");
    expect(params.get("ngpu")).toBe("4");

    const parsed = parseSearchParams(params);

    expect(parsed.runtimeId).toBe("vllm");
    expect(parsed.kvCacheDtype).toBe("fp8");
    expect(parsed.vllmGpuUtilization).toBe(0.95);
    expect(parsed.gpuCount).toBe(4);
    expect(parsed.modelId).toBe("qwen-3.5-4b");
    expect(parsed.inferenceProfileId).toBe("official-bf16");
    expect(parsed.dtype).toBe("bf16");
  });

  it("falls back to the default runtime for deprecated runtime ids", () => {
    const parsedUnsloth = parseSearchParams(new URLSearchParams("rt=unsloth"));
    const parsedSglang = parseSearchParams(new URLSearchParams("rt=sglang"));

    expect(parsedUnsloth.runtimeId).toBe(DEFAULT_INPUT.runtimeId);
    expect(parsedSglang.runtimeId).toBe(DEFAULT_INPUT.runtimeId);
  });
});
