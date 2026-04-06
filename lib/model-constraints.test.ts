import { DEFAULT_INPUT } from "@/lib/constants";
import {
  applyModelConstraints,
  getCompatibleInferenceProfiles,
  getInferenceProfileSourceUrl,
} from "@/lib/model-constraints";

describe("model constraints", () => {
  it("filters runtime-incompatible profiles and falls back to a compatible proxy", () => {
    const transformersProfiles = getCompatibleInferenceProfiles("phi-4-14b", "transformers");
    const vllmProfiles = getCompatibleInferenceProfiles("phi-4-14b", "vllm");

    expect(transformersProfiles.map((profile) => profile.id)).toContain("official-onnx-int4");
    expect(vllmProfiles.map((profile) => profile.id)).not.toContain("official-onnx-int4");

    const normalized = applyModelConstraints({
      ...DEFAULT_INPUT,
      modelId: "phi-4-14b",
      runtimeId: "vllm",
      dtype: "int4",
      inferenceProfileId: "official-onnx-int4",
    });

    expect(normalized.inferenceProfileId).toBe("proxy-int4");
    expect(normalized.dtype).toBe("int4");
  });

  it("resets runtime-only KV cache dtype settings when the runtime does not support them", () => {
    const normalized = applyModelConstraints({
      ...DEFAULT_INPUT,
      runtimeId: "transformers",
      kvCacheDtype: "fp8",
      contextLength: 32768,
      batchSize: 4,
    });

    expect(normalized.kvCacheDtype).toBe("bf16");
    expect(normalized.contextLength).toBe(4096);
    expect(normalized.batchSize).toBe(1);
  });

  it("prefers the selected checkpoint profile URL over the base model URL", () => {
    expect(
      getInferenceProfileSourceUrl(
        "qwen-2.5-7b",
        "official-awq-int4",
        "transformers",
        "int4",
      ),
    ).toBe("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/tree/main");
  });
});
