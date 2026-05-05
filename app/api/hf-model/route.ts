import { NextResponse } from "next/server";
import { models } from "@/data/models";
import { getHfRepoIdFromUrl, parseHfRepoId } from "@/lib/hf-import";
import type { Dtype, InferenceProfile, ModelSpec } from "@/lib/types";

type HfModelApiResponse = {
  id?: string;
  config?: Record<string, unknown>;
  safetensors?: {
    total?: number;
    parameters?: Record<string, number>;
  };
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const rawModelUrl = url.searchParams.get("url") ?? "";
  const repoId = parseHfRepoId(rawModelUrl);

  if (!repoId) {
    return NextResponse.json(
      { error: "Paste a Hugging Face model URL or repo id like Qwen/Qwen2.5-7B-Instruct." },
      { status: 400 },
    );
  }

  const knownModel = findKnownModel(repoId);

  if (knownModel) {
    return NextResponse.json({
      status: "known",
      modelId: knownModel.id,
      repoId,
    });
  }

  const encodedRepoId = repoId.split("/").map(encodeURIComponent).join("/");
  const response = await fetch(
    `https://huggingface.co/api/models/${encodedRepoId}`,
    {
      headers: {
        Accept: "application/json",
      },
      next: { revalidate: 3600 },
    },
  );

  if (!response.ok) {
    return NextResponse.json(
      { error: `Hugging Face returned ${response.status} for ${repoId}.` },
      { status: response.status },
    );
  }

  const hfModel = (await response.json()) as HfModelApiResponse;
  const rawConfig = await fetchRawConfig(repoId);
  const modelWithConfig: HfModelApiResponse = {
    ...hfModel,
    config: {
      ...(hfModel.config ?? {}),
      ...(rawConfig ?? {}),
    },
  };
  let model: ModelSpec;

  try {
    model = buildEstimatedModel(repoId, modelWithConfig);
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Could not infer this Hugging Face model from public config metadata.",
      },
      { status: 422 },
    );
  }

  return NextResponse.json({
    status: "estimated",
    model,
    repoId,
  });
}

async function fetchRawConfig(repoId: string) {
  const encodedRepoId = repoId.split("/").map(encodeURIComponent).join("/");
  const response = await fetch(
    `https://huggingface.co/${encodedRepoId}/raw/main/config.json`,
    {
      headers: {
        Accept: "application/json",
      },
      next: { revalidate: 3600 },
    },
  );

  if (!response.ok) {
    return null;
  }

  return (await response.json()) as Record<string, unknown>;
}

function findKnownModel(repoId: string) {
  const normalizedRepoId = repoId.toLowerCase();

  return models.find((model) => {
    const urls = [
      model.sourceUrl,
      ...model.inferenceProfiles.map((profile) => profile.sourceUrl),
    ];

    return urls.some((sourceUrl) => getHfRepoIdFromUrl(sourceUrl) === normalizedRepoId);
  });
}

function buildEstimatedModel(repoId: string, hfModel: HfModelApiResponse): ModelSpec {
  const config = hfModel.config ?? {};
  const totalParams = getTotalParams(hfModel);
  const effectiveDtype = getEffectiveDtype(hfModel);
  const weightBytes = getAverageWeightBytes(hfModel, totalParams, effectiveDtype);
  const modelType = readString(config, "model_type") ?? "transformer";
  const numLayers = readNumber(config, "num_hidden_layers", "n_layer", "num_layers");
  const hiddenSize = readNumber(config, "hidden_size", "n_embd", "d_model");
  const numAttentionHeads = readNumber(config, "num_attention_heads", "n_head");
  const numKvHeads =
    readOptionalNumber(
      config,
      "num_key_value_heads",
      "n_kv_head",
      "multi_query_group_num",
    ) ?? numAttentionHeads;
  const contextLength =
    readOptionalNumber(
      config,
      "max_position_embeddings",
      "seq_length",
      "n_positions",
      "model_max_length",
    ) ?? 32_768;
  const vocabSize = readOptionalNumber(config, "vocab_size") ?? 0;
  const displayName = repoId.split("/").at(-1)?.replace(/[-_]/g, " ") ?? repoId;
  const sourceUrl = `https://huggingface.co/${repoId}`;
  const profile: InferenceProfile = {
    id: "hf-config-estimate",
    label: `${formatDtypeLabel(effectiveDtype)} config estimate`,
    effectiveDtype,
    official: false,
    confidence: "estimated",
    note:
      "Estimated from Hugging Face config and safetensors parameter metadata. Architecture-specific kernels, custom attention layouts, and quantization packing may differ.",
    sourceUrl,
    weightMode: "direct",
    weightBytes,
  };

  return {
    id: `hf-${repoId.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}`,
    displayName,
    family: modelType.toUpperCase(),
    organization: repoId.split("/")[0] ?? "Hugging Face",
    architectureType: `${modelType} config estimate`,
    isMoe: isMoeConfig(config),
    totalParams,
    numLayers,
    hiddenSize,
    numAttentionHeads,
    numKvHeads,
    contextLength,
    vocabSize,
    license: "See Hugging Face model card",
    sourceUrl,
    shortDescription:
      "Imported from Hugging Face and estimated from public config metadata.",
    researchHighlight:
      "This model is not curated in the FitMyGPU registry yet, so the architecture is inferred from generic Hugging Face config fields.",
    memoryNote:
      "The estimate uses safetensors parameter metadata when available, then applies the standard dense transformer KV cache model.",
    inferenceProfiles: [profile],
    supportedModes: ["inference"],
  };
}

function getTotalParams(hfModel: HfModelApiResponse) {
  const total = hfModel.safetensors?.total;

  if (typeof total === "number" && total > 0) {
    return total;
  }

  throw new Error("This Hugging Face repo does not expose safetensors parameter metadata.");
}

function getEffectiveDtype(hfModel: HfModelApiResponse): Dtype {
  const params = hfModel.safetensors?.parameters ?? {};
  const ordered = Object.entries(params).sort((left, right) => right[1] - left[1]);
  const largestDtype = ordered[0]?.[0]?.toUpperCase();

  if (largestDtype === "F32" || largestDtype === "FP32") {
    return "fp32";
  }

  if (largestDtype === "F16" || largestDtype === "FP16") {
    return "fp16";
  }

  if (largestDtype === "BF16") {
    return "bf16";
  }

  return "bf16";
}

function getAverageWeightBytes(
  hfModel: HfModelApiResponse,
  totalParams: number,
  fallbackDtype: Dtype,
) {
  const params = hfModel.safetensors?.parameters ?? {};
  const weightedBytes = Object.entries(params).reduce((sum, [dtype, count]) => {
    return sum + getBytesForHfDtype(dtype, fallbackDtype) * count;
  }, 0);

  if (weightedBytes > 0) {
    return weightedBytes / totalParams;
  }

  return getBytesForDtype(fallbackDtype);
}

function getBytesForHfDtype(dtype: string, fallbackDtype: Dtype) {
  switch (dtype.toUpperCase()) {
    case "F32":
    case "FP32":
      return 4;
    case "F16":
    case "FP16":
    case "BF16":
      return 2;
    case "F8_E4M3":
    case "F8_E5M2":
    case "FP8":
      return 1;
    default:
      return getBytesForDtype(fallbackDtype);
  }
}

function getBytesForDtype(dtype: Dtype) {
  switch (dtype) {
    case "fp32":
      return 4;
    case "fp8":
    case "int8":
      return 1;
    case "int4":
      return 0.55;
    case "fp16":
    case "bf16":
      return 2;
  }
}

function readNumber(config: Record<string, unknown>, ...keys: string[]) {
  const value = readOptionalNumber(config, ...keys);

  if (value === undefined) {
    throw new Error(`Missing required config field: ${keys.join(" or ")}.`);
  }

  return value;
}

function readOptionalNumber(config: Record<string, unknown>, ...keys: string[]) {
  for (const key of keys) {
    const value = config[key];

    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }

  return undefined;
}

function readString(config: Record<string, unknown>, key: string) {
  const value = config[key];

  return typeof value === "string" ? value : undefined;
}

function isMoeConfig(config: Record<string, unknown>) {
  return Boolean(
    readOptionalNumber(
      config,
      "num_experts",
      "n_routed_experts",
      "num_local_experts",
    ),
  );
}

function formatDtypeLabel(dtype: Dtype) {
  switch (dtype) {
    case "fp32":
      return "FP32";
    case "fp16":
      return "FP16";
    case "bf16":
      return "BF16";
    case "fp8":
      return "FP8";
    case "int8":
      return "INT8";
    case "int4":
      return "4-bit";
  }
}
