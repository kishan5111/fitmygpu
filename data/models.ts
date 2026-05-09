import { ALL_RUNTIMES } from "@/lib/runtime";
import type { InferenceProfile, ModelInsightPoint, ModelSpec } from "@/lib/types";

function bytesFromCheckpointGb(checkpointGb: number, totalParams: number) {
  return (checkpointGb * 1_000_000_000) / totalParams;
}

function bytesFromCheckpointGiB(checkpointGiB: number, totalParams: number) {
  return (checkpointGiB * 1024 * 1024 * 1024) / totalParams;
}

function bytesFromCheckpointBytes(checkpointBytes: number, totalParams: number) {
  return checkpointBytes / totalParams;
}

function directProfile(
  profile: Omit<InferenceProfile, "official" | "weightMode" | "confidence"> &
    Partial<Pick<InferenceProfile, "confidence">>,
): InferenceProfile {
  return {
    ...profile,
    official: true,
    confidence: profile.confidence ?? "verified",
    weightMode: "direct",
  };
}

function point(label: string, detail: string): ModelInsightPoint {
  return { label, detail };
}

const qwen35OverviewPoints = (attentionLayers: number, totalLayers: number) => [
  point(
    "Training scope",
    "Built as a unified vision-language foundation with pre-training and post-training on multimodal tokens rather than a separate late-fusion stack.",
  ),
  point(
    "Hybrid layout",
    `${attentionLayers} of ${totalLayers} layers use gated attention while the rest use Gated DeltaNet blocks, so the stack is not a full-attention transformer end to end.`,
  ),
  point(
    "Context design",
    "Published with a native 262K context window and an architecture intended to stretch beyond that range in longer-context settings.",
  ),
];

const qwen35ResearchHighlights = [
  point(
    "Unified vision-language foundation",
    "The family is trained as one multimodal base rather than as separate text and vision branches bolted together late, which is why text-only serving still keeps the resident vision-side weights on card.",
  ),
  point(
    "Efficient hybrid architecture",
    "Gated DeltaNet layers carry sequence state while periodic gated-attention layers handle KV-heavy reasoning, so the stack aims for long-context throughput without paying dense-attention KV cost on every layer.",
  ),
  point(
    "Scalable RL generalization",
    "Qwen frames reinforcement learning and large agent-environment scaling as core to the family, with training aimed at more robust adaptation across reasoning, coding, and agent workflows.",
  ),
  point(
    "Global coverage",
    "The release emphasizes support for 201 languages and dialects, which matters for deployment quality and reinforces that the family is meant as a broad general-purpose foundation.",
  ),
  point(
    "Training infrastructure",
    "The release emphasizes near-text-only multimodal training efficiency and asynchronous RL infrastructure, signaling that the stack was built to scale rather than as a small multimodal add-on.",
  ),
];

const qwen35MemoryBehaviorPoints = (attentionLayers: number, totalLayers: number) => [
  `This text-only estimate still keeps the resident multimodal checkpoint weights on card, so the floor is higher than a pure language-only artifact of similar active size.`,
  `Only ${attentionLayers} of ${totalLayers} layers carry a standard KV cache. The remaining layers contribute a fixed sequence-state term instead, which makes long-context growth less aggressive than a dense full-attention stack.`,
  "Longer context and higher concurrency still increase memory monotonically, but more of the footprint shifts into mixed KV-plus-state behavior instead of pure transformer cache expansion.",
];

const gptOssResearchHighlights = (experts: number, activeParams: string) => [
  point(
    "Open reasoning and agents",
    "OpenAI frames GPT-OSS as an open-weight reasoning and agent model rather than as a plain chat checkpoint, with tool use and controllable reasoning effort as first-class product features.",
  ),
  point(
    "Sparse MoE backbone",
    `Each MoE block uses ${experts} experts with routed activation while the active path stays near ${activeParams} per token, so capability scales faster than per-token compute.`,
  ),
  point(
    "Mixed-weight release format",
    "The published checkpoint keeps most routed expert weights in MXFP4 while shared weights remain in BF16, which is the main deployment change versus a conventional dense BF16 export.",
  ),
  point(
    "Long-context attention recipe",
    "The stack alternates full and sliding-window attention so the model can target long reasoning traces without paying full-context KV cost on every layer.",
  ),
];

const llama31ResearchHighlights = (sizeLabel: string) => [
  point(
    "128K context family",
    `Llama 3.1 extends Meta's general-purpose open model line to a native 128K context window, which is the main deployment-facing change over earlier shorter-context Llama releases.`,
  ),
  point(
    "Multilingual instruction tuning",
    "Meta emphasizes broader multilingual coverage and stronger instruction-following across the release, not just a raw parameter bump.",
  ),
  point(
    "Tool-capable release",
    `The ${sizeLabel} checkpoint is positioned as tool-usable and production-oriented, so the release focus is practical agent and assistant behavior rather than a novel architecture change.`,
  ),
  point(
    "Grouped-query attention",
    "Grouped-query attention remains the key inference-side design choice because it preserves long-context serving practicality without a full multi-head KV burden.",
  ),
];

const qwen25ResearchHighlights = (sizeLabel: string) => [
  point(
    "Qwen2.5 capability upgrade",
    `Qwen presents the ${sizeLabel} model as part of a broader Qwen2.5 upgrade around stronger coding, math, instruction-following, and structured output behavior over Qwen2.`,
  ),
  point(
    "128K long-context release",
    "The family is explicitly pushed as a long-context line with a native 128K window, making long-document and extended-generation quality a core part of the release rather than an add-on.",
  ),
  point(
    "Broader multilingual scope",
    "Qwen highlights broader multilingual and domain coverage, reinforcing that the line is meant as a general-purpose foundation rather than a narrow English-only coding model.",
  ),
  point(
    "Dense architecture, better post-training",
    "The product story is not sparse routing or exotic attention; it is a stronger dense transformer plus better data, post-training, and deployment packaging.",
  ),
];

const nemotronResearchHighlights = (sizeLabel: string) => [
  point(
    "Reasoning-first post-training",
    `NVIDIA positions the ${sizeLabel} Nemotron model around stronger math, code, and science reasoning rather than around a new base architecture.`,
  ),
  point(
    "Qwen2.5-derived backbone",
    "The family stays close to a Qwen2.5 dense grouped-query backbone, so the main change is in post-training behavior and benchmark profile, not in memory geometry.",
  ),
  point(
    "GenSelect heavy mode",
    "The model card explicitly introduces a heavier multi-sample inference path through GenSelect, which matters because capability can scale at inference time without changing the resident model itself.",
  ),
  point(
    "Benchmark-led release framing",
    "NVIDIA markets the line primarily through reasoning benchmark results in its size class, so this is a capability-tuned release more than an architecture-tuned one.",
  ),
];

const gemma2ResearchHighlights = (sizeLabel: string) => [
  point(
    "Compact open model line",
    `Google positions Gemma 2 ${sizeLabel} as part of a lightweight open family derived from Gemini-era research, aimed at getting strong dense-model quality from smaller deployment footprints.`,
  ),
  point(
    "Efficiency over frontier scale",
    "The release emphasis is efficient open deployment and good quality-per-parameter, not sparse routing, multimodal fusion, or ultra-long-context serving.",
  ),
  point(
    "Instruction-tuned product focus",
    "The instruction variants are framed as practical developer models, so the story is real deployment usability rather than experimental architecture novelty.",
  ),
];

const mistralNemoResearchHighlights = [
  point(
    "Mistral-NVIDIA joint release",
    "Mistral Nemo is presented as a joint Mistral-NVIDIA model rather than as a routine checkpoint refresh, which is part of why the launch emphasized deployment practicality.",
  ),
  point(
    "128K context and new tokenizer",
    "The release highlights a native 128K context window and the Tekken tokenizer, both of which materially affect how the model is positioned for long-form and multilingual use.",
  ),
  point(
    "Data mix upgrade",
    "Mistral describes the model as trained with more multilingual and code-oriented data, so the upgrade story is as much about training mix as about architecture.",
  ),
  point(
    "Deployment-friendly packaging",
    "The family is explicitly pitched as a strong dense long-context model that still fits realistic single-node inference workflows, especially once lower-precision checkpoints are used.",
  ),
];

const mixtralResearchHighlights = [
  point(
    "Sparse top-2 expert routing",
    "Mixtral's core research change is sparse MoE routing: only a small subset of experts is active per token even though a much larger parameter pool stays resident.",
  ),
  point(
    "Dense-quality alternative path",
    "The release matters because it offered a practical open sparse model path at a time when most comparable open checkpoints were still fully dense.",
  ),
  point(
    "Compute and capacity decoupling",
    "Mixtral is important less because of a new attention design and more because it separates total parameter capacity from per-token compute in a way that users can feel operationally.",
  ),
];

const phi4ResearchHighlights = [
  point(
    "Reasoning-per-parameter focus",
    "Microsoft positions Phi-4 around unusually strong reasoning and coding quality for its size, so the release story is capability density rather than frontier-scale parameters.",
  ),
  point(
    "Synthetic and curated data mix",
    "The model card emphasizes the training recipe itself, especially high-quality synthetic and curated data for math, code, instruction following, and commonsense tasks.",
  ),
  point(
    "Straightforward dense deployment",
    "Phi-4 does not introduce sparse routing or hybrid attention; the practical angle is that it stays a normal dense deployment target while aiming for stronger reasoning than many peers in its class.",
  ),
];

export const models: ModelSpec[] = [
  {
    id: "gpt-oss-20b",
    displayName: "GPT-OSS 20B",
    family: "GPT-OSS",
    organization: "OpenAI",
    architectureType: "Mixture-of-experts transformer",
    isMoe: true,
    totalParams: 21_000_000_000,
    activeParams: 3_600_000_000,
    numLayers: 24,
    hiddenSize: 2880,
    numAttentionHeads: 64,
    numKvHeads: 8,
    attentionHeadDim: 64,
    cacheStrategy: "alternating_window_attention",
    denseAttentionLayerCount: 12,
    slidingWindowAttentionLayerCount: 12,
    slidingWindowContextLength: 128,
    contextLength: 128_000,
    vocabSize: 201_088,
    license: "Apache 2.0",
    sourceUrl: "https://openai.com/open-models",
    shortDescription:
      "Smaller GPT-OSS reasoning checkpoint with a routed MoE stack, 128K context, and a relatively light active path.",
    researchHighlight:
      "OpenAI positions GPT-OSS as an open-weight reasoning and agent model: Harmony-format reasoning, tool use, and MXFP4 post-training on the MoE weights are the headline changes that make the 20B model practical on smaller hardware.",
    researchHighlights: gptOssResearchHighlights(32, "3.6B params"),
    memoryNote:
      "More than 90% of GPT-OSS 20B's parameters sit in MoE weights quantized to MXFP4, while the remaining shared weights stay in BF16.",
    inferenceProfiles: [
      directProfile({
        id: "official-mixed",
        label: "Mixed MXFP4 + BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointGiB(12.8, 21_000_000_000),
        sourceUrl: "https://huggingface.co/openai/gpt-oss-20b",
        note: "OpenAI's GPT-OSS model card lists a 12.8 GiB checkpoint for gpt-oss-20b. The estimator uses that published mixed MXFP4 + BF16 resident checkpoint size directly.",
      }),
    ],
  },
  {
    id: "gpt-oss-120b",
    displayName: "GPT-OSS 120B",
    family: "GPT-OSS",
    organization: "OpenAI",
    architectureType: "Mixture-of-experts transformer",
    isMoe: true,
    totalParams: 117_000_000_000,
    activeParams: 5_100_000_000,
    numLayers: 36,
    hiddenSize: 2880,
    numAttentionHeads: 64,
    numKvHeads: 8,
    attentionHeadDim: 64,
    cacheStrategy: "alternating_window_attention",
    denseAttentionLayerCount: 18,
    slidingWindowAttentionLayerCount: 18,
    slidingWindowContextLength: 128,
    contextLength: 128_000,
    vocabSize: 201_088,
    license: "Apache 2.0",
    sourceUrl: "https://openai.com/open-models",
    shortDescription:
      "Largest GPT-OSS checkpoint in the current registry, built for higher-capacity open reasoning with a much larger resident expert pool.",
    researchHighlight:
      "The 120B release is framed as the production GPT-OSS model: configurable reasoning effort, native agent features, and MXFP4-quantized MoE weights are the main product-level changes that let it target single-80GB deployment.",
    researchHighlights: gptOssResearchHighlights(128, "5.1B params"),
    memoryNote:
      "More than 90% of GPT-OSS 120B's parameters sit in MXFP4-quantized MoE weights, while the remaining shared weights stay in BF16.",
    inferenceProfiles: [
      directProfile({
        id: "official-mixed",
        label: "Mixed MXFP4 + BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointGiB(60.8, 117_000_000_000),
        sourceUrl: "https://huggingface.co/openai/gpt-oss-120b",
        note: "OpenAI's GPT-OSS model card lists a 60.8 GiB checkpoint for gpt-oss-120b. The estimator uses that published mixed MXFP4 + BF16 resident checkpoint size directly.",
      }),
    ],
  },
  {
    id: "llama-3.1-8b",
    displayName: "Llama 3.1 8B",
    family: "Llama",
    organization: "Meta",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 8_030_000_000,
    numLayers: 32,
    hiddenSize: 4096,
    numAttentionHeads: 32,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 128_256,
    license: "Llama 3.1 Community License",
    sourceUrl: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    shortDescription:
      "Compact dense Llama model with grouped-query attention and a 128K context window.",
    researchHighlight:
      "Meta’s Llama 3.1 release centers on multilingual instruction tuning, 128K context, and built-in tool-use support while keeping grouped-query attention as the main inference-scaling choice across the family.",
    researchHighlights: llama31ResearchHighlights("8B"),
    memoryNote:
      "Dense weights dominate the footprint; grouped KV heads help prevent cache growth from exploding at long context.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: 2,
        sourceUrl: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        note: "Meta's official Llama 3.1 8B Instruct release is a BF16 checkpoint with grouped-query attention.",
      }),
    ],
  },
  {
    id: "llama-3.1-70b",
    displayName: "Llama 3.1 70B",
    family: "Llama",
    organization: "Meta",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 70_600_000_000,
    numLayers: 80,
    hiddenSize: 8192,
    numAttentionHeads: 64,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 128_256,
    license: "Llama 3.1 Community License",
    sourceUrl: "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
    shortDescription:
      "High-capacity dense Llama model that is common in serious long-context inference and fine-tuning work.",
    researchHighlight:
      "The 70B Llama 3.1 checkpoint extends the same 128K multilingual, tool-capable recipe to a much larger dense model, with grouped-query attention kept specifically for long-context inference scalability.",
    researchHighlights: llama31ResearchHighlights("70B"),
    memoryNote:
      "Most of the VRAM goes into resident dense weights, so quantization is the key lever for single-GPU inference.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: 2,
        sourceUrl: "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
        note: "Meta's official Llama 3.1 70B Instruct release is a BF16 checkpoint with grouped-query attention.",
      }),
    ],
  },
  {
    id: "qwen-2.5-7b",
    displayName: "Qwen 2.5 7B",
    family: "Qwen",
    organization: "Alibaba",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 7_610_000_000,
    numLayers: 28,
    hiddenSize: 3584,
    numAttentionHeads: 28,
    numKvHeads: 4,
    contextLength: 131_072,
    vocabSize: 152_064,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    shortDescription:
      "Small-to-mid-sized Qwen model with long context support and efficient grouped KV heads.",
    researchHighlight:
      "Qwen2.5’s official release emphasizes stronger coding, math, structured output, and long-text behavior over Qwen2, with 128K context and broader multilingual coverage as the main product-level changes.",
    researchHighlights: qwen25ResearchHighlights("7B"),
    memoryNote:
      "This is still a dense model, so resident weights set the floor; the compact KV layout mainly helps as context grows.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointGb(15.2, 7_610_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main",
        note: "The official Qwen2.5-7B-Instruct checkpoint repository is about 15.2 GB on Hugging Face.",
      }),
      directProfile({
        id: "official-gptq-int4",
        label: "Official GPTQ 4-bit checkpoint",
        effectiveDtype: "int4",
        weightBytes: bytesFromCheckpointGb(5.59, 7_610_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4/tree/main",
        note: "The official Qwen2.5-7B-Instruct-GPTQ-Int4 checkpoint repository is about 5.59 GB on Hugging Face.",
      }),
      directProfile({
        id: "official-awq-int4",
        label: "Official AWQ 4-bit checkpoint",
        effectiveDtype: "int4",
        weightBytes: bytesFromCheckpointGb(5.58, 7_610_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/tree/main",
        note: "The official Qwen2.5-7B-Instruct-AWQ checkpoint repository is about 5.58 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-2.5-14b",
    displayName: "Qwen 2.5 14B",
    family: "Qwen",
    organization: "Alibaba",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 14_700_000_000,
    numLayers: 48,
    hiddenSize: 5120,
    numAttentionHeads: 40,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 152_064,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
    shortDescription:
      "Mid-sized Qwen model with strong long-context behavior and a practical fit for 24 to 80 GB cards.",
    researchHighlight:
      "The 14B Qwen2.5 release carries the same Qwen2.5 upgrades at a more capable size: stronger instruction following, better structured outputs, 128K context, and wider multilingual support.",
    researchHighlights: qwen25ResearchHighlights("14B"),
    memoryNote:
      "The jump from 7B to 14B is mostly resident weight memory; KV cache remains relatively controlled thanks to grouped KV heads.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointGb(29.6, 14_700_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main",
        note: "The official Qwen2.5-14B-Instruct checkpoint repository is about 29.6 GB on Hugging Face.",
      }),
      directProfile({
        id: "official-gptq-int4",
        label: "Official GPTQ 4-bit checkpoint",
        effectiveDtype: "int4",
        weightBytes: bytesFromCheckpointGb(10, 14_700_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4/tree/main",
        note: "The official Qwen2.5-14B-Instruct-GPTQ-Int4 checkpoint repository is about 10 GB on Hugging Face.",
      }),
      directProfile({
        id: "official-awq-int4",
        label: "Official AWQ 4-bit checkpoint",
        effectiveDtype: "int4",
        weightBytes: bytesFromCheckpointGb(10, 14_700_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ/tree/main",
        note: "The official Qwen2.5-14B-Instruct-AWQ checkpoint repository is about 10 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-2.5-32b",
    displayName: "Qwen 2.5 32B",
    family: "Qwen",
    organization: "Alibaba",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 32_500_000_000,
    numLayers: 64,
    hiddenSize: 5120,
    numAttentionHeads: 40,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 152_064,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
    shortDescription:
      "Larger dense Qwen variant that often pushes single-GPU inference toward aggressive quantization.",
    researchHighlight:
      "At 32B, Qwen2.5 is positioned as the larger dense version of the same family improvements: stronger knowledge, coding and math, more stable long generations, and a native 128K context recipe.",
    researchHighlights: qwen25ResearchHighlights("32B"),
    memoryNote:
      "Dense resident weights dominate here, which is why 4-bit loading is usually the difference between fitting and not fitting on one card.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointGb(65.5, 32_500_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/main",
        note: "The official Qwen2.5-32B-Instruct checkpoint repository is about 65.5 GB on Hugging Face.",
      }),
      directProfile({
        id: "official-gptq-int4",
        label: "Official GPTQ 4-bit checkpoint",
        effectiveDtype: "int4",
        weightBytes: bytesFromCheckpointGb(19.4, 32_500_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4/tree/main",
        note: "The official Qwen2.5-32B-Instruct-GPTQ-Int4 checkpoint repository is about 19.4 GB on Hugging Face.",
      }),
      directProfile({
        id: "official-awq-int4",
        label: "Official AWQ 4-bit checkpoint",
        effectiveDtype: "int4",
        weightBytes: bytesFromCheckpointGb(19.3, 32_500_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ/tree/main",
        note: "The official Qwen2.5-32B-Instruct-AWQ checkpoint repository is about 19.3 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3.5-0.8b",
    displayName: "Qwen 3.5 0.8B",
    family: "Qwen",
    organization: "Alibaba",
    architectureType: "Hybrid multimodal transformer",
    isMoe: false,
    totalParams: 900_000_000,
    numLayers: 24,
    hiddenSize: 1024,
    numAttentionHeads: 8,
    numKvHeads: 2,
    attentionHeadDim: 256,
    linearNumKeyHeads: 16,
    linearNumValueHeads: 16,
    linearKeyHeadDim: 128,
    linearValueHeadDim: 128,
    linearConvKernelDim: 4,
    linearStateBytesPerElement: 4,
    contextLength: 262_144,
    modality: "multimodal",
    cacheStrategy: "hybrid_attention",
    attentionLayerCount: 6,
    vocabSize: 248_320,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-0.8B-Base",
    shortDescription:
      "Compact Qwen3.5 checkpoint with a hybrid text-plus-vision stack and a small resident footprint for text-only local experimentation.",
    researchHighlight:
      "Qwen3.5 combines a unified vision-language foundation with a hybrid DeltaNet-plus-attention layout, so the architecture story is about multimodal parity and lower long-context serving cost rather than a plain dense upgrade.",
    memoryNote:
      "This text-only estimate still counts the resident multimodal checkpoint weights; only media-token-specific memory is excluded in v1.",
    overviewPoints: qwen35OverviewPoints(6, 24),
    researchHighlights: qwen35ResearchHighlights,
    memoryBehaviorPoints: qwen35MemoryBehaviorPoints(6, 24),
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(1_746_882_752, 900_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-0.8B-Base",
        note: "The official Qwen3.5-0.8B safetensor weights total about 1.75 GB on Hugging Face, and Qwen documents Transformers and vLLM usage for the release.",
      }),
    ],
  },
  {
    id: "qwen-3.5-2b",
    displayName: "Qwen 3.5 2B",
    family: "Qwen",
    organization: "Alibaba",
    architectureType: "Hybrid multimodal transformer",
    isMoe: false,
    totalParams: 2_000_000_000,
    numLayers: 24,
    hiddenSize: 2048,
    numAttentionHeads: 8,
    numKvHeads: 2,
    attentionHeadDim: 256,
    linearNumKeyHeads: 16,
    linearNumValueHeads: 16,
    linearKeyHeadDim: 128,
    linearValueHeadDim: 128,
    linearConvKernelDim: 4,
    linearStateBytesPerElement: 4,
    contextLength: 262_144,
    modality: "multimodal",
    cacheStrategy: "hybrid_attention",
    attentionLayerCount: 6,
    vocabSize: 248_320,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-2B",
    shortDescription:
      "Small hybrid Qwen3.5 release for developers who want longer context and native multimodal training heritage without a large single-card footprint.",
    researchHighlight:
      "The 2B Qwen3.5 model keeps the family’s unified multimodal training and hybrid DeltaNet-attention recipe, so the main change versus Qwen2.5 is the architecture and training setup rather than just parameter count.",
    memoryNote:
      "Resident weights still include the multimodal components, but the hybrid stack keeps text-generation cache growth noticeably lower than a dense full-attention design.",
    overviewPoints: qwen35OverviewPoints(6, 24),
    researchHighlights: qwen35ResearchHighlights,
    memoryBehaviorPoints: qwen35MemoryBehaviorPoints(6, 24),
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(4_548_144_832, 2_000_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-2B",
        note: "The official Qwen3.5-2B safetensor weights total about 4.55 GB on Hugging Face, and Qwen documents both Transformers and vLLM serving paths.",
      }),
    ],
  },
  {
    id: "qwen-3.5-4b",
    displayName: "Qwen 3.5 4B",
    family: "Qwen",
    organization: "Alibaba",
    architectureType: "Hybrid multimodal transformer",
    isMoe: false,
    totalParams: 5_000_000_000,
    numLayers: 32,
    hiddenSize: 2560,
    numAttentionHeads: 16,
    numKvHeads: 4,
    attentionHeadDim: 256,
    linearNumKeyHeads: 16,
    linearNumValueHeads: 32,
    linearKeyHeadDim: 128,
    linearValueHeadDim: 128,
    linearConvKernelDim: 4,
    linearStateBytesPerElement: 4,
    contextLength: 262_144,
    modality: "multimodal",
    cacheStrategy: "hybrid_attention",
    attentionLayerCount: 8,
    vocabSize: 248_320,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-4B",
    shortDescription:
      "Mid-sized Qwen3.5 checkpoint with a larger resident multimodal footprint but still practical for careful single-GPU text-only serving.",
    researchHighlight:
      "The 4B Qwen3.5 release is where the family’s hybrid design starts to matter more operationally: multimodal training stays unified, but only a subset of layers pay full KV-cache cost during generation.",
    memoryNote:
      "The hybrid layout keeps cache growth lower than dense 32-layer models, but the extra multimodal resident weights raise the single-card floor.",
    overviewPoints: qwen35OverviewPoints(8, 32),
    researchHighlights: qwen35ResearchHighlights,
    memoryBehaviorPoints: qwen35MemoryBehaviorPoints(8, 32),
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(4_659_865_088, 5_000_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-4B",
        note: "The official Qwen3.5-4B safetensor weights total about 4.66 GB on Hugging Face, and Qwen documents explicit Transformers and vLLM guidance, including text-only serving in vLLM.",
      }),
    ],
  },
  {
    id: "qwen-3.5-9b",
    displayName: "Qwen 3.5 9B",
    family: "Qwen",
    organization: "Alibaba",
    architectureType: "Hybrid multimodal transformer",
    isMoe: false,
    totalParams: 10_000_000_000,
    numLayers: 32,
    hiddenSize: 4096,
    numAttentionHeads: 16,
    numKvHeads: 4,
    attentionHeadDim: 256,
    linearNumKeyHeads: 16,
    linearNumValueHeads: 32,
    linearKeyHeadDim: 128,
    linearValueHeadDim: 128,
    linearConvKernelDim: 4,
    linearStateBytesPerElement: 4,
    contextLength: 262_144,
    modality: "multimodal",
    cacheStrategy: "hybrid_attention",
    attentionLayerCount: 8,
    vocabSize: 248_320,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-9B",
    shortDescription:
      "Largest practical Qwen3.5 release for this batch, pairing a 9B language model with a resident multimodal stack that still targets single-GPU text serving.",
    researchHighlight:
      "Qwen’s own release frames the 9B model around five changes: unified multimodal training, hybrid DeltaNet-attention inference, scaled RL, broader language coverage, and a training stack built for multimodal efficiency.",
    memoryNote:
      "This estimate intentionally keeps the full multimodal checkpoint resident even for text-only use, so it is conservative relative to runtime-specific language-only shortcuts.",
    overviewPoints: qwen35OverviewPoints(8, 32),
    researchHighlights: qwen35ResearchHighlights,
    memoryBehaviorPoints: qwen35MemoryBehaviorPoints(8, 32),
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(9_653_104_368, 10_000_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-9B",
        note: "The official Qwen3.5-9B safetensor weights total about 9.65 GB on Hugging Face, and Qwen documents both Transformers and vLLM support.",
      }),
    ],
  },
  {
    id: "openreasoning-nemotron-1.5b",
    displayName: "OpenReasoning Nemotron 1.5B",
    family: "Nemotron",
    organization: "NVIDIA",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 1_540_000_000,
    numLayers: 28,
    hiddenSize: 1536,
    numAttentionHeads: 12,
    numKvHeads: 2,
    contextLength: 32_768,
    vocabSize: 151_936,
    license: "CC-BY-4.0 + Apache 2.0",
    sourceUrl: "https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B",
    shortDescription:
      "Small dense Nemotron reasoning model built on the Qwen2.5 1.5B geometry, aimed at strong math and code behavior on modest hardware.",
    researchHighlight:
      "NVIDIA presents OpenReasoning-Nemotron as a reasoning-tuned derivative of Qwen2.5 with strong math, code, and science performance, plus an explicit GenSelect heavy-inference path for combining multiple sampled solutions.",
    researchHighlights: nemotronResearchHighlights("1.5B"),
    memoryNote:
      "This behaves like a classic dense Qwen2.5-style checkpoint where resident weights dominate and KV cache follows the standard grouped-attention path.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(3_087_467_144, 1_540_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B",
        note: "The official OpenReasoning-Nemotron-1.5B safetensor weights total about 3.09 GB on Hugging Face, and NVIDIA ships it as a dense Qwen2.5-derived Transformers checkpoint.",
      }),
    ],
  },
  {
    id: "openreasoning-nemotron-7b",
    displayName: "OpenReasoning Nemotron 7B",
    family: "Nemotron",
    organization: "NVIDIA",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 7_610_000_000,
    numLayers: 28,
    hiddenSize: 3584,
    numAttentionHeads: 28,
    numKvHeads: 4,
    contextLength: 131_072,
    vocabSize: 152_064,
    license: "CC-BY-4.0 + Apache 2.0",
    sourceUrl: "https://huggingface.co/nvidia/OpenReasoning-Nemotron-7B",
    shortDescription:
      "Reasoning-tuned dense Nemotron checkpoint that tracks the familiar Qwen2.5 7B memory shape while targeting stronger math and code performance.",
    researchHighlight:
      "The 7B Nemotron release is positioned around benchmark-leading size-class reasoning results and optional GenSelect-style test-time scaling, while leaving the underlying Qwen2.5 memory geometry mostly unchanged.",
    researchHighlights: nemotronResearchHighlights("7B"),
    memoryNote:
      "Resident weights set the floor, and the grouped KV layout keeps long-context cache growth moderate relative to older full-head dense models.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(15_231_233_024, 7_610_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/nvidia/OpenReasoning-Nemotron-7B",
        note: "The official OpenReasoning-Nemotron-7B safetensor weights total about 15.23 GB on Hugging Face, and NVIDIA publishes it as a Qwen2.5-7B-derived dense Transformers checkpoint.",
      }),
    ],
  },
  {
    id: "openreasoning-nemotron-14b",
    displayName: "OpenReasoning Nemotron 14B",
    family: "Nemotron",
    organization: "NVIDIA",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 14_700_000_000,
    numLayers: 48,
    hiddenSize: 5120,
    numAttentionHeads: 40,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 152_064,
    license: "CC-BY-4.0 + Apache 2.0",
    sourceUrl: "https://huggingface.co/nvidia/OpenReasoning-Nemotron-14B",
    shortDescription:
      "Mid-sized dense Nemotron checkpoint for users who want stronger reasoning behavior than 7B without stepping straight into 32B deployment territory.",
    researchHighlight:
      "NVIDIA highlights the 14B model as one of the strongest models in its size class for reasoning benchmarks, with the main product change being post-training for long-form math, code, and science reasoning rather than a new backbone.",
    researchHighlights: nemotronResearchHighlights("14B"),
    memoryNote:
      "This is still a dense 14B-class checkpoint: weights dominate the fit decision, and context length becomes the next major lever after quantization.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(29_540_067_328, 14_700_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/nvidia/OpenReasoning-Nemotron-14B",
        note: "The official OpenReasoning-Nemotron-14B safetensor weights total about 29.54 GB on Hugging Face, and NVIDIA publishes it as a Qwen2.5-14B-derived dense Transformers checkpoint.",
      }),
    ],
  },
  {
    id: "openreasoning-nemotron-32b",
    displayName: "OpenReasoning Nemotron 32B",
    family: "Nemotron",
    organization: "NVIDIA",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 32_500_000_000,
    numLayers: 64,
    hiddenSize: 5120,
    numAttentionHeads: 40,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 152_064,
    license: "CC-BY-4.0 + Apache 2.0",
    sourceUrl: "https://huggingface.co/nvidia/OpenReasoning-Nemotron-32B",
    shortDescription:
      "Largest Nemotron checkpoint in this batch, intended as a serious reasoning model that still follows a plain dense Qwen2.5-style memory profile.",
    researchHighlight:
      "The 32B Nemotron model is the largest reasoning-tuned release in this family and is explicitly paired with GenSelect-style heavy inference, but it still rides on a dense Qwen2.5 backbone rather than introducing sparse routing.",
    researchHighlights: nemotronResearchHighlights("32B"),
    memoryNote:
      "Dense resident weights dominate immediately, so single-GPU deployment quickly becomes a quantization-and-runtime-budget problem rather than a cache problem.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(65_527_752_704, 32_500_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/nvidia/OpenReasoning-Nemotron-32B",
        note: "The official OpenReasoning-Nemotron-32B safetensor weights total about 65.53 GB on Hugging Face, and NVIDIA publishes it as a dense Qwen2.5-32B-derived Transformers checkpoint.",
      }),
    ],
  },
  {
    id: "gemma-2-9b",
    displayName: "Gemma 2 9B",
    family: "Gemma",
    organization: "Google",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 9_200_000_000,
    numLayers: 42,
    hiddenSize: 3584,
    numAttentionHeads: 16,
    numKvHeads: 8,
    contextLength: 8192,
    vocabSize: 256_000,
    license: "Gemma terms",
    sourceUrl: "https://huggingface.co/google/gemma-2-9b-it",
    shortDescription:
      "Instruction-tuned Gemma checkpoint with a relatively short native context window and efficient KV usage.",
    researchHighlight:
      "Gemma 2’s release emphasizes strong capability from comparatively compact dense models derived from Gemini-era research, with the family aimed at efficient open deployment rather than extreme context length.",
    researchHighlights: gemma2ResearchHighlights("9B"),
    memoryNote:
      "The shorter native context window keeps KV cache moderate, so the main memory driver is still the dense weight tensor.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: 2,
        sourceUrl: "https://huggingface.co/google/gemma-2-9b-it",
        note: "Google's official Gemma 2 9B Instruct release is exported in bfloat16.",
      }),
    ],
  },
  {
    id: "gemma-2-27b",
    displayName: "Gemma 2 27B",
    family: "Gemma",
    organization: "Google",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 27_000_000_000,
    numLayers: 46,
    hiddenSize: 4608,
    numAttentionHeads: 32,
    numKvHeads: 16,
    contextLength: 8192,
    vocabSize: 256_000,
    license: "Gemma terms",
    sourceUrl: "https://huggingface.co/google/gemma-2-27b-it",
    shortDescription:
      "Larger Gemma model that trades a shorter native context window for more capacity per token.",
    researchHighlight:
      "The 27B Gemma 2 model scales the same Gemma 2 recipe upward, prioritizing more capability per token within Google’s lightweight-open-model line instead of chasing sparse or ultra-long-context design.",
    researchHighlights: gemma2ResearchHighlights("27B"),
    memoryNote:
      "Because the context window is shorter, most VRAM pressure comes from resident weights rather than cache growth.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: 2,
        sourceUrl: "https://huggingface.co/google/gemma-2-27b-it",
        note: "Google's official Gemma 2 27B Instruct release is exported in bfloat16.",
      }),
    ],
  },
  {
    id: "mistral-nemo-12b",
    displayName: "Mistral Nemo 12B",
    family: "Mistral",
    organization: "Mistral AI",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 12_200_000_000,
    numLayers: 40,
    hiddenSize: 5120,
    numAttentionHeads: 32,
    numKvHeads: 8,
    contextLength: 128_000,
    vocabSize: 131_072,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
    shortDescription:
      "Long-context dense Mistral checkpoint that remains practical on a single 24 GB card with quantization.",
    researchHighlight:
      "Mistral Nemo is described as a 12B joint Mistral-NVIDIA release with 128K context, more multilingual and code-heavy training data, a new Tekken tokenizer, and a drop-in-replacement positioning versus Mistral 7B.",
    researchHighlights: mistralNemoResearchHighlights,
    memoryNote:
      "Dense weights set the baseline footprint; long-context use makes KV cache the next thing to watch after quantization.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointGb(24.5, 12_200_000_000),
        sourceUrl: "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
        note: "Mistral's official consolidated BF16 weights for Mistral Nemo are about 24.5 GB.",
      }),
      directProfile({
        id: "official-fp8",
        label: "Official FP8 checkpoint",
        effectiveDtype: "fp8",
        weightBytes: bytesFromCheckpointGb(13.6, 12_200_000_000),
        sourceUrl: "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-FP8-2407/tree/main",
        note: "Mistral's official FP8 checkpoint repository for Mistral Nemo is about 13.6 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "mixtral-8x7b",
    displayName: "Mixtral 8x7B",
    family: "Mixtral",
    organization: "Mistral AI",
    architectureType: "Mixture-of-experts transformer",
    isMoe: true,
    totalParams: 46_700_000_000,
    activeParams: 12_900_000_000,
    numLayers: 32,
    hiddenSize: 4096,
    numAttentionHeads: 32,
    numKvHeads: 8,
    contextLength: 32_768,
    vocabSize: 32_000,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
    shortDescription:
      "Sparse MoE model where runtime compute is closer to one expert pair, but VRAM still pays for resident weights.",
    researchHighlight:
      "Mixtral’s main change versus dense peers is sparse top-2 expert routing: the model behaves like a much smaller active network per token while still exposing a much larger total parameter pool for quality.",
    researchHighlights: mixtralResearchHighlights,
    memoryNote:
      "Even though only a subset of experts is active per token, single-GPU VRAM still carries the resident experts in memory.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: 2,
        sourceUrl: "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
        note: "Mistral's official Mixtral 8x7B release is a BF16 checkpoint.",
      }),
    ],
  },
  {
    id: "phi-4-14b",
    displayName: "Phi-4 14B",
    family: "Phi",
    organization: "Microsoft",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 14_700_000_000,
    numLayers: 40,
    hiddenSize: 5120,
    numAttentionHeads: 40,
    numKvHeads: 10,
    contextLength: 16_384,
    vocabSize: 100_352,
    license: "MIT",
    sourceUrl: "https://huggingface.co/microsoft/phi-4",
    shortDescription:
      "Reasoning-oriented dense Phi model with moderate context length and a straightforward single-GPU footprint.",
    researchHighlight:
      "Microsoft positions Phi-4 around high-quality synthetic and curated training data for math, coding, commonsense, and instruction-following, with the main story being unusually strong reasoning per parameter rather than a novel deployment architecture.",
    researchHighlights: phi4ResearchHighlights,
    memoryNote:
      "With a moderate context window, the model behaves like a classic dense checkpoint where weights dominate and cache stays secondary.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointGb(29.3, 14_700_000_000),
        sourceUrl: "https://huggingface.co/microsoft/phi-4/tree/main",
        note: "Microsoft's official phi-4 repository is about 29.3 GB on Hugging Face.",
      }),
      directProfile({
        id: "official-onnx-int4",
        label: "Official ONNX INT4 checkpoint",
        effectiveDtype: "int4",
        weightBytes: bytesFromCheckpointGb(8.99, 14_700_000_000),
        supportedRuntimes: ["transformers"],
        sourceUrl: "https://huggingface.co/microsoft/phi-4-onnx/tree/main/gpu/gpu-int4-rtn-block-32",
        note: "Microsoft's official phi-4 ONNX GPU INT4 checkpoint directory is about 8.99 GB on Hugging Face.",
      }),
    ],
  },
];
