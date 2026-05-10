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

const llama31TrainingReleaseContext = [
  point(
    "Family continuity",
    "Llama 3.1 stays in Meta's standard dense-transformer line rather than switching to sparse or hybrid architectures.",
  ),
  point(
    "Release packaging",
    "The instruct checkpoints are published as straightforward BF16 Hugging Face releases, which keeps the deployment story simple compared with mixed-format or multimodal families.",
  ),
  point(
    "Scaling path",
    "The family scales by model size and context length while keeping grouped-query attention as the main serving-side efficiency choice.",
  ),
];

const llama31Strengths = [
  point("General assistant use", "Strong baseline for broad instruction-following and assistant workflows."),
  point("Long context", "Native 128K context makes it practical for long retrieval and document-heavy prompts."),
  point("Tool-capable serving", "Meta explicitly positions the family for tool-using assistants and application backends."),
];

const qwen35TrainingReleaseContext = [
  point(
    "Unified release format",
    "Qwen3.5 is released as a single multimodal foundation rather than as separate text and vision checkpoints stitched together later.",
  ),
  point(
    "Architecture shift",
    "The family changes the serving geometry by mixing DeltaNet-style state layers with periodic attention layers instead of staying a plain dense-attention stack like Qwen2.5.",
  ),
  point(
    "Training stack",
    "Qwen emphasizes multimodal training efficiency and large-scale RL infrastructure as part of the release process, not just as a benchmark claim.",
  ),
];

const qwen35Strengths = [
  point("Multimodal reasoning", "Designed for a unified text-plus-vision capability profile rather than separate specialist variants."),
  point("Long-context serving", "The hybrid layout is explicitly aimed at making long-context serving cheaper than a dense full-attention stack."),
  point("Agents and coding", "Qwen positions the family as competitive across coding, reasoning, and agent-style workflows."),
];

const qwen3DenseTrainingReleaseContext = (sizeLabel: string, nativeContext: string) => [
  point(
    "Family release",
    "Qwen3 is released as a dense and MoE model family centered on switching between thinking and non-thinking modes within the same model.",
  ),
  point(
    "Training stage",
    "Qwen describes the release as a pretraining plus post-training model rather than a small instruction-only adaptation.",
  ),
  point(
    "Context packaging",
    `The ${sizeLabel} model is published with ${nativeContext} native context, and the larger dense variants explicitly extend to 131K with YaRN.`,
  ),
];

const qwen3DenseStrengths = (sizeLabel: string) => [
  point("Thinking and non-thinking use", `The ${sizeLabel} release is built to switch between deeper reasoning mode and faster general dialogue mode without changing models.`),
  point("Agent workflows", "Qwen positions the family for tool use and agent-style tasks in both thinking and non-thinking modes."),
  point("Multilingual assistant work", "The family is published with support for 100+ languages and dialects, making it a broad multilingual assistant line rather than a narrow specialist release."),
];

const qwen3MoeTrainingReleaseContext = [
  point(
    "MoE family branch",
    "Qwen3 includes dedicated MoE models alongside the dense line, keeping the same user-facing thinking/non-thinking framing while changing the serving geometry materially.",
  ),
  point(
    "Sparse activation",
    "The MoE releases expose total and activated parameter counts separately, which is the key deployment distinction versus the dense Qwen3 models.",
  ),
  point(
    "Long-context packaging",
    "The base MoE releases are published with 32K native context and 131K support with YaRN, while the 2507 update is packaged at 256K native context.",
  ),
];

const qwen3MoeStrengths = [
  point("Reasoning with lower active compute", "The MoE line is for users who want larger total capacity without paying dense-model active compute per token."),
  point("Agent and tool use", "Qwen still positions the MoE branch around agent workflows, tool calling, and mixed reasoning/general dialogue use."),
  point("Large multilingual serving", "Useful when you want very large-capacity multilingual serving without moving to a purely dense 70B+ model."),
];

const nemotronTrainingReleaseContext = [
  point(
    "Base-model inheritance",
    "OpenReasoning-Nemotron models are NVIDIA post-training releases built directly on top of Qwen2.5 dense backbones.",
  ),
  point(
    "Release method",
    "The family is released as a reasoning-tuned derivative line rather than as a new architecture family with different serving mechanics.",
  ),
  point(
    "Optional heavy mode",
    "NVIDIA pairs the base checkpoints with GenSelect-style multi-sample inference guidance, so part of the release story lives in inference strategy rather than in the resident model alone.",
  ),
];

const nemotronStrengths = [
  point("Math and science reasoning", "NVIDIA positions the family around benchmark-heavy reasoning workloads."),
  point("Code generation", "The release emphasizes code and solution-generation performance alongside math."),
  point("Test-time scaling", "GenSelect gives the family a clear path to higher-quality heavy inference when latency is less constrained."),
];

const gemma2TrainingReleaseContext = [
  point(
    "Open-model tier",
    "Gemma 2 sits in Google's smaller open model line rather than in the flagship Gemini product tier.",
  ),
  point(
    "Architecture continuity",
    "The family stays within a straightforward dense-transformer deployment pattern and does not depend on sparse or hybrid serving mechanics.",
  ),
  point(
    "Release packaging",
    "The instruction-tuned variants are packaged as practical deployment checkpoints rather than as research-preview artifacts.",
  ),
];

const gemma2Strengths = [
  point("Efficiency", "Strong quality-per-parameter for teams that want a smaller dense model footprint."),
  point("General language tasks", "Useful as a compact open baseline for assistant-style and retrieval-augmented applications."),
  point("Operational simplicity", "Straightforward dense checkpoints make the family easier to reason about than more exotic architectures."),
];

const mistralNemoTrainingReleaseContext = [
  point(
    "Joint release",
    "Mistral Nemo is a joint Mistral-NVIDIA release, which is part of why the launch emphasized production deployment characteristics.",
  ),
  point(
    "Tokenizer and context packaging",
    "The family introduces the Tekken tokenizer and a 128K context window as concrete release-level changes rather than as later add-ons.",
  ),
  point(
    "Backbone continuity",
    "The release keeps a normal dense-transformer deployment story, with the upgrade driven more by tokenizer, data mix, and context than by architecture novelty.",
  ),
];

const mistralNemoStrengths = [
  point("Long context", "Strong option for long prompts and document-heavy workflows in a dense 12B class."),
  point("Multilingual use", "The release explicitly leans into broader multilingual coverage than older smaller Mistral lines."),
  point("Code-heavy applications", "The training mix and positioning make it a common choice for code-oriented serving."),
];

const mixtralTrainingReleaseContext = [
  point(
    "Sparse release milestone",
    "Mixtral mattered as one of the first widely adopted open sparse MoE checkpoints to feel practical outside research demos.",
  ),
  point(
    "Architecture-first release",
    "The release is fundamentally about sparse routing, not about a new tokenizer, longer context, or multimodal packaging.",
  ),
  point(
    "Open deployment angle",
    "Mistral shipped it as an openly deployable sparse alternative for users who wanted higher resident capacity without dense-model compute scaling.",
  ),
];

const mixtralStrengths = [
  point("Capability per token", "Active compute stays much lower than total resident capacity, which is the main reason people reach for Mixtral."),
  point("Instruction use", "The instruct release is a strong open general assistant baseline when VRAM is available."),
  point("MoE experimentation", "Useful for teams exploring sparse routing behavior without moving to frontier closed models."),
];

const phi4TrainingReleaseContext = [
  point(
    "Data-centric release",
    "Phi-4 is framed heavily around its synthetic and curated training recipe rather than around a radical architecture change.",
  ),
  point(
    "Architecture continuity",
    "The family stays close to a conventional dense-transformer deployment story rather than introducing sparse or hybrid serving behavior.",
  ),
  point(
    "Packaging path",
    "Microsoft complements the BF16 release with an official ONNX INT4 path, so lower-VRAM deployment is part of the release packaging itself.",
  ),
];

const phi4Strengths = [
  point("Reasoning density", "Strong per-parameter reasoning is the main reason to consider Phi-4."),
  point("Coding and math", "The release is consistently framed around quantitative and code-heavy capability."),
  point("Smaller deployment footprint", "Useful when teams want a serious reasoning model without stepping into 30B+ VRAM territory."),
];

export const models: ModelSpec[] = [
  {
    id: "gpt-oss-20b",
    displayName: "GPT-OSS 20B",
    family: "GPT-OSS",
    organization: "OpenAI",
    releaseDate: "2025-08-04",
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
      "Smaller GPT-OSS release for general-purpose and reasoning use cases that need to stay within a much lighter single-card memory budget.",
    researchHighlight:
      "OpenAI positions gpt-oss-20b as the smaller GPT-OSS model for general-purpose and reasoning workloads that need a much lower memory floor than the 120B release.",
    researchHighlights: [
      point(
        "Low-memory GPT-OSS entry point",
        "The main release-level change is that GPT-OSS capability becomes practical in roughly 16 GB of memory rather than requiring an 80 GB class accelerator.",
      ),
      point(
        "Configurable reasoning effort",
        "Like the larger model, gpt-oss-20b supports low, medium, and high reasoning effort settings so latency and reasoning depth can be traded off per use case.",
      ),
      point(
        "Native agent features",
        "The smaller release still keeps the same first-class agent surface: function calling, web browsing, Python execution, and structured outputs.",
      ),
      point(
        "Full chain-of-thought access",
        "OpenAI exposes the reasoning trace for debugging and trust, even though it is not intended for direct end-user display.",
      ),
    ],
    trainingReleaseContext: [
      point(
        "Harmony-only format",
        "Both GPT-OSS models were trained on OpenAI's Harmony response format and are expected to be used with that format rather than a generic chat template.",
      ),
      point(
        "Model geometry",
        "gpt-oss-20b uses 24 layers, 21B total parameters, 3.6B active parameters per token, 32 total experts, 4 active experts per token, and a 128K context window.",
      ),
      point(
        "Quantized MoE release",
        "The MoE weights were post-trained in MXFP4, which is the release decision that makes the smaller checkpoint practical in roughly 16 GB of memory.",
      ),
      point(
        "Training data and tokenizer",
        "OpenAI describes the training mix as mostly English, text-only data with emphasis on STEM, coding, and general knowledge, tokenized with the open-sourced o200k_harmony tokenizer.",
      ),
    ],
    strengths: [
      point(
        "Smaller-memory deployment",
        "Best fit when you want GPT-OSS reasoning and agent behavior without stepping into 80 GB class hardware first.",
      ),
      point(
        "General-purpose assistant work",
        "Designed as a broad open assistant and reasoning model rather than a narrow specialist checkpoint.",
      ),
      point(
        "Fine-tuning and customization",
        "OpenAI positions the model as fine-tunable, which makes it useful when a smaller open reasoning model needs to be adapted to a specific task.",
      ),
      point(
        "Commercial deployment",
        "The Apache 2.0 license keeps experimentation and product deployment straightforward for teams that want permissive usage terms.",
      ),
    ],
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
    releaseDate: "2025-08-04",
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
      "Production GPT-OSS release for general-purpose and higher-reasoning workloads that can fit on a single 80 GB class GPU.",
    researchHighlight:
      "OpenAI positions gpt-oss-120b as the production GPT-OSS model for general-purpose and high-reasoning use cases that fit on a single 80 GB accelerator.",
    researchHighlights: [
      point(
        "Single-80GB target",
        "The release is explicitly positioned around fitting production-grade reasoning into one 80 GB class GPU such as an H100 or MI300X, which is the main operational change versus a typical 100B+ open model.",
      ),
      point(
        "Configurable reasoning effort",
        "OpenAI exposes low, medium, and high reasoning effort settings so latency and reasoning depth can be traded off at inference time instead of using one fixed behavior.",
      ),
      point(
        "Native agent features",
        "The model is released with native support for function calling, web browsing, Python execution, and structured outputs rather than treating those as wrapper-level add-ons.",
      ),
      point(
        "Full chain-of-thought access",
        "The release provides access to the model's reasoning trace for debugging and auditability, though OpenAI notes it is not intended for direct end-user display.",
      ),
    ],
    trainingReleaseContext: [
      point(
        "Harmony-only format",
        "Both GPT-OSS models were trained on OpenAI's Harmony response format and are expected to be used with that format rather than a generic chat template.",
      ),
      point(
        "Model geometry",
        "gpt-oss-120b uses 36 layers, 117B total parameters, 5.1B active parameters per token, 128 total experts, 4 active experts per token, and a 128K context window.",
      ),
      point(
        "Quantized MoE release",
        "The MoE weights were post-trained in MXFP4, which is the packaging decision that makes the 120B checkpoint practical on a single 80 GB GPU.",
      ),
      point(
        "Training data and tokenizer",
        "OpenAI describes the training mix as mostly English, text-only data with emphasis on STEM, coding, and general knowledge, tokenized with the open-sourced o200k_harmony tokenizer.",
      ),
    ],
    strengths: [
      point(
        "Production general-purpose serving",
        "Best fit when you want one open model that can cover broad assistant, coding, and reasoning workloads without moving to multi-GPU serving first.",
      ),
      point(
        "High-reasoning workloads",
        "Strong match for use cases that benefit from controllable deeper reasoning rather than the fastest possible low-latency answers.",
      ),
      point(
        "Fine-tuning and customization",
        "OpenAI explicitly positions the model as fine-tunable, which matters if you want to adapt one large reasoning-capable checkpoint to a narrower production task.",
      ),
      point(
        "Commercial deployment",
        "The Apache 2.0 license makes it unusually straightforward to experiment, customize, and deploy commercially without copyleft friction.",
      ),
    ],
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
    releaseDate: "2024-07-18",
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
    trainingReleaseContext: llama31TrainingReleaseContext,
    strengths: llama31Strengths,
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
    releaseDate: "2024-07-16",
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
    trainingReleaseContext: llama31TrainingReleaseContext,
    strengths: llama31Strengths,
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
    id: "qwen-2.5-0.5b",
    displayName: "Qwen 2.5 0.5B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2024-09-16",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 490_000_000,
    numLayers: 24,
    hiddenSize: 896,
    numAttentionHeads: 14,
    numKvHeads: 2,
    contextLength: 32_768,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
    shortDescription:
      "Instruction-tuned 0.5B Qwen2.5 model for lightweight assistant, structured-output, and long-prompt use in very small dense deployments.",
    researchHighlight:
      "The 0.5B Qwen2.5 release brings the same family improvements in coding, mathematics, instruction following, structured outputs, and multilingual coverage down to a much smaller footprint.",
    researchHighlights: [
      point(
        "Smallest Qwen2.5 instruct entry",
        "The main product change here is accessibility: the Qwen2.5 capability set is pushed into a model small enough for much lighter local and edge-style deployments.",
      ),
      point(
        "Structured-output focus",
        "Even at 0.5B, Qwen still emphasizes stronger JSON and structured-data behavior, which matters more practically than raw benchmark scale at this size.",
      ),
      point(
        "Long-prompt support",
        "The model keeps a 32K context window, which is notable for a checkpoint this small and makes it more useful than a short-context miniature model.",
      ),
    ],
    trainingReleaseContext: [
      point(
        "Family release",
        "Qwen2.5 was released as a broad language-model line spanning base and instruction-tuned checkpoints from 0.5B to 72B parameters.",
      ),
      point(
        "Model architecture",
        "The 0.5B instruct model is a causal language model built as a dense transformer with RoPE, SwiGLU, RMSNorm, attention QKV bias, and tied word embeddings.",
      ),
      point(
        "0.5B model geometry",
        "The checkpoint has 0.49B total parameters, 0.36B non-embedding parameters, 24 layers, 14 query heads, 2 KV heads, a 32,768-token context window, and up to 8,192 generated tokens.",
      ),
      point(
        "Training stage",
        "Qwen describes the release as a pretraining plus post-training model rather than a tiny instruction-only adaptation on top of an older base.",
      ),
    ],
    strengths: [
      point(
        "Very small deployments",
        "Best fit when VRAM or latency budgets are tight and you still want a modern instruction-tuned open model with structured-output support.",
      ),
      point(
        "Structured outputs",
        "Useful for lightweight JSON, extraction, and formatting tasks where a small but instruction-aligned model is enough.",
      ),
      point(
        "Long prompts on small hardware",
        "The 32K context window makes it more practical for retrieval-heavy or prompt-heavy tasks than many other tiny open checkpoints.",
      ),
    ],
    memoryNote:
      "At this size the resident weight floor is low, so long context and runtime overhead start to matter proportionally more than on larger dense models.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(494_032_768, 490_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
        note: "The official Qwen2.5-0.5B-Instruct safetensor weights total about 0.49 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-2.5-1.5b",
    displayName: "Qwen 2.5 1.5B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2024-09-17",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 1_540_000_000,
    numLayers: 28,
    hiddenSize: 1536,
    numAttentionHeads: 12,
    numKvHeads: 2,
    contextLength: 32_768,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
    shortDescription:
      "Instruction-tuned 1.5B Qwen2.5 model for lightweight coding, math, structured-output, and assistant tasks in a small dense deployment footprint.",
    researchHighlight:
      "The 1.5B Qwen2.5 release keeps the same family improvements as the larger models while staying in a much smaller deployment class than 7B and above.",
    researchHighlights: [
      point(
        "Small but more capable than 0.5B",
        "The 1.5B model is the first Qwen2.5 size where users often expect a more useful general assistant while still staying in a very lightweight VRAM class.",
      ),
      point(
        "Coding and math uplift",
        "Qwen still frames the family around stronger coding and mathematics than Qwen2, which matters more here because 1.5B is often used as a practical small local model.",
      ),
      point(
        "Structured-output support",
        "JSON and structured-data handling remain part of the product story rather than being reserved only for the larger checkpoints.",
      ),
    ],
    trainingReleaseContext: [
      point(
        "Family release",
        "Qwen2.5 was released as a broad language-model line spanning base and instruction-tuned checkpoints from 0.5B to 72B parameters.",
      ),
      point(
        "Model architecture",
        "The 1.5B instruct model is a causal language model built as a dense transformer with RoPE, SwiGLU, RMSNorm, attention QKV bias, and tied word embeddings.",
      ),
      point(
        "1.5B model geometry",
        "The checkpoint has 1.54B total parameters, 1.31B non-embedding parameters, 28 layers, 12 query heads, 2 KV heads, a 32,768-token context window, and up to 8,192 generated tokens.",
      ),
      point(
        "Training stage",
        "Qwen describes the release as a pretraining plus post-training model rather than a small instruction-only adaptation on top of an older base.",
      ),
    ],
    strengths: [
      point(
        "Small general assistant use",
        "Useful when you want a more capable lightweight assistant model than 0.5B without moving all the way to 7B-class memory costs.",
      ),
      point(
        "Structured outputs",
        "A reasonable fit for lighter JSON, extraction, and formatting workflows on small hardware.",
      ),
      point(
        "Small coding and math tasks",
        "Good for modest technical and code-oriented tasks when a very small open model is required.",
      ),
    ],
    memoryNote:
      "Resident weights are still modest at this size, so long context and runtime overhead matter proportionally more than on mid-size dense checkpoints.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(1_543_714_304, 1_540_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
        note: "The official Qwen2.5-1.5B-Instruct safetensor weights total about 1.54 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-2.5-3b",
    displayName: "Qwen 2.5 3B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2024-09-17",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 3_090_000_000,
    numLayers: 36,
    hiddenSize: 2048,
    numAttentionHeads: 16,
    numKvHeads: 2,
    contextLength: 32_768,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
    shortDescription:
      "Instruction-tuned 3B Qwen2.5 model for stronger small-model coding, math, structured-output, and assistant use in a compact dense footprint.",
    researchHighlight:
      "The 3B Qwen2.5 release sits between the very small models and the 7B class, giving users more dense capacity while still staying in a relatively compact deployment range.",
    researchHighlights: [
      point(
        "Middle of the small-model ladder",
        "The 3B model is the practical bridge between miniature Qwen2.5 checkpoints and the more capable 7B class.",
      ),
      point(
        "Coding and math uplift",
        "Qwen's family-wide improvements in code and mathematics become more useful here because 3B often represents a realistic balance between capability and footprint.",
      ),
      point(
        "Structured-output support",
        "The release still emphasizes tables, JSON, and structured-data handling, which helps the 3B model stay useful for application workflows beyond plain chat.",
      ),
    ],
    trainingReleaseContext: [
      point(
        "Family release",
        "Qwen2.5 was released as a broad language-model line spanning base and instruction-tuned checkpoints from 0.5B to 72B parameters.",
      ),
      point(
        "Model architecture",
        "The 3B instruct model is a causal language model built as a dense transformer with RoPE, SwiGLU, RMSNorm, attention QKV bias, and tied word embeddings.",
      ),
      point(
        "3B model geometry",
        "The checkpoint has 3.09B total parameters, 2.77B non-embedding parameters, 36 layers, 16 query heads, 2 KV heads, a 32,768-token context window, and up to 8,192 generated tokens.",
      ),
      point(
        "Training stage",
        "Qwen describes the release as a pretraining plus post-training model rather than a small instruction-only adaptation on top of an older base.",
      ),
    ],
    strengths: [
      point(
        "Small-model capability balance",
        "Useful when 0.5B or 1.5B are too small, but you still want to stay below the heavier 7B-class deployment footprint.",
      ),
      point(
        "Coding and structured tasks",
        "A practical choice for smaller code, extraction, JSON, and tool-oriented workflows on limited hardware.",
      ),
      point(
        "Long prompts on compact hardware",
        "The 32K context window keeps it viable for retrieval-augmented or prompt-heavy tasks while still staying small.",
      ),
    ],
    memoryNote:
      "At 3B, resident weights matter more than on the tiny checkpoints, but the model is still small enough that context and runtime reserve remain visible parts of the total.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(3_085_938_688, 3_090_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",
        note: "The official Qwen2.5-3B-Instruct safetensor weights total about 3.09 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-2.5-7b",
    displayName: "Qwen 2.5 7B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2024-09-16",
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
      "Instruction-tuned 7B Qwen2.5 model for long-context, coding, math, and structured-output workloads in a straightforward dense deployment shape.",
    researchHighlight:
      "Qwen2.5 improves on Qwen2 with stronger knowledge, coding, mathematics, instruction following, long-text generation, structured-data understanding, and JSON-style structured outputs.",
    researchHighlights: [
      point(
        "Stronger coding and mathematics",
        "Qwen highlights a large jump in coding and math capability over Qwen2, driven by stronger domain-specialized training within the family.",
      ),
      point(
        "Better instruction following",
        "The instruction-tuned 7B release is explicitly framed as more reliable on role setting, system prompts, and condition-following than the earlier line.",
      ),
      point(
        "Structured-output reliability",
        "Qwen calls out better handling of tables, structured data, and especially JSON generation, which is one of the main practical reasons people use the family.",
      ),
      point(
        "Long-context release",
        "The model supports 128K context and is described as significantly stronger at generating long outputs over 8K tokens than the previous generation.",
      ),
    ],
    trainingReleaseContext: [
      point(
        "Family release",
        "Qwen2.5 was released as a broad language-model line spanning base and instruction-tuned checkpoints from 0.5B to 72B parameters.",
      ),
      point(
        "Model architecture",
        "The 7B instruct model is a causal language model built as a dense transformer with RoPE, SwiGLU, RMSNorm, and QKV bias in attention.",
      ),
      point(
        "7B model geometry",
        "The checkpoint has 7.61B total parameters, 6.53B non-embedding parameters, 28 layers, 28 query heads, 4 KV heads, a 131,072-token context window, and up to 8,192 generated tokens.",
      ),
      point(
        "Multilingual scope",
        "Qwen describes the family as supporting more than 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, and Arabic.",
      ),
    ],
    strengths: [
      point(
        "Coding and math",
        "Strong fit for code generation, technical problem solving, and quantitative reasoning relative to many other dense 7B-class open models.",
      ),
      point(
        "Structured outputs",
        "Especially useful when you want reliable JSON generation, table understanding, or other structured-response behavior.",
      ),
      point(
        "Long prompts and long outputs",
        "A good choice for long-context prompting and longer-form generations because the model is explicitly tuned for 128K context and 8K-token generation.",
      ),
      point(
        "Multilingual assistant use",
        "The release is positioned as a broad multilingual assistant model rather than a narrow English-only specialist checkpoint.",
      ),
    ],
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
    releaseDate: "2024-09-16",
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
      "Instruction-tuned 14B Qwen2.5 model for long-context, coding, math, and structured-output workloads with a larger dense capacity than the 7B release.",
    researchHighlight:
      "The 14B Qwen2.5 release carries the same Qwen2.5 family improvements at a larger scale, with stronger capacity for coding, mathematics, structured outputs, and long-context assistant behavior.",
    researchHighlights: [
      point(
        "Larger-capacity Qwen2.5",
        "The 14B model keeps the same Qwen2.5 capability upgrades as the 7B release but adds more dense capacity for users who want a stronger open general-purpose model without jumping to 32B.",
      ),
      point(
        "Coding and mathematics focus",
        "Qwen explicitly frames the family as significantly stronger than Qwen2 on coding and mathematics, and the 14B size is one of the practical points where that extra capacity becomes more visible.",
      ),
      point(
        "Structured-output reliability",
        "The release still emphasizes stronger handling of tables, structured data, and JSON-style outputs, which is one of the main deployment reasons to choose Qwen2.5.",
      ),
      point(
        "Long-context assistant behavior",
        "The model keeps the same 128K context and 8K generation framing, so it is still positioned for longer prompts and longer-form completions rather than short-context chat only.",
      ),
    ],
    trainingReleaseContext: [
      point(
        "Family release",
        "Qwen2.5 was released as a broad language-model line spanning base and instruction-tuned checkpoints from 0.5B to 72B parameters.",
      ),
      point(
        "Model architecture",
        "The 14B instruct model is a causal language model built as a dense transformer with RoPE, SwiGLU, RMSNorm, and QKV bias in attention.",
      ),
      point(
        "14B model geometry",
        "The checkpoint has 14.7B total parameters, 13.1B non-embedding parameters, 48 layers, 40 query heads, 8 KV heads, a 131,072-token context window, and up to 8,192 generated tokens.",
      ),
      point(
        "Training stage",
        "Qwen describes the release as a pretraining plus post-training model rather than a small instruction-only adaptation on top of an older base.",
      ),
    ],
    strengths: [
      point(
        "Coding and math",
        "Better fit than 7B-class checkpoints when you want more headroom on code generation, technical tasks, and quantitative reasoning while staying below 32B.",
      ),
      point(
        "Structured outputs",
        "Still a strong option for JSON, table understanding, and other structured-response workflows where Qwen2.5 is commonly used.",
      ),
      point(
        "Long prompts and long outputs",
        "The 128K context and 8K generation framing make it practical for long retrieval contexts, summarization, and document-heavy assistant tasks.",
      ),
      point(
        "General multilingual assistant use",
        "Useful when you want a broader multilingual dense model with more capacity than 7B without taking on the much larger resident cost of 32B-class models.",
      ),
    ],
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
    releaseDate: "2024-09-17",
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
      "Instruction-tuned 32B Qwen2.5 model for higher-capacity long-context, coding, math, and structured-output workloads in a large dense deployment shape.",
    researchHighlight:
      "At 32B, Qwen2.5 keeps the same family improvements as the smaller releases but adds substantially more dense capacity for stronger coding, mathematics, structured outputs, and long-context assistant use.",
    researchHighlights: [
      point(
        "Largest dense Qwen2.5 release in this slice",
        "The 32B model is the larger dense step-up from 7B and 14B for users who want more capability without moving into hybrid or sparse architectures.",
      ),
      point(
        "Coding and mathematics headroom",
        "Qwen presents the family as significantly stronger than Qwen2 on coding and mathematics, and 32B is the size where that extra dense capacity becomes a more explicit product choice.",
      ),
      point(
        "Structured-output reliability",
        "The release still leans heavily on stronger structured data understanding and JSON-style output generation, which remains one of the family's most practical deployment advantages.",
      ),
      point(
        "Long-context assistant behavior",
        "The model keeps the same 128K context and 8K generation framing, so it remains aimed at longer prompts and longer-form completions rather than short-context-only chat.",
      ),
    ],
    trainingReleaseContext: [
      point(
        "Family release",
        "Qwen2.5 was released as a broad language-model line spanning base and instruction-tuned checkpoints from 0.5B to 72B parameters.",
      ),
      point(
        "Model architecture",
        "The 32B instruct model is a causal language model built as a dense transformer with RoPE, SwiGLU, RMSNorm, and QKV bias in attention.",
      ),
      point(
        "32B model geometry",
        "The checkpoint has 32.5B total parameters, 31.0B non-embedding parameters, 64 layers, 40 query heads, 8 KV heads, a 131,072-token context window, and up to 8,192 generated tokens.",
      ),
      point(
        "Training stage",
        "Qwen describes the release as a pretraining plus post-training model rather than a small instruction-only adaptation on top of an older base.",
      ),
    ],
    strengths: [
      point(
        "Higher-capacity coding and math",
        "Best fit when 7B or 14B class models are not enough for code generation, technical tasks, or more demanding quantitative reasoning.",
      ),
      point(
        "Structured outputs at larger scale",
        "Still a strong option for JSON, table understanding, and other structured-response workflows, but with more dense capacity behind those tasks.",
      ),
      point(
        "Long prompts and long outputs",
        "The 128K context and 8K generation framing keep it practical for long retrieval contexts, summarization, and document-heavy assistant tasks.",
      ),
      point(
        "Large multilingual assistant baseline",
        "Useful when you want a broad multilingual dense model with materially more capability than 7B or 14B, and you are willing to pay the larger resident-weight cost.",
      ),
    ],
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
    id: "qwen-2.5-72b",
    displayName: "Qwen 2.5 72B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2024-09-16",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 72_700_000_000,
    numLayers: 80,
    hiddenSize: 8192,
    numAttentionHeads: 64,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 152_064,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
    shortDescription:
      "Instruction-tuned 72B Qwen2.5 model for the highest-capacity dense Qwen2.5 long-context, coding, math, and structured-output workloads.",
    researchHighlight:
      "The 72B Qwen2.5 release is the largest dense model in the family, carrying the same Qwen2.5 improvements into a much larger-capacity deployment target.",
    researchHighlights: [
      point(
        "Largest dense Qwen2.5 release",
        "The 72B model is the top dense-capacity endpoint of the Qwen2.5 line, for users who want the strongest version of the same core family improvements.",
      ),
      point(
        "Coding and mathematics at larger scale",
        "Qwen frames the whole family as improved on coding and math over Qwen2, and 72B is the point where that extra dense capacity becomes a major product decision rather than a small step-up.",
      ),
      point(
        "Structured-output reliability",
        "The release still emphasizes structured-data understanding and JSON generation, but now at a scale more likely to be used in serious production-quality assistant and workflow systems.",
      ),
      point(
        "Long-context dense alternative",
        "The model keeps the same 128K context and 8K generation framing while remaining a plain dense transformer instead of moving to sparse or hybrid serving geometry.",
      ),
    ],
    trainingReleaseContext: [
      point(
        "Family release",
        "Qwen2.5 was released as a broad language-model line spanning base and instruction-tuned checkpoints from 0.5B to 72B parameters.",
      ),
      point(
        "Model architecture",
        "The 72B instruct model is a causal language model built as a dense transformer with RoPE, SwiGLU, RMSNorm, and attention QKV bias.",
      ),
      point(
        "72B model geometry",
        "The checkpoint has 72.7B total parameters, 70.0B non-embedding parameters, 80 layers, 64 query heads, 8 KV heads, a 131,072-token context window, and up to 8,192 generated tokens.",
      ),
      point(
        "Training stage",
        "Qwen describes the release as a pretraining plus post-training model rather than a small instruction-only adaptation on top of an older base.",
      ),
    ],
    strengths: [
      point(
        "Highest-capacity dense Qwen2.5 use",
        "Best fit when smaller Qwen2.5 checkpoints are not enough and you want the strongest dense version of the family for coding, reasoning, and assistant work.",
      ),
      point(
        "Large-scale structured-output systems",
        "Useful for high-quality JSON, table, and structured-response workflows when model capacity matters more than keeping the deployment footprint small.",
      ),
      point(
        "Long-context assistant backends",
        "The 128K context window keeps it practical for document-heavy and retrieval-heavy assistant systems, assuming the larger resident footprint is acceptable.",
      ),
      point(
        "Broad multilingual dense serving",
        "A strong choice when you want a large multilingual dense model without moving into MoE or hybrid-architecture tradeoffs.",
      ),
    ],
    memoryNote:
      "At 72B, the resident dense weight floor dominates immediately, so runtime choice and quantization become the main levers once context is already long.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(72_706_203_648, 72_700_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
        note: "The official Qwen2.5-72B-Instruct safetensor weights total about 72.71 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-0.6b",
    displayName: "Qwen 3 0.6B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-04-27",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 600_000_000,
    numLayers: 28,
    hiddenSize: 1024,
    numAttentionHeads: 16,
    numKvHeads: 8,
    contextLength: 32_768,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-0.6B",
    shortDescription:
      "Smallest dense Qwen3 release with switchable thinking and non-thinking modes in a very light deployment footprint.",
    researchHighlight:
      "Qwen3 introduces one of the family’s main product changes at every size: the same model can switch between thinking mode for harder reasoning and a faster non-thinking mode for general dialogue.",
    researchHighlights: [
      point("Thinking-mode switch", "Qwen3’s defining change is seamless switching between deeper reasoning and faster non-thinking dialogue within the same checkpoint."),
      point("Reasoning uplift", "Qwen positions the line as stronger than QwQ in thinking mode and stronger than Qwen2.5 instruct models in non-thinking mode on reasoning-heavy tasks."),
      point("Agent and multilingual focus", "The release also emphasizes stronger agent use and support for 100+ languages and dialects, even at smaller sizes."),
    ],
    trainingReleaseContext: qwen3DenseTrainingReleaseContext("0.6B", "32K"),
    strengths: qwen3DenseStrengths("0.6B"),
    memoryNote:
      "At this size the resident weight floor stays small, so runtime reserve and long-context cache can become a larger fraction of total VRAM than on bigger dense models.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(751_632_384, 600_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-0.6B",
        note: "The official Qwen3-0.6B safetensor weights total about 0.75 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-1.7b",
    displayName: "Qwen 3 1.7B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-04-27",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 1_700_000_000,
    numLayers: 28,
    hiddenSize: 2048,
    numAttentionHeads: 16,
    numKvHeads: 8,
    contextLength: 32_768,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-1.7B",
    shortDescription:
      "Small dense Qwen3 release for lightweight reasoning, agent, and multilingual assistant use with switchable thinking modes.",
    researchHighlight:
      "The 1.7B release keeps the core Qwen3 identity: one model can move between deeper reasoning mode and faster general dialogue mode depending on the prompt and latency target.",
    researchHighlights: [
      point("Thinking-mode switch", "The same checkpoint can explicitly shift between reasoning-heavy and non-thinking behavior instead of forcing one fixed inference style."),
      point("Reasoning and instruction uplift", "Qwen presents the family as stronger than prior QwQ and Qwen2.5 instruct baselines across reasoning, coding, and instruction following."),
      point("Agent-ready small model", "Even the smaller dense releases are positioned for tool use and multilingual assistant workflows rather than only basic chat."),
    ],
    trainingReleaseContext: qwen3DenseTrainingReleaseContext("1.7B", "32K"),
    strengths: qwen3DenseStrengths("1.7B"),
    memoryNote:
      "Resident weights are still modest at 1.7B, so the model stays easy to fit; context growth and runtime reserve matter proportionally more than on larger dense Qwen3 checkpoints.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(2_031_739_904, 1_700_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-1.7B",
        note: "The official Qwen3-1.7B safetensor weights total about 2.03 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-4b",
    displayName: "Qwen 3 4B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-04-27",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 4_000_000_000,
    numLayers: 36,
    hiddenSize: 2560,
    numAttentionHeads: 32,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-4B",
    shortDescription:
      "Dense Qwen3 release with switchable thinking modes, stronger reasoning, and 131K extended-context support through YaRN.",
    researchHighlight:
      "The 4B release is where Qwen3 starts to look like a serious reasoning and agent model while still staying in a manageable dense single-node deployment class.",
    researchHighlights: [
      point("Thinking-mode switch", "The 4B model keeps the family’s dual-mode design, letting one checkpoint cover deeper reasoning and faster everyday dialogue."),
      point("Reasoning and agent uplift", "Qwen emphasizes stronger reasoning, tool use, and agent-task performance over earlier Qwen2.5 instruct models."),
      point("Extended context with YaRN", "The model is published with 32K native context and 131K support through YaRN, which is part of its practical release story."),
    ],
    trainingReleaseContext: qwen3DenseTrainingReleaseContext("4B", "32K"),
    strengths: qwen3DenseStrengths("4B"),
    memoryNote:
      "This is still a dense model, so weights set the floor; the main extra lever is that longer contexts can extend to 131K when YaRN-style serving is used.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(8_044_936_192, 4_000_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-4B",
        note: "The official Qwen3-4B safetensor weights total about 8.04 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-8b",
    displayName: "Qwen 3 8B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-04-27",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 8_200_000_000,
    numLayers: 36,
    hiddenSize: 4096,
    numAttentionHeads: 32,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-8B",
    shortDescription:
      "Dense Qwen3 release for stronger general-purpose reasoning, agent, and multilingual assistant use with switchable thinking modes.",
    researchHighlight:
      "The 8B release is one of the main practical dense Qwen3 checkpoints: large enough to matter for reasoning and agent tasks, but still far easier to deploy than the biggest dense or MoE variants.",
    researchHighlights: [
      point("Thinking-mode switch", "The 8B model preserves Qwen3’s ability to move between deeper reasoning mode and faster non-thinking dialogue."),
      point("Reasoning and instruction uplift", "Qwen positions the line as stronger than earlier Qwen2.5 instruct releases across mathematics, code, commonsense reasoning, and instruction following."),
      point("Extended context with YaRN", "The checkpoint keeps 32K native context and extends to 131K with YaRN, which matters for long-prompt deployment planning."),
    ],
    trainingReleaseContext: qwen3DenseTrainingReleaseContext("8B", "32K"),
    strengths: qwen3DenseStrengths("8B"),
    memoryNote:
      "Weights dominate the dense 8B footprint, but the model still stays manageable enough that runtime choice and context length both visibly affect total VRAM.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(8_190_735_360, 8_200_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-8B",
        note: "The official Qwen3-8B safetensor weights total about 8.19 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-14b",
    displayName: "Qwen 3 14B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-04-27",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 14_800_000_000,
    numLayers: 40,
    hiddenSize: 5120,
    numAttentionHeads: 40,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-14B",
    shortDescription:
      "Dense Qwen3 release for higher-capacity reasoning, agent, and multilingual assistant workloads with switchable thinking modes.",
    researchHighlight:
      "The 14B release keeps the same Qwen3 product story but moves it into a more capable dense serving tier for users who want more headroom than 8B without jumping to 32B or MoE.",
    researchHighlights: [
      point("Thinking-mode switch", "The checkpoint keeps one-model switching between deeper reasoning and faster general-purpose dialogue."),
      point("Reasoning and alignment uplift", "Qwen emphasizes stronger reasoning, instruction following, role-play, creative writing, and human preference alignment than earlier generations."),
      point("Extended context with YaRN", "The 14B release keeps the 32K native and 131K-with-YaRN context framing of the larger dense Qwen3 line."),
    ],
    trainingReleaseContext: qwen3DenseTrainingReleaseContext("14B", "32K"),
    strengths: qwen3DenseStrengths("14B"),
    memoryNote:
      "At 14B, resident weights dominate the floor more clearly, while long-context serving still depends on runtime reserve and whether the extended YaRN window is used.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(29_536_614_400, 14_800_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-14B",
        note: "The official Qwen3-14B safetensor weights total about 29.54 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-32b",
    displayName: "Qwen 3 32B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-04-27",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 32_800_000_000,
    numLayers: 64,
    hiddenSize: 5120,
    numAttentionHeads: 64,
    numKvHeads: 8,
    contextLength: 131_072,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-32B",
    shortDescription:
      "Largest dense Qwen3 release for high-capacity reasoning, agent, and multilingual assistant workloads with switchable thinking modes.",
    researchHighlight:
      "The 32B release is the largest dense Qwen3 checkpoint before moving into the MoE branch, giving the family’s reasoning and agent improvements much more dense capacity behind them.",
    researchHighlights: [
      point("Thinking-mode switch", "The model keeps the same dual-mode reasoning and dialogue story, but at a much larger dense scale."),
      point("Reasoning, alignment, and agents", "Qwen presents the family as stronger on reasoning, instruction following, creative tasks, and agent workflows than prior Qwen lines."),
      point("Extended context with YaRN", "The 32B release keeps 32K native context and 131K support with YaRN, which matters because long-context growth becomes more expensive at this size."),
    ],
    trainingReleaseContext: qwen3DenseTrainingReleaseContext("32B", "32K"),
    strengths: qwen3DenseStrengths("32B"),
    memoryNote:
      "At 32B, the dense resident-weight floor dominates quickly, so quantization and runtime reserve become central once you move beyond short contexts.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(65_524_246_528, 32_800_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-32B",
        note: "The official Qwen3-32B safetensor weights total about 65.52 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-30b-a3b",
    displayName: "Qwen 3 30B A3B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-04-27",
    architectureType: "Mixture-of-experts transformer",
    isMoe: true,
    totalParams: 30_500_000_000,
    activeParams: 3_300_000_000,
    numLayers: 48,
    hiddenSize: 2048,
    numAttentionHeads: 32,
    numKvHeads: 4,
    contextLength: 131_072,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-30B-A3B",
    shortDescription:
      "Qwen3 MoE release with 30.5B total parameters and 3.3B active parameters, built for lower active compute than a comparable dense model.",
    researchHighlight:
      "Qwen3-30B-A3B is the smaller MoE branch of Qwen3, combining the family’s thinking-mode and agent story with a much lower active path than a dense 30B-class checkpoint.",
    researchHighlights: [
      point("MoE branch of Qwen3", "This model moves Qwen3 into a sparse MoE serving geometry while keeping the same user-facing thinking/non-thinking framing."),
      point("Low active path", "Only 3.3B parameters are activated per token out of 30.5B total, which is the central deployment distinction versus the dense Qwen3 line."),
      point("Agent and reasoning focus", "Qwen still positions the model for reasoning, instruction following, and complex agent workflows rather than only general chat."),
    ],
    trainingReleaseContext: qwen3MoeTrainingReleaseContext,
    strengths: qwen3MoeStrengths,
    memoryNote:
      "Resident VRAM tracks the full 30.5B parameter pool even though token compute is closer to the 3.3B activated path, so MoE changes compute pressure more than the weight floor.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(61_064_245_248, 30_500_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-30B-A3B",
        note: "The official Qwen3-30B-A3B safetensor weights total about 61.06 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-235b-a22b",
    displayName: "Qwen 3 235B A22B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-04-27",
    architectureType: "Mixture-of-experts transformer",
    isMoe: true,
    totalParams: 235_000_000_000,
    activeParams: 22_000_000_000,
    numLayers: 94,
    hiddenSize: 4096,
    numAttentionHeads: 64,
    numKvHeads: 4,
    contextLength: 131_072,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-235B-A22B",
    shortDescription:
      "Largest Qwen3 MoE release with 235B total parameters and 22B activated parameters, aimed at frontier-scale open reasoning and agent use.",
    researchHighlight:
      "Qwen3-235B-A22B is the flagship MoE branch of Qwen3, combining the family’s switchable thinking modes with a much larger resident expert pool than the smaller dense and MoE releases.",
    researchHighlights: [
      point("Flagship Qwen3 MoE", "This is the largest-capacity Qwen3 release in the current line, intended as the top open model in the family."),
      point("Sparse activation at scale", "The model keeps 235B total parameters but only 22B activated per token, which is the core reason to deploy it as an MoE rather than a dense frontier-scale model."),
      point("Reasoning and agent focus", "Qwen still frames the flagship around reasoning, instruction following, and agent-style workflows rather than only benchmark scale."),
    ],
    trainingReleaseContext: qwen3MoeTrainingReleaseContext,
    strengths: qwen3MoeStrengths,
    memoryNote:
      "Even with only 22B activated per token, the full 235B resident expert pool dominates VRAM immediately, so this is primarily a multi-card or very-large-card deployment target.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(470_187_269_120, 235_000_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-235B-A22B",
        note: "The official Qwen3-235B-A22B safetensor weights total about 470.19 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-4b-thinking-2507",
    displayName: "Qwen 3 4B Thinking 2507",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-08-05",
    architectureType: "Dense decoder-only transformer",
    isMoe: false,
    totalParams: 4_000_000_000,
    numLayers: 36,
    hiddenSize: 2560,
    numAttentionHeads: 32,
    numKvHeads: 8,
    contextLength: 262_144,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507",
    shortDescription:
      "Qwen3 update focused on deeper reasoning and longer native context, tuned specifically for more complex thinking-heavy workloads.",
    researchHighlight:
      "Qwen3-4B-Thinking-2507 is the reasoning-focused follow-up to Qwen3-4B, with stronger complex-task performance and a native 256K context window.",
    researchHighlights: [
      point("Reasoning-focused update", "Qwen describes this release as a scaled-up thinking-capability update rather than a general-purpose refresh."),
      point("256K native context", "The 2507 update moves the model to 256K native context, which is one of the clearest deployment changes from the base 4B release."),
      point("Deeper thinking length", "Qwen explicitly notes a longer thinking length and recommends the model for highly complex reasoning tasks."),
    ],
    trainingReleaseContext: [
      point("Release lineage", "This is an updated reasoning-oriented version of Qwen3-4B rather than a separate new architecture family."),
      point("Model geometry", "The update keeps the same 4.0B parameter, 36-layer, 32Q/8KV dense geometry as the base 4B model."),
      point("Context packaging", "Unlike the base model’s 32K native context with YaRN extension, this update is packaged with 256K native context."),
    ],
    strengths: [
      point("Complex reasoning", "Best fit for logic, mathematics, science, coding, and other tasks where longer reasoning traces help."),
      point("Long-context understanding", "The 256K native context makes it more useful for very long inputs than the base Qwen3 dense line."),
      point("Tool and instruction use", "Qwen also positions the update as stronger on instruction following and tool usage, not only on benchmark reasoning."),
    ],
    memoryNote:
      "This remains a dense 4B model, but the 256K native context means KV growth can become a much larger part of the total than on the base 32K-native release.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(4_022_468_096, 4_000_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507",
        note: "The official Qwen3-4B-Thinking-2507 safetensor weights total about 4.02 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3-30b-a3b-instruct-2507",
    displayName: "Qwen 3 30B A3B Instruct 2507",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2025-07-28",
    architectureType: "Mixture-of-experts transformer",
    isMoe: true,
    totalParams: 30_500_000_000,
    activeParams: 3_300_000_000,
    numLayers: 48,
    hiddenSize: 2048,
    numAttentionHeads: 32,
    numKvHeads: 4,
    contextLength: 262_144,
    vocabSize: 151_936,
    license: "Apache 2.0",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507",
    shortDescription:
      "Non-thinking Qwen3 MoE update with stronger general capabilities, better alignment, and native 256K context packaging.",
    researchHighlight:
      "Qwen3-30B-A3B-Instruct-2507 is the non-thinking update to the smaller Qwen3 MoE branch, tuned for stronger general assistant and tool-using behavior rather than explicit think-block reasoning.",
    researchHighlights: [
      point("Non-thinking update", "This release is explicitly the non-thinking-mode update and no longer requires users to force thinking off at inference time."),
      point("General-capability uplift", "Qwen describes stronger instruction following, logical reasoning, comprehension, mathematics, science, coding, and tool use than the earlier non-thinking version."),
      point("256K native context", "The update is packaged with 256K native context, making long-context serving more central than in the base 30B-A3B release."),
    ],
    trainingReleaseContext: [
      point("Release lineage", "This is an updated non-thinking-mode variant of Qwen3-30B-A3B rather than a brand-new architecture branch."),
      point("MoE geometry", "The model keeps the same 30.5B total / 3.3B active parameter geometry, 48 layers, 128 experts, and 8 activated experts as the base A3B release."),
      point("Output behavior", "Qwen notes that this update no longer emits <think></think> blocks and is intended as a cleaner non-thinking deployment target."),
    ],
    strengths: [
      point("General assistant quality", "Best fit when you want the Qwen3 MoE branch without exposing explicit thinking-mode behavior in outputs."),
      point("Tool and workflow use", "Qwen emphasizes stronger tool usage, instruction following, and text generation alignment in this update."),
      point("Long-context non-thinking serving", "The 256K native context makes it useful for long-input assistant workflows where explicit reasoning blocks are not desired."),
    ],
    memoryNote:
      "Resident VRAM still tracks the full 30.5B MoE checkpoint, but the 256K native context means cache growth becomes much more visible during long-context serving than in the base 32K-native release.",
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(61_064_245_248, 30_500_000_000),
        sourceUrl: "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507",
        note: "The official Qwen3-30B-A3B-Instruct-2507 safetensor weights total about 61.06 GB on Hugging Face.",
      }),
    ],
  },
  {
    id: "qwen-3.5-0.8b",
    displayName: "Qwen 3.5 0.8B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2026-02-28",
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
    trainingReleaseContext: qwen35TrainingReleaseContext,
    strengths: qwen35Strengths,
    memoryBehaviorPoints: qwen35MemoryBehaviorPoints(6, 24),
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(1_746_882_752, 900_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-0.8B-Base",
        note: "The official Qwen3.5-0.8B checkpoint totals about 1.75 GB on Hugging Face, and Qwen documents Transformers and vLLM usage for the release.",
      }),
    ],
  },
  {
    id: "qwen-3.5-2b",
    displayName: "Qwen 3.5 2B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2026-02-28",
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
    trainingReleaseContext: qwen35TrainingReleaseContext,
    strengths: qwen35Strengths,
    memoryBehaviorPoints: qwen35MemoryBehaviorPoints(6, 24),
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(4_548_144_832, 2_000_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-2B",
        note: "The official Qwen3.5-2B checkpoint totals about 4.55 GB on Hugging Face, and Qwen documents both Transformers and vLLM serving paths.",
      }),
    ],
  },
  {
    id: "qwen-3.5-4b",
    displayName: "Qwen 3.5 4B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2026-02-27",
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
    trainingReleaseContext: qwen35TrainingReleaseContext,
    strengths: qwen35Strengths,
    memoryBehaviorPoints: qwen35MemoryBehaviorPoints(8, 32),
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(9_319_737_856, 5_000_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-4B",
        note: "The official Qwen3.5-4B checkpoint totals about 9.32 GB on Hugging Face, and Qwen documents explicit Transformers and vLLM guidance, including text-only serving in vLLM.",
      }),
    ],
  },
  {
    id: "qwen-3.5-9b",
    displayName: "Qwen 3.5 9B",
    family: "Qwen",
    organization: "Alibaba",
    releaseDate: "2026-02-27",
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
    trainingReleaseContext: qwen35TrainingReleaseContext,
    strengths: qwen35Strengths,
    memoryBehaviorPoints: qwen35MemoryBehaviorPoints(8, 32),
    inferenceProfiles: [
      directProfile({
        id: "official-bf16",
        label: "Official BF16 checkpoint",
        effectiveDtype: "bf16",
        weightBytes: bytesFromCheckpointBytes(19_306_216_416, 10_000_000_000),
        supportedRuntimes: ALL_RUNTIMES,
        sourceUrl: "https://huggingface.co/Qwen/Qwen3.5-9B",
        note: "The official Qwen3.5-9B checkpoint totals about 19.31 GB on Hugging Face, and Qwen documents both Transformers and vLLM support.",
      }),
    ],
  },
  {
    id: "openreasoning-nemotron-1.5b",
    displayName: "OpenReasoning Nemotron 1.5B",
    family: "Nemotron",
    organization: "NVIDIA",
    releaseDate: "2025-07-15",
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
    trainingReleaseContext: nemotronTrainingReleaseContext,
    strengths: nemotronStrengths,
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
    releaseDate: "2025-07-15",
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
    trainingReleaseContext: nemotronTrainingReleaseContext,
    strengths: nemotronStrengths,
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
    releaseDate: "2025-07-15",
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
    trainingReleaseContext: nemotronTrainingReleaseContext,
    strengths: nemotronStrengths,
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
    releaseDate: "2025-07-15",
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
    trainingReleaseContext: nemotronTrainingReleaseContext,
    strengths: nemotronStrengths,
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
    releaseDate: "2024-06-24",
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
    trainingReleaseContext: gemma2TrainingReleaseContext,
    strengths: gemma2Strengths,
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
    releaseDate: "2024-06-24",
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
    trainingReleaseContext: gemma2TrainingReleaseContext,
    strengths: gemma2Strengths,
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
    releaseDate: "2024-07-17",
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
    trainingReleaseContext: mistralNemoTrainingReleaseContext,
    strengths: mistralNemoStrengths,
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
    releaseDate: "2023-12-10",
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
    trainingReleaseContext: mixtralTrainingReleaseContext,
    strengths: mixtralStrengths,
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
    releaseDate: "2024-12-11",
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
    trainingReleaseContext: phi4TrainingReleaseContext,
    strengths: phi4Strengths,
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
