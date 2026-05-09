import type { BlogPost } from "@/lib/types";

export const posts: BlogPost[] = [
  {
    slug: "how-inference-vram-is-calculated",
    title: "How inference VRAM is calculated",
    summary:
      "A practical breakdown of the terms that usually matter for inference memory: weights, KV cache, state, and runtime reserve.",
    publishedAt: "2026-05-10",
    category: "architecture-note",
    companyIds: ["openai", "qwen", "meta-llama", "nvidia"],
    content: [
      {
        type: "paragraph",
        text: "Inference VRAM is not just model size. The real footprint is usually a sum of resident weights, cache or recurrent state, and a runtime reserve that keeps the engine usable under load.",
      },
      {
        type: "paragraph",
        text: "That is why a 7B checkpoint can fit comfortably in one runtime and fail in another, even when both are loading the same weights. The raw checkpoint is only the floor.",
      },
      {
        type: "paragraph",
        text: "The simplest mistake people make is treating parameter count as the answer. Parameter count helps, but inference memory is really about what must stay resident on the card while requests are actively being served.",
      },
      {
        type: "heading",
        text: "The core terms",
      },
      {
        type: "svg",
        variant: "vram-breakdown",
      },
      {
        type: "list",
        items: [
          "Weights: the resident checkpoint in GPU memory.",
          "KV cache: memory that grows with active context and concurrency for attention layers.",
          "State memory: extra recurrent or state-space buffers for hybrid architectures.",
          "Runtime reserve: allocator overhead, kernels, scratch buffers, and engine-specific headroom.",
        ],
      },
      {
        type: "heading",
        text: "A simple dense-transformer view",
      },
      {
        type: "paragraph",
        text: "For a standard grouped-query transformer, the useful mental model is: resident weights plus KV cache plus a safety reserve. Longer context and more concurrent requests mainly grow the KV term.",
      },
      {
        type: "paragraph",
        text: "If you only remember one part of the formula, remember that cache growth is linear in the practical serving knobs. Double the context or double the concurrency and the KV term roughly doubles too.",
      },
      {
        type: "code",
        language: "text",
        code:
          "total_vram ~= weights + kv_cache + overhead\n\nkv_cache ~= batch * context * layers * 2 * kv_heads * head_dim * bytes_per_element",
      },
      {
        type: "paragraph",
        text: "That is why lowering context or concurrency often works immediately for serving. Those are linear levers on the cache term.",
      },
      {
        type: "heading",
        text: "Why checkpoint format matters",
      },
      {
        type: "paragraph",
        text: "Checkpoint format mostly changes the resident weight term. BF16, FP8, INT8, INT4, or mixed formats like MXFP4 all change how many bytes each stored parameter consumes, but they do not magically remove runtime overhead or serving cache.",
      },
      {
        type: "paragraph",
        text: "That is why a 4-bit or mixed-format checkpoint can dramatically lower the fit floor for a big model, while long context and concurrency can still push the full serving estimate back up.",
      },
      {
        type: "heading",
        text: "Where model architecture changes the math",
      },
      {
        type: "paragraph",
        text: "Not every model should use the same cache formula. Hybrid architectures, sliding-window attention, MoE routing, and compressed-weight checkpoints all change which memory terms matter and how they scale.",
      },
      {
        type: "list",
        items: [
          "MoE models still keep total resident weights in memory, even if only part of the model is active per token.",
          "Sliding-window attention limits some cache growth to a bounded window instead of full context.",
          "Hybrid attention or state-space models may use KV cache for only part of the stack and a separate fixed state term for the rest.",
          "Quantized or mixed-format checkpoints mainly change the resident weight term, not the runtime behavior of the cache.",
        ],
      },
      {
        type: "heading",
        text: "Why runtime matters too",
      },
      {
        type: "paragraph",
        text: "A runtime is not just a loader. vLLM, for example, usually plans around a configurable GPU memory utilization budget instead of pretending it owns the full card. That makes the estimate more honest for serving, but it also means the required card VRAM is larger than the raw tensor sum.",
      },
      {
        type: "paragraph",
        text: "This matters because production inference is not just tensor storage. Schedulers, paging behavior, allocator fragmentation, compiled kernels, and scratch buffers all consume part of the card even when the model math looks clean on paper.",
      },
      {
        type: "code",
        language: "text",
        code:
          "required_card_vram ~= core_estimate / gpu_memory_utilization",
      },
      {
        type: "paragraph",
        text: "The practical takeaway is simple: model architecture controls the memory terms, and runtime controls how much of the card is realistically available.",
      },
      {
        type: "heading",
        text: "What FitMyGPU is trying to do",
      },
      {
        type: "paragraph",
        text: "The goal is not to pretend every architecture or runtime can be represented by one universal formula. The goal is to make the assumptions explicit enough that the estimate is useful and extend the memory model only when there is enough architecture evidence to justify it.",
      },
      {
        type: "paragraph",
        text: "That is why different model families can end up with different strategies for cache, state, or resident weight handling. The right question is not whether there is a single perfect formula. The right question is whether the formula matches the architecture closely enough to help someone choose the right GPU.",
      },
    ],
  },
];
