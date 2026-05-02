export type ModelSpec = {
  id: string;
  family: ModelFamily;
  name: string;
  paramsB: number;
  kvBytesPerTokenFp16: number;
};

export const MODEL_FAMILIES = [
  "Llama",
  "Qwen",
  "DeepSeek",
  "Mistral",
  "Gemma",
  "OpenAI",
  "GLM",
  "MiniMax",
  "Moonshot",
] as const;

export type ModelFamily = (typeof MODEL_FAMILIES)[number];

// kvBytesPerTokenFp16 = 4 × n_layers × n_kv_heads × head_dim for GQA/MHA models.
// MLA-style (DeepSeek V3+): (kv_lora_rank + qk_rope_head_dim) × n_layers × 2.
// Hybrid Mamba/attention models count only attention layers.
// All values verified against each model's HF config.json unless noted.
export const MODELS: ModelSpec[] = [
  // Llama (Meta)
  { id: "llama-3.1-8b",     family: "Llama",     name: "Llama 3.1 8B",                       paramsB: 8.03, kvBytesPerTokenFp16: 131072 },
  { id: "llama-3.1-70b",    family: "Llama",     name: "Llama 3.1 70B",                      paramsB: 70.6, kvBytesPerTokenFp16: 327680 },
  { id: "llama-3.1-405b",   family: "Llama",     name: "Llama 3.1 405B",                     paramsB: 405,  kvBytesPerTokenFp16: 516096 },
  { id: "llama-3.3-70b",    family: "Llama",     name: "Llama 3.3 70B",                      paramsB: 70.6, kvBytesPerTokenFp16: 327680 },
  { id: "llama-4-scout",    family: "Llama",     name: "Llama 4 Scout (MoE 109B/17B)",       paramsB: 109,  kvBytesPerTokenFp16: 196608 },
  { id: "llama-4-maverick", family: "Llama",     name: "Llama 4 Maverick (MoE 400B/17B)",    paramsB: 400,  kvBytesPerTokenFp16: 196608 },

  // Qwen (Alibaba)
  { id: "qwen-2.5-7b",        family: "Qwen", name: "Qwen 2.5 7B",                       paramsB: 7.62, kvBytesPerTokenFp16: 57344 },
  { id: "qwen-2.5-14b",       family: "Qwen", name: "Qwen 2.5 14B",                      paramsB: 14.7, kvBytesPerTokenFp16: 196608 },
  { id: "qwen-2.5-32b",       family: "Qwen", name: "Qwen 2.5 32B",                      paramsB: 32.5, kvBytesPerTokenFp16: 262144 },
  { id: "qwen-2.5-72b",       family: "Qwen", name: "Qwen 2.5 72B",                      paramsB: 72.7, kvBytesPerTokenFp16: 327680 },
  { id: "qwen-3-8b",          family: "Qwen", name: "Qwen 3 8B",                         paramsB: 8.2,  kvBytesPerTokenFp16: 147456 },
  { id: "qwen-3-32b",         family: "Qwen", name: "Qwen 3 32B",                        paramsB: 32,   kvBytesPerTokenFp16: 262144 },
  { id: "qwen-3-30b-a3b",     family: "Qwen", name: "Qwen 3 30B-A3B (MoE 30B/3B)",       paramsB: 30,   kvBytesPerTokenFp16: 98304 },
  { id: "qwen-3.6-27b",       family: "Qwen", name: "Qwen 3.6 27B (Hybrid DeltaNet)",    paramsB: 28,   kvBytesPerTokenFp16: 16384 },  // 16 of 64 layers full-attention
  { id: "qwen-3.6-35b-a3b",   family: "Qwen", name: "Qwen 3.6 35B-A3B (Hybrid MoE 36B/3B)", paramsB: 36, kvBytesPerTokenFp16: 20480 },  // 10 of 40 layers full-attention

  // DeepSeek
  { id: "deepseek-v3.1",     family: "DeepSeek", name: "DeepSeek V3.1 (MoE 671B/37B, MLA)",     paramsB: 671,  kvBytesPerTokenFp16: 70272 },
  { id: "deepseek-r1",       family: "DeepSeek", name: "DeepSeek R1 (MoE 671B/37B, MLA)",       paramsB: 671,  kvBytesPerTokenFp16: 70272 },
  { id: "deepseek-v4-flash", family: "DeepSeek", name: "DeepSeek V4 Flash (MoE 284B/13B)",      paramsB: 284,  kvBytesPerTokenFp16: 49536 },
  { id: "deepseek-v4-pro",   family: "DeepSeek", name: "DeepSeek V4 Pro (MoE 1.6T/49B)",        paramsB: 1600, kvBytesPerTokenFp16: 70272 },

  // Mistral
  { id: "mistral-7b-v0.3", family: "Mistral", name: "Mistral 7B v0.3",                  paramsB: 7.24, kvBytesPerTokenFp16: 131072 },
  { id: "mistral-small-3", family: "Mistral", name: "Mistral Small 3 (24B)",            paramsB: 23.6, kvBytesPerTokenFp16: 163840 },
  { id: "mixtral-8x7b",    family: "Mistral", name: "Mixtral 8x7B (MoE 47B/13B)",       paramsB: 46.7, kvBytesPerTokenFp16: 131072 },
  { id: "mixtral-8x22b",   family: "Mistral", name: "Mixtral 8x22B (MoE 141B/39B)",     paramsB: 141,  kvBytesPerTokenFp16: 229376 },

  // Gemma (Google)
  { id: "gemma-2-9b",      family: "Gemma", name: "Gemma 2 9B",                          paramsB: 9.24, kvBytesPerTokenFp16: 344064 },
  { id: "gemma-2-27b",     family: "Gemma", name: "Gemma 2 27B",                         paramsB: 27.2, kvBytesPerTokenFp16: 376832 },
  { id: "gemma-3-12b",     family: "Gemma", name: "Gemma 3 12B",                         paramsB: 12,   kvBytesPerTokenFp16: 393216 },
  { id: "gemma-3-27b",     family: "Gemma", name: "Gemma 3 27B",                         paramsB: 27,   kvBytesPerTokenFp16: 507904 },
  { id: "gemma-4-e2b",     family: "Gemma", name: "Gemma 4 E2B (multimodal)",            paramsB: 5,    kvBytesPerTokenFp16: 35840 },
  { id: "gemma-4-e4b",     family: "Gemma", name: "Gemma 4 E4B (multimodal)",            paramsB: 8,    kvBytesPerTokenFp16: 86016 },
  { id: "gemma-4-26b-a4b", family: "Gemma", name: "Gemma 4 26B-A4B (MoE 27B/4B)",        paramsB: 27,   kvBytesPerTokenFp16: 245760 },
  { id: "gemma-4-31b",     family: "Gemma", name: "Gemma 4 31B",                         paramsB: 33,   kvBytesPerTokenFp16: 983040 },

  // OpenAI
  { id: "gpt-oss-20b",  family: "OpenAI", name: "GPT-OSS 20B (MoE)",  paramsB: 20,  kvBytesPerTokenFp16: 49152 },
  { id: "gpt-oss-120b", family: "OpenAI", name: "GPT-OSS 120B (MoE)", paramsB: 120, kvBytesPerTokenFp16: 73728 },

  // GLM (Zhipu / zai-org)
  { id: "glm-4.5", family: "GLM", name: "GLM-4.5 (MoE 355B/32B)", paramsB: 355, kvBytesPerTokenFp16: 376832 },
  { id: "glm-4.6", family: "GLM", name: "GLM-4.6 (MoE 355B/32B)", paramsB: 355, kvBytesPerTokenFp16: 376832 },

  // MiniMax
  { id: "minimax-m2",   family: "MiniMax", name: "MiniMax M2 (MoE 230B/10B)",   paramsB: 230, kvBytesPerTokenFp16: 253952 },
  { id: "minimax-m2.7", family: "MiniMax", name: "MiniMax M2.7 (MoE 230B/10B)", paramsB: 230, kvBytesPerTokenFp16: 253952 },

  // Moonshot (Kimi) — DeepSeek V3-derived, MLA attention
  { id: "kimi-k2.5", family: "Moonshot", name: "Kimi K2.5 (MoE 1.1T/87B, MLA)", paramsB: 1100, kvBytesPerTokenFp16: 70272 },
  { id: "kimi-k2.6", family: "Moonshot", name: "Kimi K2.6 (MoE 1.1T/87B, MLA)", paramsB: 1100, kvBytesPerTokenFp16: 70272 },
];

export const CUSTOM_MODEL_ID = "custom";

export function findModel(id: string): ModelSpec | undefined {
  return MODELS.find((m) => m.id === id);
}
