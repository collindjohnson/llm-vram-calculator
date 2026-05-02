# LLM VRAM Calculator

A back-of-the-envelope calculator for figuring out whether a given language model will fit in your GPU's VRAM — for inference, LoRA, or QLoRA fine-tuning.

Includes per-architecture KV cache values for current open-weights models (Llama, Qwen, DeepSeek, Mistral, Gemma, GPT-OSS, GLM, MiniMax, Kimi) computed from each model's Hugging Face `config.json`. Custom mode falls back to a sub-linear approximation calibrated for modern GQA architectures.

## Run locally

```bash
git clone https://github.com/collindjohnson/llm-vram-calculator.git
cd llm-vram-calculator
npm install
npm run dev
```

Then open http://localhost:3000.

## How it works

Memory usage breaks down into four buckets:

- **Model weights** — `parameters × bytes_per_parameter`. A 7B model in bfloat16 (2 bytes) is ~14 GB. For MoE models, all experts must be resident, so total params (not active) drives weight memory.
- **KV cache** — Attention state cached during inference. Scales linearly with context length but only sub-linearly with model size, since it depends on `n_layers × n_kv_heads × head_dim` rather than total parameters. Modern GQA architectures (Llama-3+, Qwen 2.5+, Mistral) cache far less per token than legacy MHA at the same parameter count.
- **Training overhead** — Only present during fine-tuning. ~0.6× weights for LoRA, ~0.4× for QLoRA, ~3× for full fine-tuning (gradients + Adam optimizer states + activations).
- **Activations + buffer** — A flat 1-2 GB for working memory.

## Math reference

```
weights_gb     = params_billions × (bits / 8)

# Model preset (exact, from the selected model's attention architecture):
kv_cache_gb    = (2 × n_layers × n_kv_heads × head_dim × 2)
                 × context_length × (bits / 16) / 1e9

# Custom mode (sub-linear approximation, calibrated for modern GQA):
kv_cache_gb    = √params_billions × 40000
                 × context_length × (bits / 16) / 1e9

training_gb    = weights_gb × { 0 inference, 0.6 LoRA, 0.4 QLoRA, 3 full }
activations_gb = 1 if inference else 2
total_gb       = weights_gb + kv_cache_gb + training_gb + activations_gb
usable_gb      = total_vram_gb × 0.9
```

## Limitations

These are estimates, not guarantees. Real memory usage depends on framework choice (vLLM, llama.cpp, PyTorch, TensorRT-LLM), batch size, attention variant (FlashAttention, paged attention), and whatever else is running on the GPU. Treat **comfortable fit** as accurate, **tight but works** as "test before you commit," and **won't fit** as a hard wall.

The KV cache assumes the cache is stored at the same precision as the weights — common when running 4-bit weights with 4-bit KV cache to fit long contexts in VRAM. Most production setups keep KV in fp16/bf16 regardless of weight precision; if you're running fp16 KV with 4-bit weights, multiply the reported KV cache by 4.

DeepSeek V3.1, R1, and V4 use Multi-Head Latent Attention (or its V4 sparse-index successor), which compresses KV substantially — they're listed with their effective per-token KV size. Qwen 3.6 (27B and 35B-A3B) interleaves Gated DeltaNet linear-attention layers with full-attention layers in a 3:1 ratio, so only 1-in-4 layers grow KV with context; the listed sizes reflect that.

## Contributing

Pull requests welcome.

**Adding a model**: append an entry to `app/models.ts`. The `kvBytesPerTokenFp16` field comes from the model's Hugging Face `config.json`:

- **Standard GQA/MHA**: `4 × num_hidden_layers × num_key_value_heads × head_dim` (head_dim is `hidden_size / num_attention_heads` if not explicitly set).
- **MLA (DeepSeek V3+)**: `(kv_lora_rank + qk_rope_head_dim) × num_hidden_layers × 2`.
- **Hybrid attention/linear** (Qwen 3.6, Jamba-style): only count the full-attention layers, not the linear/Mamba ones.

Cite the relevant `config.json` link in your PR. New model families need to be added to `MODEL_FAMILIES` in the same file.

**Improving the formulas**: the custom-mode KV approximation (`√params × 40000` bytes/token in fp16) is a coarse fit; targeted improvements (e.g. classifying by GQA group ratio, or per-family multipliers) are welcome — bring data.

## Deploy

The site is a stock Next.js app, so it deploys cleanly to Vercel, Cloudflare Pages, or any host that runs Node 20+. For Vercel: import the repo, accept defaults, done.

## License

MIT — see [LICENSE](LICENSE).
