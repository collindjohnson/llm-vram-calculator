import LLMCalculator from "./LLMCalculator";

const REPO_URL = "https://github.com/collindjohnson/llm-vram-calculator";

export default function Home() {
  return (
    <div className="mx-auto max-w-[720px] px-6 pt-12 pb-16">
      <header className="mb-6 flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-[36px] font-medium tracking-tight text-coal dark:text-paper mb-1">
            LLM VRAM Calculator
          </h1>
          <p className="text-slate text-base">interactive tool · MIT licensed</p>
        </div>
        <a
          href={REPO_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-slate hover:text-green dark:hover:text-lime transition-colors underline underline-offset-4"
        >
          View source on GitHub →
        </a>
      </header>

      <p className="text-base mb-4">
        A back-of-the-envelope calculator for figuring out whether a given
        language model will fit in your GPU&apos;s VRAM — for inference, LoRA, or
        QLoRA fine-tuning.
      </p>

      <p className="text-base mb-2 text-graphite dark:text-stone">
        This tool assumes ~90% of total VRAM is usable, leaving headroom for
        CUDA context, the framework runtime, and any other GPU workloads
        (display, encoder, etc.). For multi-GPU setups, enter the VRAM of a
        single card if the model fits on one, or the combined VRAM if you&apos;re
        using tensor/pipeline parallelism.
      </p>

      <LLMCalculator />

      <h2 className="text-[22px] font-medium tracking-tight text-coal dark:text-paper mt-10 mb-3">
        How it works
      </h2>

      <p className="mb-3">Memory usage breaks down into four buckets:</p>

      <ul className="list-disc pl-6 space-y-2 mb-6 marker:text-slate">
        <li>
          <strong className="text-coal dark:text-paper">Model weights</strong>{" "}
          — <code className="font-mono text-[13px]">parameters × bytes_per_parameter</code>.
          A 7B model in bfloat16 (2 bytes) is ~14 GB. For MoE models, all
          experts must be resident, so total params (not active) drives weight
          memory.
        </li>
        <li>
          <strong className="text-coal dark:text-paper">KV cache</strong> —
          Attention state cached during inference. Scales linearly with context
          length but only sub-linearly with model size, since it depends on{" "}
          <code className="font-mono text-[13px]">n_layers × n_kv_heads × head_dim</code>{" "}
          rather than total parameters. Modern GQA architectures (Llama-3+,
          Qwen 2.5+, Mistral) cache far less per token than legacy MHA at the
          same parameter count.
        </li>
        <li>
          <strong className="text-coal dark:text-paper">Training overhead</strong>{" "}
          — Only present during fine-tuning. ~0.6× weights for LoRA, ~0.4× for
          QLoRA, ~3× for full fine-tuning (gradients + Adam optimizer states +
          activations).
        </li>
        <li>
          <strong className="text-coal dark:text-paper">Activations + buffer</strong>{" "}
          — A flat 1-2 GB for working memory.
        </li>
      </ul>

      <h2 className="text-[22px] font-medium tracking-tight text-coal dark:text-paper mt-8 mb-3">
        Math reference
      </h2>

      <pre className="bg-stone/40 dark:bg-forest/30 rounded-[12px] p-4 text-[13px] font-mono overflow-x-auto mb-6">
        <code>{`weights_gb     = params_billions × (bits / 8)

# Model preset (exact, from the selected model's attention architecture):
kv_cache_gb    = (2 × n_layers × n_kv_heads × head_dim × 2)
                 × context_length × (bits / 16) / 1e9

# Custom mode (sub-linear approximation, calibrated for modern GQA):
kv_cache_gb    = √params_billions × 40000
                 × context_length × (bits / 16) / 1e9

training_gb    = weights_gb × { 0 inference, 0.6 LoRA, 0.4 QLoRA, 3 full }
activations_gb = 1 if inference else 2
total_gb       = weights_gb + kv_cache_gb + training_gb + activations_gb
usable_gb      = total_vram_gb × 0.9`}</code>
      </pre>

      <h2 className="text-[22px] font-medium tracking-tight text-coal dark:text-paper mt-8 mb-3">
        Limitations
      </h2>

      <p className="mb-3">
        These are estimates, not guarantees. Real memory usage depends on
        framework choice (vLLM, llama.cpp, PyTorch, TensorRT-LLM), batch size,
        attention variant (FlashAttention, paged attention), and whatever else
        is running on the GPU. Treat <strong>comfortable fit</strong> as
        accurate, <strong>tight but works</strong> as &ldquo;test before you
        commit,&rdquo; and <strong>won&apos;t fit</strong> as a hard wall.
      </p>

      <p className="mb-3">
        The KV cache assumes the cache is stored at the same precision as the
        weights — common when running 4-bit weights with 4-bit KV cache to fit
        long contexts in VRAM. Most production setups keep KV in fp16/bf16
        regardless of weight precision; if you&apos;re running fp16 KV with
        4-bit weights, multiply the reported KV cache by 4.
      </p>

      <p className="mb-3">
        DeepSeek V3.1, R1, and V4 use Multi-Head Latent Attention (or its V4
        sparse-index successor), which compresses KV substantially — they&apos;re
        listed with their effective per-token KV size. Qwen 3.6 (27B and
        35B-A3B) interleaves Gated DeltaNet linear-attention layers with
        full-attention layers in a 3:1 ratio, so only 1-in-4 layers grow KV
        with context; the listed sizes reflect that.
      </p>

      <h2 className="text-[22px] font-medium tracking-tight text-coal dark:text-paper mt-8 mb-3">
        Contributing
      </h2>

      <p className="mb-3">
        Spotted a wrong KV cache size, missing model, or improvement to the
        approximations?{" "}
        <a
          href={`${REPO_URL}/issues`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-green dark:text-lime underline underline-offset-4"
        >
          Open an issue
        </a>{" "}
        or{" "}
        <a
          href={`${REPO_URL}/pulls`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-green dark:text-lime underline underline-offset-4"
        >
          send a PR
        </a>
        . Per-architecture KV values come straight from each model&apos;s
        Hugging Face <code className="font-mono text-[13px]">config.json</code>;
        cite that in your PR.
      </p>
    </div>
  );
}
