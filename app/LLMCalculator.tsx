"use client";

import { memo, useCallback, useDeferredValue, useMemo, useState } from "react";
import {
  CUSTOM_MODEL_ID,
  findModel,
  MODEL_FAMILIES,
  MODELS,
} from "./models";

function formatGB(gb: number): string {
  if (gb < 1) return `${(gb * 1024).toFixed(0)} MB`;
  return `${gb.toFixed(1)} GB`;
}

type UseCase = "inference" | "lora" | "qlora" | "full";

type State = {
  vram: number;
  model: string;
  params: number;
  bits: number;
  useCase: UseCase;
  ctx: number;
};

type Calc = {
  weightsGB: number;
  kvGB: number;
  trainGB: number;
  actGB: number;
  totalGB: number;
  availGB: number;
};

function kvCacheGB(state: State, bytesPerParam: number) {
  const preset = findModel(state.model);
  const bytesPerTokenFp16 = preset
    ? preset.kvBytesPerTokenFp16
    : 40000 * Math.sqrt(state.params);
  const bytesPerToken = bytesPerTokenFp16 * (bytesPerParam / 2);
  return (bytesPerToken * state.ctx) / 1e9;
}

function calculate(state: State): Calc {
  const bytesPerParam = state.bits / 8;
  const weightsGB = state.params * bytesPerParam;
  const kvGB = kvCacheGB(state, bytesPerParam);

  let trainGB = 0;
  if (state.useCase === "lora") trainGB = weightsGB * 0.6;
  else if (state.useCase === "qlora") trainGB = weightsGB * 0.4;
  else if (state.useCase === "full") trainGB = weightsGB * 3;

  const actGB = state.useCase === "inference" ? 1 : 2;
  const totalGB = weightsGB + kvGB + trainGB + actGB;
  const availGB = state.vram * 0.9;

  return { weightsGB, kvGB, trainGB, actGB, totalGB, availGB };
}

type StatusKey = "fit" | "tight" | "no";

function getStatus(totalGB: number, availGB: number): {
  key: StatusKey;
  label: string;
  className: string;
} {
  if (totalGB <= availGB * 0.7) {
    return {
      key: "fit",
      label: "Comfortable fit",
      className: "bg-mint text-forest",
    };
  } else if (totalGB <= availGB) {
    return {
      key: "tight",
      label: "Tight but works",
      className: "bg-cream text-burnt",
    };
  } else {
    return {
      key: "no",
      label: "Won't fit",
      className: "bg-red-100 text-red-800",
    };
  }
}

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

type PairConfig = {
  id: string;
  label: string;
  unit: string;
  inputMin: number;
  inputMax?: number;
  inputStep: number;
  sliderMin: number;
  sliderMax: number;
  sliderStep: number;
  sliderMinLabel: string;
  sliderMaxLabel: string;
};

// VRAM has no upper input cap — multi-GPU sums and future hardware can exceed
// any number we'd pick. Slider stays bounded for ergonomics; the number field
// accepts anything ≥ inputMin.
const VRAM_CONFIG: PairConfig = {
  id: "vram",
  label: "GPU VRAM",
  unit: "GB",
  inputMin: 4,
  inputStep: 1,
  sliderMin: 8,
  sliderMax: 96,
  sliderStep: 4,
  sliderMinLabel: "8",
  sliderMaxLabel: "96",
};

const PARAMS_CONFIG: PairConfig = {
  id: "params",
  label: "Model size (parameters)",
  unit: "B",
  inputMin: 0.01,
  inputMax: 2000,
  inputStep: 0.1,
  sliderMin: 0.1,
  sliderMax: 120,
  sliderStep: 0.1,
  sliderMinLabel: "0.1B",
  sliderMaxLabel: "120B",
};

const BITS_CONFIG: PairConfig = {
  id: "bits",
  label: "Precision (bits)",
  unit: "bits",
  inputMin: 2,
  inputMax: 32,
  inputStep: 1,
  sliderMin: 2,
  sliderMax: 32,
  sliderStep: 1,
  sliderMinLabel: "2",
  sliderMaxLabel: "32",
};

const CTX_CONFIG: PairConfig = {
  id: "ctx",
  label: "Context length (tokens)",
  unit: "tok",
  inputMin: 128,
  inputMax: 1000000,
  inputStep: 128,
  sliderMin: 512,
  sliderMax: 200000,
  sliderStep: 512,
  sliderMinLabel: "512",
  sliderMaxLabel: "200K",
};

type PairProps = {
  config: PairConfig;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
};

const NumberSliderPair = memo(function NumberSliderPair({
  config,
  value,
  onChange,
  disabled = false,
}: PairProps) {
  const commit = (raw: number) => {
    if (Number.isNaN(raw)) return;
    const max = config.inputMax ?? Number.POSITIVE_INFINITY;
    onChange(clamp(raw, config.inputMin, max));
  };

  const sliderValue = clamp(value, config.sliderMin, config.sliderMax);
  const dimmed = disabled ? "opacity-60" : "";

  return (
    <div className={`flex flex-col gap-1.5 ${dimmed}`}>
      <label
        htmlFor={`${config.id}-input`}
        className="text-[13px] text-slate dark:text-stone/80"
      >
        {config.label}
      </label>
      <div className="flex items-center gap-2">
        <input
          id={`${config.id}-input`}
          type="number"
          min={config.inputMin}
          max={config.inputMax}
          step={config.inputStep}
          value={value}
          disabled={disabled}
          onChange={(e) => commit(parseFloat(e.target.value))}
          className="flex-1 h-9 px-2.5 text-sm font-mono text-right rounded-md
            bg-paper dark:bg-coal text-coal dark:text-paper
            border border-stone dark:border-forest
            hover:border-graphite dark:hover:border-stone
            focus:outline-none focus:border-green dark:focus:border-lime
            disabled:cursor-not-allowed"
        />
        <span className="text-[13px] text-slate min-w-[36px]">{config.unit}</span>
      </div>
      <input
        type="range"
        min={config.sliderMin}
        max={config.sliderMax}
        step={config.sliderStep}
        value={sliderValue}
        disabled={disabled}
        onChange={(e) => commit(parseFloat(e.target.value))}
        className="w-full accent-green dark:accent-lime touch-pan-y disabled:cursor-not-allowed"
        aria-label={`${config.label} slider`}
      />
      <div className="flex justify-between text-[11px] text-slate px-0.5">
        <span>{config.sliderMinLabel}</span>
        <span>{config.sliderMaxLabel}</span>
      </div>
    </div>
  );
});

export default function LLMCalculator() {
  const [state, setState] = useState<State>({
    vram: 24,
    model: CUSTOM_MODEL_ID,
    params: 7,
    bits: 16,
    useCase: "inference",
    ctx: 8192,
  });

  const deferredState = useDeferredValue(state);
  const r = useMemo(() => calculate(deferredState), [deferredState]);
  const status = useMemo(
    () => getStatus(r.totalGB, r.availGB),
    [r.totalGB, r.availGB],
  );

  const setVram = useCallback(
    (v: number) => setState((s) => ({ ...s, vram: v })),
    [],
  );
  const setParams = useCallback(
    (v: number) =>
      setState((s) =>
        s.params === v ? s : { ...s, params: v, model: CUSTOM_MODEL_ID },
      ),
    [],
  );
  const setBits = useCallback(
    (v: number) => setState((s) => ({ ...s, bits: v })),
    [],
  );
  const setCtx = useCallback(
    (v: number) => setState((s) => ({ ...s, ctx: v })),
    [],
  );
  const setUseCase = useCallback(
    (v: UseCase) => setState((s) => ({ ...s, useCase: v })),
    [],
  );
  const setModel = useCallback((id: string) => {
    const preset = findModel(id);
    setState((s) =>
      preset
        ? { ...s, model: preset.id, params: preset.paramsB }
        : { ...s, model: CUSTOM_MODEL_ID },
    );
  }, []);

  const isPreset = state.model !== CUSTOM_MODEL_ID;

  const numberCellClass =
    "text-right font-mono tabular-nums text-graphite dark:text-stone min-w-[5.5ch]";

  return (
    <div className="not-prose my-8 space-y-6 [contain:content]">
      <section className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="sm:col-span-2 flex flex-col gap-1.5">
          <label
            htmlFor="model"
            className="text-[13px] text-slate dark:text-stone/80"
          >
            Model preset
          </label>
          <select
            id="model"
            value={state.model}
            onChange={(e) => setModel(e.target.value)}
            className="h-9 px-2.5 text-sm rounded-md
              bg-paper dark:bg-coal text-coal dark:text-paper
              border border-stone dark:border-forest
              hover:border-graphite dark:hover:border-stone
              focus:outline-none focus:border-green dark:focus:border-lime"
          >
            <option value={CUSTOM_MODEL_ID}>Custom (set parameters manually)</option>
            {MODEL_FAMILIES.map((family) => (
              <optgroup key={family} label={family}>
                {MODELS.filter((m) => m.family === family).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          <p className="text-[11px] text-slate px-0.5">
            {isPreset
              ? "KV cache computed from this model's attention architecture."
              : "KV cache estimated from a sub-linear approximation calibrated for modern GQA models."}
          </p>
        </div>
        <NumberSliderPair config={VRAM_CONFIG} value={state.vram} onChange={setVram} />
        <NumberSliderPair config={PARAMS_CONFIG} value={state.params} onChange={setParams} disabled={isPreset} />
        <NumberSliderPair config={BITS_CONFIG} value={state.bits} onChange={setBits} />
        <div className="flex flex-col gap-1.5">
          <label
            htmlFor="usecase"
            className="text-[13px] text-slate dark:text-stone/80"
          >
            Use case
          </label>
          <select
            id="usecase"
            value={state.useCase}
            onChange={(e) => setUseCase(e.target.value as UseCase)}
            className="h-9 px-2.5 text-sm rounded-md
              bg-paper dark:bg-coal text-coal dark:text-paper
              border border-stone dark:border-forest
              hover:border-graphite dark:hover:border-stone
              focus:outline-none focus:border-green dark:focus:border-lime"
          >
            <option value="inference">Inference only (running it)</option>
            <option value="lora">LoRA fine-tuning</option>
            <option value="qlora">QLoRA fine-tuning (4-bit base)</option>
            <option value="full">Full fine-tuning (rarely possible)</option>
          </select>
        </div>
        <div className="sm:col-span-2">
          <NumberSliderPair config={CTX_CONFIG} value={state.ctx} onChange={setCtx} />
        </div>
      </section>

      <section className="rounded-[12px] p-5 bg-stone/40 dark:bg-forest/30">
        <div className="flex items-baseline justify-between mb-3 gap-4">
          <span className="text-[13px] text-slate dark:text-stone/80">
            Estimated total memory needed
          </span>
          <span className="text-[24px] font-medium font-mono tabular-nums text-coal dark:text-paper text-right min-w-[7ch]">
            {formatGB(r.totalGB)}
          </span>
        </div>
        <div className="flex items-baseline justify-between mb-4 gap-4">
          <span className="text-[13px] text-slate dark:text-stone/80">
            Usable VRAM (90%, leaves headroom for CUDA + framework overhead)
          </span>
          <span className="text-base font-medium font-mono tabular-nums text-graphite dark:text-stone text-right min-w-[7ch]">
            {formatGB(r.availGB)}
          </span>
        </div>
        <div
          className={`inline-block text-sm font-medium px-3 py-1.5 rounded-md ${status.className}`}
        >
          {status.label}
        </div>
      </section>

      <section className="border border-stone dark:border-forest rounded-[12px] p-5">
        <h2 className="text-base font-medium text-coal dark:text-paper mb-3 mt-0">
          Memory breakdown
        </h2>
        <div className="grid grid-cols-[1fr_auto] gap-y-1.5 gap-x-4 text-[13px]">
          <span className="text-slate">Model weights</span>
          <span className={numberCellClass}>{formatGB(r.weightsGB)}</span>
          <span className="text-slate">KV cache (context)</span>
          <span className={numberCellClass}>{formatGB(r.kvGB)}</span>
          <span className="text-slate">Training overhead</span>
          <span className={numberCellClass}>
            {deferredState.useCase === "inference"
              ? "— (inference)"
              : formatGB(r.trainGB)}
          </span>
          <span className="text-slate">Activations + buffer</span>
          <span className={numberCellClass}>{formatGB(r.actGB)}</span>
        </div>
      </section>
    </div>
  );
}
