"use client";

import {
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import { gpus } from "@/data/gpus";
import { models } from "@/data/models";
import {
  dtypeOptions,
  modeOptions,
  trainingTypeOptions,
} from "@/lib/constants";
import {
  formatBandwidth,
  formatDtype,
  formatGb,
  formatInteger,
  formatParamCount,
} from "@/lib/format";
import { estimateVram } from "@/lib/estimator";
import {
  applyModelConstraints,
  getInferenceProfile,
  getInferenceProfiles,
  modelSupportsMode,
} from "@/lib/model-constraints";
import { normalizeEstimateInput, serializeEstimateInput } from "@/lib/query-state";
import type { EstimateInput, EstimateResult, Mode, ModelSpec } from "@/lib/types";

type Props = {
  initialInput: EstimateInput;
  initialResult: EstimateResult | null;
};

const fieldClassName =
  "w-full rounded-[1.35rem] border border-[var(--line)] bg-white/65 px-4 py-3 text-[0.96rem] text-[var(--ink)] outline-none transition focus:border-[var(--line-strong)] focus:bg-white";
const cardClassName = "surface-card section-reveal rounded-[2rem] p-6 md:p-8";

export function WillItFitApp({ initialInput, initialResult }: Props) {
  const router = useRouter();
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const hasCalculatedRef = useRef(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [formState, setFormState] = useState(initialInput);
  const [result, setResult] = useState<EstimateResult | null>(initialResult);
  const selectedModel = models.find((model) => model.id === formState.modelId) ?? models[0];
  const selectedInferenceProfile = getInferenceProfile(
    selectedModel,
    formState.inferenceProfileId,
    formState.dtype,
  );
  const modelOptions = models.map((model) => ({
    label: model.displayName,
    value: model.id,
  }));
  const inferenceProfileOptions = getInferenceProfiles(selectedModel).map((profile) => ({
    label: profile.label,
    value: profile.id,
  }));
  const dtypeFieldOptions = dtypeOptions.map((option) => ({
    label: option.label,
    value: option.value,
  }));
  const trainingFieldOptions = trainingTypeOptions.map((option) => ({
    label: option.label,
    value: option.value,
  }));
  const gpuFieldOptions = gpus.map((gpu) => ({
    label: gpu.displayName,
    value: gpu.id,
  }));
  const modeFieldOptions = modeOptions.map((option) => ({
    disabled: !modelSupportsMode(selectedModel, option.value),
    label: option.label,
    value: option.value,
  }));

  const scrollToResults = useEffectEvent(() => {
    resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  useEffect(() => {
    if (!result || !hasCalculatedRef.current) {
      return;
    }

    scrollToResults();
  }, [result]);

  function updateField<K extends keyof EstimateInput>(
    key: K,
    value: EstimateInput[K],
  ) {
    setFormState((current) => {
      const next = { ...current, [key]: value };

      if (key === "mode" && value === "inference") {
        next.trainingType = current.trainingType;
      }

      return next;
    });
  }

  function handleModelChange(modelId: string) {
    setFormState((current) => applyModelConstraints({ ...current, modelId }));
  }

  function handleModeChange(nextMode: Mode) {
    setFormState((current) => applyModelConstraints({ ...current, mode: nextMode }));
  }

  function handleInferenceProfileChange(profileId: string) {
    const profile = getInferenceProfile(selectedModel, profileId, formState.dtype);

    setFormState((current) =>
      applyModelConstraints({
        ...current,
        dtype: profile.effectiveDtype,
        inferenceProfileId: profile.id,
      }),
    );
  }

  function handleNumberChange<K extends "contextLength" | "batchSize" | "customVramGb">(
    key: K,
    rawValue: string,
  ) {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) {
      return;
    }

    updateField(key, parsed as EstimateInput[K]);
  }

  function handleCalculate() {
    const normalized = applyModelConstraints(normalizeEstimateInput(formState));
    const nextResult = estimateVram(normalized);
    const params = serializeEstimateInput(normalized);

    hasCalculatedRef.current = true;
    setFormState(normalized);
    setResult(nextResult);

    startTransition(() => {
      router.replace(`/?${params.toString()}`, { scroll: false });
    });
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-[74rem] flex-col gap-8 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
      <section className={cardClassName}>
        <div className="mb-8 flex flex-col gap-4">
          <div className="max-w-3xl space-y-3">
            <h1 className="hero-title text-4xl leading-none text-[var(--ink)] md:text-[4.2rem]">
              Will It Fit?
            </h1>
            <p className="max-w-2xl text-base leading-7 text-[var(--muted)] md:text-lg">
              Estimate GPU VRAM for model inference and training with a compact,
              explainable breakdown. The math stays simple on purpose.
            </p>
          </div>
        </div>

        <form
          className="space-y-7"
          onSubmit={(event) => {
            event.preventDefault();
            handleCalculate();
          }}
        >
          <div className="grid gap-5 md:grid-cols-2">
            <div className="space-y-3">
              <FieldLabel label="Model" />
              <SelectField
                onChange={handleModelChange}
                options={modelOptions}
                value={formState.modelId}
              />
            </div>

            <div className="space-y-3">
              <FieldLabel
                label={
                  formState.mode === "inference"
                    ? "Checkpoint profile"
                    : "Dtype / quantization"
                }
              />
              {formState.mode === "inference" ? (
                <SelectField
                  disabled={Boolean(selectedModel.fixedDtype)}
                  onChange={handleInferenceProfileChange}
                  options={inferenceProfileOptions}
                  value={selectedInferenceProfile.id}
                />
              ) : (
                <SelectField
                  disabled={Boolean(selectedModel.fixedDtype)}
                  onChange={(value) => updateField("dtype", value as EstimateInput["dtype"])}
                  options={dtypeFieldOptions}
                  value={formState.dtype}
                />
              )}
            </div>

            <p className="text-sm leading-6 text-[var(--muted)] md:col-span-2">
              {formatModelAtGlance(selectedModel)}
            </p>

            {selectedModel.fixedDtype ? (
              <p className="text-sm leading-6 text-[var(--muted)] md:col-span-2">
                This checkpoint is fixed to {formatDtype(selectedModel.fixedDtype)} in
                v0.
              </p>
            ) : null}

            {formState.mode === "inference" ? (
              <p className="text-sm leading-6 text-[var(--muted)] md:col-span-2">
                {selectedInferenceProfile.official ? "Official" : "Proxy"} profile:{" "}
                {selectedInferenceProfile.label}. {selectedInferenceProfile.note}
              </p>
            ) : null}

            <div className="space-y-3 md:col-span-2">
              <FieldLabel
                label="Mode"
                hint="Inference is weight + KV cache. Training adds activations, gradients, and optimizer state."
              />
              <SegmentedField
                onChange={(value) => handleModeChange(value as Mode)}
                options={modeFieldOptions}
                value={formState.mode}
              />
            </div>

            {!modelSupportsMode(selectedModel, "training") ? (
              <p className="text-sm leading-6 text-[var(--muted)] md:col-span-2">
                This model is inference-only in v0.
              </p>
            ) : null}

            {formState.mode === "training" ? (
              <div className="space-y-3">
                <FieldLabel label="Training type" />
                <SelectField
                  onChange={(value) =>
                    updateField("trainingType", value as EstimateInput["trainingType"])
                  }
                  options={trainingFieldOptions}
                  value={formState.trainingType}
                />
              </div>
            ) : null}

            <div className="space-y-3">
              <FieldLabel label="GPU" />
              <SelectField
                onChange={(value) => updateField("gpuId", value)}
                options={gpuFieldOptions}
                value={formState.gpuId}
              />
              {formState.gpuId === "custom" ? (
                <p className="text-sm leading-6 text-[var(--muted)]">
                  Enter the custom VRAM value in Advanced options below.
                </p>
              ) : null}
            </div>
          </div>

          <div
            className={cx(
              "rounded-[1.8rem] border border-[var(--line)] bg-white/50 px-4 py-2.5 transition-[background-color,border-color,box-shadow] duration-300 ease-out md:px-5 md:py-3",
              advancedOpen && "bg-white/62 shadow-[0_16px_36px_rgba(30,36,42,0.06)]",
            )}
          >
            <button
              aria-expanded={advancedOpen}
              className="flex w-full items-center justify-between gap-3 text-left"
              onClick={() => setAdvancedOpen((current) => !current)}
              type="button"
            >
              <div>
                <p className="text-sm font-medium text-[var(--ink)]">
                  Advanced options
                </p>
                <p className="mt-0.5 text-[0.82rem] leading-5 text-[var(--muted)]">
                  Context length and batch size. Extra fields appear when they are relevant.
                </p>
              </div>
              <span className="flex h-8 w-8 items-center justify-center rounded-full border border-[var(--line)] bg-white/72">
                <ChevronIcon open={advancedOpen} />
              </span>
            </button>

            <AnimatedExpand open={advancedOpen}>
              <div className="grid gap-5 pt-5 md:grid-cols-2">
                <div className="space-y-3">
                  <FieldLabel label="Context length" />
                  <input
                    className={fieldClassName}
                    min={256}
                    onChange={(event) =>
                      handleNumberChange("contextLength", event.target.value)
                    }
                    step={256}
                    type="number"
                    value={formState.contextLength}
                  />
                </div>

                <div className="space-y-3">
                  <FieldLabel label="Batch size" />
                  <input
                    className={fieldClassName}
                    min={1}
                    onChange={(event) =>
                      handleNumberChange("batchSize", event.target.value)
                    }
                    step={1}
                    type="number"
                    value={formState.batchSize}
                  />
                </div>

                <AnimatedExpand
                  className="md:col-span-1"
                  open={formState.gpuId === "custom"}
                >
                  <div className="space-y-3">
                    <FieldLabel label="Custom GPU VRAM (GB)" />
                    <input
                      className={fieldClassName}
                      min={1}
                      onChange={(event) =>
                        handleNumberChange("customVramGb", event.target.value)
                      }
                      step={1}
                      type="number"
                      value={formState.customVramGb}
                    />
                  </div>
                </AnimatedExpand>

                <AnimatedExpand
                  className="md:col-span-2"
                  open={formState.mode === "training"}
                >
                  <div className="grid gap-5 md:grid-cols-2">
                    <ToggleRow
                      checked={formState.gradientCheckpointing}
                      description="Halves the activation factor in the v0 training estimate."
                      label="Gradient checkpointing"
                      onChange={() =>
                        updateField(
                          "gradientCheckpointing",
                          !formState.gradientCheckpointing,
                        )
                      }
                    />
                    <ToggleRow
                      checked={formState.sequencePacking}
                      description="Applies a 15% activation discount as a compact packing proxy."
                      label="Sequence packing"
                      onChange={() =>
                        updateField("sequencePacking", !formState.sequencePacking)
                      }
                    />
                  </div>
                </AnimatedExpand>
              </div>
            </AnimatedExpand>
          </div>

          <div className="flex flex-col gap-4 border-t border-[var(--line)] pt-6">
            <button
              className="inline-flex w-full items-center justify-center rounded-full bg-[var(--ink)] px-5 py-3.5 text-sm font-medium text-white transition hover:translate-y-[-1px] hover:shadow-[var(--shadow-soft)] md:w-auto"
              type="submit"
            >
              Calculate VRAM
            </button>
            <p className="max-w-2xl text-sm leading-6 text-[var(--muted)]">
              Quick mental model: a dense 7B model in FP16 starts around 14 GB
              just for weights. Long context and training state are what push it
              beyond a 24 GB card.
            </p>
          </div>
        </form>
      </section>

      {result ? (
        <div className="space-y-4" ref={resultsRef}>
          {result.warnings.length > 0 ? (
            <section className={`${cardClassName} border-[rgba(154,79,67,0.18)]`}>
              <SectionTitle
                eyebrow="Warnings"
                title="This estimate uses one or more documented proxies"
              />
              <div className="mt-4 space-y-3">
                {result.warnings.map((warning) => (
                  <p
                    key={warning}
                    className="rounded-[1.35rem] bg-[var(--danger-soft)] px-4 py-3 text-sm leading-6 text-[var(--danger)]"
                  >
                    {warning}
                  </p>
                ))}
              </div>
            </section>
          ) : null}

          <section
            className={cx(
              cardClassName,
              result.fits
                ? "border-[rgba(30,105,82,0.18)]"
                : "border-[rgba(154,79,67,0.18)]",
            )}
          >
            <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(18rem,0.85fr)]">
              <div className="space-y-4">
                <span
                  className={cx(
                    "inline-flex rounded-full px-3 py-1 text-[0.72rem] mono",
                    result.fits
                      ? "bg-[var(--success-soft)] text-[var(--success)]"
                      : "bg-[var(--danger-soft)] text-[var(--danger)]",
                  )}
                >
                  {result.fits
                    ? "Fits on selected GPU"
                    : "Does not fit on selected GPU"}
                </span>
                <div>
                  <p className="text-sm text-[var(--muted)]">
                    {result.model.displayName} · {result.calculationProfile}
                  </p>
                  <h2 className="hero-title mt-2 text-3xl leading-tight md:text-4xl">
                    {formatGb(result.totalBytes)}
                  </h2>
                  <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted)]">
                    Against {result.gpu.displayName}, this leaves{" "}
                    {result.fits
                      ? `${formatGb(result.headroomBytes)} of headroom.`
                      : `${formatGb(result.deficitBytes)} of deficit.`}
                  </p>
                </div>
              </div>

              <div className="rounded-[1.6rem] bg-white/62 p-5">
                <p className="eyebrow text-[0.62rem] text-[var(--muted)]">
                  Quick read
                </p>
                <div className="mt-4 space-y-3">
                  <DetailRow
                    label="Selected GPU"
                    value={result.gpu.displayName}
                  />
                  <DetailRow label="Class" value={result.gpu.classType} />
                  <DetailRow
                    label="Bandwidth"
                    value={formatBandwidth(result.gpu.memoryBandwidthGbps)}
                  />
                  <DetailRow
                    label="Available VRAM"
                    value={`${result.gpu.vramGb.toFixed(0)} GB`}
                  />
                  <DetailRow
                    label="Estimated total"
                    value={formatGb(result.totalBytes)}
                  />
                  <DetailRow
                    label={result.fits ? "Headroom" : "Deficit"}
                    value={formatGb(result.fits ? result.headroomBytes : result.deficitBytes)}
                  />
                </div>
              </div>
            </div>
          </section>

          <section className={cardClassName}>
            <SectionTitle eyebrow="Breakdown" title="Where the memory goes" />
            <div className="mt-6 space-y-4">
              {result.breakdown.map((item) => (
                <div key={item.key} className="rounded-[1.6rem] bg-white/62 p-4">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <h3 className="text-sm font-medium text-[var(--ink)]">
                        {item.label}
                      </h3>
                      <p className="mt-1 text-sm leading-6 text-[var(--muted)]">
                        {item.note}
                      </p>
                    </div>
                    <p className="mono text-sm text-[var(--ink)]">
                      {formatGb(item.bytes)}
                    </p>
                  </div>
                  <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[rgba(30,36,42,0.08)]">
                    <div
                      className="h-full rounded-full bg-[var(--accent)]"
                      style={{
                        width: `${Math.max(6, (item.bytes / result.totalBytes) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-5 rounded-[1.6rem] bg-white/55 p-4 text-sm leading-7 text-[var(--muted)]">
              <p>Weights = parameter count × bytes per parameter.</p>
              {result.input.mode === "inference" ? (
                <p>KV cache grows with context length, layers, and batch size.</p>
              ) : (
                <p>Training adds activations, gradients, and optimizer memory on top of resident weights.</p>
              )}
              <p>Quantization mainly shrinks the frozen weight footprint; activations and runtime buffers stay closer to 16-bit math.</p>
            </div>

            <details className="mt-4 rounded-[1.6rem] border border-[var(--line)] bg-white/55 p-4">
              <summary className="cursor-pointer list-none text-sm font-medium text-[var(--ink)]">
                Show the substituted formulas
              </summary>
              <div className="mt-4 space-y-3">
                {result.math.map((line) => (
                  <div key={line.label} className="rounded-[1.35rem] bg-white/72 p-4">
                    <div className="flex flex-col gap-1">
                      <p className="text-sm font-medium text-[var(--ink)]">
                        {line.label}
                      </p>
                      <p className="mono text-xs leading-6 text-[var(--muted)]">
                        {line.symbolic}
                      </p>
                      <p className="mono text-sm leading-6 text-[var(--ink)]">
                        {line.substituted}
                      </p>
                      <p className="text-sm leading-6 text-[var(--muted)]">
                        {line.note}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </details>
          </section>

          <section className={cardClassName}>
            <SectionTitle eyebrow="Tips" title="What to change next" />
            <div className="mt-5 space-y-3">
              {result.tips.map((tip) => (
                <p
                  key={tip}
                  className="rounded-[1.35rem] bg-white/62 px-4 py-3 text-sm leading-6 text-[var(--muted)]"
                >
                  {tip}
                </p>
              ))}
            </div>
          </section>

          <section className={cardClassName}>
            <SectionTitle eyebrow="Model" title="Selected model" />
            <div className="mt-5 rounded-[1.75rem] bg-white/55 p-6 md:p-7">
              <h3 className="text-xl text-[var(--ink)]">
                {result.model.displayName}
              </h3>
              <p className="mt-3 max-w-4xl text-sm leading-7 text-[var(--muted)]">
                {result.model.shortDescription}
              </p>
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <div className="rounded-[1.4rem] bg-white/62 p-4">
                  <p className="eyebrow text-[0.62rem] text-[var(--muted)]">
                    Research highlight
                  </p>
                  <p className="mt-2 text-sm leading-7 text-[var(--ink)]">
                    {result.model.researchHighlight}
                  </p>
                </div>
                <div className="rounded-[1.4rem] bg-white/62 p-4">
                  <p className="eyebrow text-[0.62rem] text-[var(--muted)]">
                    Memory note
                  </p>
                  <p className="mt-2 text-sm leading-7 text-[var(--ink)]">
                    {result.model.memoryNote}
                  </p>
                </div>
              </div>
              {result.notes.length > 0 ? (
                <div className="mt-4 rounded-[1.4rem] bg-white/62 p-4">
                  <p className="eyebrow text-[0.62rem] text-[var(--muted)]">
                    Estimator notes
                  </p>
                  <div className="mt-2 space-y-2 text-sm leading-7 text-[var(--ink)]">
                    {result.notes.map((note) => (
                      <p key={note}>{note}</p>
                    ))}
                  </div>
                </div>
              ) : null}
              <div className="mt-6 grid gap-x-10 gap-y-4 md:grid-cols-2">
                <DetailRow
                  label="Architecture"
                  value={result.model.architectureType}
                />
                <DetailRow
                  label="Total params"
                  value={formatParamCount(result.model.totalParams)}
                />
                <DetailRow
                  label="Active params"
                  value={
                    result.model.activeParams
                      ? formatParamCount(result.model.activeParams)
                      : "Dense model"
                  }
                />
                <DetailRow label="Layers" value={result.model.numLayers.toString()} />
                <DetailRow
                  label="Hidden size"
                  value={formatInteger(result.model.hiddenSize)}
                />
                <DetailRow
                  label="Attention heads"
                  value={result.model.numAttentionHeads.toString()}
                />
                <DetailRow
                  label="KV heads"
                  value={(result.model.numKvHeads ?? result.model.numAttentionHeads).toString()}
                />
                <DetailRow
                  label="Context length"
                  value={formatInteger(result.model.contextLength)}
                />
                <DetailRow label="License" value={result.model.license} />
              </div>
              <a
                className="mt-5 inline-flex rounded-full border border-[var(--line)] px-4 py-2 text-sm text-[var(--ink)] transition hover:border-[var(--line-strong)]"
                href={result.model.sourceUrl}
                rel="noreferrer"
                target="_blank"
              >
                Open source link
              </a>
            </div>
          </section>

          <div className="px-2 pt-1 text-sm leading-7 text-[var(--muted)]">
            <p className="eyebrow text-[0.68rem] text-[var(--muted)]">
              Assumptions
            </p>
            <div className="mt-3 space-y-2">
              <p>
                The calculator works in raw bytes, displays decimal GB, and keeps
                conservative runtime headroom.
              </p>
            </div>
            <div className="mt-5 border-t border-[var(--line)] pt-4 text-center text-sm leading-6 text-[var(--ink)]">
              <a
                className="underline decoration-[var(--line-strong)] underline-offset-4 transition hover:text-[var(--accent)]"
                href="https://buymeacoffee.com/kishanvavdara"
                rel="noreferrer"
                target="_blank"
              >
                If you found this helpful, show us support here ☕
              </a>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}

function FieldLabel({ hint, label }: { hint?: string; label: string }) {
  return (
    <div className="space-y-1">
      <label className="eyebrow text-[0.68rem] text-[var(--muted)]">{label}</label>
      {hint ? <p className="text-sm leading-6 text-[var(--muted)]">{hint}</p> : null}
    </div>
  );
}

function SelectField({
  disabled = false,
  onChange,
  options,
  value,
}: {
  disabled?: boolean;
  onChange: (value: string) => void;
  options: Array<{ label: string; value: string }>;
  value: string;
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const selectedOption = options.find((option) => option.value === value) ?? options[0];
  const isOpen = !disabled && open;

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleEscape);
    };
  }, []);

  return (
    <div className={cx("relative", isOpen && "z-40")} ref={containerRef}>
      <button
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        className={cx(
          fieldClassName,
          "flex items-center justify-between gap-4 text-left",
          disabled && "cursor-not-allowed opacity-55",
        )}
        disabled={disabled}
        onClick={() => {
          if (disabled) {
            return;
          }

          setOpen((current) => !current);
        }}
        type="button"
      >
        <span className="truncate">{selectedOption?.label}</span>
        <ChevronIcon open={isOpen} />
      </button>

      <div
        className={cx(
          "absolute left-0 right-0 top-[calc(100%+0.55rem)] z-50 origin-top overflow-hidden rounded-[1.35rem] border border-[var(--line)] bg-[rgba(255,252,247,0.96)] p-2 shadow-[0_24px_60px_rgba(30,36,42,0.12)] backdrop-blur-xl transition-[opacity,transform] duration-250 ease-[cubic-bezier(0.22,1,0.36,1)] will-change-transform",
          isOpen
            ? "pointer-events-auto translate-y-0 scale-100 opacity-100"
            : "pointer-events-none -translate-y-1.5 scale-[0.985] opacity-0",
        )}
      >
        <div
          className={cx(
            "max-h-72 overflow-y-auto overscroll-contain pr-1 transition-[opacity,transform] duration-250 ease-[cubic-bezier(0.22,1,0.36,1)]",
            isOpen ? "translate-y-0 opacity-100" : "-translate-y-1 opacity-0",
          )}
          role="listbox"
        >
          {options.map((option) => (
            <button
              aria-selected={option.value === value}
              className={cx(
                "flex min-h-11 w-full items-center justify-between rounded-[1rem] px-3.5 py-3 text-sm text-[var(--ink)] transition",
                option.value === value
                  ? "bg-[rgba(30,36,42,0.08)]"
                  : "hover:bg-[rgba(30,36,42,0.05)]",
              )}
              key={option.value}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
              role="option"
              type="button"
            >
              <span className="truncate">{option.label}</span>
              {option.value === value ? (
                <span className="mono text-[0.72rem] text-[var(--muted)]">set</span>
              ) : null}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function AnimatedExpand({
  className,
  children,
  open,
}: {
  className?: string;
  children: ReactNode;
  open: boolean;
}) {
  return (
    <div
      aria-hidden={!open}
      className={cx(
        "grid transition-[grid-template-rows,opacity] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)]",
        className,
        open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
      )}
    >
      <div className="overflow-hidden">
        <div
          className={cx(
            "transition-[opacity,transform] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)]",
            open ? "translate-y-0 opacity-100" : "-translate-y-2 opacity-0",
          )}
        >
          {children}
        </div>
      </div>
    </div>
  );
}

function SegmentedField({
  onChange,
  options,
  value,
}: {
  onChange: (value: string) => void;
  options: Array<{ disabled?: boolean; label: string; value: string }>;
  value: string;
}) {
  const activeIndex = Math.max(
    options.findIndex((option) => option.value === value),
    0,
  );

  return (
    <div
      className="relative inline-grid w-full rounded-full border border-[var(--line)] bg-white/60 p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.28)] md:max-w-[22rem]"
      style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
    >
      <span
        aria-hidden="true"
        className="absolute bottom-1 left-1 top-1 rounded-full bg-[var(--ink)] shadow-[var(--shadow-soft)] transition-transform duration-250 ease-out"
        style={{
          transform: `translateX(${activeIndex * 100}%)`,
          width: `calc((100% - 0.5rem) / ${options.length})`,
        }}
      />

      {options.map((option) => (
        <button
          key={option.value}
          className={cx(
            "relative z-10 rounded-full px-4 py-2.5 text-sm transition-colors duration-200",
            option.disabled
              ? "cursor-not-allowed text-[var(--muted)]/55"
              : value === option.value
                ? "text-white"
                : "text-[var(--muted)] hover:text-[var(--ink)]",
          )}
          disabled={option.disabled}
          onClick={() => onChange(option.value)}
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function ToggleRow({
  checked,
  description,
  label,
  onChange,
}: {
  checked: boolean;
  description: string;
  label: string;
  onChange: () => void;
}) {
  return (
    <button
      className="flex items-center justify-between gap-4 rounded-[1.35rem] border border-[var(--line)] bg-white/62 px-4 py-3 text-left"
      onClick={onChange}
      type="button"
    >
      <div>
        <p className="text-sm font-medium text-[var(--ink)]">{label}</p>
        <p className="text-sm leading-6 text-[var(--muted)]">{description}</p>
      </div>
      <span
        className={cx(
          "relative h-7 w-12 rounded-full transition",
          checked ? "bg-[var(--accent)]" : "bg-[rgba(30,36,42,0.12)]",
        )}
      >
        <span
          className={cx(
            "absolute top-1 h-5 w-5 rounded-full bg-white transition",
            checked ? "left-6" : "left-1",
          )}
        />
      </span>
    </button>
  );
}

function SectionTitle({
  eyebrow,
  title,
}: {
  eyebrow: string;
  title: string;
}) {
  return (
    <div className="space-y-2">
      <p className="eyebrow text-[0.68rem] text-[var(--muted)]">{eyebrow}</p>
      <h2 className="hero-title text-[2rem] leading-none text-[var(--ink)]">
        {title}
      </h2>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-[var(--line)] pb-3 last:border-b-0 last:pb-0">
      <p className="text-sm leading-6 text-[var(--muted)]">{label}</p>
      <p className="text-right text-sm leading-6 text-[var(--ink)]">{value}</p>
    </div>
  );
}

function formatModelAtGlance(model: ModelSpec) {
  const kvHeads = model.numKvHeads ?? model.numAttentionHeads;
  const parameterLine = model.activeParams
    ? `${formatParamCount(model.totalParams)} total • ${formatParamCount(model.activeParams)} active`
    : `${formatParamCount(model.totalParams)} dense`;

  return `${parameterLine} • ${formatInteger(model.contextLength)} context • ${kvHeads} KV heads`;
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      aria-hidden="true"
      className={cx("h-4 w-4 shrink-0 text-[var(--muted)] transition", open && "rotate-180")}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M6 9L12 15L18 9"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}
