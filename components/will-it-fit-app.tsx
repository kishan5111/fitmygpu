"use client";

import {
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { DetailRow } from "@/components/detail-row";
import { HomeExplore } from "@/components/home-explore";
import { ModelSpecGrid } from "@/components/model-spec-grid";
import { gpus } from "@/data/gpus";
import { models } from "@/data/models";
import {
  formatBandwidth,
  formatDtype,
  formatGb,
  formatInteger,
} from "@/lib/format";
import { estimateVram, estimateVramForModel } from "@/lib/estimator";
import type { HfImportResult } from "@/lib/hf-import";
import {
  applyModelConstraints,
  getAllowedLoadDtypes,
  getCompatibleInferenceProfile,
  getCompatibleInferenceProfiles,
  getInferenceProfileSourceUrl,
} from "@/lib/model-constraints";
import {
  formatModelAtGlance,
  isMultimodalModel,
} from "@/lib/model-display";
import {
  kvCacheDtypeOptions,
  runtimeOptions,
  runtimeSupportsKvCacheDtype,
} from "@/lib/runtime";
import { normalizeEstimateInput, serializeEstimateInput } from "@/lib/query-state";
import { dtypeOptions } from "@/lib/constants";
import type {
  EstimateInput,
  EstimateResult,
  ModelSpec,
} from "@/lib/types";

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
  const [hfModelUrl, setHfModelUrl] = useState("");
  const [hfImportMessage, setHfImportMessage] = useState<string | null>(null);
  const [hfImportPending, setHfImportPending] = useState(false);
  const [importedModel, setImportedModel] = useState<ModelSpec | null>(null);
  const [formState, setFormState] = useState(() =>
    normalizeEstimateInput(initialInput),
  );
  const [result, setResult] = useState<EstimateResult | null>(initialResult);
  const localSelectedModel = models.find((model) => model.id === formState.modelId);
  const selectedModel =
    importedModel?.id === formState.modelId ? importedModel : localSelectedModel ?? models[0];
  const compatibleInferenceProfiles = getCompatibleInferenceProfiles(
    selectedModel,
    formState.runtimeId,
  );
  const selectedInferenceProfile = getCompatibleInferenceProfile(
    selectedModel,
    formState.runtimeId,
    formState.inferenceProfileId,
    formState.dtype,
  );
  const selectedRuntime = runtimeOptions.find(
    (runtime) => runtime.id === formState.runtimeId,
  ) ?? runtimeOptions[0];
  const showsServingControls = formState.runtimeId !== "transformers";
  const showsKvCacheDtype = runtimeSupportsKvCacheDtype(formState.runtimeId);
  const showsAdvancedOptions = showsServingControls || formState.gpuId === "custom";
  const hasCompatibleProfiles = compatibleInferenceProfiles.length > 0;
  const modelOptions = [
    ...(importedModel
      ? [
          {
            label: `${importedModel.displayName} (HF estimate)`,
            value: importedModel.id,
          },
        ]
      : []),
    ...[...models].sort(compareModelDropdownOrder).map((model) => ({
      label: model.displayName,
      value: model.id,
    })),
  ];
  const runtimeFieldOptions = runtimeOptions.map((runtime) => ({
    label: runtime.label,
    value: runtime.id,
  }));
  const kvCacheDtypeFieldOptions = kvCacheDtypeOptions.map((option) => ({
    label: option.label,
    value: option.value,
  }));
  const inferenceProfileOptions = compatibleInferenceProfiles.map((profile) => ({
    label: profile.label,
    value: profile.id,
  }));
  const loadDtypeOptions = selectedInferenceProfile
    ? dtypeOptions
        .filter((option) => getAllowedLoadDtypes(selectedInferenceProfile).includes(option.value))
        .map((option) => ({ label: option.label, value: option.value }))
    : [];
  const gpuFieldOptions = gpus.map((gpu) => ({
    label: gpu.displayName,
    value: gpu.id,
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
    setFormState((current) => ({ ...current, [key]: value }));
  }

  function handleModelChange(modelId: string) {
    const nextModel = importedModel?.id === modelId ? importedModel : undefined;

    if (!nextModel) {
      setImportedModel(null);
    }

    setFormState((current) =>
      applyConstraintsForModel({ ...current, modelId }, nextModel),
    );
  }

  function handleInferenceProfileChange(profileId: string) {
    const profile = getCompatibleInferenceProfile(
      selectedModel,
      formState.runtimeId,
      profileId,
      formState.dtype,
    );

    if (!profile) {
      return;
    }

    setFormState((current) =>
      applyConstraintsForModel(
        {
          ...current,
          inferenceProfileId: profile.id,
        },
        selectedModel,
      ),
    );
  }

  function handleRuntimeChange(runtimeId: string) {
    setFormState((current) =>
      applyConstraintsForModel(
        {
          ...current,
          runtimeId: runtimeId as EstimateInput["runtimeId"],
        },
        selectedModel,
      ),
    );
  }

  function handleNumberChange<
    K extends
      | "contextLength"
      | "batchSize"
      | "gpuCount"
      | "customVramGb"
      | "vllmGpuUtilization",
  >(
    key: K,
    rawValue: string,
  ) {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) {
      return;
    }

    updateField(key, parsed as EstimateInput[K]);
  }

  async function handleHfImport() {
    setHfImportPending(true);
    setHfImportMessage(null);

    try {
      const response = await fetch(`/api/hf-model?url=${encodeURIComponent(hfModelUrl)}`);
      const payload = (await response.json()) as HfImportResult | { error?: string };

      if (!response.ok || "error" in payload) {
        const errorMessage = "error" in payload ? payload.error : undefined;

        setHfImportMessage(errorMessage ?? "Could not import that Hugging Face model.");
        return;
      }

      const importResult = payload as HfImportResult;

      if (importResult.status === "known") {
        setImportedModel(null);
        setFormState((current) =>
          applyModelConstraints({ ...current, modelId: importResult.modelId }),
        );
        setHfImportMessage(`Matched ${importResult.repoId} to a verified registry entry.`);
        return;
      }

      const estimatedModel = importResult.model;
      const firstProfile = estimatedModel.inferenceProfiles[0];
      setImportedModel(estimatedModel);
      setFormState((current) =>
        applyConstraintsForModel(
          {
            ...current,
            modelId: estimatedModel.id,
            dtype: firstProfile?.effectiveDtype ?? current.dtype,
            inferenceProfileId: firstProfile?.id ?? "",
          },
          estimatedModel,
        ),
      );
      setHfImportMessage(`Imported ${importResult.repoId} as an estimated config profile.`);
    } catch {
      setHfImportMessage("Could not reach the Hugging Face import endpoint.");
    } finally {
      setHfImportPending(false);
    }
  }

  function handleCalculate() {
    if (!hasCompatibleProfiles) {
      return;
    }

    const normalized = applyConstraintsForModel(
      normalizeEstimateInput(formState),
      selectedModel,
    );
    const nextResult =
      importedModel?.id === normalized.modelId
        ? estimateVramForModel(normalized, importedModel)
        : estimateVram(normalized);
    const params = serializeEstimateInput(normalized);

    hasCalculatedRef.current = true;
    setFormState(normalized);
    setResult(nextResult);

    startTransition(() => {
      router.replace(`/?${params.toString()}`, { scroll: false });
    });
  }

  return (
    <>
      <main className="mx-auto flex min-h-screen max-w-[74rem] flex-col gap-8 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
        <section className={cardClassName}>
        <div className="mb-8 flex flex-col gap-4">
          <div className="max-w-3xl space-y-3">
            <h1 className="hero-title text-4xl leading-none text-[var(--ink)] md:text-[4.2rem]">
              Will It Fit?
            </h1>
            <p className="max-w-2xl text-base leading-7 text-[var(--muted)] md:text-lg">
              Estimate text inference VRAM across Transformers and vLLM with a
              compact, explainable breakdown.
            </p>
            <p className="max-w-2xl text-sm leading-6 text-[var(--muted)] md:text-base">
              Model pages and release notes are included here too.
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
          <div className="rounded-[1.6rem] border border-[var(--line)] bg-white/50 p-4">
            <FieldLabel label="Hugging Face URL" />
            <div className="mt-3 flex flex-col gap-3 md:flex-row">
              <input
                className={fieldClassName}
                onChange={(event) => setHfModelUrl(event.target.value)}
                placeholder="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"
                type="text"
                value={hfModelUrl}
              />
              <button
                className="inline-flex items-center justify-center rounded-full border border-[var(--line)] px-5 py-3 text-sm font-medium text-[var(--ink)] transition hover:border-[var(--line-strong)] disabled:cursor-not-allowed disabled:opacity-50"
                disabled={hfImportPending || hfModelUrl.trim().length === 0}
                onClick={handleHfImport}
                type="button"
              >
                {hfImportPending ? "Importing" : "Import"}
              </button>
            </div>
            {hfImportMessage ? (
              <p className="mt-3 text-sm leading-6 text-[var(--muted)]">
                {hfImportMessage}
              </p>
            ) : null}
          </div>

          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
            <div className="space-y-3">
              <FieldLabel label="Model" />
              <SelectField
                onChange={handleModelChange}
                options={modelOptions}
                value={formState.modelId}
              />
            </div>

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

            <div className="space-y-3">
              <FieldLabel label="Runtime" />
              <SelectField
                onChange={handleRuntimeChange}
                options={runtimeFieldOptions}
                value={formState.runtimeId}
              />
            </div>

            <p className="text-sm leading-6 text-[var(--muted)] md:col-span-2">
              {formatModelAtGlance(selectedModel)}
            </p>

            {selectedModel.fixedDtype ? (
              <p className="text-sm leading-6 text-[var(--muted)] md:col-span-2">
                This checkpoint is fixed to {formatDtype(selectedModel.fixedDtype)} in
                the current release.
              </p>
            ) : null}

            {hasCompatibleProfiles && selectedInferenceProfile ? (
              null
            ) : (
              <p className="rounded-[1.35rem] bg-[var(--danger-soft)] px-4 py-3 text-sm leading-6 text-[var(--danger)] md:col-span-2 xl:col-span-4">
                {selectedModel.displayName} does not have a compatible checkpoint profile
                for {selectedRuntime.label} in v1 yet. Pick a different runtime or model.
              </p>
            )}

            {isMultimodalModel(selectedModel) ? (
              <div className="rounded-[1.5rem] border border-[var(--line)] bg-white/55 px-4 py-3 md:col-span-2 xl:col-span-4">
                <span className="inline-flex rounded-full bg-[var(--accent-soft)] px-3 py-1 text-[0.72rem] mono text-[var(--accent)]">
                  Text-only estimate
                </span>
                <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                  This multimodal checkpoint is estimated only for text requests in
                  v1. Resident vision and projector weights stay counted, but image
                  and video token memory is excluded.
                </p>
              </div>
            ) : null}

          </div>

          {showsAdvancedOptions ? (
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
                    {showsServingControls ? "Serving options" : "Custom GPU"}
                  </p>
                  <p className="mt-0.5 text-[0.82rem] leading-5 text-[var(--muted)]">
                    {showsServingControls
                      ? "Context length and concurrent requests. Extra fields appear when they are relevant."
                      : "Set nominal VRAM for a custom card."}
                  </p>
                </div>
                <span className="flex h-8 w-8 items-center justify-center rounded-full border border-[var(--line)] bg-white/72">
                  <ChevronIcon open={advancedOpen} />
                </span>
              </button>

              <AnimatedExpand open={advancedOpen}>
                <div className="grid gap-5 pt-5 md:grid-cols-2">
                  {showsServingControls ? (
                    <div className="space-y-3">
                      <FieldLabel label="Checkpoint profile" />
                      {hasCompatibleProfiles && selectedInferenceProfile ? (
                        <SelectField
                          disabled={Boolean(selectedModel.fixedDtype)}
                          onChange={handleInferenceProfileChange}
                          options={inferenceProfileOptions}
                          value={selectedInferenceProfile.id}
                        />
                      ) : (
                        <div
                          className={`${fieldClassName} cursor-not-allowed text-[var(--muted)] opacity-70`}
                        >
                          No compatible profile in v1
                        </div>
                      )}
                    </div>
                  ) : null}

                  {showsServingControls && selectedInferenceProfile ? (
                    <div className="space-y-3">
                      <FieldLabel label="Load dtype" />
                      <SelectField
                        onChange={(value) =>
                          updateField("dtype", value as EstimateInput["dtype"])
                        }
                        options={loadDtypeOptions}
                        value={formState.dtype}
                      />
                    </div>
                  ) : null}

                  {showsKvCacheDtype ? (
                    <div className="space-y-3">
                      <FieldLabel label="KV cache dtype" />
                      <SelectField
                        onChange={(value) =>
                          updateField("kvCacheDtype", value as EstimateInput["kvCacheDtype"])
                        }
                        options={kvCacheDtypeFieldOptions}
                        value={formState.kvCacheDtype}
                      />
                    </div>
                  ) : null}

                  {showsServingControls ? (
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
                  ) : null}

                  {showsServingControls ? (
                    <div className="space-y-3">
                      <FieldLabel label="Concurrent requests" />
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
                  ) : null}

                  {formState.runtimeId === "vllm" ? (
                    <div className="space-y-3">
                      <FieldLabel label="GPU count" />
                      <input
                        className={fieldClassName}
                        max={72}
                        min={1}
                        onChange={(event) =>
                          handleNumberChange("gpuCount", event.target.value)
                        }
                        step={1}
                        type="number"
                        value={formState.gpuCount}
                      />
                    </div>
                  ) : null}

                  {formState.runtimeId === "vllm" ? (
                    <div className="space-y-3">
                      <FieldLabel label="GPU memory utilization" />
                      <input
                        className={fieldClassName}
                        max={0.99}
                        min={0.5}
                        onChange={(event) =>
                          handleNumberChange("vllmGpuUtilization", event.target.value)
                        }
                        step={0.01}
                        type="number"
                        value={formState.vllmGpuUtilization}
                      />
                    </div>
                  ) : null}

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
                </div>
              </AnimatedExpand>
            </div>
          ) : null}

          <div className="flex flex-col gap-4 border-t border-[var(--line)] pt-6">
            <button
              className="inline-flex w-full items-center justify-center rounded-full bg-[var(--ink)] px-5 py-3.5 text-sm font-medium text-white transition enabled:hover:translate-y-[-1px] enabled:hover:shadow-[var(--shadow-soft)] disabled:cursor-not-allowed disabled:opacity-45 md:w-auto"
              disabled={!hasCompatibleProfiles}
              type="submit"
            >
              Calculate VRAM
            </button>
            <p className="text-sm leading-6 text-[var(--muted)]">
              vLLM estimates use the selected GPU memory utilization. Transformers
              stays a fixed 4K single-request baseline.
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
                {isMultimodalModel(result.model) ? (
                  <span className="inline-flex rounded-full bg-[var(--accent-soft)] px-3 py-1 text-[0.72rem] mono text-[var(--accent)]">
                    Text-only estimate
                  </span>
                ) : null}
                <div>
                  <p className="text-sm text-[var(--muted)]">
                    {result.model.displayName} · {result.runtime.label} · {result.calculationProfile}
                  </p>
                  <p className="mt-2 text-sm text-[var(--muted)]">
                    {result.fitMetricLabel}
                  </p>
                  <h2 className="hero-title mt-2 text-3xl leading-tight md:text-4xl">
                    {formatGb(result.requiredGpuBytes)}
                  </h2>
                  <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted)]">
                    Core estimate: {formatGb(result.totalBytes)}.{" "}
                    Against {formatGpuSelection(result)}, this leaves{" "}
                    {result.fits
                      ? `${formatGb(result.headroomBytes)} of headroom.`
                      : `${formatGb(result.deficitBytes)} of deficit.`}
                  </p>
                  {result.maxConcurrencyAtContext !== undefined ? (
                    <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted)]">
                      At {formatInteger(result.effectiveContextLength)} tokens, the
                      estimated max concurrency is{" "}
                      {formatInteger(result.maxConcurrencyAtContext)} concurrent
                      {result.maxConcurrencyAtContext === 1 ? " request" : " requests"}.
                    </p>
                  ) : null}
                  <div className="mt-4 flex flex-wrap gap-3">
                    {isRegistryModel(result.model) ? (
                      <Link
                        className="inline-flex rounded-full border border-[var(--line)] px-4 py-2 text-sm text-[var(--ink)] transition hover:border-[var(--line-strong)]"
                        href={getModelNotesHref(result.input)}
                      >
                        About model
                      </Link>
                    ) : (
                      <a
                        className="inline-flex rounded-full border border-[var(--line)] px-4 py-2 text-sm text-[var(--ink)] transition hover:border-[var(--line-strong)]"
                        href={result.model.sourceUrl}
                        rel="noreferrer"
                        target="_blank"
                      >
                        Open on Hugging Face
                      </a>
                    )}
                  </div>
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
                  <DetailRow label="Runtime" value={result.runtime.label} />
                  {result.input.runtimeId !== "transformers" ? (
                    <DetailRow
                      label="Load dtype"
                      value={formatDtype(result.effectiveDtype)}
                    />
                  ) : null}
                  {result.input.runtimeId === "vllm" ? (
                    <DetailRow
                      label="GPU utilization"
                      value={result.input.vllmGpuUtilization.toString()}
                    />
                  ) : null}
                  {result.input.runtimeId === "vllm" ? (
                    <DetailRow
                      label="GPU count"
                      value={formatInteger(result.input.gpuCount)}
                    />
                  ) : null}
                  {result.input.runtimeId === "transformers" ? (
                    <DetailRow label="Serving mode" value="Single request baseline" />
                  ) : null}
                  {runtimeSupportsKvCacheDtype(result.input.runtimeId) ? (
                    <DetailRow
                      label="KV cache dtype"
                      value={formatDtype(result.input.kvCacheDtype)}
                    />
                  ) : null}
                  {result.input.runtimeId !== "transformers" ? (
                    <DetailRow
                      label="Context length"
                      value={formatInteger(result.effectiveContextLength)}
                    />
                  ) : null}
                  {result.input.runtimeId !== "transformers" ? (
                    <DetailRow
                      label="Current concurrency"
                      value={formatInteger(result.input.batchSize)}
                    />
                  ) : null}
                  {result.maxConcurrencyAtContext !== undefined ? (
                    <DetailRow
                      label="Max concurrency @ context"
                      value={formatInteger(result.maxConcurrencyAtContext)}
                    />
                  ) : null}
                  <DetailRow label="Class" value={result.gpu.classType} />
                  <DetailRow
                    label="Bandwidth"
                    value={formatBandwidth(result.gpu.memoryBandwidthGbps)}
                  />
                  <DetailRow
                    label="Nominal VRAM"
                    value={`${result.gpu.vramGb.toFixed(0)} GB`}
                  />
                  {result.input.gpuCount > 1 ? (
                    <DetailRow
                      label="Aggregate VRAM"
                      value={formatGb(result.gpuBytes)}
                    />
                  ) : null}
                  <DetailRow
                    label="Core estimate"
                    value={formatGb(result.totalBytes)}
                  />
                  <DetailRow
                    label={result.fitMetricLabel}
                    value={formatGb(result.requiredGpuBytes)}
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
            <SectionTitle eyebrow="GPU compare" title="What fits this setup" />
            <div className="mt-5 overflow-hidden rounded-[1.6rem] border border-[var(--line)] bg-white/55">
              <div className="grid grid-cols-[minmax(9rem,1.2fr)_0.6fr_0.7fr_0.7fr] gap-3 border-b border-[var(--line)] px-4 py-3 text-[0.72rem] mono text-[var(--muted)]">
                <span>GPU</span>
                <span>Status</span>
                <span>Headroom</span>
                <span>Max conc.</span>
              </div>
              {getGpuComparisonResults(result).map((comparison) => (
                <div
                  className="grid grid-cols-[minmax(9rem,1.2fr)_0.6fr_0.7fr_0.7fr] gap-3 border-b border-[var(--line)] px-4 py-3 text-sm last:border-b-0"
                  key={comparison.gpu.id}
                >
                  <span className="text-[var(--ink)]">
                    {comparison.result.input.gpuCount > 1
                      ? `${formatInteger(comparison.result.input.gpuCount)} × ${comparison.gpu.displayName}`
                      : comparison.gpu.displayName}
                  </span>
                  <span
                    className={comparison.result.fits ? "text-[var(--success)]" : "text-[var(--danger)]"}
                  >
                    {comparison.result.fits ? "Fits" : "OOM"}
                  </span>
                  <span className="text-[var(--muted)]">
                    {comparison.result.fits
                      ? formatGb(comparison.result.headroomBytes)
                      : `-${formatGb(comparison.result.deficitBytes)}`}
                  </span>
                  <span className="text-[var(--muted)]">
                    {comparison.result.maxConcurrencyAtContext === undefined
                      ? "-"
                      : formatInteger(comparison.result.maxConcurrencyAtContext)}
                  </span>
                </div>
              ))}
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
                <p>
                  {result.input.runtimeId === "transformers"
                    ? "Transformers is pinned to a fixed 4K single-request baseline in this release."
                    : "KV cache grows with context length, KV-bearing layers, concurrent requests, and the selected KV cache dtype."}
                </p>
              ) : (
                <p>Training adds activations, gradients, and optimizer memory on top of resident weights.</p>
              )}
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
            <SectionTitle eyebrow="Model" title="Selected model" />
            <div className="mt-5 rounded-[1.75rem] bg-white/55 p-6 md:p-7">
              <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div>
                  <h3 className="text-xl text-[var(--ink)]">{result.model.displayName}</h3>
                  <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                    {formatModelAtGlance(result.model)}
                  </p>
                </div>
                {isRegistryModel(result.model) ? (
                  <Link
                    className="inline-flex rounded-full border border-[var(--line)] px-4 py-2 text-sm text-[var(--ink)] transition hover:border-[var(--line-strong)]"
                    href={getModelNotesHref(result.input)}
                  >
                    About model
                  </Link>
                ) : (
                  <a
                    className="inline-flex rounded-full border border-[var(--line)] px-4 py-2 text-sm text-[var(--ink)] transition hover:border-[var(--line-strong)]"
                    href={result.model.sourceUrl}
                    rel="noreferrer"
                    target="_blank"
                  >
                    Open on Hugging Face
                  </a>
                )}
              </div>
              <ModelSpecGrid className="mt-6" model={result.model} />
              <div className="mt-5 flex flex-wrap gap-3">
                <a
                  className="inline-flex rounded-full border border-[var(--line)] px-4 py-2 text-sm text-[var(--ink)] transition hover:border-[var(--line-strong)]"
                  href={getInferenceProfileSourceUrl(
                    result.model,
                    result.effectiveInferenceProfileId,
                    result.input.runtimeId,
                    result.effectiveDtype,
                  )}
                  rel="noreferrer"
                  target="_blank"
                >
                  Open selected checkpoint
                </a>
              </div>
            </div>
          </section>

          <div className="px-2 pt-1 text-sm leading-7 text-[var(--muted)]">
            <div className="border-t border-[var(--line)] pt-4 text-center text-sm leading-6 text-[var(--ink)]">
              <a
                className="underline decoration-[var(--line-strong)] underline-offset-4 transition hover:text-[var(--accent)]"
                href="https://buymeacoffee.com/kishanvavdara"
                rel="noreferrer"
                target="_blank"
              >
                If you found this useful, support the project here ☕
              </a>
            </div>
          </div>
          </div>
        ) : null}
      </main>
      {result ? null : <HomeExplore />}
    </>
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
        <span className="min-w-0 flex-1 truncate" title={selectedOption?.label}>
          {selectedOption?.label}
        </span>
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
                "flex min-h-11 w-full items-start justify-between gap-3 rounded-[1rem] px-3.5 py-3 text-left text-sm text-[var(--ink)] transition",
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
              <span className="flex-1 whitespace-normal break-words leading-5">
                {option.label}
              </span>
              {option.value === value ? (
                <span className="mono pt-0.5 text-[0.72rem] text-[var(--muted)]">set</span>
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

function compareModelDropdownOrder(left: ModelSpec, right: ModelSpec) {
  const familyRankDiff = getModelFamilyRank(right) - getModelFamilyRank(left);

  if (familyRankDiff !== 0) {
    return familyRankDiff;
  }

  const versionDiff = getModelVersionScore(right.displayName) - getModelVersionScore(left.displayName);

  if (versionDiff !== 0) {
    return versionDiff;
  }

  const sizeDiff = getModelSizeScore(right.displayName) - getModelSizeScore(left.displayName);

  if (sizeDiff !== 0) {
    return sizeDiff;
  }

  return left.displayName.localeCompare(right.displayName);
}

function getModelFamilyRank(model: ModelSpec) {
  const familyRank: Record<string, number> = {
    "GPT-OSS": 90,
    Qwen: 80,
    Nemotron: 70,
    Llama: 60,
    Phi: 50,
    Gemma: 40,
    Mistral: 30,
    Mixtral: 20,
  };

  return familyRank[model.family] ?? 0;
}

function getModelVersionScore(displayName: string) {
  const match = displayName.match(/(\d+(?:\.\d+)?)/);

  return match ? Number.parseFloat(match[1]) : 0;
}

function getModelSizeScore(displayName: string) {
  const match = displayName.match(/(\d+(?:\.\d+)?)B/i);

  return match ? Number.parseFloat(match[1]) : 0;
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

function getModelNotesHref(input: EstimateInput) {
  return `/models/${input.modelId}?${serializeEstimateInput(input).toString()}`;
}

function applyConstraintsForModel(input: EstimateInput, model?: ModelSpec): EstimateInput {
  if (!model || isRegistryModel(model)) {
    return applyModelConstraints(input);
  }

  const profile =
    getCompatibleInferenceProfile(
      model,
      input.runtimeId,
      input.inferenceProfileId,
      input.dtype,
    ) ?? model.inferenceProfiles[0];

  return {
    ...input,
    mode: "inference" as const,
    contextLength: input.runtimeId === "transformers" ? 4096 : input.contextLength,
    batchSize: input.runtimeId === "transformers" ? 1 : input.batchSize,
    gpuCount: input.runtimeId === "transformers" ? 1 : input.gpuCount,
    kvCacheDtype: runtimeSupportsKvCacheDtype(input.runtimeId)
      ? input.kvCacheDtype
      : "bf16" as const,
    dtype:
      profile && getAllowedLoadDtypes(profile).includes(input.dtype)
        ? input.dtype
        : profile?.effectiveDtype ?? input.dtype,
    inferenceProfileId: profile?.id ?? "",
  };
}

function getGpuComparisonResults(result: EstimateResult) {
  return gpus
    .filter((gpu) => gpu.id !== "custom")
    .map((gpu) => ({
      gpu,
      result: estimateVramForModel(
        {
          ...result.input,
          gpuId: gpu.id,
        },
        result.model,
      ),
    }))
    .sort((left, right) => {
      if (left.result.fits !== right.result.fits) {
        return left.result.fits ? -1 : 1;
      }

      if (left.result.fits) {
        return right.result.headroomBytes - left.result.headroomBytes;
      }

      return left.result.deficitBytes - right.result.deficitBytes;
    });
}

function formatGpuSelection(result: EstimateResult) {
  if (result.input.gpuCount <= 1) {
    return result.gpu.displayName;
  }

  return `${formatInteger(result.input.gpuCount)} × ${result.gpu.displayName}`;
}

function isRegistryModel(model: ModelSpec) {
  return models.some((entry) => entry.id === model.id);
}
