import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { DetailRow } from "@/components/detail-row";
import { ModelSpecGrid } from "@/components/model-spec-grid";
import { models } from "@/data/models";
import { canEstimateInput, estimateVram } from "@/lib/estimator";
import { formatDtype, formatGb, formatInteger } from "@/lib/format";
import {
  applyModelConstraints,
  getCompatibleInferenceProfile,
} from "@/lib/model-constraints";
import {
  formatModelAtGlance,
  getInferenceProfileRuntimeLabels,
  getProfileConfidenceLabel,
  isMultimodalModel,
} from "@/lib/model-display";
import { runtimeSupportsKvCacheDtype } from "@/lib/runtime";
import {
  hasMeaningfulSearchParams,
  parseSearchParams,
  serializeEstimateInput,
} from "@/lib/query-state";

type PageProps = {
  params: Promise<{ modelId: string }>;
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};

export function generateStaticParams() {
  return models.map((model) => ({ modelId: model.id }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { modelId } = await params;
  const model = models.find((entry) => entry.id === modelId);

  if (!model) {
    return {
      title: "Model notes | FitMyGPU",
    };
  }

  return {
    title: `${model.displayName} | Model notes | FitMyGPU`,
    description: model.shortDescription,
  };
}

export default async function ModelNotesPage({ params, searchParams }: PageProps) {
  const { modelId } = await params;
  const model = models.find((entry) => entry.id === modelId);

  if (!model) {
    notFound();
  }

  const resolvedSearchParams = (await searchParams) ?? {};
  const hasContext = hasMeaningfulSearchParams(resolvedSearchParams);
  const contextInput = applyModelConstraints({
    ...parseSearchParams(resolvedSearchParams),
    modelId: model.id,
  });
  const contextResult =
    hasContext && canEstimateInput(contextInput) ? estimateVram(contextInput) : null;
  const selectedProfile = getCompatibleInferenceProfile(
    model,
    contextResult?.input.runtimeId ?? contextInput.runtimeId,
    contextResult?.effectiveInferenceProfileId ?? contextInput.inferenceProfileId,
    contextResult?.effectiveDtype ?? contextInput.dtype,
  );
  const backHref = hasContext
    ? `/?${serializeEstimateInput(contextResult?.input ?? contextInput).toString()}`
    : `/?model=${model.id}`;
  const modelLinks = [
    model.sourceUrl,
    ...model.inferenceProfiles.map((profile) => profile.sourceUrl),
  ].filter(Boolean);
  const sourceLinks = Array.from(new Set(modelLinks));

  return (
    <main className="mx-auto flex min-h-screen max-w-[74rem] flex-col gap-6 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <Link
          className="eyebrow inline-flex text-[0.68rem] text-[var(--muted)] transition hover:text-[var(--accent)]"
          href={backHref}
        >
          Back to calculator
        </Link>
        <div className="mt-4 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl space-y-3">
            <p className="eyebrow text-[0.68rem] text-[var(--muted)]">Model notes</p>
            <h1 className="hero-title text-4xl leading-none text-[var(--ink)] md:text-[4rem]">
              {model.displayName}
            </h1>
            <p className="max-w-3xl text-base leading-7 text-[var(--muted)] md:text-lg">
              {model.shortDescription}
            </p>
            <p className="text-sm leading-6 text-[var(--muted)]">
              {formatModelAtGlance(model)}
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <a
              className="inline-flex rounded-full border border-[var(--line)] px-4 py-2 text-sm text-[var(--ink)] transition hover:border-[var(--line-strong)]"
              href={model.sourceUrl}
              rel="noreferrer"
              target="_blank"
            >
              Open base model
            </a>
            {selectedProfile ? (
              <a
                className="inline-flex rounded-full border border-[var(--line)] px-4 py-2 text-sm text-[var(--ink)] transition hover:border-[var(--line-strong)]"
                href={selectedProfile.sourceUrl || model.sourceUrl}
                rel="noreferrer"
                target="_blank"
              >
                Open selected checkpoint
              </a>
            ) : null}
          </div>
        </div>
      </section>

      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Architecture" title="Model spec" />
        <div className="mt-5 rounded-[1.75rem] bg-white/55 p-6 md:p-7">
          <ModelSpecGrid model={model} />
        </div>
      </section>

      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Why it matters" title="Why memory behaves this way" />
        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <div className="rounded-[1.6rem] bg-white/62 p-5">
            <p className="eyebrow text-[0.62rem] text-[var(--muted)]">Research highlight</p>
            <p className="mt-3 text-sm leading-7 text-[var(--ink)]">
              {model.researchHighlight}
            </p>
          </div>
          <div className="rounded-[1.6rem] bg-white/62 p-5">
            <p className="eyebrow text-[0.62rem] text-[var(--muted)]">Memory note</p>
            <p className="mt-3 text-sm leading-7 text-[var(--ink)]">
              {model.memoryNote}
            </p>
          </div>
        </div>
      </section>

      {contextResult ? (
        <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
          <SectionTitle eyebrow="Calculator context" title="Current selection" />
          <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
            <div className="rounded-[1.6rem] bg-white/62 p-5">
              <div className="space-y-3">
                <DetailRow label="Runtime" value={contextResult.runtime.label} />
                <DetailRow label="GPU" value={contextResult.gpu.displayName} />
                {contextResult.input.gpuCount > 1 ? (
                  <DetailRow
                    label="GPU count"
                    value={formatInteger(contextResult.input.gpuCount)}
                  />
                ) : null}
                {contextResult.input.gpuCount > 1 ? (
                  <DetailRow
                    label="Aggregate VRAM"
                    value={formatGb(contextResult.gpuBytes)}
                  />
                ) : null}
                <DetailRow
                  label="Checkpoint profile"
                  value={contextResult.calculationProfile}
                />
                <DetailRow label="Weights dtype" value={formatDtype(contextResult.effectiveDtype)} />
                {runtimeSupportsKvCacheDtype(contextResult.input.runtimeId) ? (
                  <DetailRow
                    label="KV cache dtype"
                    value={formatDtype(contextResult.input.kvCacheDtype)}
                  />
                ) : null}
                <DetailRow
                  label="Modality"
                  value={
                    isMultimodalModel(model) ? "Text-only estimate in calculator" : "Text"
                  }
                />
              </div>
            </div>
            <div className="space-y-4">
              {contextResult.runtimeNotes.length > 0 ? (
                <div className="rounded-[1.6rem] bg-white/62 p-5">
                  <p className="eyebrow text-[0.62rem] text-[var(--muted)]">
                    Runtime assumptions
                  </p>
                  <div className="mt-3 space-y-2 text-sm leading-7 text-[var(--ink)]">
                    {contextResult.runtimeNotes.map((note) => (
                      <p key={note}>{note}</p>
                    ))}
                  </div>
                </div>
              ) : null}
              {contextResult.notes.length > 0 ? (
                <div className="rounded-[1.6rem] bg-white/62 p-5">
                  <p className="eyebrow text-[0.62rem] text-[var(--muted)]">
                    Estimator notes
                  </p>
                  <div className="mt-3 space-y-2 text-sm leading-7 text-[var(--ink)]">
                    {contextResult.notes.map((note) => (
                      <p key={note}>{note}</p>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </section>
      ) : null}

      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Checkpoints" title="Official profiles" />
        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          {model.inferenceProfiles.map((profile) => {
            const runtimeLabels = getInferenceProfileRuntimeLabels(profile);
            const isSelected = selectedProfile?.id === profile.id;

            return (
              <div key={profile.id} className="rounded-[1.6rem] bg-white/62 p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="text-base font-medium text-[var(--ink)]">
                      {profile.label}
                    </h3>
                    <p className="mt-1 text-sm leading-6 text-[var(--muted)]">
                      {formatDtype(profile.effectiveDtype)} checkpoint
                    </p>
                  </div>
                  {isSelected ? (
                    <span className="inline-flex rounded-full bg-[var(--accent-soft)] px-3 py-1 text-[0.72rem] mono text-[var(--accent)]">
                      Current
                    </span>
                  ) : null}
                  <span className="inline-flex rounded-full border border-[var(--line)] px-3 py-1 text-[0.72rem] mono text-[var(--muted)]">
                    {getProfileConfidenceLabel(profile)}
                  </span>
                </div>
                <p className="mt-4 text-sm leading-7 text-[var(--ink)]">{profile.note}</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {runtimeLabels.map((label) => (
                    <span
                      key={label}
                      className="inline-flex rounded-full border border-[var(--line)] px-3 py-1 text-[0.72rem] mono text-[var(--muted)]"
                    >
                      {label}
                    </span>
                  ))}
                </div>
                <a
                  className="mt-5 inline-flex rounded-full border border-[var(--line)] px-4 py-2 text-sm text-[var(--ink)] transition hover:border-[var(--line-strong)]"
                  href={profile.sourceUrl || model.sourceUrl}
                  rel="noreferrer"
                  target="_blank"
                >
                  Open checkpoint
                </a>
              </div>
            );
          })}
        </div>
      </section>

      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Sources" title="Reference links" />
        <div className="mt-5 space-y-3">
          {sourceLinks.map((url) => (
            <a
              key={url}
              className="flex items-start justify-between gap-4 rounded-[1.35rem] bg-white/62 px-4 py-3 text-sm leading-6 text-[var(--ink)] transition hover:bg-white/72"
              href={url}
              rel="noreferrer"
              target="_blank"
            >
              <span className="min-w-0 break-all">{url}</span>
              <span className="mono shrink-0 text-[0.72rem] text-[var(--muted)]">open</span>
            </a>
          ))}
        </div>
      </section>
    </main>
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
