import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";
import { DetailRow } from "@/components/detail-row";
import { models } from "@/data/models";
import { formatInteger, formatParamCount } from "@/lib/format";
import {
  getKvBearingLayers,
  isMultimodalModel,
} from "@/lib/model-display";
import {
  getCompanyForModel,
} from "@/lib/site-content";

type PageProps = {
  params: Promise<{ modelId: string }>;
};

export function generateStaticParams() {
  return models.map((model) => ({ modelId: model.id }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { modelId } = await params;
  const model = models.find((entry) => entry.id === modelId);

  return {
    title: model ? `${model.displayName} | FitMyGPU` : "Model | FitMyGPU",
    description: model?.shortDescription ?? "Model notes on FitMyGPU.",
  };
}

export default async function ModelPage({ params }: PageProps) {
  const { modelId } = await params;
  const model = models.find((entry) => entry.id === modelId);

  if (!model) {
    notFound();
  }

  const sourceLinks = Array.from(
    new Set([model.sourceUrl, ...model.inferenceProfiles.map((profile) => profile.sourceUrl)].filter(Boolean)),
  );
  const company = getCompanyForModel(model);

  return (
    <main className="page-reveal mx-auto flex min-h-screen max-w-[74rem] flex-col gap-6 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <Link
          className="eyebrow inline-flex text-[0.68rem] text-[var(--muted)] transition hover:text-[var(--accent)]"
          href={`/?model=${model.id}`}
        >
          Back to calculator
        </Link>
        <div className="mt-4 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl space-y-3">
            <p className="eyebrow text-[0.68rem] text-[var(--muted)]">
              {company?.name ?? model.organization}
            </p>
            <h1 className="hero-title text-4xl leading-none text-[var(--ink)] md:text-[4rem]">
              {model.displayName}
            </h1>
            <p className="max-w-3xl text-base leading-7 text-[var(--muted)] md:text-lg">
              {model.shortDescription}
            </p>
          </div>
        </div>
      </section>

      <section className="space-y-6">
        <Card eyebrow="Overview and architecture" title="What it is">
          <div className="rounded-[1.6rem] bg-white/58 p-5">
            <div className="grid gap-x-10 gap-y-4 md:grid-cols-2">
              <DetailRow label="Company" value={company?.name ?? model.organization} />
              <DetailRow label="Family" value={model.family} />
              <DetailRow label="Release date" value={formatReleaseDate(model.releaseDate)} />
              <DetailRow label="Architecture" value={model.architectureType} />
              <DetailRow label="License" value={model.license} />
              <DetailRow
                label="Modality"
                value={isMultimodalModel(model) ? "Multimodal (text-only estimate)" : "Text"}
              />
              <DetailRow
                label="Context window"
                value={formatInteger(model.contextLength)}
              />
              <DetailRow label="Total params" value={formatParamCount(model.totalParams)} />
              <DetailRow
                label="Active params"
                value={model.activeParams ? formatParamCount(model.activeParams) : "Dense model"}
              />
              <DetailRow label="Layers" value={model.numLayers.toString()} />
              <DetailRow label="Hidden size" value={formatInteger(model.hiddenSize)} />
              <DetailRow label="Attention heads" value={model.numAttentionHeads.toString()} />
              <DetailRow
                label="KV heads"
                value={(model.numKvHeads ?? model.numAttentionHeads).toString()}
              />
              <DetailRow label="KV-bearing layers" value={getKvBearingLayers(model).toString()} />
            </div>
          </div>
          {model.overviewPoints?.length ? (
            <InsightList className="mt-5" points={model.overviewPoints} />
          ) : null}
        </Card>

        <Card eyebrow="Research highlight" title="What improved">
          {model.researchHighlights?.length ? (
            <InsightList points={model.researchHighlights} />
          ) : (
            <p>{model.researchHighlight}</p>
          )}
        </Card>

        {model.trainingReleaseContext?.length ? (
          <Card eyebrow="Training and release context" title="How it was released">
            <InsightList points={model.trainingReleaseContext} />
          </Card>
        ) : null}

        {model.strengths?.length ? (
          <Card eyebrow="Where it is strong" title="Where it is strong">
            <InsightList points={model.strengths} />
          </Card>
        ) : null}

        <Card eyebrow="Memory behavior" title="What dominates VRAM">
          {model.memoryBehaviorPoints?.length ? (
            <div className="space-y-3">
              {model.memoryBehaviorPoints.map((point) => (
                <p key={point}>{point}</p>
              ))}
            </div>
          ) : (
            <p>{model.memoryNote}</p>
          )}
          {isMultimodalModel(model) ? (
            <p className="mt-3">
              FitMyGPU currently treats this as a text-only estimate. Resident multimodal weights remain counted, but media-token overhead is excluded.
            </p>
          ) : null}
        </Card>

        <Card eyebrow="Sources" title="Where this page is grounded">
          <div className="space-y-3">
            {sourceLinks.map((url) => (
              <a
                key={url}
                className="flex items-start justify-between gap-4 rounded-[1.35rem] bg-white/58 px-4 py-3 text-sm leading-6 text-[var(--ink)] transition hover:bg-white/72"
                href={url}
                rel="noreferrer"
                target="_blank"
              >
                <span className="min-w-0 break-all">{url}</span>
                <span className="mono shrink-0 text-[0.72rem] text-[var(--muted)]">open</span>
              </a>
            ))}
          </div>
        </Card>
      </section>
    </main>
  );
}

function InsightList({
  className,
  points,
}: {
  className?: string;
  points: Array<{ label: string; detail: string }>;
}) {
  return (
    <div className={cx("space-y-3", className)}>
      {points.map((point) => (
        <div key={point.label} className="rounded-[1.35rem] bg-white/58 px-4 py-4">
          <p className="text-sm font-medium text-[var(--ink)]">{point.label}</p>
          <p className="mt-1 text-sm leading-7 text-[var(--muted)]">{point.detail}</p>
        </div>
      ))}
    </div>
  );
}

function Card({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
      <p className="eyebrow text-[0.68rem] text-[var(--muted)]">{eyebrow}</p>
      <h2 className="hero-title mt-3 text-[2rem] leading-none text-[var(--ink)]">
        {title}
      </h2>
      <div className="mt-4 text-sm leading-7 text-[var(--ink)]">{children}</div>
    </section>
  );
}

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

function formatReleaseDate(value?: string) {
  if (!value) {
    return "Unknown";
  }

  const date = new Date(`${value}T00:00:00Z`);

  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        timeZone: "UTC",
      });
}
