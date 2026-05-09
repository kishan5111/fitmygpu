import type { Metadata } from "next";
import Link from "next/link";
import { models } from "@/data/models";
import { formatModelAtGlance } from "@/lib/model-display";
import {
  getCompanies,
  getCompanyForModel,
  getModelsForCompany,
} from "@/lib/site-content";

export const metadata: Metadata = {
  title: "Models | FitMyGPU",
  description: "Browse model pages with concise architecture, memory, and runtime notes.",
};

export default function ModelsPage() {
  const featuredModels = [
    "gpt-oss-120b",
    "qwen-3.5-9b",
    "openreasoning-nemotron-14b",
    "llama-3.1-70b",
    "qwen-2.5-32b",
    "gpt-oss-20b",
  ]
    .map((modelId) => models.find((model) => model.id === modelId))
    .filter((model): model is (typeof models)[number] => Boolean(model));
  const browseCompanies = getCompanies()
    .map((company) => ({
      company,
      modelCount: getModelsForCompany(company.id).length,
    }))
    .sort((left, right) => right.modelCount - left.modelCount || left.company.name.localeCompare(right.company.name))
    .slice(0, 8);

  return (
    <main className="page-reveal mx-auto flex min-h-screen max-w-[74rem] flex-col gap-6 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <p className="eyebrow text-[0.68rem] text-[var(--muted)]">Models</p>
        <h1 className="hero-title mt-3 text-4xl leading-none text-[var(--ink)] md:text-[4rem]">
          Browse the model registry
        </h1>
        <p className="mt-4 max-w-3xl text-base leading-7 text-[var(--muted)] md:text-lg">
          Start with the newest or most useful models, browse by company, or jump to the full index below.
        </p>
      </section>

      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Start here" title="Models to explore first" />
        <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {featuredModels.map((model) => (
            <Link
              key={model.id}
              className="rounded-[1.6rem] border border-[var(--line)] bg-white/58 p-5 transition hover:border-[var(--line-strong)] hover:bg-white/72"
              href={`/models/${model.id}`}
            >
              <p className="text-sm text-[var(--muted)]">
                {getCompanyForModel(model)?.name ?? model.organization}
              </p>
              <h2 className="mt-2 text-xl font-medium text-[var(--ink)]">
                {model.displayName}
              </h2>
              <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                {model.shortDescription}
              </p>
              <p className="mt-3 text-sm leading-6 text-[var(--muted)]">
                {formatModelAtGlance(model)}
              </p>
            </Link>
          ))}
        </div>
      </section>

      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Companies" title="Browse by source" />
        <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {browseCompanies.map(({ company, modelCount }) => (
            <Link
              key={company.id}
              className="rounded-[1.6rem] border border-[var(--line)] bg-white/58 p-5 transition hover:border-[var(--line-strong)] hover:bg-white/72"
              href={`/companies/${company.id}`}
            >
              <h2 className="text-lg font-medium text-[var(--ink)]">{company.name}</h2>
              <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                {company.shortDescription}
              </p>
              <p className="mt-3 text-sm text-[var(--muted)]">
                {modelCount > 0 ? `${modelCount} models covered` : "Coverage coming next"}
              </p>
            </Link>
          ))}
        </div>
      </section>

      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Index" title="All models" />
        <div className="mt-5 overflow-hidden rounded-[1.6rem] border border-[var(--line)] bg-white/55">
          <div className="grid grid-cols-[minmax(10rem,1.2fr)_minmax(8rem,0.8fr)_minmax(0,1.4fr)] gap-4 border-b border-[var(--line)] px-4 py-3 text-[0.72rem] mono text-[var(--muted)]">
            <span>Model</span>
            <span>Company</span>
            <span>At a glance</span>
          </div>
          {models.map((model) => (
            <Link
              key={model.id}
              className="grid grid-cols-[minmax(10rem,1.2fr)_minmax(8rem,0.8fr)_minmax(0,1.4fr)] gap-4 border-b border-[var(--line)] px-4 py-3 text-sm transition hover:bg-white/72 last:border-b-0"
              href={`/models/${model.id}`}
            >
              <span className="font-medium text-[var(--ink)]">{model.displayName}</span>
              <span className="text-[var(--muted)]">
                {getCompanyForModel(model)?.name ?? model.organization}
              </span>
              <span className="text-[var(--muted)]">{formatModelAtGlance(model)}</span>
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}

function SectionTitle({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <div className="space-y-2">
      <p className="eyebrow text-[0.68rem] text-[var(--muted)]">{eyebrow}</p>
      <h2 className="hero-title text-[2rem] leading-none text-[var(--ink)]">{title}</h2>
    </div>
  );
}
