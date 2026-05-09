import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { formatModelAtGlance } from "@/lib/model-display";
import {
  getCompanies,
  getCompany,
  getCompanyModelGroups,
  getModelsForCompany,
  sortCompanyModels,
} from "@/lib/site-content";

type PageProps = {
  params: Promise<{ companyId: string }>;
};

export function generateStaticParams() {
  return getCompanies().map((company) => ({ companyId: company.id }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { companyId } = await params;
  const company = getCompany(companyId);

  return {
    title: company ? `${company.name} | FitMyGPU` : "Company | FitMyGPU",
    description: company?.shortDescription ?? "Model company coverage on FitMyGPU.",
  };
}

export default async function CompanyPage({ params }: PageProps) {
  const { companyId } = await params;
  const company = getCompany(companyId);

  if (!company) {
    notFound();
  }

  const companyModels = getModelsForCompany(company.id);
  const orderedModels = sortCompanyModels(companyModels);
  const latestModel = orderedModels[0];
  const modelGroups = getCompanyModelGroups(companyModels);
  return (
    <main className="page-reveal mx-auto flex min-h-screen max-w-[74rem] flex-col gap-6 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <p className="eyebrow text-[0.68rem] text-[var(--muted)]">Company</p>
        <h1 className="hero-title mt-3 text-4xl leading-none text-[var(--ink)] md:text-[4rem]">
          {company.name}
        </h1>
        <p className="mt-4 max-w-3xl text-base leading-7 text-[var(--muted)] md:text-lg">
          {company.shortDescription}
        </p>
      </section>

      {latestModel ? (
        <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
          <SectionTitle eyebrow="Start here" title="Latest model" />
          <div className="mt-5">
            <Link
              className="rounded-[1.7rem] border border-[var(--line)] bg-white/58 p-6 transition hover:border-[var(--line-strong)] hover:bg-white/72"
              href={`/models/${latestModel.id}`}
            >
              <p className="text-sm text-[var(--muted)]">{latestModel.family}</p>
              <h2 className="mt-2 text-2xl font-medium text-[var(--ink)]">
                {latestModel.displayName}
              </h2>
              <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
                {latestModel.shortDescription}
              </p>
              <p className="mt-4 text-sm leading-6 text-[var(--muted)]">
                {formatModelAtGlance(latestModel)}
              </p>
            </Link>
          </div>
        </section>
      ) : null}

      <section className="space-y-6">
        {modelGroups.length > 0 ? (
          modelGroups.map((group) => (
            <section
              key={group.series}
              className="surface-card section-reveal rounded-[2rem] p-6 md:p-8"
            >
              <SectionTitle eyebrow="Series" title={group.series} />
              <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {group.models.map((model) => (
                  <Link
                    key={model.id}
                    className="rounded-[1.6rem] border border-[var(--line)] bg-white/58 p-5 transition hover:border-[var(--line-strong)] hover:bg-white/72"
                    href={`/models/${model.id}`}
                  >
                    <h2 className="text-lg font-medium text-[var(--ink)]">{model.displayName}</h2>
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
          ))
        ) : (
          <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
            <SectionTitle eyebrow="Models" title="Current coverage" />
            <p className="mt-5 text-sm leading-6 text-[var(--muted)]">
              No registry models from this company are linked yet. This page is ready for future coverage.
            </p>
          </section>
        )}
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
