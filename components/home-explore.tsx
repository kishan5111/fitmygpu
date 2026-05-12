"use client";

import Link from "next/link";
import { models } from "@/data/models";
import { formatModelAtGlance } from "@/lib/model-display";
import {
  getCompanies,
  getCompanyForModel,
  getPosts,
} from "@/lib/site-content";

export function HomeExplore() {
  const recentModels = models.slice(0, 6);
  const recentPosts = getPosts().slice(0, 3);
  const companies = getCompanies().slice(0, 8);

  return (
    <div className="mx-auto flex max-w-[74rem] flex-col gap-6 px-4 pb-10 sm:px-6 lg:px-10 lg:pb-14">
      <section className="surface-card rounded-[2rem] p-6 md:p-8">
        <p className="max-w-3xl text-sm leading-7 text-[var(--muted)] md:text-base">
          Use the calculator first, then go deeper with model pages and release
          notes when you want architecture context, memory behavior, or newer
          inference updates.
        </p>
      </section>

      <section className="surface-card rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Model research" title="About models" />
        <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {recentModels.map((model) => (
            <Link
              key={model.id}
              className="rounded-[1.6rem] border border-[var(--line)] bg-white/58 p-5 transition hover:border-[var(--line-strong)] hover:bg-white/72"
              href={`/models/${model.id}`}
            >
              <p className="text-sm text-[var(--muted)]">
                {getCompanyForModel(model)?.name ?? model.organization}
              </p>
              <h3 className="mt-2 text-lg font-medium text-[var(--ink)]">
                {model.displayName}
              </h3>
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

      <section className="surface-card rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Companies" title="Browse by model source" />
        <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {companies.map((company) => (
            <Link
              key={company.id}
              className="rounded-[1.6rem] border border-[var(--line)] bg-white/58 p-5 transition hover:border-[var(--line-strong)] hover:bg-white/72"
              href={`/companies/${company.id}`}
            >
              <h3 className="text-base font-medium text-[var(--ink)]">{company.name}</h3>
              <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                {company.shortDescription}
              </p>
            </Link>
          ))}
        </div>
      </section>

      <section className="surface-card rounded-[2rem] p-6 md:p-8">
        <SectionTitle eyebrow="Latest releases" title="Model and inference notes" />
        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          {recentPosts.map((post) => (
            <Link
              key={post.slug}
              className="rounded-[1.6rem] border border-[var(--line)] bg-white/58 p-5 transition hover:border-[var(--line-strong)] hover:bg-white/72"
              href={`/blog/${post.slug}`}
            >
              <p className="eyebrow text-[0.62rem] text-[var(--muted)]">
                {post.publishedAt}
              </p>
              <h3 className="mt-2 text-lg font-medium text-[var(--ink)]">
                {post.title}
              </h3>
              <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                {post.summary}
              </p>
            </Link>
          ))}
        </div>
      </section>
    </div>
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
