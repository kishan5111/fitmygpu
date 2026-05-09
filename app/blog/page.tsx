import type { Metadata } from "next";
import Link from "next/link";
import { getPosts } from "@/lib/site-content";

export const metadata: Metadata = {
  title: "Blog | FitMyGPU",
  description: "Short posts about model releases, inference runtimes, and VRAM behavior.",
};

export default function BlogIndexPage() {
  const posts = getPosts();

  return (
    <main className="page-reveal mx-auto flex min-h-screen max-w-[74rem] flex-col gap-6 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
      <section className="surface-card section-reveal rounded-[2rem] p-6 md:p-8">
        <p className="eyebrow text-[0.68rem] text-[var(--muted)]">Blog</p>
        <h1 className="hero-title mt-3 text-[2.65rem] leading-none text-[var(--ink)] md:text-[3.45rem]">
          Inference notes and model updates
        </h1>
        <p className="mt-4 max-w-3xl text-base leading-7 text-[var(--muted)] md:text-lg">
          Notes on model releases, inference changes, and how the calculator works.
        </p>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        {posts.map((post) => (
          <Link
            key={post.slug}
            className="surface-card section-reveal rounded-[1.8rem] p-6 transition hover:border-[var(--line-strong)]"
            href={`/blog/${post.slug}`}
          >
            <p className="eyebrow text-[0.62rem] text-[var(--muted)]">{post.publishedAt}</p>
            <h2 className="mt-3 text-2xl font-medium text-[var(--ink)]">{post.title}</h2>
            <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{post.summary}</p>
          </Link>
        ))}
      </section>
    </main>
  );
}
