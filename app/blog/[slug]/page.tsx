import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getCompany, getPost, getPosts } from "@/lib/site-content";
import type { BlogContentBlock } from "@/lib/types";

type PageProps = {
  params: Promise<{ slug: string }>;
};

export function generateStaticParams() {
  return getPosts().map((post) => ({ slug: post.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const post = getPost(slug);

  return {
    title: post ? `${post.title} | FitMyGPU` : "Post | FitMyGPU",
    description: post?.summary ?? "FitMyGPU blog post.",
  };
}

export default async function BlogPostPage({ params }: PageProps) {
  const { slug } = await params;
  const post = getPost(slug);

  if (!post) {
    notFound();
  }

  return (
    <main className="page-reveal mx-auto flex min-h-screen max-w-[62rem] flex-col gap-7 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
      <section className="section-reveal">
        <Link
          className="eyebrow inline-flex text-[0.68rem] text-[var(--muted)] transition hover:text-[var(--accent)]"
          href="/blog"
        >
          Back to blog
        </Link>
        <p className="eyebrow mt-4 text-[0.68rem] text-[var(--muted)]">{post.publishedAt}</p>
        <h1 className="hero-title mt-3 text-[2.95rem] leading-none text-[var(--ink)] md:text-[4.15rem]">
          {post.title}
        </h1>
        <p className="mt-4 text-base leading-7 text-[var(--muted)] md:text-lg">
          {post.summary}
        </p>
        <div className="mt-5 flex flex-wrap gap-2">
          {post.companyIds.map((companyId) => (
            <Link
              key={companyId}
              className="inline-flex rounded-full border border-[var(--line)] px-3 py-1 text-[0.72rem] mono text-[var(--muted)]"
              href={`/companies/${companyId}`}
            >
              {getCompany(companyId)?.name ?? companyId}
            </Link>
          ))}
        </div>
      </section>

      <article className="section-reveal space-y-7 text-[1.04rem] leading-8 text-[var(--ink)]">
        {post.content.map((block, index) => (
          <ContentBlock block={block} key={`${block.type}-${index}`} />
        ))}
      </article>
    </main>
  );
}

function ContentBlock({ block }: { block: BlogContentBlock }) {
  switch (block.type) {
    case "paragraph":
      return <p>{block.text}</p>;
    case "heading":
      return (
        <h2 className="hero-title pt-3 text-[2rem] leading-none text-[var(--ink)]">
          {block.text}
        </h2>
      );
    case "list":
      return (
        <ul className="space-y-3 pl-5 text-[var(--ink)]">
          {block.items.map((item) => (
            <li key={item} className="list-disc">
              {item}
            </li>
          ))}
        </ul>
      );
    case "code":
      return (
        <pre className="overflow-x-auto border-l-2 border-[var(--line-strong)] pl-5 text-sm leading-7 text-[var(--ink)]">
          <code>{block.code}</code>
        </pre>
      );
    case "svg":
      return <InlineDiagram variant={block.variant} />;
  }
}

function InlineDiagram({ variant }: { variant: "vram-breakdown" }) {
  if (variant !== "vram-breakdown") {
    return null;
  }

  return (
    <div className="overflow-hidden py-2">
      <svg
        aria-label="Inference VRAM breakdown diagram"
        className="h-auto w-full"
        viewBox="0 0 860 236"
      >
        <rect fill="rgba(45,108,102,0.16)" height="92" rx="22" width="290" x="0" y="64" />
        <rect fill="rgba(204,140,83,0.18)" height="92" rx="22" width="250" x="305" y="64" />
        <rect fill="rgba(36,42,48,0.12)" height="92" rx="22" width="160" x="570" y="64" />
        <rect fill="rgba(154,79,67,0.14)" height="92" rx="22" width="115" x="745" y="64" />

        <text fill="#1e242a" fontFamily="IBM Plex Mono, monospace" fontSize="16" x="28" y="104">
          weights
        </text>
        <text fill="#1e242a" fontFamily="IBM Plex Mono, monospace" fontSize="16" x="333" y="104">
          KV cache / state
        </text>
        <text fill="#1e242a" fontFamily="IBM Plex Mono, monospace" fontSize="16" x="598" y="104">
          overhead
        </text>
        <text fill="#1e242a" fontFamily="IBM Plex Mono, monospace" fontSize="16" x="771" y="104">
          reserve
        </text>

        <text fill="#66707b" fontFamily="Avenir Next, sans-serif" fontSize="14" x="28" y="128">
          resident checkpoint
        </text>
        <text fill="#66707b" fontFamily="Avenir Next, sans-serif" fontSize="14" x="333" y="126">
          grows with context
        </text>
        <text fill="#66707b" fontFamily="Avenir Next, sans-serif" fontSize="14" x="333" y="145">
          or sequence state
        </text>
        <text fill="#66707b" fontFamily="Avenir Next, sans-serif" fontSize="14" x="598" y="128">
          kernels and scratch
        </text>
        <text fill="#66707b" fontFamily="Avenir Next, sans-serif" fontSize="14" x="760" y="128">
          engine budget
        </text>
      </svg>
    </div>
  );
}
