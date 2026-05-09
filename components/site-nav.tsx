import Link from "next/link";

const links = [
  { href: "/", label: "Calculator" },
  { href: "/models", label: "Models" },
  { href: "/blog", label: "Blog" },
];

export function SiteNav() {
  return (
    <header className="glass-nav sticky top-0 z-20 border-b border-[var(--line)]">
      <div className="mx-auto flex max-w-[74rem] items-center justify-between px-4 py-4 sm:px-6 lg:px-10">
        <Link className="hero-title text-xl text-[var(--ink)]" href="/">
          FitMyGPU
        </Link>
        <nav className="flex items-center gap-5 text-sm text-[var(--muted)]">
          {links.map((link) => (
            <Link
              key={link.href}
              className="transition hover:text-[var(--ink)]"
              href={link.href}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
