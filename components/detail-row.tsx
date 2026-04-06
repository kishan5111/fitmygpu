import type { ReactNode } from "react";

export function DetailRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-[var(--line)] pb-3 last:border-b-0 last:pb-0">
      <p className="text-sm leading-6 text-[var(--muted)]">{label}</p>
      <div className="text-right text-sm leading-6 text-[var(--ink)]">{value}</div>
    </div>
  );
}
