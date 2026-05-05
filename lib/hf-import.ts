import type { ModelSpec } from "@/lib/types";

export type HfImportResult =
  | {
      status: "known";
      modelId: string;
      repoId: string;
    }
  | {
      status: "estimated";
      model: ModelSpec;
      repoId: string;
    };

export function parseHfRepoId(rawValue: string): string | null {
  const value = rawValue.trim();

  if (!value) {
    return null;
  }

  const normalized = value
    .replace(/^https?:\/\/huggingface\.co\//i, "")
    .replace(/^huggingface\.co\//i, "")
    .replace(/^\/+/, "");
  const [path] = normalized.split(/[?#]/);
  const [owner, name] = path?.split("/") ?? [];

  if (!owner || !name) {
    return null;
  }

  return `${owner}/${name}`;
}

export function getHfRepoIdFromUrl(url: string) {
  return parseHfRepoId(url)?.toLowerCase();
}
