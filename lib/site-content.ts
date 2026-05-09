import { companies } from "@/data/companies";
import { models } from "@/data/models";
import { posts } from "@/data/posts";
import type { BlogPost, CompanySpec, ModelSpec } from "@/lib/types";

const companyMap = new Map(companies.map((company) => [company.id, company]));
const postMap = new Map(posts.map((post) => [post.slug, post]));

export function getCompanies(): CompanySpec[] {
  return companies;
}

export function getCompany(companyId: string): CompanySpec | undefined {
  return companyMap.get(companyId);
}

export function getCompanyForModel(model: ModelSpec): CompanySpec | undefined {
  return getCompany(getModelCompanyId(model));
}

export function getModelCompanyId(model: ModelSpec): string {
  if (model.companyId) {
    return model.companyId;
  }

  if (model.id.startsWith("gpt-oss")) {
    return "openai";
  }

  if (model.id.startsWith("llama-")) {
    return "meta-llama";
  }

  if (model.id.startsWith("qwen-")) {
    return "qwen";
  }

  if (model.id.startsWith("nemotron-")) {
    return "nvidia";
  }

  if (model.id.startsWith("phi-")) {
    return "microsoft-phi";
  }

  if (model.id.startsWith("gemma-")) {
    return "gemma";
  }

  if (model.id.startsWith("mistral-") || model.id.startsWith("mixtral-")) {
    return "mistral";
  }

  if (model.organization === "OpenAI") {
    return "openai";
  }

  if (model.organization === "Meta") {
    return "meta-llama";
  }

  if (model.organization === "NVIDIA") {
    return "nvidia";
  }

  return "qwen";
}

export function getModelsForCompany(companyId: string): ModelSpec[] {
  return models.filter((model) => getModelCompanyId(model) === companyId);
}

export function getPosts(): BlogPost[] {
  return [...posts].sort((left, right) => right.publishedAt.localeCompare(left.publishedAt));
}

export function getPost(slug: string): BlogPost | undefined {
  return postMap.get(slug);
}

export function getPostsForCompany(companyId: string): BlogPost[] {
  return getPosts().filter((post) => post.companyIds.includes(companyId));
}

export function getPostsForModel(modelId: string): BlogPost[] {
  const model = models.find((entry) => entry.id === modelId);
  if (!model) {
    return [];
  }

  return getPosts().filter(
    (post) =>
      post.modelIds?.includes(modelId) ||
      post.companyIds.includes(getModelCompanyId(model)),
  );
}

export function getCompanyModelSeries(model: ModelSpec): string {
  return (
    model.displayName
      .replace(/\s+\d+(?:\.\d+)?B$/i, "")
      .replace(/\s+\d+x\d+B$/i, "")
      .trim() || model.family
  );
}

export function sortCompanyModels(modelsForCompany: ModelSpec[]): ModelSpec[] {
  return [...modelsForCompany].sort((left, right) => {
    const seriesDiff = getSeriesSortValue(right) - getSeriesSortValue(left);
    if (seriesDiff !== 0) {
      return seriesDiff;
    }

    if (left.totalParams !== right.totalParams) {
      return right.totalParams - left.totalParams;
    }

    return left.displayName.localeCompare(right.displayName);
  });
}

export function getCompanyModelGroups(modelsForCompany: ModelSpec[]) {
  const orderedModels = sortCompanyModels(modelsForCompany);
  const groups = new Map<string, ModelSpec[]>();

  for (const model of orderedModels) {
    const series = getCompanyModelSeries(model);
    groups.set(series, [...(groups.get(series) ?? []), model]);
  }

  return Array.from(groups.entries()).map(([series, groupedModels]) => ({
    series,
    models: groupedModels,
  }));
}

function getSeriesSortValue(model: ModelSpec): number {
  const series = getCompanyModelSeries(model);
  const match = series.match(/\d+(?:\.\d+)?/);

  if (!match) {
    return 0;
  }

  return Number.parseFloat(match[0]);
}
