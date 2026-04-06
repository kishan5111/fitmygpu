import type { MetadataRoute } from "next";
import { models } from "@/data/models";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: "https://fitmygpu.com",
      lastModified: new Date(),
      changeFrequency: "weekly",
      priority: 1,
    },
    ...models.map((model) => ({
      url: `https://fitmygpu.com/models/${model.id}`,
      lastModified: new Date(),
      changeFrequency: "weekly" as const,
      priority: 0.7,
    })),
  ];
}
