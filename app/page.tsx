import { WillItFitApp } from "@/components/will-it-fit-app";
import { canEstimateInput, estimateVram } from "@/lib/estimator";
import { applyModelConstraints } from "@/lib/model-constraints";
import {
  hasMeaningfulSearchParams,
  parseSearchParams,
} from "@/lib/query-state";

type PageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};

export default async function Page({ searchParams }: PageProps) {
  const resolvedSearchParams = (await searchParams) ?? {};
  const initialInput = applyModelConstraints(parseSearchParams(resolvedSearchParams));
  const initialResult = hasMeaningfulSearchParams(resolvedSearchParams)
    ? canEstimateInput(initialInput)
      ? estimateVram(initialInput)
      : null
    : null;

  return <WillItFitApp initialInput={initialInput} initialResult={initialResult} />;
}
