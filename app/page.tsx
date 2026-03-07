import { WillItFitApp } from "@/components/will-it-fit-app";
import { estimateVram } from "@/lib/estimator";
import {
  hasMeaningfulSearchParams,
  parseSearchParams,
} from "@/lib/query-state";

type PageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};

export default async function Page({ searchParams }: PageProps) {
  const resolvedSearchParams = (await searchParams) ?? {};
  const initialInput = parseSearchParams(resolvedSearchParams);
  const initialResult = hasMeaningfulSearchParams(resolvedSearchParams)
    ? estimateVram(initialInput)
    : null;

  return (
    <WillItFitApp initialInput={initialInput} initialResult={initialResult} />
  );
}
