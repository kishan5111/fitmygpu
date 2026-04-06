import { DetailRow } from "@/components/detail-row";
import { formatInteger, formatParamCount } from "@/lib/format";
import { getKvBearingLayers, isMultimodalModel } from "@/lib/model-display";
import type { ModelSpec } from "@/lib/types";

export function ModelSpecGrid({
  className,
  model,
}: {
  className?: string;
  model: ModelSpec;
}) {
  return (
    <div className={cx("grid gap-x-10 gap-y-4 md:grid-cols-2", className)}>
      <DetailRow label="Architecture" value={model.architectureType} />
      <DetailRow label="Total params" value={formatParamCount(model.totalParams)} />
      <DetailRow
        label="Active params"
        value={model.activeParams ? formatParamCount(model.activeParams) : "Dense model"}
      />
      <DetailRow label="Layers" value={model.numLayers.toString()} />
      <DetailRow label="Hidden size" value={formatInteger(model.hiddenSize)} />
      <DetailRow label="Attention heads" value={model.numAttentionHeads.toString()} />
      <DetailRow
        label="KV heads"
        value={(model.numKvHeads ?? model.numAttentionHeads).toString()}
      />
      <DetailRow label="KV-bearing layers" value={getKvBearingLayers(model).toString()} />
      <DetailRow label="Context length" value={formatInteger(model.contextLength)} />
      <DetailRow
        label="Modality"
        value={isMultimodalModel(model) ? "Multimodal, text-only estimate" : "Text"}
      />
      <DetailRow label="License" value={model.license} />
    </div>
  );
}

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}
