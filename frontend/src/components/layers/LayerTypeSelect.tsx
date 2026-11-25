import type { LayerType } from "../../types/config";
import { useBuilderStore } from "../../store/useBuilderStore";

const LAYER_TYPES: LayerType[] = [
  "Linear", "Conv2d", "MaxPool2d", "Flatten", "LSTM", "GRU",
];

const TYPE_COLORS: Record<LayerType, string> = {
  Linear:   "text-zinc-300",
  Conv2d:   "text-yellow-400",
  MaxPool2d:"text-red-400",
  Flatten:  "text-cyan-400",
  LSTM:     "text-purple-400",
  GRU:      "text-indigo-400",
};

interface Props {
  id:   string;
  type: LayerType;
}

export function LayerTypeSelect({ id, type }: Props) {
  const setLayerType = useBuilderStore((s) => s.setLayerType);

  return (
    <select
      value={type}
      onChange={(e) => setLayerType(id, e.target.value as LayerType)}
      className={`
        bg-transparent border-none outline-none cursor-pointer text-sm font-semibold
        ${TYPE_COLORS[type]}
        focus:ring-1 focus:ring-zinc-600 rounded px-1
      `}
    >
      {LAYER_TYPES.map((t) => (
        <option key={t} value={t} className="bg-zinc-900 text-zinc-100">
          {t}
        </option>
      ))}
    </select>
  );
}
