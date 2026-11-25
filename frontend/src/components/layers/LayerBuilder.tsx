import { useState, useCallback, useEffect } from "react";
import {
  DndContext,
  closestCenter,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  verticalListSortingStrategy,
  arrayMove,
} from "@dnd-kit/sortable";
import { useQuery } from "@tanstack/react-query";
import { Plus } from "lucide-react";
import { useBuilderStore } from "../../store/useBuilderStore";
import { fetchDatasets, validateConfig } from "../../api/client";
import { LayerCard } from "./LayerCard";
import type { LayerType, ValidationResult } from "../../types/config";

const ADD_TYPES: LayerType[] = ["Linear", "Conv2d", "MaxPool2d", "Flatten", "LSTM", "GRU"];

const TYPE_LABEL: Record<LayerType, string> = {
  Linear:   "Linear",
  Conv2d:   "Conv2d",
  MaxPool2d:"MaxPool2d",
  Flatten:  "Flatten",
  LSTM:     "LSTM",
  GRU:      "GRU",
};

export function LayerBuilder() {
  const layers        = useBuilderStore((s) => s.layers);
  const hyperparams   = useBuilderStore((s) => s.hyperparams);
  const addLayer      = useBuilderStore((s) => s.addLayer);
  const reorderLayers = useBuilderStore((s) => s.reorderLayers);

  const { data: datasetsData } = useQuery({
    queryKey: ["datasets"],
    queryFn: fetchDatasets,
  });

  const [validation, setValidation] = useState<ValidationResult | null>(null);

  const runValidation = useCallback(async () => {
    if (layers.length === 0) { setValidation(null); return; }
    const meta = datasetsData?.meta[hyperparams.dataset];
    if (!meta) return;
    try {
      const result = await validateConfig(
        meta.input_shape ?? null,
        meta.input_dim,
        meta.task,
        meta.n_classes,
        layers.map(({ id: _id, ...rest }) => rest),
      );
      setValidation(result);
    } catch {
      setValidation(null);
    }
  }, [layers, hyperparams.dataset, datasetsData]);

  useEffect(() => { runValidation(); }, [runValidation]);

  // dnd-kit setup
  const sensors = useSensors(useSensor(PointerSensor));

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const oldIdx = layers.findIndex((l) => l.id === active.id);
    const newIdx = layers.findIndex((l) => l.id === over.id);
    reorderLayers(arrayMove(layers, oldIdx, newIdx));
  }

  const shapeFor = (index: number): number[] | null | undefined => {
    if (!validation) return undefined;
    return validation.shapes[index];
  };

  return (
    <div className="flex flex-col gap-3">
      {/* Sortable layer list */}
      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <SortableContext items={layers.map((l) => l.id)} strategy={verticalListSortingStrategy}>
          {layers.map((layer, i) => (
            <LayerCard key={layer.id} layer={layer} shape={shapeFor(i)} />
          ))}
        </SortableContext>
      </DndContext>

      {layers.length === 0 && (
        <p className="text-sm text-zinc-600 text-center py-4 select-none">
          No layers yet — add one below.
        </p>
      )}

      {/* Validation error banner */}
      {validation && !validation.valid && (
        <div className="text-xs text-red-400 bg-red-950/40 border border-red-800 rounded px-3 py-2">
          {validation.error}
        </div>
      )}

      {/* Add layer buttons */}
      <div className="flex flex-wrap gap-2 pt-1">
        {ADD_TYPES.map((t) => (
          <button
            key={t}
            onClick={() => addLayer(t)}
            className="
              flex items-center gap-1 px-2.5 py-1 rounded
              bg-zinc-800 hover:bg-zinc-700 border border-zinc-700
              text-xs text-zinc-300 transition-colors
            "
          >
            <Plus size={12} />
            {TYPE_LABEL[t]}
          </button>
        ))}
      </div>
    </div>
  );
}
