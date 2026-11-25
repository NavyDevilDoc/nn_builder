import { useState, useCallback, useEffect, Fragment } from "react";
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
import { Plus, X } from "lucide-react";
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

// ── Insert divider ────────────────────────────────────────────────────────────

function InsertDivider({ beforeId }: { beforeId: string }) {
  const [open, setOpen] = useState(false);
  const insertLayerBefore = useBuilderStore((s) => s.insertLayerBefore);

  function insert(type: LayerType) {
    insertLayerBefore(beforeId, type);
    setOpen(false);
  }

  if (!open) {
    return (
      <div
        className="flex items-center gap-2 py-1 group/ins cursor-pointer"
        onClick={() => setOpen(true)}
        role="button"
        aria-label="Insert layer here"
      >
        <div className="flex-1 h-px bg-zinc-800 group-hover/ins:bg-zinc-600 transition-colors" />
        <Plus size={11} className="text-zinc-700 group-hover/ins:text-zinc-400 transition-colors flex-shrink-0" />
        <div className="flex-1 h-px bg-zinc-800 group-hover/ins:bg-zinc-600 transition-colors" />
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5 px-2 py-1.5 my-0.5 bg-zinc-900/80 border border-dashed border-zinc-700 rounded">
      {ADD_TYPES.map((t) => (
        <button
          key={t}
          onClick={() => insert(t)}
          className="
            flex items-center gap-1 px-2 py-0.5 rounded
            bg-zinc-800 hover:bg-zinc-700 border border-zinc-700
            text-xs text-zinc-300 transition-colors
          "
        >
          {TYPE_LABEL[t]}
        </button>
      ))}
      <button
        onClick={() => setOpen(false)}
        className="ml-auto text-zinc-600 hover:text-zinc-400 transition-colors"
        aria-label="Cancel"
      >
        <X size={12} />
      </button>
    </div>
  );
}

// ── LayerBuilder ──────────────────────────────────────────────────────────────

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
          <div className="flex flex-col">
            {layers.map((layer, i) => (
              <Fragment key={layer.id}>
                <InsertDivider beforeId={layer.id} />
                <LayerCard layer={layer} shape={shapeFor(i)} />
              </Fragment>
            ))}
          </div>
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
