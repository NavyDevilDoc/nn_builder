import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { GripVertical, Trash2 } from "lucide-react";
import type { Activation, LayerConfig } from "../../types/config";
import { useBuilderStore } from "../../store/useBuilderStore";
import { LayerTypeSelect } from "./LayerTypeSelect";

// ── Shared micro-components ───────────────────────────────────────────────────

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-xs text-zinc-400 select-none">{children}</span>
  );
}

function NumberInput({
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (n: number) => void;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
      className="
        w-20 bg-zinc-800 border border-zinc-700 rounded px-2 py-0.5
        text-sm text-zinc-100 text-right
        focus:outline-none focus:ring-1 focus:ring-zinc-500
      "
    />
  );
}

function Checkbox({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-zinc-400 w-3.5 h-3.5"
      />
      <span className="text-xs text-zinc-400">{label}</span>
    </label>
  );
}

const ACTIVATIONS: Activation[] = ["ReLU", "Tanh", "Sigmoid", "LeakyReLU", "Linear"];

function ActivationSelect({
  value,
  onChange,
}: {
  value: Activation;
  onChange: (a: Activation) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as Activation)}
      className="
        bg-zinc-800 border border-zinc-700 rounded px-2 py-0.5
        text-sm text-zinc-100
        focus:outline-none focus:ring-1 focus:ring-zinc-500
      "
    >
      {ACTIVATIONS.map((a) => (
        <option key={a} value={a} className="bg-zinc-900">
          {a}
        </option>
      ))}
    </select>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center justify-between gap-3">{children}</div>;
}

// ── Shape badge ───────────────────────────────────────────────────────────────

function ShapeBadge({ shape }: { shape: number[] | null | undefined }) {
  if (shape === undefined) return null;
  const text = shape === null ? "?" : `[${shape.join(" × ")}]`;
  return (
    <span className="ml-auto text-xs font-mono text-zinc-500 select-none">
      {text}
    </span>
  );
}

// ── Per-type control panels ───────────────────────────────────────────────────

function LinearControls({ layer, update }: { layer: LayerConfig; update: (p: Partial<LayerConfig>) => void }) {
  return (
    <>
      <Row>
        <Label>Units</Label>
        <NumberInput value={layer.units ?? 64} min={1} onChange={(n) => update({ units: n })} />
      </Row>
      <Row>
        <Label>Activation</Label>
        <ActivationSelect
          value={layer.activation ?? "ReLU"}
          onChange={(a) => update({ activation: a })}
        />
      </Row>
      <Row>
        <Checkbox
          checked={layer.batch_norm ?? false}
          onChange={(v) => update({ batch_norm: v })}
          label="Batch norm"
        />
      </Row>
      <Row>
        <Checkbox
          checked={layer.dropout ?? false}
          onChange={(v) => update({ dropout: v })}
          label="Dropout"
        />
        {layer.dropout && (
          <NumberInput
            value={layer.dropout_rate ?? 0.0}
            min={0}
            max={0.99}
            step={0.05}
            onChange={(n) => update({ dropout_rate: n })}
          />
        )}
      </Row>
    </>
  );
}

function Conv2dControls({ layer, update }: { layer: LayerConfig; update: (p: Partial<LayerConfig>) => void }) {
  return (
    <>
      <Row>
        <Label>Out channels</Label>
        <NumberInput value={layer.out_channels ?? 32} min={1} onChange={(n) => update({ out_channels: n })} />
      </Row>
      <Row>
        <Label>Kernel size</Label>
        <NumberInput value={layer.kernel_size ?? 3} min={1} onChange={(n) => update({ kernel_size: n })} />
      </Row>
      <Row>
        <Label>Activation</Label>
        <ActivationSelect
          value={layer.activation ?? "ReLU"}
          onChange={(a) => update({ activation: a })}
        />
      </Row>
      <Row>
        <Checkbox
          checked={layer.batch_norm ?? false}
          onChange={(v) => update({ batch_norm: v })}
          label="Batch norm"
        />
      </Row>
    </>
  );
}

function MaxPool2dControls({ layer, update }: { layer: LayerConfig; update: (p: Partial<LayerConfig>) => void }) {
  return (
    <Row>
      <Label>Kernel size</Label>
      <NumberInput value={layer.kernel_size ?? 2} min={1} onChange={(n) => update({ kernel_size: n })} />
    </Row>
  );
}

function RNNControls({ layer, update }: { layer: LayerConfig; update: (p: Partial<LayerConfig>) => void }) {
  const numLayers = layer.num_layers ?? 1;
  return (
    <>
      <Row>
        <Label>Hidden size</Label>
        <NumberInput value={layer.hidden_size ?? 64} min={1} onChange={(n) => update({ hidden_size: n })} />
      </Row>
      <Row>
        <Label>Num layers</Label>
        <NumberInput value={numLayers} min={1} max={8} onChange={(n) => update({ num_layers: n })} />
      </Row>
      <Row>
        <Checkbox
          checked={layer.bidirectional ?? false}
          onChange={(v) => update({ bidirectional: v })}
          label="Bidirectional"
        />
      </Row>
      {numLayers > 1 && (
        <Row>
          <Label>RNN dropout</Label>
          <NumberInput
            value={layer.rnn_dropout ?? 0.0}
            min={0}
            max={0.99}
            step={0.05}
            onChange={(n) => update({ rnn_dropout: n })}
          />
        </Row>
      )}
    </>
  );
}

// ── LayerCard ─────────────────────────────────────────────────────────────────

interface Props {
  layer: LayerConfig;
  shape?: number[] | null;
}

export function LayerCard({ layer, shape }: Props) {
  const updateLayer  = useBuilderStore((s) => s.updateLayer);
  const removeLayer  = useBuilderStore((s) => s.removeLayer);

  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: layer.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };

  const update = (patch: Partial<LayerConfig>) => updateLayer(layer.id, patch);

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="
        bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-3
        flex flex-col gap-2.5 select-none
      "
    >
      {/* Header row */}
      <div className="flex items-center gap-2">
        <button
          {...attributes}
          {...listeners}
          className="text-zinc-600 hover:text-zinc-400 cursor-grab active:cursor-grabbing touch-none"
          aria-label="Drag to reorder"
        >
          <GripVertical size={15} />
        </button>

        <LayerTypeSelect id={layer.id} type={layer.type} />

        <ShapeBadge shape={shape} />

        <button
          onClick={() => removeLayer(layer.id)}
          className="ml-auto text-zinc-600 hover:text-red-400 transition-colors"
          aria-label="Remove layer"
        >
          <Trash2 size={14} />
        </button>
      </div>

      {/* Controls */}
      {layer.type === "Linear"    && <LinearControls    layer={layer} update={update} />}
      {layer.type === "Conv2d"    && <Conv2dControls    layer={layer} update={update} />}
      {layer.type === "MaxPool2d" && <MaxPool2dControls layer={layer} update={update} />}
      {layer.type === "Flatten"   && null}
      {(layer.type === "LSTM" || layer.type === "GRU") && <RNNControls layer={layer} update={update} />}
    </div>
  );
}
