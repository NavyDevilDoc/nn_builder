import { useBuilderStore } from "../store/useBuilderStore";
import type { BuilderConfig } from "../types/config";

type Hyperparams = Omit<BuilderConfig, "hidden_layers">;

// ── Shared controls ───────────────────────────────────────────────────────────

function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2 border-b border-zinc-800 last:border-0">
      <span className="text-xs text-zinc-400 shrink-0">{label}</span>
      <div className="flex items-center gap-2">{children}</div>
    </div>
  );
}

function Num({
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
        w-24 bg-zinc-800 border border-zinc-700 rounded px-2 py-1
        text-sm text-zinc-100 text-right
        focus:outline-none focus:ring-1 focus:ring-zinc-500
      "
    />
  );
}

function Check({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex items-center gap-1.5 cursor-pointer select-none">
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

function Sel<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: T[];
  onChange: (v: T) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      className="
        bg-zinc-800 border border-zinc-700 rounded px-2 py-1
        text-sm text-zinc-100
        focus:outline-none focus:ring-1 focus:ring-zinc-500
      "
    >
      {options.map((o) => (
        <option key={o} value={o} className="bg-zinc-900">
          {o}
        </option>
      ))}
    </select>
  );
}

// ── HyperparamPanel ───────────────────────────────────────────────────────────

export function HyperparamPanel() {
  const hyperparams      = useBuilderStore((s) => s.hyperparams);
  const updateHyperparams = useBuilderStore((s) => s.updateHyperparams);

  const set = <K extends keyof Hyperparams>(key: K, val: Hyperparams[K]) =>
    updateHyperparams({ [key]: val } as Partial<Hyperparams>);

  const isFullBatch = hyperparams.batch_size === "full";

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3">
      <h2 className="mb-1 text-xs font-medium uppercase tracking-widest text-zinc-500">
        Hyperparameters
      </h2>

      <FieldRow label="Optimizer">
        <Sel
          value={hyperparams.optimizer}
          options={["Adam", "SGD", "Momentum"]}
          onChange={(v) => set("optimizer", v)}
        />
      </FieldRow>

      <FieldRow label="Learning rate">
        <Num
          value={hyperparams.lr}
          min={0.00001}
          max={1}
          step={0.001}
          onChange={(n) => set("lr", n)}
        />
      </FieldRow>

      <FieldRow label="Epochs">
        <Num value={hyperparams.epochs} min={1} max={10000} onChange={(n) => set("epochs", n)} />
      </FieldRow>

      <FieldRow label="Batch size">
        <Check
          checked={isFullBatch}
          onChange={(v) => set("batch_size", v ? "full" : 32)}
          label="Full"
        />
        {!isFullBatch && (
          <Num
            value={hyperparams.batch_size as number}
            min={1}
            onChange={(n) => set("batch_size", n)}
          />
        )}
      </FieldRow>

      <FieldRow label="Weight decay">
        <Num
          value={hyperparams.weight_decay}
          min={0}
          step={0.0001}
          onChange={(n) => set("weight_decay", n)}
        />
      </FieldRow>

      <FieldRow label="Seed">
        <Num value={hyperparams.seed} min={0} onChange={(n) => set("seed", n)} />
      </FieldRow>

      <FieldRow label="Early stopping">
        <Check
          checked={hyperparams.early_stopping}
          onChange={(v) => set("early_stopping", v)}
          label="Enabled"
        />
      </FieldRow>

      {hyperparams.early_stopping && (
        <>
          <FieldRow label="ES patience">
            <Num value={hyperparams.es_patience} min={1} onChange={(n) => set("es_patience", n)} />
          </FieldRow>
          <FieldRow label="ES min delta">
            <Num
              value={hyperparams.es_min_delta}
              min={0}
              step={0.0001}
              onChange={(n) => set("es_min_delta", n)}
            />
          </FieldRow>
        </>
      )}

      <FieldRow label="Device">
        <Sel
          value={hyperparams.device}
          options={["cpu", "cuda"]}
          onChange={(v) => set("device", v)}
        />
      </FieldRow>
    </div>
  );
}
