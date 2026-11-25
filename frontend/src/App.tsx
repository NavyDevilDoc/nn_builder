import { useQuery } from "@tanstack/react-query";
import { fetchDatasets } from "./api/client";
import { useBuilderStore } from "./store/useBuilderStore";
import { LayerBuilder }   from "./components/layers/LayerBuilder";
import { HyperparamPanel } from "./components/HyperparamPanel";
import { TrainingPanel }  from "./components/TrainingPanel";
import { ResultsPanel }   from "./components/ResultsPanel";
import { CodeViewer }     from "./components/CodeViewer";
import { ArchDiagram }    from "./components/ArchDiagram";
import type { DatasetInfo } from "./types/config";

// ── Dataset selector ──────────────────────────────────────────────────────────

function DatasetMeta({ info }: { info: DatasetInfo }) {
  return (
    <span className="flex gap-1.5 flex-wrap text-xs text-zinc-500">
      <span className="rounded bg-zinc-800 px-1.5 py-0.5">{info.task}</span>
      <span className="rounded bg-zinc-800 px-1.5 py-0.5">{info.n_classes} classes</span>
      <span className="rounded bg-zinc-800 px-1.5 py-0.5">dim={info.input_dim}</span>
      {info.input_shape && (
        <span className="rounded bg-zinc-800 px-1.5 py-0.5">
          {info.input_shape.join("×")}
        </span>
      )}
    </span>
  );
}

function DatasetSelector() {
  const { data, isLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: fetchDatasets,
  });

  const selected          = useBuilderStore((s) => s.hyperparams.dataset);
  const updateHyperparams = useBuilderStore((s) => s.updateHyperparams);

  if (isLoading) return <p className="text-sm text-zinc-500">Loading datasets…</p>;
  if (!data) return null;

  const meta = data.meta[selected];

  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-medium uppercase tracking-widest text-zinc-500">
        Dataset
      </label>
      <div className="flex items-center gap-3 flex-wrap">
        <select
          value={selected}
          onChange={(e) => updateHyperparams({ dataset: e.target.value })}
          className="
            bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5
            text-sm text-zinc-100 font-mono
            focus:outline-none focus:ring-1 focus:ring-zinc-500
          "
        >
          {data.names.map((name) => (
            <option key={name} value={name} className="bg-zinc-900">
              {name}
            </option>
          ))}
        </select>
        {meta && <DatasetMeta info={meta} />}
      </div>
      {meta && <p className="text-xs text-zinc-500">{meta.description}</p>}
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      <header className="border-b border-zinc-800 px-8 py-4 shrink-0">
        <h1 className="text-xl font-semibold tracking-tight">Neural Network Builder</h1>
      </header>

      {/* Two-column layout — stacks on narrow screens */}
      <div className="flex-1 grid grid-cols-1 xl:grid-cols-2 gap-0 min-h-0">

        {/* ── Left: config ── */}
        <div className="border-r border-zinc-800 overflow-y-auto px-6 py-6 flex flex-col gap-6">
          <DatasetSelector />

          <section className="flex flex-col gap-3">
            <h2 className="text-xs font-medium uppercase tracking-widest text-zinc-500">
              Hidden Layers
            </h2>
            <LayerBuilder />
          </section>

          <HyperparamPanel />
        </div>

        {/* ── Right: outputs ── */}
        <div className="overflow-y-auto px-6 py-6 flex flex-col gap-6">
          <section className="flex flex-col gap-3">
            <h2 className="text-xs font-medium uppercase tracking-widest text-zinc-500">
              Architecture
            </h2>
            <ArchDiagram />
          </section>

          <TrainingPanel />
          <ResultsPanel />
          <CodeViewer />
        </div>

      </div>
    </div>
  );
}
