import { useState, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Moon, Sun, Download, Upload } from "lucide-react";
import { fetchDatasets, fetchDatasetSample } from "./api/client";
import { useBuilderStore } from "./store/useBuilderStore";
import { LayerBuilder }   from "./components/layers/LayerBuilder";
import { HyperparamPanel } from "./components/HyperparamPanel";
import { TrainingPanel }  from "./components/TrainingPanel";
import { ResultsPanel }   from "./components/ResultsPanel";
import { CodeViewer }     from "./components/CodeViewer";
import { ArchDiagram }    from "./components/ArchDiagram";
import { effectiveSeqLen, type DatasetInfo, type LayerConfig } from "./types/config";
import type { BuilderConfig } from "./types/config";

// ── Dataset selector ──────────────────────────────────────────────────────────

function DatasetMeta({ info }: { info: DatasetInfo }) {
  return (
    <span className="flex gap-1.5 flex-wrap text-xs text-zinc-500">
      <span className="rounded bg-zinc-800 px-1.5 py-0.5">{info.task}</span>
      {info.task !== "regression" && (
        <span className="rounded bg-zinc-800 px-1.5 py-0.5">{info.n_classes} classes</span>
      )}
      <span className="rounded bg-zinc-800 px-1.5 py-0.5">
        {info.seq_len !== null ? `features=${info.input_dim}` : `dim=${info.input_dim}`}
      </span>
      {info.seq_len !== null && (
        <span className="rounded bg-zinc-800 px-1.5 py-0.5">sequential</span>
      )}
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
  const seqLenOverride    = useBuilderStore((s) => s.hyperparams.seq_len);
  const updateHyperparams = useBuilderStore((s) => s.updateHyperparams);

  const meta = data?.meta[selected];

  const { data: sampleData } = useQuery({
    queryKey: ["sample", selected],
    queryFn: () => fetchDatasetSample(selected),
    enabled: !!meta?.input_shape,
    staleTime: Infinity,
  });

  if (isLoading) return <p className="text-sm text-zinc-500">Loading datasets…</p>;
  if (!data) return null;

  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-medium uppercase tracking-widest text-zinc-500">
        Dataset
      </label>
      <div className="flex items-center gap-3 flex-wrap">
        <select
          value={selected}
          // A window-length override only makes sense for the dataset it was set on
          onChange={(e) => updateHyperparams({ dataset: e.target.value, seq_len: null })}
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
      {meta && meta.seq_len !== null && (
        <label className="flex items-center gap-2 text-xs text-zinc-400">
          Window length (seq_len)
          <input
            type="number"
            min={2}
            max={500}
            value={effectiveSeqLen(meta, seqLenOverride) ?? ""}
            onChange={(e) => {
              const n = Math.round(Number(e.target.value));
              if (n >= 2) updateHyperparams({ seq_len: n === meta.seq_len ? null : n });
            }}
            className="
              w-20 bg-zinc-800 border border-zinc-700 rounded px-2 py-1
              text-sm text-zinc-100 text-left
              focus:outline-none focus:ring-1 focus:ring-zinc-500
            "
          />
          {seqLenOverride !== null && seqLenOverride !== undefined && (
            <button
              onClick={() => updateHyperparams({ seq_len: null })}
              className="text-zinc-500 hover:text-zinc-300 underline-offset-2 hover:underline"
            >
              reset to {meta.seq_len}
            </button>
          )}
        </label>
      )}
      {sampleData && (
        <img
          src={sampleData.image}
          alt={`${selected} sample grid`}
          className="w-32 h-32 rounded border border-zinc-700 object-cover"
        />
      )}
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [dark, setDark] = useState(() =>
    document.documentElement.classList.contains("dark")
  );
  const fileInputRef = useRef<HTMLInputElement>(null);
  const loadConfig   = useBuilderStore((s) => s.loadConfig);

  function toggleDark() {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    setDark(next);
  }

  function handleExport() {
    const { layers, hyperparams } = useBuilderStore.getState();
    const json = JSON.stringify({ layers, hyperparams }, null, 2);
    const url  = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const a    = Object.assign(document.createElement("a"), { href: url, download: "nn_builder_config.json" });
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleImport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string) as {
          layers: LayerConfig[];
          hyperparams: Omit<BuilderConfig, "hidden_layers">;
        };
        if (Array.isArray(data.layers) && data.hyperparams) {
          loadConfig(data.layers, data.hyperparams);
        }
      } catch { /* ignore malformed JSON */ }
    };
    reader.readAsText(file);
    e.target.value = "";
  }

  return (
    <div className="h-screen bg-zinc-950 text-zinc-100 flex flex-col overflow-hidden">
      <header className="border-b border-zinc-800 px-8 py-4 shrink-0 flex items-center justify-between">
        <h1 className="text-xl font-semibold tracking-tight">Neural Network Builder</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExport}
            title="Export config as JSON"
            className="p-1.5 rounded text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800 transition-colors"
          >
            <Download size={16} />
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            title="Import config from JSON"
            className="p-1.5 rounded text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800 transition-colors"
          >
            <Upload size={16} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            className="hidden"
            onChange={handleImport}
          />
          <div className="w-px h-4 bg-zinc-700 mx-1" />
          <button
            onClick={toggleDark}
            title="Toggle dark / light mode"
            className="p-1.5 rounded text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800 transition-colors"
          >
            {dark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
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
