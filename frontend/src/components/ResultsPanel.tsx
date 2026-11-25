import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from "recharts";
import { ImageIcon, Loader2 } from "lucide-react";
import { useBuilderStore } from "../store/useBuilderStore";
import { fetchConfusionMatrix, fetchDecisionBoundary } from "../api/client";

// ── Chart helpers ─────────────────────────────────────────────────────────────

function buildChartData(history: {
  train_loss: number[];
  val_loss:   number[];
  train_acc:  number[];
  val_acc:    number[];
}) {
  return history.train_loss.map((_, i) => ({
    epoch:      i + 1,
    train_loss: +history.train_loss[i].toFixed(4),
    val_loss:   +history.val_loss[i].toFixed(4),
    train_acc:  +(history.train_acc[i] * 100).toFixed(2),
    val_acc:    +(history.val_acc[i] * 100).toFixed(2),
  }));
}

const CHART_PROPS = {
  margin: { top: 4, right: 8, left: -16, bottom: 0 },
};

function ChartContainer({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-zinc-500 select-none">{title}</span>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          {children as React.ReactElement}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ── Visualization image ───────────────────────────────────────────────────────

function VizImage({ src, alt }: { src: string; alt: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="text-xs text-zinc-500 select-none">{alt}</span>
      <img
        src={src}
        alt={alt}
        className="rounded border border-zinc-700 max-w-full"
      />
    </div>
  );
}

// ── ResultsPanel ──────────────────────────────────────────────────────────────

export function ResultsPanel() {
  const training         = useBuilderStore((s) => s.training);
  const results          = useBuilderStore((s) => s.results);
  const getConfig        = useBuilderStore((s) => s.getConfig);
  const setConfusionMatrix  = useBuilderStore((s) => s.setConfusionMatrix);
  const setDecisionBoundary = useBuilderStore((s) => s.setDecisionBoundary);

  const [loadingViz, setLoadingViz] = useState(false);
  const [vizError,   setVizError]   = useState<string | null>(null);

  const hasHistory = training.history.train_loss.length > 0;
  const sessionId  = training.sessionId;
  const canVisualize = !training.active && sessionId && sessionId !== "pending";

  if (!hasHistory) return null;

  const chartData = buildChartData(training.history);

  async function handleVisualize() {
    if (!canVisualize) return;
    setLoadingViz(true);
    setVizError(null);
    const config = getConfig();
    try {
      const [cmRes, dbRes] = await Promise.allSettled([
        fetchConfusionMatrix(sessionId!, config),
        fetchDecisionBoundary(sessionId!, config),
      ]);
      if (cmRes.status === "fulfilled") setConfusionMatrix(cmRes.value.image);
      else setVizError(`Confusion matrix: ${(cmRes.reason as Error).message}`);
      if (dbRes.status === "fulfilled") setDecisionBoundary(dbRes.value.image);
      // decision boundary silently skipped for non-2D datasets — no error
    } catch (err) {
      setVizError((err as Error).message);
    } finally {
      setLoadingViz(false);
    }
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 flex flex-col gap-5">
      <h2 className="text-xs font-medium uppercase tracking-widest text-zinc-500">
        Results
      </h2>

      {/* Loss curves */}
      <ChartContainer title="Loss">
        <LineChart data={chartData} {...CHART_PROPS}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis dataKey="epoch" tick={{ fill: "#71717a", fontSize: 11 }} />
          <YAxis tick={{ fill: "#71717a", fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6 }}
            labelStyle={{ color: "#a1a1aa" }}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }} />
          <Line type="monotone" dataKey="train_loss" name="train" stroke="#6366f1" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="val_loss"   name="val"   stroke="#f59e0b" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ChartContainer>

      {/* Accuracy curves */}
      <ChartContainer title="Accuracy (%)">
        <LineChart data={chartData} {...CHART_PROPS}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis dataKey="epoch" tick={{ fill: "#71717a", fontSize: 11 }} />
          <YAxis tick={{ fill: "#71717a", fontSize: 11 }} domain={[0, 100]} />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 6 }}
            labelStyle={{ color: "#a1a1aa" }}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }} />
          <Line type="monotone" dataKey="train_acc" name="train" stroke="#22c55e" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="val_acc"   name="val"   stroke="#06b6d4" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ChartContainer>

      {/* Visualizations */}
      {canVisualize && (
        <div className="flex flex-col gap-3">
          <button
            onClick={handleVisualize}
            disabled={loadingViz}
            className="
              self-start flex items-center gap-2 px-3 py-1.5 rounded
              bg-zinc-800 hover:bg-zinc-700 border border-zinc-700
              text-xs text-zinc-300 transition-colors
              disabled:opacity-50 disabled:cursor-not-allowed
            "
          >
            {loadingViz
              ? <Loader2 size={12} className="animate-spin" />
              : <ImageIcon size={12} />}
            {loadingViz ? "Generating…" : "Generate visualizations"}
          </button>

          {vizError && (
            <p className="text-xs text-red-400">{vizError}</p>
          )}

          {(results.confusionMatrix || results.decisionBoundary) && (
            <div className="flex flex-wrap gap-6 justify-center">
              {results.confusionMatrix  && (
                <VizImage src={results.confusionMatrix}  alt="Confusion matrix" />
              )}
              {results.decisionBoundary && (
                <VizImage src={results.decisionBoundary} alt="Decision boundary" />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
