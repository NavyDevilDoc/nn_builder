import { useRef, useState } from "react";
import { Play, Square, AlertCircle } from "lucide-react";
import { useBuilderStore } from "../store/useBuilderStore";
import { streamTraining, cancelTraining } from "../api/client";
import type { EpochEvent } from "../types/config";

function fmt(n: number, decimals = 4) {
  return n.toFixed(decimals);
}

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5 min-w-[72px]">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className="text-sm font-mono text-zinc-200">{value}</span>
    </div>
  );
}

export function TrainingPanel() {
  const training      = useBuilderStore((s) => s.training);
  const hyperparams   = useBuilderStore((s) => s.hyperparams);
  const layers        = useBuilderStore((s) => s.layers);
  const fixedSeed     = useBuilderStore((s) => s.fixedSeed);
  const getConfig     = useBuilderStore((s) => s.getConfig);
  const startTraining  = useBuilderStore((s) => s.startTraining);
  const setSessionId   = useBuilderStore((s) => s.setSessionId);
  const pushEpoch      = useBuilderStore((s) => s.pushEpoch);
  const finishTraining = useBuilderStore((s) => s.finishTraining);

  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Latest epoch for live stat display
  const history = training.history;
  const lastIdx = history.train_loss.length - 1;
  const hasStats = lastIdx >= 0;

  const totalEpochs = hyperparams.epochs;
  const progress = totalEpochs > 0
    ? Math.min(100, (training.epochsDone / totalEpochs) * 100)
    : 0;

  async function handleTrain() {
    if (layers.length === 0) {
      setError("Add at least one hidden layer before training.");
      return;
    }
    setError(null);

    const abort = new AbortController();
    abortRef.current = abort;

    // sessionId is generated server-side and returned in the "done" event;
    // use a placeholder until then so startTraining can reset history now.
    startTraining("pending");

    try {
      const base   = getConfig();
      const config = fixedSeed ? base : { ...base, seed: Math.floor(Math.random() * 2 ** 31) };
      for await (const event of streamTraining(config, abort.signal)) {
        if (event.type === "epoch") {
          pushEpoch(event as EpochEvent);
        } else if (event.type === "done") {
          setSessionId(event.session_id);
          finishTraining();
        } else if (event.type === "error") {
          setError(event.message);
          finishTraining();
          return;
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setError((err as Error).message);
      }
      finishTraining();
    } finally {
      abortRef.current = null;
    }
  }

  function handleStop() {
    if (abortRef.current) {
      abortRef.current.abort();
    }
    if (training.sessionId && training.sessionId !== "pending") {
      cancelTraining(training.sessionId);
    }
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 flex flex-col gap-3">
      {/* Header row */}
      <div className="flex items-center gap-3">
        {!training.active ? (
          <button
            onClick={handleTrain}
            className="
              flex items-center gap-2 px-4 py-2 rounded
              bg-emerald-700 hover:bg-emerald-600 active:bg-emerald-800
              text-sm font-medium text-white transition-colors
            "
          >
            <Play size={14} />
            Train
          </button>
        ) : (
          <button
            onClick={handleStop}
            className="
              flex items-center gap-2 px-4 py-2 rounded
              bg-zinc-700 hover:bg-zinc-600 active:bg-zinc-800
              text-sm font-medium text-zinc-200 transition-colors
            "
          >
            <Square size={14} />
            Stop
          </button>
        )}

        {training.active && (
          <span className="text-xs text-zinc-400 tabular-nums">
            Epoch {training.epochsDone} / {totalEpochs}
          </span>
        )}

        {!training.active && training.epochsDone > 0 && (
          <span className="text-xs text-zinc-500 tabular-nums">
            Completed {training.epochsDone} epoch{training.epochsDone !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Progress bar — visible while active or after first epoch */}
      {(training.active || training.epochsDone > 0) && (
        <div className="h-1.5 w-full rounded-full bg-zinc-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-emerald-600 transition-all duration-200"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Live stats */}
      {hasStats && (
        <div className="flex flex-wrap gap-4 pt-1">
          <StatPill label="train loss" value={fmt(history.train_loss[lastIdx])} />
          <StatPill label="val loss"   value={fmt(history.val_loss[lastIdx])} />
          <StatPill label="train acc"  value={`${(history.train_acc[lastIdx] * 100).toFixed(1)}%`} />
          <StatPill label="val acc"    value={`${(history.val_acc[lastIdx] * 100).toFixed(1)}%`} />
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="flex items-start gap-2 text-xs text-red-400 bg-red-950/40 border border-red-800 rounded px-3 py-2">
          <AlertCircle size={13} className="mt-0.5 shrink-0" />
          {error}
        </div>
      )}
    </div>
  );
}
