import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import SyntaxHighlighter from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { ChevronDown, ChevronUp, Copy, Check } from "lucide-react";
import { useBuilderStore } from "../store/useBuilderStore";
import { fetchCode } from "../api/client";

export function CodeViewer() {
  const [open,   setOpen]   = useState(false);
  const [copied, setCopied] = useState(false);

  const getConfig = useBuilderStore((s) => s.getConfig);
  const layers    = useBuilderStore((s) => s.layers);
  const hyperparams = useBuilderStore((s) => s.hyperparams);

  // Re-fetch whenever the config changes; disabled when panel is closed.
  const config = getConfig();
  const { data, isLoading, error } = useQuery({
    queryKey: ["code", config],
    queryFn:  () => fetchCode(config),
    enabled:  open && layers.length > 0,
    staleTime: 0,
  });

  async function handleCopy() {
    if (!data?.code) return;
    await navigator.clipboard.writeText(data.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  const hasLayers = layers.length > 0;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
      {/* Toggle header */}
      <button
        onClick={() => setOpen((o) => !o)}
        disabled={!hasLayers}
        className="
          w-full flex items-center justify-between px-4 py-3
          text-xs font-medium uppercase tracking-widest text-zinc-500
          hover:text-zinc-300 hover:bg-zinc-800/50 transition-colors
          disabled:cursor-not-allowed disabled:opacity-40
        "
      >
        <span>Generated Code</span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {/* Body */}
      {open && (
        <div className="border-t border-zinc-800">
          {isLoading && (
            <p className="px-4 py-6 text-xs text-zinc-500 text-center">Generating…</p>
          )}
          {error && (
            <p className="px-4 py-4 text-xs text-red-400">
              {(error as Error).message}
            </p>
          )}
          {data?.code && (
            <div className="relative">
              {/* Copy button */}
              <button
                onClick={handleCopy}
                className="
                  absolute top-3 right-3 z-10
                  flex items-center gap-1.5 px-2 py-1 rounded
                  bg-zinc-700 hover:bg-zinc-600 border border-zinc-600
                  text-xs text-zinc-300 transition-colors
                "
              >
                {copied ? <Check size={11} /> : <Copy size={11} />}
                {copied ? "Copied" : "Copy"}
              </button>

              <SyntaxHighlighter
                language="python"
                style={atomOneDark}
                customStyle={{
                  margin: 0,
                  padding: "1rem",
                  background: "#111113",
                  fontSize: "0.75rem",
                  lineHeight: "1.6",
                  borderRadius: 0,
                }}
                showLineNumbers
                lineNumberStyle={{ color: "#52525b", minWidth: "2.5rem" }}
              >
                {data.code}
              </SyntaxHighlighter>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
