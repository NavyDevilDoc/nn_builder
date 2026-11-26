import { useMemo } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  type Node,
  type Edge,
} from "reactflow";
import "reactflow/dist/style.css";
import { useBuilderStore } from "../store/useBuilderStore";
import { useQuery } from "@tanstack/react-query";
import { fetchDatasets, validateConfig } from "../api/client";
import { effectiveSeqLen, type LayerConfig, type LayerType } from "../types/config";

// ── Colors matching LayerTypeSelect ──────────────────────────────────────────

const TYPE_COLOR: Record<LayerType, { border: string; text: string; bg: string }> = {
  Linear:   { border: "#71717a", text: "#d4d4d8", bg: "#27272a" },
  Conv2d:   { border: "#ca8a04", text: "#fbbf24", bg: "#1c1800" },
  MaxPool2d:{ border: "#b91c1c", text: "#f87171", bg: "#1c0a0a" },
  Flatten:  { border: "#0e7490", text: "#22d3ee", bg: "#021d22" },
  LSTM:     { border: "#7c3aed", text: "#c084fc", bg: "#130d22" },
  GRU:      { border: "#4338ca", text: "#818cf8", bg: "#0d0f22" },
};

function layerLabel(layer: LayerConfig): string {
  switch (layer.type) {
    case "Linear":    return `Linear\n${layer.units ?? 64}`;
    case "Conv2d":    return `Conv2d\n${layer.out_channels ?? 32}ch k=${layer.kernel_size ?? 3}`;
    case "MaxPool2d": return `MaxPool2d\nk=${layer.kernel_size ?? 2}`;
    case "Flatten":   return "Flatten";
    case "LSTM":      return `LSTM\nh=${layer.hidden_size ?? 64}`;
    case "GRU":       return `GRU\nh=${layer.hidden_size ?? 64}`;
  }
}

function shapeLabel(shape: number[] | null | undefined): string {
  if (shape === undefined || shape === null) return "?";
  return shape.join(" × ");
}

const NODE_W = 130;
const NODE_H = 56;
const GAP    = 80;

// ── Custom node ───────────────────────────────────────────────────────────────

function LayerNode({ data }: { data: { layer: LayerConfig; label: string } }) {
  const c = TYPE_COLOR[data.layer.type];
  const lines = data.label.split("\n");
  return (
    <div
      style={{
        width: NODE_W,
        height: NODE_H,
        background: c.bg,
        border: `1px solid ${c.border}`,
        borderRadius: 8,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "4px 8px",
        gap: 2,
      }}
    >
      <span style={{ color: c.text, fontSize: 12, fontWeight: 600 }}>{lines[0]}</span>
      {lines[1] && (
        <span style={{ color: c.border, fontSize: 10, fontFamily: "monospace" }}>
          {lines[1]}
        </span>
      )}
    </div>
  );
}

const NODE_TYPES = { layer: LayerNode };

// ── ArchDiagram ───────────────────────────────────────────────────────────────

export function ArchDiagram() {
  const layers     = useBuilderStore((s) => s.layers);
  const hyperparams = useBuilderStore((s) => s.hyperparams);

  const { data: datasetsData } = useQuery({
    queryKey: ["datasets"],
    queryFn:  fetchDatasets,
  });

  const { data: validation } = useQuery({
    queryKey: ["validate-diagram", layers, hyperparams.dataset, hyperparams.seq_len],
    queryFn: async () => {
      const meta = datasetsData?.meta[hyperparams.dataset];
      if (!meta || layers.length === 0) return null;
      return validateConfig(
        meta.input_shape ?? null,
        meta.input_dim,
        meta.task,
        meta.n_classes,
        layers.map(({ id: _id, ...rest }) => rest),
        effectiveSeqLen(meta, hyperparams.seq_len),
      );
    },
    enabled: !!datasetsData && layers.length > 0,
  });

  const { nodes, edges } = useMemo(() => {
    const shapes = validation?.shapes ?? [];
    const centerX = NODE_W / 2;
    const totalItems = layers.length + 2; // input + hidden + output
    const totalH = totalItems * NODE_H + (totalItems - 1) * GAP;
    let y = 0;

    const nodes: Node[] = [];
    const edges: Edge[] = [];

    // ── Input node ────────────────────────────────────────────────────────────
    const inputShape = shapes[0];
    nodes.push({
      id: "input",
      type: "default",
      position: { x: 0, y },
      data: { label: `Input\n${shapeLabel(inputShape)}` },
      style: {
        width: NODE_W, height: NODE_H,
        background: "#18181b", border: "1px solid #3f3f46",
        borderRadius: 8, fontSize: 11, color: "#a1a1aa",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", lineHeight: 1.5,
      },
    });
    y += NODE_H + GAP;

    // ── Hidden layer nodes ────────────────────────────────────────────────────
    layers.forEach((layer, i) => {
      const nodeId = layer.id;
      nodes.push({
        id: nodeId,
        type: "layer",
        position: { x: 0, y },
        data: { layer, label: layerLabel(layer) },
      });

      const prevId = i === 0 ? "input" : layers[i - 1].id;
      const edgeShape = shapes[i]; // shape ENTERING this layer
      edges.push({
        id:     `e-${prevId}-${nodeId}`,
        source: prevId,
        target: nodeId,
        label:  shapeLabel(edgeShape),
        labelStyle: { fill: "#71717a", fontSize: 10 },
        labelBgStyle: { fill: "#09090b" },
        style:  { stroke: "#3f3f46" },
      });

      y += NODE_H + GAP;
    });

    // ── Output node ───────────────────────────────────────────────────────────
    const outputShape = shapes[layers.length + 1];
    nodes.push({
      id: "output",
      type: "default",
      position: { x: 0, y },
      data: { label: `Output\n${shapeLabel(outputShape)}` },
      style: {
        width: NODE_W, height: NODE_H,
        background: "#052e16", border: "1px solid #16a34a",
        borderRadius: 8, fontSize: 11, color: "#4ade80",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", lineHeight: 1.5,
      },
    });

    const lastId = layers.length > 0 ? layers[layers.length - 1].id : "input";
    const preOutputShape = shapes[layers.length]; // shape entering output layer
    edges.push({
      id:     `e-${lastId}-output`,
      source: lastId,
      target: "output",
      label:  shapeLabel(preOutputShape),
      labelStyle: { fill: "#71717a", fontSize: 10 },
      labelBgStyle: { fill: "#09090b" },
      style:  { stroke: "#3f3f46" },
    });

    // Centre all nodes horizontally
    nodes.forEach((n) => { n.position.x = centerX - NODE_W / 2; });

    return { nodes, edges, totalH };
  }, [layers, validation]);

  if (layers.length === 0) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 h-48 flex items-center justify-center">
        <p className="text-xs text-zinc-600 select-none">Add layers to see the diagram</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-zinc-800 overflow-hidden h-72">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={true}
        zoomOnScroll={false}
        zoomOnPinch={true}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#27272a" variant={BackgroundVariant.Dots} gap={20} size={1} />
      </ReactFlow>
    </div>
  );
}
