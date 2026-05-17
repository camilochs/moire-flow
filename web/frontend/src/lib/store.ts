import { create } from "zustand";
import type {
  Edge as RFEdge,
  Node as RFNode,
  XYPosition,
} from "@xyflow/react";
import { addEdge, applyEdgeChanges, applyNodeChanges } from "@xyflow/react";
import { api, type BoxDescriptor, type WorkflowEdge, type WorkflowSpec } from "./api";

export interface BoxNodeData extends Record<string, unknown> {
  box: BoxDescriptor;
  params: Record<string, unknown>;
  inputs: Record<string, unknown>;
}

export type BoxNode = RFNode<BoxNodeData, "box">;
export type BoxEdge = RFEdge;

interface FlowStore {
  catalog: BoxDescriptor[];
  loadingCatalog: boolean;
  errorCatalog: string | null;
  nodes: BoxNode[];
  edges: BoxEdge[];
  selectedNodeId: string | null;
  runResults: Record<string, unknown> | null;
  runError: string | null;
  running: boolean;

  loadCatalog: () => Promise<void>;

  // canvas
  onNodesChange: (changes: Parameters<typeof applyNodeChanges>[0]) => void;
  onEdgesChange: (changes: Parameters<typeof applyEdgeChanges>[0]) => void;
  onConnect: (connection: Parameters<typeof addEdge>[0]) => void;
  addBox: (box: BoxDescriptor, position: XYPosition) => void;
  removeNode: (id: string) => void;
  setSelected: (id: string | null) => void;
  updateNodeParams: (id: string, params: Record<string, unknown>) => void;
  updateNodeInputs: (id: string, inputs: Record<string, unknown>) => void;

  // execution / export
  setRunResults: (r: Record<string, unknown> | null, e: string | null) => void;
  setRunning: (b: boolean) => void;
  buildSpec: () => WorkflowSpec;
  loadSpec: (spec: WorkflowSpec) => void;
  clear: () => void;
}

let _nodeCounter = 0;
function nodeId(boxName: string): string {
  _nodeCounter += 1;
  return `${boxName}_${_nodeCounter}`;
}

export const useFlow = create<FlowStore>((set, get) => ({
  catalog: [],
  loadingCatalog: false,
  errorCatalog: null,
  nodes: [],
  edges: [],
  selectedNodeId: null,
  runResults: null,
  runError: null,
  running: false,

  loadCatalog: async () => {
    set({ loadingCatalog: true, errorCatalog: null });
    try {
      const catalog = await api.listBoxes();
      set({ catalog, loadingCatalog: false });
    } catch (err) {
      set({
        loadingCatalog: false,
        errorCatalog: err instanceof Error ? err.message : String(err),
      });
    }
  },

  onNodesChange: (changes) =>
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) as BoxNode[] })),
  onEdgesChange: (changes) =>
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })),
  onConnect: (connection) =>
    set((s) => ({ edges: addEdge({ ...connection, animated: false }, s.edges) })),

  addBox: (box, position) => {
    const id = nodeId(box.name);
    const params = _defaultsFromSchema(box.params_schema);
    const node: BoxNode = {
      id,
      type: "box",
      position,
      data: { box, params, inputs: {} },
    };
    set((s) => ({ nodes: [...s.nodes, node], selectedNodeId: id }));
  },

  removeNode: (id) =>
    set((s) => ({
      nodes: s.nodes.filter((n) => n.id !== id),
      edges: s.edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: s.selectedNodeId === id ? null : s.selectedNodeId,
    })),

  setSelected: (id) => set({ selectedNodeId: id }),

  updateNodeParams: (id, params) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, params } } : n
      ),
    })),

  updateNodeInputs: (id, inputs) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, inputs } } : n
      ),
    })),

  setRunResults: (results, error) =>
    set({ runResults: results, runError: error }),
  setRunning: (running) => set({ running }),

  buildSpec: () => {
    const s = get();
    const nodes = s.nodes.map((n) => ({
      id: n.id,
      box_name: n.data.box.name,
      params: n.data.params,
      inputs: n.data.inputs,
    }));
    const edges: WorkflowEdge[] = s.edges.map((e) => ({
      from_node: e.source,
      from_field: (e.sourceHandle ?? "").replace(/^out:/, ""),
      to_node: e.target,
      to_field: (e.targetHandle ?? "").replace(/^in:/, ""),
    }));
    return { nodes, edges };
  },

  loadSpec: (spec) => {
    const catalog = get().catalog;
    const byName = new Map(catalog.map((b) => [b.name, b]));
    const nodes: BoxNode[] = spec.nodes
      .map((n, idx) => {
        const box = byName.get(n.box_name);
        if (!box) return null;
        return {
          id: n.id,
          type: "box" as const,
          position: { x: 40 + 280 * idx, y: 60 + 40 * (idx % 3) },
          data: {
            box,
            params: { ..._defaultsFromSchema(box.params_schema), ...n.params },
            inputs: n.inputs,
          },
        };
      })
      .filter(Boolean) as BoxNode[];
    const edges: BoxEdge[] = spec.edges.map((e, i) => ({
      id: `e_${i}`,
      source: e.from_node,
      sourceHandle: `out:${e.from_field}`,
      target: e.to_node,
      targetHandle: `in:${e.to_field}`,
    }));
    set({ nodes, edges, selectedNodeId: null });
  },

  clear: () => set({ nodes: [], edges: [], selectedNodeId: null, runResults: null, runError: null }),
}));

function _defaultsFromSchema(schema: { properties?: Record<string, { default?: unknown }> }): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, sub] of Object.entries(schema.properties ?? {})) {
    if (sub && Object.prototype.hasOwnProperty.call(sub, "default")) {
      out[key] = sub.default;
    }
  }
  return out;
}
