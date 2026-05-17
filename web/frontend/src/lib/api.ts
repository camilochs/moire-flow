export type JSONSchema = Record<string, unknown> & {
  properties?: Record<string, JSONSchema>;
  required?: string[];
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
};

export interface BoxDescriptor {
  name: string;
  description: string;
  inputs_schema: JSONSchema;
  params_schema: JSONSchema;
  outputs_schema: JSONSchema;
  inputs: string[];
  outputs: string[];
  params: string[];
}

export interface WorkflowNode {
  id: string;
  box_name: string;
  params: Record<string, unknown>;
  inputs: Record<string, unknown>;
}

export interface WorkflowEdge {
  from_node: string;
  from_field: string;
  to_node: string;
  to_field: string;
}

export interface WorkflowSpec {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

export interface RunResponse {
  ok: boolean;
  results: Record<string, unknown>;
  error: string | null;
}

const API_BASE = "/api";

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function jpost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  health: () => jget<{ status: string; version: string }>("/health"),
  listBoxes: () => jget<BoxDescriptor[]>("/boxes"),
  describe: (name: string) => jget<BoxDescriptor>(`/boxes/${name}`),
  validate: (spec: WorkflowSpec) =>
    jpost<{ ok: boolean; errors: string[]; nodes: number; edges: number }>(
      "/workflows/validate",
      spec
    ),
  run: (spec: WorkflowSpec, external_inputs?: Record<string, Record<string, unknown>>) =>
    jpost<RunResponse>("/workflows/run", { spec, external_inputs }),
};
