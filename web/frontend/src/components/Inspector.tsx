import { useMemo } from "react";
import { Trash2, Info } from "lucide-react";
import { useFlow } from "../lib/store";
import { prettyBoxName } from "./Sidebar";
import type { JSONSchema } from "../lib/api";

export function Inspector() {
  const { nodes, selectedNodeId, updateNodeParams, removeNode, runResults } =
    useFlow();
  const node = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) ?? null,
    [nodes, selectedNodeId]
  );

  if (!node) {
    return (
      <aside className="flex h-full w-80 flex-col border-l border-zinc-200 dark:border-zinc-800 panel">
        <Header title="Inspector" />
        <div className="flex flex-1 flex-col items-center justify-center text-center
                        px-8 text-xs text-zinc-500 dark:text-zinc-400">
          <Info className="h-5 w-5 mb-2 text-zinc-400" />
          Select a node on the canvas to configure its parameters.
        </div>
      </aside>
    );
  }

  const { box, params } = node.data;
  const props = (box.params_schema.properties ?? {}) as Record<string, JSONSchema>;
  const nodeResult = runResults?.[node.id];

  return (
    <aside className="flex h-full w-80 flex-col border-l border-zinc-200 dark:border-zinc-800 panel">
      <Header title={prettyBoxName(box.name)} />
      <div className="px-4 py-3 border-b border-zinc-200 dark:border-zinc-800
                      flex items-start justify-between gap-3">
        <p className="text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
          {box.description}
        </p>
        <button
          aria-label="Remove node"
          onClick={() => removeNode(node.id)}
          className="rounded-md p-1.5 text-zinc-500 hover:bg-red-50 dark:hover:bg-red-950/40
                     hover:text-red-600 transition-colors"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
        <div className="field-label mb-1">Node ID</div>
        <code className="text-xs font-mono text-zinc-800 dark:text-zinc-200">
          {node.id}
        </code>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        <div className="field-label">Parameters</div>
        {Object.keys(props).length === 0 && (
          <p className="text-xs text-zinc-400">(no parameters)</p>
        )}
        {Object.entries(props).map(([key, subschema]) => (
          <ParamField
            key={key}
            name={key}
            schema={subschema}
            value={params[key]}
            onChange={(v) =>
              updateNodeParams(node.id, { ...params, [key]: v })
            }
          />
        ))}
      </div>
      {nodeResult !== undefined && (
        <div className="px-4 py-3 border-t border-zinc-200 dark:border-zinc-800
                        bg-elevated dark:bg-elevated-dark max-h-48 overflow-y-auto">
          <div className="field-label mb-1.5">Last run output</div>
          <pre className="text-[10px] leading-snug font-mono text-zinc-700 dark:text-zinc-300
                          whitespace-pre-wrap break-all">
            {JSON.stringify(nodeResult, null, 2).slice(0, 1000)}
          </pre>
        </div>
      )}
    </aside>
  );
}

function Header({ title }: { title: string }) {
  return (
    <div className="flex items-center px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
      <span className="text-sm font-semibold tracking-tight">{title}</span>
    </div>
  );
}

function ParamField({
  name,
  schema,
  value,
  onChange,
}: {
  name: string;
  schema: JSONSchema;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const t = inferType(schema);
  const label = schema.title ?? name;

  if (t === "boolean") {
    return (
      <label className="flex items-center justify-between gap-3 cursor-pointer">
        <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
          {label}
        </span>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 rounded border-zinc-300 dark:border-zinc-600
                     text-brand-500 focus:ring-brand-500/30"
        />
      </label>
    );
  }

  if (schema.enum && Array.isArray(schema.enum)) {
    return (
      <div className="space-y-1">
        <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
          {label}
        </label>
        <select
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          className="input"
        >
          {(schema.enum as unknown[]).map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {String(opt)}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (t === "integer" || t === "number") {
    return (
      <div className="space-y-1">
        <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
          {label}
        </label>
        <input
          type="number"
          step={t === "integer" ? 1 : "any"}
          value={value === undefined || value === null ? "" : (value as number)}
          onChange={(e) => {
            const v = e.target.value;
            if (v === "") return onChange(null);
            const parsed = t === "integer" ? parseInt(v, 10) : parseFloat(v);
            if (!Number.isNaN(parsed)) onChange(parsed);
          }}
          className="input"
        />
      </div>
    );
  }

  if (t === "object" || t === "array") {
    return (
      <div className="space-y-1">
        <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
          {label}
        </label>
        <textarea
          value={value === undefined ? "" : JSON.stringify(value, null, 2)}
          onChange={(e) => {
            try {
              onChange(JSON.parse(e.target.value));
            } catch {
              /* swallow until valid */
            }
          }}
          rows={3}
          className="input resize-y"
        />
      </div>
    );
  }

  // string fallback
  return (
    <div className="space-y-1">
      <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
        {label}
      </label>
      <input
        type="text"
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(e) => onChange(e.target.value)}
        className="input"
      />
    </div>
  );
}

function inferType(schema: JSONSchema): string {
  if (Array.isArray(schema.type)) return String(schema.type[0]);
  if (typeof schema.type === "string") return schema.type;
  if ("anyOf" in schema && Array.isArray((schema as { anyOf?: unknown }).anyOf)) {
    const anyOf = (schema as { anyOf: JSONSchema[] }).anyOf;
    const nonNull = anyOf.find((s) => s.type && s.type !== "null");
    if (nonNull?.type) return String(nonNull.type);
  }
  return "string";
}
