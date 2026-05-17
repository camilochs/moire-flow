import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useFlow, type BoxNode as BoxNodeType } from "../lib/store";
import { prettyBoxName } from "./Sidebar";

function BoxNodeImpl({ data, selected, id }: NodeProps<BoxNodeType>) {
  const { box } = data;
  const setSelected = useFlow((s) => s.setSelected);

  return (
    <div
      onClick={() => setSelected(id)}
      className={`group relative min-w-[220px] rounded-xl border bg-surface dark:bg-surface-dark
                  shadow-soft transition-all
                  ${selected
                    ? "border-brand-500 shadow-glow"
                    : "border-zinc-200 dark:border-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-600"}`}
    >
      <div className="flex items-center justify-between gap-2 rounded-t-xl
                      bg-gradient-to-r from-brand-500/10 via-brand-500/5 to-transparent
                      dark:from-brand-500/20 dark:via-brand-500/10 dark:to-transparent
                      border-b border-zinc-200 dark:border-zinc-700
                      px-3 py-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="h-1.5 w-1.5 rounded-full bg-brand-500 shadow-[0_0_8px] shadow-brand-500/70 flex-shrink-0" />
          <span className="truncate text-sm font-semibold text-zinc-800 dark:text-zinc-100">
            {prettyBoxName(box.name)}
          </span>
        </div>
        <span className="chip">{box.params.length}p</span>
      </div>
      <div className="grid grid-cols-2 gap-1 px-2 py-2 text-[11px]">
        <ul className="flex flex-col gap-1">
          {box.inputs.map((field) => (
            <li
              key={`in-${field}`}
              className="relative rounded-sm pl-3 pr-1 py-0.5
                         text-zinc-600 dark:text-zinc-400 font-mono"
            >
              <Handle
                id={`in:${field}`}
                type="target"
                position={Position.Left}
                style={{ left: -6 }}
              />
              {field}
            </li>
          ))}
        </ul>
        <ul className="flex flex-col gap-1 text-right">
          {box.outputs.map((field) => (
            <li
              key={`out-${field}`}
              className="relative rounded-sm pl-1 pr-3 py-0.5
                         text-zinc-600 dark:text-zinc-400 font-mono"
            >
              <Handle
                id={`out:${field}`}
                type="source"
                position={Position.Right}
                style={{ right: -6 }}
              />
              {field}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export const BoxNode = memo(BoxNodeImpl);
