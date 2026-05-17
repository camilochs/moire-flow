import { useEffect } from "react";
import { Boxes, Search, AlertCircle } from "lucide-react";
import { useFlow } from "../lib/store";
import type { BoxDescriptor } from "../lib/api";
import { useState, useMemo } from "react";

export function Sidebar() {
  const { catalog, loadingCatalog, errorCatalog, loadCatalog } = useFlow();
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (catalog.length === 0 && !loadingCatalog) {
      loadCatalog();
    }
  }, [catalog.length, loadingCatalog, loadCatalog]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return catalog;
    return catalog.filter(
      (b) =>
        b.name.toLowerCase().includes(q) ||
        b.description.toLowerCase().includes(q),
    );
  }, [catalog, query]);

  return (
    <aside className="flex h-full w-72 flex-col border-r border-zinc-200 dark:border-zinc-800 panel">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
        <Boxes className="h-4 w-4 text-brand-500" />
        <span className="text-sm font-semibold tracking-tight">Box catalog</span>
        <span className="chip ml-auto">{catalog.length}</span>
      </div>
      <div className="px-3 py-2 border-b border-zinc-200 dark:border-zinc-800">
        <div className="relative">
          <Search className="absolute left-2 top-1.5 h-3.5 w-3.5 text-zinc-400" />
          <input
            className="w-full rounded-md bg-elevated dark:bg-elevated-dark
                       pl-7 pr-2 py-1.5 text-xs
                       text-zinc-800 dark:text-zinc-200
                       placeholder:text-zinc-400 dark:placeholder:text-zinc-500
                       border border-transparent focus:border-brand-500
                       focus:outline-none focus:ring-2 focus:ring-brand-500/20"
            placeholder="Filter boxes…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-2 py-2">
        {errorCatalog && (
          <div className="mx-2 my-3 flex items-start gap-2 rounded-md bg-red-50 dark:bg-red-950/30
                          text-red-700 dark:text-red-300 text-xs px-2.5 py-2">
            <AlertCircle className="h-3.5 w-3.5 mt-0.5" />
            <span>Failed to load catalog: {errorCatalog}</span>
          </div>
        )}
        {loadingCatalog && (
          <div className="text-xs text-zinc-500 px-3 py-2">Loading…</div>
        )}
        <ul className="flex flex-col gap-1">
          {filtered.map((box) => (
            <BoxCard key={box.name} box={box} />
          ))}
        </ul>
      </div>
      <div className="px-4 py-2 border-t border-zinc-200 dark:border-zinc-800
                      text-[10px] text-zinc-500 dark:text-zinc-500">
        Drag boxes onto the canvas.
      </div>
    </aside>
  );
}

function BoxCard({ box }: { box: BoxDescriptor }) {
  const onDragStart = (e: React.DragEvent<HTMLLIElement>) => {
    e.dataTransfer.setData("application/x-moire-box", box.name);
    e.dataTransfer.effectAllowed = "move";
  };
  return (
    <li
      draggable
      onDragStart={onDragStart}
      className="group cursor-grab active:cursor-grabbing select-none
                 rounded-md px-3 py-2 text-left
                 hover:bg-elevated dark:hover:bg-elevated-dark
                 border border-transparent hover:border-zinc-200
                 dark:hover:border-zinc-700 transition-colors"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-zinc-800 dark:text-zinc-100">
          {prettyBoxName(box.name)}
        </span>
        <span className="text-[10px] font-mono text-zinc-400">
          {box.inputs.length}↦{box.outputs.length}
        </span>
      </div>
      <p className="mt-0.5 text-[11px] leading-snug text-zinc-500 dark:text-zinc-400 line-clamp-2">
        {box.description}
      </p>
    </li>
  );
}

export function prettyBoxName(name: string): string {
  return name
    .split("_")
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(" ");
}
