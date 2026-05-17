import { Moon, Sun, Play, Download, Upload, Trash2, Atom, Loader2 } from "lucide-react";
import { useRef } from "react";
import { useTheme } from "../lib/theme";
import { useFlow } from "../lib/store";
import { api, type WorkflowSpec } from "../lib/api";

export function TopBar() {
  const { theme, toggle } = useTheme();
  const {
    buildSpec,
    loadSpec,
    clear,
    setRunResults,
    setRunning,
    running,
    runError,
    nodes,
  } = useFlow();
  const fileInput = useRef<HTMLInputElement>(null);

  const onExport = () => {
    const spec = buildSpec();
    const blob = new Blob([JSON.stringify(spec, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "moire-flow-spec.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const onImportClick = () => fileInput.current?.click();
  const onImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    try {
      const spec = JSON.parse(text) as WorkflowSpec;
      loadSpec(spec);
    } catch (err) {
      console.error("Bad spec:", err);
    } finally {
      e.target.value = "";
    }
  };

  const onRun = async () => {
    setRunning(true);
    setRunResults(null, null);
    try {
      const spec = buildSpec();
      const out = await api.run(spec);
      if (out.ok) {
        setRunResults(out.results, null);
      } else {
        setRunResults(null, out.error ?? "Run failed");
      }
    } catch (err) {
      setRunResults(null, err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-zinc-200 dark:border-zinc-800 panel px-4">
      <div className="flex items-center gap-2 min-w-[200px]">
        <div className="relative flex h-8 w-8 items-center justify-center
                        rounded-lg bg-gradient-to-br from-brand-500 to-brand-700
                        shadow-glow">
          <Atom className="h-4 w-4 text-white" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold tracking-tight">moire-flow</div>
          <div className="text-[10px] uppercase tracking-widest text-zinc-500 dark:text-zinc-400">
            workflow studio
          </div>
        </div>
      </div>

      {runError && (
        <div className="rounded-md bg-red-50 dark:bg-red-950/30
                        text-red-700 dark:text-red-300 text-xs px-2.5 py-1
                        border border-red-200/60 dark:border-red-900/60 max-w-md truncate">
          {runError}
        </div>
      )}

      <div className="ml-auto flex items-center gap-1.5">
        <input
          ref={fileInput}
          type="file"
          accept="application/json"
          className="hidden"
          onChange={onImportFile}
        />
        <button onClick={onImportClick} className="btn">
          <Upload className="h-3.5 w-3.5" />
          Import
        </button>
        <button onClick={onExport} className="btn" disabled={nodes.length === 0}>
          <Download className="h-3.5 w-3.5" />
          Export
        </button>
        <button onClick={clear} className="btn" disabled={nodes.length === 0}>
          <Trash2 className="h-3.5 w-3.5" />
          Clear
        </button>
        <button
          onClick={onRun}
          disabled={running || nodes.length === 0}
          className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {running ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Play className="h-3.5 w-3.5" />
          )}
          Run
        </button>
        <div className="mx-1 h-6 w-px bg-zinc-200 dark:bg-zinc-700" />
        <button
          onClick={toggle}
          aria-label="Toggle theme"
          className="rounded-md p-1.5 text-zinc-600 dark:text-zinc-300
                     hover:bg-elevated dark:hover:bg-elevated-dark transition-colors"
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </button>
      </div>
    </header>
  );
}
