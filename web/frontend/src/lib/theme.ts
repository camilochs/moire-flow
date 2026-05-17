import { create } from "zustand";

type Theme = "light" | "dark";
const STORAGE_KEY = "moire-flow-theme";

function readInitial(): Theme {
  if (typeof window === "undefined") return "dark";
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "dark") root.classList.add("dark");
  else root.classList.remove("dark");
  localStorage.setItem(STORAGE_KEY, theme);
}

interface ThemeStore {
  theme: Theme;
  toggle: () => void;
  set: (t: Theme) => void;
}

const initial = readInitial();
applyTheme(initial);

export const useTheme = create<ThemeStore>((set, get) => ({
  theme: initial,
  toggle: () => {
    const next = get().theme === "dark" ? "light" : "dark";
    applyTheme(next);
    set({ theme: next });
  },
  set: (theme) => {
    applyTheme(theme);
    set({ theme });
  },
}));
