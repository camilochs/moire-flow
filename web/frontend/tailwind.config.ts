import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      colors: {
        // Surfaces — neutral grays with cool undertone
        canvas: {
          DEFAULT: "rgb(250 250 251)",
          dark: "rgb(9 9 11)", // zinc-950
        },
        surface: {
          DEFAULT: "rgb(255 255 255)",
          dark: "rgb(24 24 27)", // zinc-900
        },
        elevated: {
          DEFAULT: "rgb(248 250 252)",
          dark: "rgb(39 39 42)", // zinc-800
        },
        // Accent — indigo
        brand: {
          50: "rgb(238 242 255)",
          100: "rgb(224 231 255)",
          400: "rgb(129 140 248)",
          500: "rgb(99 102 241)",
          600: "rgb(79 70 229)",
          700: "rgb(67 56 202)",
        },
      },
      boxShadow: {
        soft: "0 1px 2px 0 rgb(0 0 0 / 0.04), 0 1px 3px 0 rgb(0 0 0 / 0.06)",
        elev:
          "0 4px 6px -1px rgb(0 0 0 / 0.06), 0 2px 4px -2px rgb(0 0 0 / 0.04)",
        glow: "0 0 0 1px rgb(99 102 241 / 0.18), 0 4px 12px rgb(99 102 241 / 0.18)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(2px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 150ms ease-out",
      },
    },
  },
  plugins: [],
} satisfies Config;
