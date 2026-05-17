# moire-flow web

Visual workflow studio for moire-flow. Drag boxes from the catalog, connect
inputs/outputs, edit params in the inspector, and run the pipeline.

![Studio demo](docs/demo.gif)

A short walkthrough of the UI (also available as
[`docs/demo.mp4`](docs/demo.mp4)): empty canvas in dark mode, filter the
catalog, import a sample workflow, select a node, edit a parameter, toggle
light/dark theme, click Run to see the result appear in the inspector.

## Architecture

```
web/
├── backend/        FastAPI app exposing BOX_REGISTRY + WorkflowEngine
└── frontend/       Vite + React + TypeScript + Tailwind + React Flow
```

The backend reads schemas directly from `moire_flow.boxes.BOX_REGISTRY`, so
adding a new box in `src/` automatically makes it available in the UI — no
duplication.

## Run in dev

In two terminals:

```bash
# 1. Backend (port 8765)
uv run uvicorn web.backend.server:app --reload --port 8765

# 2. Frontend (port 5173, proxies /api → 8765)
cd web/frontend
npm install     # first time only
npm run dev
```

Open <http://localhost:5173>.

## Build for prod

```bash
cd web/frontend
npm run build     # → dist/
```

Serve `dist/` behind nginx and point `/api` to the FastAPI app.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | Service health + version |
| GET  | `/api/boxes`  | Catalog: every box with its three Pydantic schemas |
| GET  | `/api/boxes/{name}` | Single box descriptor |
| POST | `/api/workflows/validate` | Validate a WorkflowSpec without executing |
| POST | `/api/workflows/run` | Execute a WorkflowSpec, return serialized outputs |

## Theme

Light + dark, persisted to `localStorage` under `moire-flow-theme`. Initial
value falls back to `prefers-color-scheme`. The pre-render script in
`index.html` applies the class before React mounts to avoid a theme flash.

## Adding a new box to the UI

1. Add the box in `src/moire_flow/boxes/` and decorate it with `@register_box`.
2. Restart the backend.
3. The new box appears in the sidebar automatically with its schemas.
