"""FastAPI server exposing the moire-flow box catalog + executor.

Run with:
    uv run uvicorn web.backend.server:app --reload --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from moire_flow import __version__
from moire_flow.boxes import BOX_REGISTRY
from moire_flow.engine import WorkflowEngine, WorkflowSpec

logger = logging.getLogger("moire-flow.server")

app = FastAPI(
    title="moire-flow",
    version=__version__,
    description="Modular workflow engine for moire lattice matching + MD setup.",
)

# CORS is permissive in dev — tighten via env in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/boxes")
def list_boxes() -> list[dict[str, Any]]:
    """Catalog of every registered box with its JSON schemas."""
    out: list[dict[str, Any]] = []
    for name in sorted(BOX_REGISTRY):
        cls = BOX_REGISTRY[name]
        out.append(_box_summary(name, cls))
    return out


@app.get("/api/boxes/{box_name}")
def describe_box(box_name: str) -> dict[str, Any]:
    if box_name not in BOX_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown box: {box_name}")
    return _box_summary(box_name, BOX_REGISTRY[box_name])


class ValidateResponse(BaseModel):
    ok: bool
    errors: list[str] = []
    nodes: int
    edges: int


@app.post("/api/workflows/validate", response_model=ValidateResponse)
def validate_workflow(spec_dict: dict) -> ValidateResponse:
    try:
        spec = WorkflowSpec.model_validate(spec_dict)
    except Exception as exc:
        return ValidateResponse(ok=False, errors=[str(exc)], nodes=0, edges=0)
    return ValidateResponse(ok=True, nodes=len(spec.nodes), edges=len(spec.edges))


class RunRequest(BaseModel):
    spec: dict
    external_inputs: dict[str, dict[str, Any]] | None = None


class RunResponse(BaseModel):
    ok: bool
    results: dict[str, Any] = {}
    error: str | None = None


@app.post("/api/workflows/run", response_model=RunResponse)
def run_workflow(req: RunRequest) -> RunResponse:
    try:
        spec = WorkflowSpec.model_validate(req.spec)
    except Exception as exc:
        return RunResponse(ok=False, error=f"Invalid spec: {exc}")
    try:
        results = WorkflowEngine().run(spec, external_inputs=req.external_inputs)
    except Exception as exc:
        logger.exception("workflow execution failed")
        return RunResponse(ok=False, error=str(exc))
    serialized: dict[str, Any] = {}
    for node_id, model in results.items():
        try:
            serialized[node_id] = model.model_dump(mode="json")
        except Exception as exc:
            serialized[node_id] = {"_unserializable": str(exc), "_type": type(model).__name__}
    return RunResponse(ok=True, results=serialized)


def _box_summary(name: str, cls: type) -> dict[str, Any]:
    return {
        "name": name,
        "description": cls.description,
        "inputs_schema": cls.inputs_schema.model_json_schema(),
        "params_schema": cls.params_schema.model_json_schema(),
        "outputs_schema": cls.outputs_schema.model_json_schema(),
        "inputs": list(cls.inputs_schema.model_fields),
        "outputs": list(cls.outputs_schema.model_fields),
        "params": list(cls.params_schema.model_fields),
    }


__all__ = ["app"]
