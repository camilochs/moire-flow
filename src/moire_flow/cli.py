"""moire-flow CLI: list registered boxes and execute a WorkflowSpec.

Usage:
    moire-flow list-boxes
    moire-flow describe <box_name>
    moire-flow run <spec.json> [--out DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from moire_flow import __version__
from moire_flow.boxes import BOX_REGISTRY
from moire_flow.engine import WorkflowEngine, WorkflowSpec

app = typer.Typer(
    name="moire-flow",
    help="Modular workflow engine for moire lattice matching + MD setup.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the moire-flow version."""
    typer.echo(__version__)


@app.command("list-boxes")
def list_boxes() -> None:
    """List the names of all registered boxes."""
    for name in sorted(BOX_REGISTRY):
        cls = BOX_REGISTRY[name]
        typer.echo(f"{name:<24} — {cls.description}")


@app.command()
def describe(box_name: str) -> None:
    """Print the JSON schemas for a box's inputs / params / outputs."""
    if box_name not in BOX_REGISTRY:
        typer.echo(
            f"Unknown box {box_name!r}. Available: {sorted(BOX_REGISTRY)}",
            err=True,
        )
        raise typer.Exit(code=1)
    cls = BOX_REGISTRY[box_name]
    out = {
        "name": cls.name,
        "description": cls.description,
        "inputs_schema": cls.inputs_schema.model_json_schema(),
        "params_schema": cls.params_schema.model_json_schema(),
        "outputs_schema": cls.outputs_schema.model_json_schema(),
    }
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command()
def run(
    spec_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to a WorkflowSpec JSON file"),
    out: Path | None = typer.Option(None, "--out", help="Write each node's output JSON under this directory"),
) -> None:
    """Execute a workflow specification."""
    spec_text = spec_path.read_text()
    spec = WorkflowSpec.model_validate_json(spec_text)
    typer.echo(f"[moire-flow] running {len(spec.nodes)} nodes from {spec_path}")
    results = WorkflowEngine().run(spec)
    typer.echo(f"[moire-flow] done — produced outputs for: {sorted(results)}")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for node_id, model in results.items():
            try:
                (out / f"{node_id}.json").write_text(model.model_dump_json(indent=2))
            except Exception as exc:  # non-serializable output is allowed
                typer.echo(f"[warn] could not serialize {node_id}: {exc}", err=True)


if __name__ == "__main__":
    app()
