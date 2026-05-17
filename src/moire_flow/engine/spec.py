"""WorkflowSpec: declarative description of a moire-flow pipeline.

A `WorkflowSpec` is JSON-serializable and consumable by both the M9 web UI
and the CLI engine. It is *portable*: no runtime capabilities (LAMMPS
availability, MLIP models on disk) are recorded here.

A node references a Box by `box_name` (must exist in `BOX_REGISTRY`) and
declares its `params`. The `inputs` field can either embed a literal value
for a Pydantic input field, or leave it unset — in which case the field is
resolved through an `Edge` from an upstream node's output.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from moire_flow.boxes import BOX_REGISTRY


class Edge(BaseModel):
    """Connect an upstream node's output field to a downstream node's input field."""

    from_node: str
    from_field: str
    to_node: str
    to_field: str


class Node(BaseModel):
    """One step in the workflow."""

    id: str
    box_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowSpec(BaseModel):
    """Declarative pipeline."""

    nodes: list[Node]
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "WorkflowSpec":
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate node ids: {ids}")
        id_to_box: dict[str, str] = {}
        for n in self.nodes:
            if n.box_name not in BOX_REGISTRY:
                raise ValueError(f"Node {n.id!r}: unknown box {n.box_name!r}")
            id_to_box[n.id] = n.box_name
        for e in self.edges:
            if e.from_node not in id_to_box:
                raise ValueError(f"Edge references unknown from_node {e.from_node!r}")
            if e.to_node not in id_to_box:
                raise ValueError(f"Edge references unknown to_node {e.to_node!r}")
            up_cls = BOX_REGISTRY[id_to_box[e.from_node]]
            down_cls = BOX_REGISTRY[id_to_box[e.to_node]]
            up_fields = set(up_cls.outputs_schema.model_fields)
            down_fields = set(down_cls.inputs_schema.model_fields)
            if e.from_field not in up_fields:
                raise ValueError(
                    f"Edge {e.from_node}.{e.from_field} → "
                    f"{e.to_node}.{e.to_field}: output {e.from_field!r} "
                    f"not in {up_cls.name!r} outputs {sorted(up_fields)}"
                )
            if e.to_field not in down_fields:
                raise ValueError(
                    f"Edge {e.from_node}.{e.from_field} → "
                    f"{e.to_node}.{e.to_field}: input {e.to_field!r} "
                    f"not in {down_cls.name!r} inputs {sorted(down_fields)}"
                )
        return self


__all__ = ["Edge", "Node", "WorkflowSpec"]
