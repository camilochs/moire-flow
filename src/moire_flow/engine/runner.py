"""WorkflowEngine: topologically execute a WorkflowSpec.

For each node:
1. Build the input dict: literal `inputs` from the node merged with
   resolved upstream outputs (via edges).
2. Validate against the Box's `inputs_schema` and `params_schema`.
3. Call `box.run(inputs_model, params_model)`.
4. Store the output in the per-node result registry.

The engine has no notion of "case name" or "source mode" — those concerns
live one layer above (in user code that builds the WorkflowSpec). See
ARCHITECTURE.md decisions.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from pydantic import BaseModel

from moire_flow.boxes import BOX_REGISTRY, Box

from .spec import Edge, Node, WorkflowSpec


def _topological_order(nodes: list[Node], edges: list[Edge]) -> list[str]:
    indeg: dict[str, int] = defaultdict(int)
    children: dict[str, list[str]] = defaultdict(list)
    ids = {n.id for n in nodes}
    for n in nodes:
        indeg.setdefault(n.id, 0)
    for e in edges:
        if e.from_node == e.to_node:
            raise ValueError(f"Self-loop on node {e.from_node!r}")
        children[e.from_node].append(e.to_node)
        indeg[e.to_node] += 1
    queue = deque(sorted([nid for nid in ids if indeg[nid] == 0]))
    order: list[str] = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in children[nid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    if len(order) != len(ids):
        missing = ids - set(order)
        raise ValueError(f"Cycle detected — these nodes never become reachable: {sorted(missing)}")
    return order


class WorkflowEngine:
    """Stateless executor. Reusable across runs."""

    def run(
        self,
        spec: WorkflowSpec,
        external_inputs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, BaseModel]:
        """Execute `spec` and return `{node_id: output_pydantic_model}`.

        `external_inputs[node_id]` may supply literal inputs that aren't
        embedded in the spec itself (useful for binding paths/files at
        runtime without putting them in the portable WorkflowSpec).
        """
        external_inputs = external_inputs or {}
        order = _topological_order(spec.nodes, spec.edges)
        nodes_by_id = {n.id: n for n in spec.nodes}
        results: dict[str, BaseModel] = {}

        # Index edges by destination for O(1) input resolution.
        edges_by_dst: dict[str, list[Edge]] = defaultdict(list)
        for e in spec.edges:
            edges_by_dst[e.to_node].append(e)

        for node_id in order:
            node = nodes_by_id[node_id]
            box_cls: type[Box] = BOX_REGISTRY[node.box_name]
            inputs_dict = self._resolve_inputs(node, edges_by_dst[node_id], results, external_inputs)
            inputs_model = box_cls.inputs_schema(**inputs_dict)
            params_model = box_cls.params_schema(**node.params)
            output_model = box_cls().run(inputs_model, params_model)
            results[node_id] = output_model
        return results

    @staticmethod
    def _resolve_inputs(
        node: Node,
        in_edges: list[Edge],
        results: dict[str, BaseModel],
        external_inputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(node.inputs)
        for e in in_edges:
            upstream = results[e.from_node]
            value = getattr(upstream, e.from_field)
            resolved[e.to_field] = value
        if node.id in external_inputs:
            resolved.update(external_inputs[node.id])
        return resolved


__all__ = ["WorkflowEngine", "_topological_order"]
