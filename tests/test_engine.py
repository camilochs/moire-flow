"""WorkflowEngine: validation + topological order + end-to-end smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moire_flow.engine import Edge, Node, WorkflowEngine, WorkflowSpec
from moire_flow.engine.runner import _topological_order


# ---------- Topological-order tests ----------

def test_topological_order_simple_chain():
    nodes = [Node(id="a", box_name="lattice_transformer"),
             Node(id="b", box_name="lattice_matcher")]
    edges = [Edge(from_node="a", from_field="Blat",
                  to_node="b", to_field="Blat")]
    order = _topological_order(nodes, edges)
    assert order == ["a", "b"]


def test_topological_order_detects_cycle():
    nodes = [Node(id="a", box_name="lattice_transformer"),
             Node(id="b", box_name="lattice_transformer")]
    edges = [
        Edge(from_node="a", from_field="Blat", to_node="b", to_field="Alat"),
        Edge(from_node="b", from_field="Blat", to_node="a", to_field="Alat"),
    ]
    with pytest.raises(ValueError, match="Cycle"):
        _topological_order(nodes, edges)


def test_self_loop_rejected():
    nodes = [Node(id="a", box_name="lattice_transformer")]
    edges = [Edge(from_node="a", from_field="Blat", to_node="a", to_field="Alat")]
    with pytest.raises(ValueError, match="Self-loop"):
        _topological_order(nodes, edges)


# ---------- WorkflowSpec validation ----------

def test_duplicate_node_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        WorkflowSpec(
            nodes=[
                Node(id="a", box_name="lattice_transformer"),
                Node(id="a", box_name="lattice_matcher"),
            ]
        )


def test_unknown_box_rejected():
    with pytest.raises(ValueError, match="unknown box"):
        WorkflowSpec(nodes=[Node(id="a", box_name="nonexistent_box")])


def test_edge_field_validated_against_schemas():
    """An edge that points at a non-existent output field must be rejected."""
    with pytest.raises(ValueError, match="output"):
        WorkflowSpec(
            nodes=[
                Node(id="a", box_name="lattice_transformer",
                     params={"transform_type": "identity"}),
                Node(id="b", box_name="lattice_matcher"),
            ],
            edges=[
                Edge(from_node="a", from_field="NOT_A_FIELD",
                     to_node="b", to_field="Blat"),
            ],
        )


# ---------- End-to-end pipeline ----------

ALAT = np.array([[3.16, 0.0], [0.0, 3.16]])
BASIS = np.array([[0.0, 0.0], [1.58, 1.58]])


def test_minimal_two_node_pipeline():
    """LatticeTransformer (rotation 30°) → LatticeMatcher (rotation_only)."""
    spec = WorkflowSpec(
        nodes=[
            Node(
                id="transform",
                box_name="lattice_transformer",
                params={"transform_type": "rotation", "theta_deg": 30.0},
                inputs={"Alat": ALAT.tolist(), "basis_A": BASIS.tolist(),
                        "species_A": ["Mo", "S"]},
            ),
            Node(
                id="match",
                box_name="lattice_matcher",
                params={
                    "method": "rotation_only",
                    "N": 4,
                    "theta_steps": 13,
                    "bf_top_k": 5,
                },
                inputs={"Alat": ALAT.tolist()},
            ),
        ],
        edges=[
            Edge(from_node="transform", from_field="Blat",
                 to_node="match", to_field="Blat"),
        ],
    )
    out = WorkflowEngine().run(spec)
    matcher_out = out["match"]
    assert len(matcher_out.solutions) > 0


def test_full_pipeline_through_lammps(tmp_path: Path):
    """End-to-end: synthetic MoS2 identity case → 6 LAMMPS files."""
    spec = WorkflowSpec(
        nodes=[
            Node(
                id="transform",
                box_name="lattice_transformer",
                params={"transform_type": "identity"},
                inputs={"Alat": ALAT.tolist(), "basis_A": BASIS.tolist(),
                        "species_A": ["Mo", "S"]},
            ),
            Node(
                id="match",
                box_name="lattice_matcher",
                params={
                    "method": "rotation_only",
                    "N": 4,
                    "theta_steps": 7,
                    "bf_top_k": 3,
                    "dim": (4, 4),
                },
                inputs={"Alat": ALAT.tolist()},
            ),
        ],
        edges=[
            Edge(from_node="transform", from_field="Blat",
                 to_node="match", to_field="Blat"),
        ],
    )
    results = WorkflowEngine().run(spec)
    assert "match" in results
    sols = results["match"].solutions
    assert len(sols) > 0
    # Continue manually (the spec only describes 2 nodes here for clarity;
    # extending the DAG to 9 nodes works the same way).
    from moire_flow.boxes import (
        AtomAssembler, AtomAssemblerInputs, AtomAssemblerParams,
        LammpsInputWriter, LammpsInputWriterInputs, LammpsInputWriterParams,
        MDSupercellBuilder, MDSupercellBuilderInputs, MDSupercellBuilderParams,
        PotentialAssigner, PotentialAssignerInputs, PotentialAssignerParams,
    )
    bilayer = AtomAssembler().run(
        AtomAssemblerInputs(
            solution=sols[0],
            basis_A=BASIS, basis_B=BASIS,
            Alat=ALAT, Blat=ALAT,
            species_A=["Mo", "S"], species_B=["Mo", "S"],
        ),
        AtomAssemblerParams(),
    )
    md = MDSupercellBuilder().run(
        MDSupercellBuilderInputs(bilayer=bilayer),
        MDSupercellBuilderParams(),
    )
    plan = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=md,
            species_A=["Mo", "S"], species_B=["Mo", "S"],
        ),
        PotentialAssignerParams(),
    )
    run_plan = LammpsInputWriter().run(
        LammpsInputWriterInputs(
            md_structure=md,
            potential_plan=plan,
            species_A=bilayer.species_A,
            species_B=bilayer.species_B,
            n_atoms_A=len(bilayer.atoms_A),
        ),
        LammpsInputWriterParams(output_dir=tmp_path / "lammps"),
    )
    assert run_plan.bilayer_data.exists()
    assert run_plan.layer_A_script.exists()
