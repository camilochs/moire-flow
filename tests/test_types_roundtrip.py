"""Core-types JSON round-trip + schema serializability."""

from __future__ import annotations

import numpy as np

from moire_flow.core import (
    BilayerAtoms,
    LayerPair,
    MatchingSolution,
    MDStructure,
    Structure,
    ValidationMetrics,
)


def test_structure_roundtrip():
    s = Structure(
        atoms=np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 5.0]]),
        species=["Mo", "S"],
        cell=np.eye(3) * 3.16,
    )
    s2 = Structure.model_validate_json(s.model_dump_json())
    assert s2.atoms.shape == (2, 3)
    assert s2.atoms.dtype == np.float64
    assert s2.species == ["Mo", "S"]
    np.testing.assert_array_equal(s2.cell, s.cell)


def test_layerpair_roundtrip():
    layer = Structure(
        atoms=np.zeros((1, 3)),
        species=["C"],
        cell=np.eye(3),
    )
    lp = LayerPair(
        layer_A=layer,
        layer_B=layer,
        sc_vecs=np.array([[3.0, 0.0], [0.0, 3.0]]),
        replication=(2, 2),
        z_mean_A=0.0,
        z_mean_B=3.3,
    )
    lp2 = LayerPair.model_validate_json(lp.model_dump_json())
    assert lp2.replication == (2, 2)
    assert lp2.sc_vecs.shape == (2, 2)


def test_matching_solution_roundtrip():
    sol = MatchingSolution(
        v1=np.array([1.0, 0.0]),
        v2=np.array([0.0, 1.0]),
        s1=1.0,
        s2=1.0,
        theta_deg=15.0,
        cost=0.01,
        mismatch=0.005,
        area=10.0,
        oBlat=np.eye(2),
    )
    sol2 = MatchingSolution.model_validate_json(sol.model_dump_json())
    assert sol2.theta_deg == 15.0
    assert sol2.v1.shape == (2,)


def test_bilayer_atoms_roundtrip():
    ba = BilayerAtoms(
        atoms_A=np.zeros((2, 2)),
        atoms_B=np.zeros((2, 2)),
        species_A=["Mo", "Mo"],
        species_B=["S", "S"],
        sc_vecs=np.eye(2),
    )
    ba2 = BilayerAtoms.model_validate_json(ba.model_dump_json())
    assert ba2.atoms_A.shape == (2, 2)


def test_validation_metrics_roundtrip():
    vm = ValidationMetrics(
        fractional_rmsd=0.01,
        distance_signature_error=0.02,
        coordination_match=0.95,
        passes={"theta_ok": True, "rmsd_ok": False},
    )
    vm2 = ValidationMetrics.model_validate_json(vm.model_dump_json())
    assert vm2.passes == {"theta_ok": True, "rmsd_ok": False}


def test_md_structure_roundtrip():
    md = MDStructure(
        atoms=np.zeros((1, 3)),
        species=["Mo"],
        cell=np.eye(3),
        vacuum_z=15.0,
        interlayer_z=3.3,
    )
    md2 = MDStructure.model_validate_json(md.model_dump_json())
    assert md2.vacuum_z == 15.0


def test_schemas_are_serializable():
    for cls in (Structure, LayerPair, MatchingSolution, BilayerAtoms, MDStructure):
        schema = cls.model_json_schema()
        assert "title" in schema
        assert "properties" in schema
