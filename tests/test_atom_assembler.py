"""AtomAssembler: identity case + supercell scaling."""

from __future__ import annotations

import numpy as np
import pytest

from moire_flow.boxes import (
    AtomAssembler,
    AtomAssemblerInputs,
    AtomAssemblerParams,
    LatticeMatcher,
    LatticeMatcherInputs,
    LatticeMatcherParams,
)
from moire_flow.core.algebra2d import R
from moire_flow.core.types import MatchingSolution


ALAT = np.array([[3.16, 0.0], [1.58, 2.737]])
BASIS = np.array([[0.0, 0.0], [2.107, 1.825]])
SPECIES = ["Mo", "S"]


def test_identity_assembler_returns_basis_count_per_unit_area():
    """When sc = Alat (1 unit cell), expect exactly len(basis) atoms per layer."""
    sol = MatchingSolution(
        v1=ALAT[0],
        v2=ALAT[1],
        s1=0.0, s2=0.0, theta_deg=0.0, cost=0.0, mismatch=0.0,
        area=abs(ALAT[0, 0] * ALAT[1, 1] - ALAT[0, 1] * ALAT[1, 0]),
        oBlat=ALAT,
    )
    out = AtomAssembler().run(
        AtomAssemblerInputs(
            solution=sol,
            basis_A=BASIS,
            basis_B=BASIS,
            Alat=ALAT,
            Blat=ALAT,
            species_A=SPECIES,
            species_B=SPECIES,
        ),
        AtomAssemblerParams(),
    )
    assert out.atoms_A.shape[0] == 2
    assert out.atoms_B.shape[0] == 2
    assert out.species_A == SPECIES
    assert out.species_B == SPECIES


def test_double_supercell_doubles_atom_count():
    """sc = 2*Alat (area×4) → 4× basis atoms per layer."""
    sol = MatchingSolution(
        v1=2 * ALAT[0],
        v2=2 * ALAT[1],
        s1=0.0, s2=0.0, theta_deg=0.0, cost=0.0, mismatch=0.0,
        area=4 * abs(ALAT[0, 0] * ALAT[1, 1] - ALAT[0, 1] * ALAT[1, 0]),
        oBlat=ALAT,
    )
    out = AtomAssembler().run(
        AtomAssemblerInputs(
            solution=sol,
            basis_A=BASIS,
            basis_B=BASIS,
            Alat=ALAT,
            Blat=ALAT,
            species_A=SPECIES,
            species_B=SPECIES,
        ),
        AtomAssemblerParams(),
    )
    assert out.atoms_A.shape[0] == 8
    assert out.atoms_B.shape[0] == 8


def test_pipeline_lattice_matcher_to_assembler():
    """End-to-end: matcher finds a solution, assembler builds the bilayer."""
    matcher_out = LatticeMatcher().run(
        LatticeMatcherInputs(Alat=ALAT, Blat=ALAT),
        LatticeMatcherParams(
            method="rotation_only", N=4, theta_steps=13, bf_top_k=5, dim=(6, 6)
        ),
    )
    assert len(matcher_out.solutions) > 0
    sol = matcher_out.solutions[0]
    out = AtomAssembler().run(
        AtomAssemblerInputs(
            solution=sol,
            basis_A=BASIS,
            basis_B=BASIS,
            Alat=ALAT,
            Blat=ALAT,
            species_A=SPECIES,
            species_B=SPECIES,
        ),
        AtomAssemblerParams(),
    )
    assert out.atoms_A.shape[0] > 0
    assert out.atoms_B.shape[0] > 0
    np.testing.assert_array_equal(out.sc_vecs, np.array([sol.v1, sol.v2]))
