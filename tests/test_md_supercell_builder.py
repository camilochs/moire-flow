"""MDSupercellBuilder: 2D bilayer → 3D LAMMPS-ready MDStructure."""

from __future__ import annotations

import numpy as np

from moire_flow.boxes import (
    MDSupercellBuilder,
    MDSupercellBuilderInputs,
    MDSupercellBuilderParams,
)
from moire_flow.core.types import BilayerAtoms


def _make_bilayer() -> BilayerAtoms:
    return BilayerAtoms(
        atoms_A=np.array([[0.0, 0.0], [1.58, 1.58]]),
        atoms_B=np.array([[0.0, 0.0], [1.58, 1.58]]),
        species_A=["Mo", "S"],
        species_B=["Mo", "S"],
        sc_vecs=np.array([[3.16, 0.0], [0.0, 3.16]]),
    )


def test_basic_promotion_to_3d():
    md = MDSupercellBuilder().run(
        MDSupercellBuilderInputs(bilayer=_make_bilayer()),
        MDSupercellBuilderParams(vacuum_z=15.0, interlayer_z=3.3),
    )
    assert md.atoms.shape == (4, 3)
    # First two atoms (layer A) at z=0
    np.testing.assert_allclose(md.atoms[:2, 2], 0.0)
    # Last two atoms (layer B) at z=3.3
    np.testing.assert_allclose(md.atoms[2:, 2], 3.3)


def test_cell_carries_2d_supercell_and_vacuum():
    md = MDSupercellBuilder().run(
        MDSupercellBuilderInputs(bilayer=_make_bilayer()),
        MDSupercellBuilderParams(vacuum_z=15.0, interlayer_z=3.3),
    )
    np.testing.assert_allclose(md.cell[0, :2], [3.16, 0.0])
    np.testing.assert_allclose(md.cell[1, :2], [0.0, 3.16])
    # cell z = interlayer + vacuum
    assert md.cell[2, 2] == 3.3 + 15.0
    assert md.vacuum_z == 15.0
    assert md.interlayer_z == 3.3


def test_species_concatenation_order():
    md = MDSupercellBuilder().run(
        MDSupercellBuilderInputs(bilayer=_make_bilayer()),
        MDSupercellBuilderParams(),
    )
    assert md.species == ["Mo", "S", "Mo", "S"]


def test_padding_cells_replicates_in_plane():
    """With padding_cells >= 1, the supercell is tiled in plane."""
    md = MDSupercellBuilder().run(
        MDSupercellBuilderInputs(bilayer=_make_bilayer()),
        MDSupercellBuilderParams(padding_cells=2, lj_cutoff=8.0),
    )
    # 8.0 * 1.2 / 3.16 ≈ 3.04 → ceil = 4 → 4×4 in-plane replication
    assert md.atoms.shape[0] >= 32
    assert md.cell[0, 0] >= 3.16  # at least the original


def test_round_trip_json():
    md = MDSupercellBuilder().run(
        MDSupercellBuilderInputs(bilayer=_make_bilayer()),
        MDSupercellBuilderParams(),
    )
    from moire_flow.core.types import MDStructure
    again = MDStructure.model_validate_json(md.model_dump_json())
    np.testing.assert_array_equal(again.atoms, md.atoms)
    assert again.species == md.species


def test_z_offset_shifts_layers():
    md = MDSupercellBuilder().run(
        MDSupercellBuilderInputs(bilayer=_make_bilayer()),
        MDSupercellBuilderParams(z_offset=5.0, interlayer_z=3.3),
    )
    np.testing.assert_allclose(md.atoms[:2, 2], 5.0)
    np.testing.assert_allclose(md.atoms[2:, 2], 8.3)
