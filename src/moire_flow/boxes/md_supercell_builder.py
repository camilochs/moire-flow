"""MDSupercellBuilder: promote a 2D BilayerAtoms to a 3D LAMMPS-ready cell.

Inputs are atoms already placed inside the matched supercell (output of
AtomAssembler). The box does not re-tile the lattice — that responsibility
lives upstream (BilayerSplitter handles import-time tiling). Here we only:
- place layer A at z = z_offset (default 0.0)
- place layer B at z = z_offset + interlayer_z
- expose a triclinic 3x3 cell with vacuum padding along z
- (optionally) replicate the supercell in-plane by `padding_cells` to reach
  a chosen LJ cutoff via `lj_cutoff`
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

from moire_flow.core.types import BilayerAtoms, MDStructure

from .base import Box, register_box


class MDSupercellBuilderInputs(BaseModel):
    bilayer: BilayerAtoms


class MDSupercellBuilderParams(BaseModel):
    vacuum_z: float = Field(default=15.0, gt=0.0)
    interlayer_z: float = Field(default=3.3, gt=0.0)
    padding_cells: int = Field(default=0, ge=0)
    lj_cutoff: float = Field(default=8.0, gt=0.0)
    tile_factor: float = Field(default=1.2, ge=1.0)
    z_offset: float = 0.0


def _replicate(
    atoms_2d: np.ndarray, species: list[str], sc_vecs: np.ndarray, nx: int, ny: int
) -> tuple[np.ndarray, list[str]]:
    if nx == 1 and ny == 1:
        return atoms_2d, list(species)
    v1, v2 = sc_vecs[0], sc_vecs[1]
    tiled: list[np.ndarray] = []
    tiled_species: list[str] = []
    for i in range(nx):
        for j in range(ny):
            shift = i * v1 + j * v2
            tiled.append(atoms_2d + shift)
            tiled_species.extend(species)
    return np.vstack(tiled), tiled_species


def _replication_for_cutoff(
    sc_vecs: np.ndarray, lj_cutoff: float, factor: float, padding_cells: int
) -> tuple[int, int]:
    """Choose nx, ny so each lateral edge reaches `factor * lj_cutoff` Å."""
    min_len = factor * lj_cutoff
    ax = float(np.linalg.norm(sc_vecs[0]))
    by = float(np.linalg.norm(sc_vecs[1]))
    nx = max(1, math.ceil(min_len / ax)) if padding_cells > 0 else 1
    ny = max(1, math.ceil(min_len / by)) if padding_cells > 0 else 1
    nx = max(nx, 1 + padding_cells)
    ny = max(ny, 1 + padding_cells)
    return nx, ny


@register_box
class MDSupercellBuilder(
    Box[MDSupercellBuilderInputs, MDSupercellBuilderParams, MDStructure]
):
    name = "md_supercell_builder"
    description = "Promote a 2D BilayerAtoms supercell into a 3D LAMMPS-ready MDStructure."
    inputs_schema = MDSupercellBuilderInputs
    params_schema = MDSupercellBuilderParams
    outputs_schema = MDStructure

    def run(
        self,
        inputs: MDSupercellBuilderInputs,
        params: MDSupercellBuilderParams,
    ) -> MDStructure:
        bi = inputs.bilayer
        sc_vecs = np.asarray(bi.sc_vecs, dtype=np.float64)
        atoms_A_2d = np.asarray(bi.atoms_A, dtype=np.float64)
        atoms_B_2d = np.asarray(bi.atoms_B, dtype=np.float64)

        nx, ny = _replication_for_cutoff(
            sc_vecs, params.lj_cutoff, params.tile_factor, params.padding_cells
        )
        atoms_A_2d, species_A = _replicate(atoms_A_2d, list(bi.species_A), sc_vecs, nx, ny)
        atoms_B_2d, species_B = _replicate(atoms_B_2d, list(bi.species_B), sc_vecs, nx, ny)
        sc_vecs_scaled = np.array([nx * sc_vecs[0], ny * sc_vecs[1]], dtype=np.float64)

        z_A = float(params.z_offset)
        z_B = float(params.z_offset + params.interlayer_z)
        atoms_A_3d = np.column_stack([atoms_A_2d, np.full(len(atoms_A_2d), z_A)])
        atoms_B_3d = np.column_stack([atoms_B_2d, np.full(len(atoms_B_2d), z_B)])

        atoms = np.vstack([atoms_A_3d, atoms_B_3d]).astype(np.float64)
        species = species_A + species_B

        # Build a triclinic 3x3 cell: rows 0,1 are sc_vecs lifted to 3D, row 2
        # is the vacuum-padded z box.
        cell = np.zeros((3, 3), dtype=np.float64)
        cell[0, :2] = sc_vecs_scaled[0]
        cell[1, :2] = sc_vecs_scaled[1]
        z_extent = (z_B - z_A) + params.vacuum_z
        cell[2, 2] = z_extent

        return MDStructure(
            atoms=atoms,
            species=species,
            cell=cell,
            vacuum_z=float(params.vacuum_z),
            interlayer_z=float(params.interlayer_z),
        )


__all__ = ["MDSupercellBuilder", "MDSupercellBuilderInputs", "MDSupercellBuilderParams"]
