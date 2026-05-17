"""AtomAssembler: fill the matched supercell with atoms of both layers.

Port of `build_atoms_for_best` (reference 2166-2173).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from moire_flow.core.algebra2d import transform_basis, wrap_basis_to_cell
from moire_flow.core.matching import atoms_in_supercell
from moire_flow.core.types import BilayerAtoms, MatchingSolution, NDArray2D, NDArray2x2

from .base import Box, register_box


class AtomAssemblerInputs(BaseModel):
    solution: MatchingSolution
    basis_A: NDArray2D
    basis_B: NDArray2D
    Alat: NDArray2x2
    Blat: NDArray2x2
    species_A: list[str]
    species_B: list[str]


class AtomAssemblerParams(BaseModel):
    pass


@register_box
class AtomAssembler(Box[AtomAssemblerInputs, AtomAssemblerParams, BilayerAtoms]):
    name = "atom_assembler"
    description = "Place A and B basis atoms inside the matched supercell."
    inputs_schema = AtomAssemblerInputs
    params_schema = AtomAssemblerParams
    outputs_schema = BilayerAtoms

    def run(
        self,
        inputs: AtomAssemblerInputs,
        params: AtomAssemblerParams,
    ) -> BilayerAtoms:
        sol = inputs.solution
        sc_vecs = np.array([sol.v1, sol.v2], dtype=np.float64)
        Alat = np.asarray(inputs.Alat, dtype=np.float64)
        Blat = np.asarray(inputs.Blat, dtype=np.float64)
        oBlat = np.asarray(sol.oBlat, dtype=np.float64)
        basis_A = np.asarray(inputs.basis_A, dtype=np.float64)
        basis_B = np.asarray(inputs.basis_B, dtype=np.float64)

        basis_A_wrapped = wrap_basis_to_cell(basis_A, Alat)
        basis_B_transformed = transform_basis(basis_B, Blat, oBlat)

        atoms_A = atoms_in_supercell(sc_vecs, Alat, basis_A_wrapped)
        atoms_B = atoms_in_supercell(sc_vecs, oBlat, basis_B_transformed)

        species_A_filled = _replicate_species(inputs.species_A, len(atoms_A), len(basis_A))
        species_B_filled = _replicate_species(inputs.species_B, len(atoms_B), len(basis_B))

        return BilayerAtoms(
            atoms_A=atoms_A,
            atoms_B=atoms_B,
            species_A=species_A_filled,
            species_B=species_B_filled,
            sc_vecs=sc_vecs,
        )


def _replicate_species(species: list[str], n_atoms: int, basis_len: int) -> list[str]:
    """Repeat the basis-species pattern to match the supercell atom count."""
    if basis_len == 0:
        return []
    if n_atoms % basis_len != 0:
        # Supercell is not an integer multiple of basis — fall back to a per-atom
        # round-robin assignment so lengths still match.
        return [species[i % basis_len] for i in range(n_atoms)]
    factor = n_atoms // basis_len
    return list(species) * factor


__all__ = ["AtomAssembler", "AtomAssemblerInputs", "AtomAssemblerParams"]
