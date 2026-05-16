"""XYZ reader → `Structure`. Thin wrapper over ASE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from moire_flow.core.types import Structure


def read_xyz(path: Path | str) -> Structure:
    """Read an XYZ file. If no cell is stored, returns the identity 3×3."""
    atoms = ase_read(str(path), format="extxyz")
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    if not np.any(cell):
        cell = np.eye(3, dtype=np.float64)
    return Structure(
        atoms=np.asarray(atoms.get_positions(), dtype=np.float64),
        species=list(atoms.get_chemical_symbols()),
        cell=cell,
    )


__all__ = ["read_xyz"]
