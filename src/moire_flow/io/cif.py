"""CIF reader → `Structure`. Thin wrapper over ASE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from moire_flow.core.types import Structure


def read_cif(path: Path | str) -> Structure:
    """Read a CIF file and return a `Structure` with native 3×3 cell."""
    atoms = ase_read(str(path), format="cif")
    return Structure(
        atoms=np.asarray(atoms.get_positions(), dtype=np.float64),
        species=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=np.float64),
    )


__all__ = ["read_cif"]
