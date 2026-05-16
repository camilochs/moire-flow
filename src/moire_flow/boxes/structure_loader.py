"""StructureLoader: file path → Structure.

Pilot box for M1. Dispatches to an `io/` adapter by file extension or by
explicit `format` parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from moire_flow.core.types import Structure
from moire_flow.io.cif import read_cif
from moire_flow.io.lammps_data import read_lammps_data
from moire_flow.io.xyz import read_xyz

from .base import Box, register_box

Format = Literal["auto", "cif", "lammps_data", "xyz"]


class StructureLoaderInputs(BaseModel):
    path: Path


class StructureLoaderParams(BaseModel):
    format: Format = "auto"
    wrap_to_cell: bool = False
    duplicate_tolerance: float = Field(default=1e-3, ge=0.0)
    strict_elements: bool = False

    @field_validator("duplicate_tolerance")
    @classmethod
    def _check_tol(cls, v: float) -> float:
        if v < 0:
            raise ValueError("duplicate_tolerance must be non-negative")
        return v


_EXT_TO_FORMAT: dict[str, Format] = {
    ".cif": "cif",
    ".data": "lammps_data",
    ".lmp": "lammps_data",
    ".xyz": "xyz",
    ".extxyz": "xyz",
}


def _detect_format(path: Path) -> Format:
    suf = path.suffix.lower()
    if suf in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[suf]
    raise ValueError(
        f"Cannot auto-detect format from extension '{suf}'. "
        f"Set `format` explicitly to one of: cif, lammps_data, xyz."
    )


def _dedup(struct: Structure, tol: float) -> Structure:
    """Remove duplicate atoms (within `tol` Å in any direction)."""
    if tol <= 0 or struct.atoms.shape[0] < 2:
        return struct
    keep = np.ones(struct.atoms.shape[0], dtype=bool)
    pos = struct.atoms
    for i in range(1, pos.shape[0]):
        if not keep[i]:
            continue
        d = pos[:i][keep[:i]] - pos[i]
        if np.any(np.all(np.abs(d) < tol, axis=1)):
            keep[i] = False
    if keep.all():
        return struct
    return Structure(
        atoms=pos[keep],
        species=[s for s, k in zip(struct.species, keep) if k],
        cell=struct.cell,
        energy=struct.energy,
        forces=struct.forces[keep] if struct.forces is not None else None,
    )


def _wrap(struct: Structure) -> Structure:
    """Wrap fractional coords into [0, 1)."""
    inv = np.linalg.inv(struct.cell)
    frac = struct.atoms @ inv
    frac -= np.floor(frac)
    wrapped = frac @ struct.cell
    return Structure(
        atoms=wrapped,
        species=struct.species,
        cell=struct.cell,
        energy=struct.energy,
        forces=struct.forces,
    )


@register_box
class StructureLoader(Box[StructureLoaderInputs, StructureLoaderParams, Structure]):
    name = "structure_loader"
    description = "Load a structural file (CIF, LAMMPS .data, XYZ) into a Structure."
    inputs_schema = StructureLoaderInputs
    params_schema = StructureLoaderParams
    outputs_schema = Structure

    def run(self, inputs: StructureLoaderInputs, params: StructureLoaderParams) -> Structure:
        path = Path(inputs.path)
        if not path.exists():
            raise FileNotFoundError(f"No such structure file: {path}")
        fmt: Format = params.format if params.format != "auto" else _detect_format(path)
        if fmt == "cif":
            struct = read_cif(path)
        elif fmt == "lammps_data":
            struct = read_lammps_data(path)
        elif fmt == "xyz":
            struct = read_xyz(path)
        else:  # pragma: no cover - typing exhaustive
            raise ValueError(f"Unsupported format: {fmt}")
        if params.strict_elements and any(s.startswith("X") for s in struct.species):
            raise ValueError(
                f"strict_elements=True: unresolved elements in {path.name}: "
                f"{[s for s in struct.species if s.startswith('X')]}"
            )
        if params.duplicate_tolerance > 0:
            struct = _dedup(struct, params.duplicate_tolerance)
        if params.wrap_to_cell:
            struct = _wrap(struct)
        return struct


__all__ = [
    "StructureLoader",
    "StructureLoaderInputs",
    "StructureLoaderParams",
]
