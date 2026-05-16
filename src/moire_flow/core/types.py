"""Core data types shared by all boxes.

Pydantic v2 models with numpydantic-validated numpy arrays. All models are
JSON-serializable (`model_dump_json` / `model_json_schema`) so the web UI
(M9) can consume them via `BOX_REGISTRY`.

Shape conventions:
- 2D coordinates: (N, 2) float64
- 3D coordinates: (N, 3) float64
- 2D lattice/supercell vectors stored row-wise: (2, 2)
- 3D lattice/supercell vectors stored row-wise: (3, 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field

NDArray2D = Annotated[NDArray[Shape["*, 2"], np.float64], Field(description="(N, 2) float64")]
NDArray3D = Annotated[NDArray[Shape["*, 3"], np.float64], Field(description="(N, 3) float64")]
NDArray2x2 = Annotated[NDArray[Shape["2, 2"], np.float64], Field(description="(2, 2) float64")]
NDArray3x3 = Annotated[NDArray[Shape["3, 3"], np.float64], Field(description="(3, 3) float64")]


class _BoxModel(BaseModel):
    """Base model: arbitrary types (numpy arrays) + frozen for purity."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)


class Structure(_BoxModel):
    """Raw structural data loaded from a file. Monolayer or bilayer."""

    atoms: NDArray3D
    species: list[str]
    cell: NDArray3x3
    energy: float | None = None
    forces: NDArray3D | None = None


class LayerPair(_BoxModel):
    """Two layers extracted from a bilayer Structure, sharing a supercell."""

    layer_A: Structure
    layer_B: Structure
    sc_vecs: NDArray2x2
    replication: tuple[int, int]
    z_mean_A: float
    z_mean_B: float


class MatchingSolution(_BoxModel):
    """One candidate supercell from LatticeMatcher."""

    v1: NDArray2x2
    v2: NDArray2x2
    s1: float
    s2: float
    theta_deg: float
    cost: float
    mismatch: float
    area: float
    oBlat: NDArray2x2


class BilayerAtoms(_BoxModel):
    """Atoms placed inside the matched supercell, 2D coordinates."""

    atoms_A: NDArray2D
    atoms_B: NDArray2D
    species_A: list[str]
    species_B: list[str]
    sc_vecs: NDArray2x2


class ValidationTarget(_BoxModel):
    """Optional ground truth for synthetic cases."""

    target_theta_deg: float | None = None
    target_s1: float | None = None
    target_s2: float | None = None
    theta_period_deg: float = 60.0
    allow_sign_flip: bool = True


class ValidationMetrics(_BoxModel):
    """Output of the Validator box."""

    theta_error_sym_deg: float | None = None
    s1_error: float | None = None
    s2_error: float | None = None
    fractional_rmsd: float
    distance_signature_error: float
    coordination_match: float
    passes: dict[str, bool]


class MDStructure(_BoxModel):
    """A 2D BilayerAtoms promoted to 3D with vacuum Z for LAMMPS."""

    atoms: NDArray3D
    species: list[str]
    cell: NDArray3x3
    vacuum_z: float
    interlayer_z: float | None = None
    data_file: Path | None = None


class PotentialPlan(_BoxModel):
    """Resolved potential assignment for a bilayer."""

    intralayer_A: dict
    intralayer_B: dict
    interlayer: dict


class LammpsRunPlan(_BoxModel):
    """Paths produced by LammpsInputWriter."""

    layer_A_data: Path
    layer_B_data: Path
    bilayer_data: Path
    layer_A_script: Path
    layer_B_script: Path
    bilayer_script: Path
    work_dir: Path


__all__ = [
    "NDArray2D",
    "NDArray3D",
    "NDArray2x2",
    "NDArray3x3",
    "Structure",
    "LayerPair",
    "MatchingSolution",
    "BilayerAtoms",
    "ValidationTarget",
    "ValidationMetrics",
    "MDStructure",
    "PotentialPlan",
    "LammpsRunPlan",
]
