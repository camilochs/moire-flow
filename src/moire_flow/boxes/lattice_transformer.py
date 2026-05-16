"""LatticeTransformer: synthesize a transformed lattice + basis from a reference.

Used to build synthetic bilayer test cases where layer B is a known
deformation of layer A. Five `transform_type` modes mirror the original
`build_transformed_lattice` / `build_transformed_basis` (reference 1767-1788).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, model_validator

from moire_flow.core.algebra2d import R, S, transform_basis
from moire_flow.core.types import NDArray2D, NDArray2x2

from .base import Box, register_box

TransformType = Literal[
    "identity",
    "rotation",
    "isotropic_scale",
    "anisotropic_strain",
    "strain_plus_rotation",
]


class LatticeTransformerInputs(BaseModel):
    Alat: NDArray2x2
    basis_A: NDArray2D
    species_A: list[str]


class LatticeTransformerParams(BaseModel):
    transform_type: TransformType
    theta_deg: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    scale: float = 1.0

    @model_validator(mode="after")
    def _validate_params(self) -> "LatticeTransformerParams":
        if self.transform_type == "isotropic_scale" and self.scale == 0:
            raise ValueError("scale must be non-zero for isotropic_scale")
        return self


class TransformedLattice(BaseModel):
    Blat: NDArray2x2
    basis_B: NDArray2D
    species_B: list[str]


def _apply_transform(Alat: np.ndarray, p: LatticeTransformerParams) -> np.ndarray:
    if p.transform_type == "identity":
        return Alat.copy()
    if p.transform_type == "rotation":
        return Alat.dot(R(np.radians(p.theta_deg)).T)
    if p.transform_type == "isotropic_scale":
        return float(p.scale) * Alat
    if p.transform_type == "anisotropic_strain":
        return S(p.s1, p.s2).dot(Alat)
    if p.transform_type == "strain_plus_rotation":
        return S(p.s1, p.s2).dot(Alat.dot(R(np.radians(p.theta_deg)).T))
    raise ValueError(f"Unknown transform_type: {p.transform_type}")


@register_box
class LatticeTransformer(
    Box[LatticeTransformerInputs, LatticeTransformerParams, TransformedLattice]
):
    name = "lattice_transformer"
    description = "Apply a rigid/strain transformation to a 2D lattice + basis."
    inputs_schema = LatticeTransformerInputs
    params_schema = LatticeTransformerParams
    outputs_schema = TransformedLattice

    def run(
        self,
        inputs: LatticeTransformerInputs,
        params: LatticeTransformerParams,
    ) -> TransformedLattice:
        Alat = np.asarray(inputs.Alat, dtype=np.float64)
        basis_A = np.asarray(inputs.basis_A, dtype=np.float64)
        Blat = _apply_transform(Alat, params)
        basis_B = transform_basis(basis_A, Alat, Blat)
        return TransformedLattice(
            Blat=Blat,
            basis_B=basis_B,
            species_B=list(inputs.species_A),
        )


__all__ = [
    "LatticeTransformer",
    "LatticeTransformerInputs",
    "LatticeTransformerParams",
    "TransformedLattice",
    "TransformType",
]
