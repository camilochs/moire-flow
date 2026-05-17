"""BilayerSplitter: split a bilayer Structure into two layers + a 2D supercell.

Port of `split_layers_from_z` + `lateral_lengths_from_sc` +
`normalize_prebuilt_bilayer` (reference 97-229).
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

from moire_flow.core.types import LayerPair, NDArray2D, Structure

from .base import Box, register_box


class BilayerSplitterInputs(BaseModel):
    structure: Structure


class BilayerSplitterParams(BaseModel):
    lj_cutoff: float = Field(default=8.0, gt=0.0)
    tile_factor: float = Field(default=1.2, ge=1.0)
    max_atoms: int = Field(default=2000, ge=1)
    z_axis_index: int = Field(default=2, ge=0, le=2)


def _lateral_lengths(sc_vecs: np.ndarray) -> tuple[float, float, float]:
    """Effective in-plane lengths (ax, by, bx) for tiling decisions."""
    v1, v2 = np.asarray(sc_vecs[0], dtype=np.float64), np.asarray(sc_vecs[1], dtype=np.float64)
    ax = float(np.linalg.norm(v1))
    ex = v1 / ax
    bx = float(np.dot(v2, ex))
    by = float(np.sqrt(max(float(np.dot(v2, v2)) - bx**2, 0.0)))
    return ax, by, bx


def _tile_layer(
    atoms2d: np.ndarray, species: list[str], sc_vecs: np.ndarray, nx: int, ny: int
) -> tuple[np.ndarray, list[str]]:
    v1, v2 = sc_vecs[0], sc_vecs[1]
    tiled: list[np.ndarray] = []
    tiled_species: list[str] = []
    for i in range(nx):
        for j in range(ny):
            shift = i * v1 + j * v2
            tiled.append(atoms2d + shift)
            tiled_species.extend(species)
    return np.vstack(tiled), tiled_species


def _split_layers_from_z(
    pos: np.ndarray, species: list[str], z_axis: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], float, float]:
    other = [a for a in range(3) if a != z_axis]
    z = pos[:, z_axis]
    z_mid = 0.5 * (float(z.min()) + float(z.max()))
    idx_A = z < z_mid
    idx_B = ~idx_A
    if idx_A.sum() == 0 or idx_B.sum() == 0:
        raise ValueError("layer split failed: one side is empty")
    atoms_A = pos[idx_A][:, other]
    atoms_B = pos[idx_B][:, other]
    species_A = [s for s, k in zip(species, idx_A) if k]
    species_B = [s for s, k in zip(species, idx_B) if k]
    zA_mean, zB_mean = float(np.mean(z[idx_A])), float(np.mean(z[idx_B]))
    # Always: A is the lower layer, B is the upper.
    if zA_mean > zB_mean:
        atoms_A, atoms_B = atoms_B, atoms_A
        species_A, species_B = species_B, species_A
        zA_mean, zB_mean = zB_mean, zA_mean
    return atoms_A, atoms_B, species_A, species_B, zA_mean, zB_mean


def _normalize_replication(
    atoms_A: np.ndarray,
    atoms_B: np.ndarray,
    species_A: list[str],
    species_B: list[str],
    sc_vecs: np.ndarray,
    params: BilayerSplitterParams,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray, tuple[int, int]]:
    ax, by, bx = _lateral_lengths(sc_vecs)
    min_len = params.tile_factor * params.lj_cutoff
    nx = math.ceil(min_len / ax) if ax < min_len else 1
    ny = math.ceil(min_len / by) if by < min_len else 1
    if abs(bx) > 1e-10:
        nx_skew = math.ceil(ny * abs(bx) / ax) + 1
        if nx_skew > nx:
            nx = nx_skew

    n_total_new = (len(atoms_A) + len(atoms_B)) * nx * ny
    if n_total_new > params.max_atoms or (nx == 1 and ny == 1):
        return atoms_A, atoms_B, species_A, species_B, sc_vecs, (1, 1)

    atoms_A_t, species_A_t = _tile_layer(atoms_A, species_A, sc_vecs, nx, ny)
    atoms_B_t, species_B_t = _tile_layer(atoms_B, species_B, sc_vecs, nx, ny)
    sc_new = np.array([nx * sc_vecs[0], ny * sc_vecs[1]], dtype=np.float64)
    return atoms_A_t, atoms_B_t, species_A_t, species_B_t, sc_new, (nx, ny)


@register_box
class BilayerSplitter(Box[BilayerSplitterInputs, BilayerSplitterParams, LayerPair]):
    name = "bilayer_splitter"
    description = "Split a bilayer Structure into two layers + a shared 2D supercell."
    inputs_schema = BilayerSplitterInputs
    params_schema = BilayerSplitterParams
    outputs_schema = LayerPair

    def run(
        self, inputs: BilayerSplitterInputs, params: BilayerSplitterParams
    ) -> LayerPair:
        s = inputs.structure
        pos = np.asarray(s.atoms, dtype=np.float64)
        atoms_A_2d, atoms_B_2d, species_A, species_B, zA, zB = _split_layers_from_z(
            pos, list(s.species), params.z_axis_index
        )
        sc_vecs_2d = np.asarray(s.cell[:2, :2], dtype=np.float64)
        atoms_A_2d, atoms_B_2d, species_A, species_B, sc_vecs_2d, replication = (
            _normalize_replication(
                atoms_A_2d, atoms_B_2d, species_A, species_B, sc_vecs_2d, params
            )
        )

        layer_A = _layer_to_structure(atoms_A_2d, species_A, zA, sc_vecs_2d)
        layer_B = _layer_to_structure(atoms_B_2d, species_B, zB, sc_vecs_2d)
        return LayerPair(
            layer_A=layer_A,
            layer_B=layer_B,
            sc_vecs=sc_vecs_2d,
            replication=replication,
            z_mean_A=zA,
            z_mean_B=zB,
        )


def _layer_to_structure(
    atoms2d: np.ndarray, species: list[str], z_mean: float, sc_vecs_2d: np.ndarray
) -> Structure:
    """Promote a 2D layer back to a Structure with the shared cell + averaged Z."""
    atoms3d = np.column_stack([atoms2d, np.full(len(atoms2d), z_mean)])
    cell = np.eye(3, dtype=np.float64)
    cell[:2, :2] = sc_vecs_2d
    return Structure(atoms=atoms3d, species=species, cell=cell)


__all__ = ["BilayerSplitter", "BilayerSplitterInputs", "BilayerSplitterParams"]
