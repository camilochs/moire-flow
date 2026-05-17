"""LatticeMatcher: search supercell-commensurate solutions.

Three methods (selected via `params.method`):
- `continuous`: differential evolution over (s1, s2, theta), then extract
  near-integer lattice vectors and pair them. Port of
  `run_continuous_optimization` (reference 1345-1454).
- `brute_force`: grid over (s1, s2, theta) × integer-coeff pairs.
  Port of `run_brute_force` (reference 1500-1586).
- `rotation_only`: pure-rotation reference scan with zero strain.
  Port of `run_brute_force_rotation_only_reference` (reference 1589-1661).

Output: list of `MatchingSolution`, sorted by area (smallest first).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy import optimize

from moire_flow.core.algebra2d import R, S
from moire_flow.core.matching import (
    build_a_lattice_candidates,
    change_basis,
    compute_mismatch,
    cost_funcion,
    find_scvect_pairs,
    get_lavects,
    param_grid_from_bounds,
    supercell_points,
    vec2d_angle_deg,
    vec2d_cross,
)
from moire_flow.core.types import MatchingSolution, NDArray2x2

from .base import Box, register_box

Method = Literal["continuous", "brute_force", "rotation_only"]


class LatticeMatcherInputs(BaseModel):
    Alat: NDArray2x2
    Blat: NDArray2x2


class LatticeMatcherParams(BaseModel):
    method: Method = "continuous"
    dim: tuple[int, int] = (10, 10)
    bounds: list[tuple[float, float]] = Field(
        default_factory=lambda: [(-0.05, 0.05), (-0.05, 0.05), (-np.pi / 2, np.pi / 2)]
    )
    # Continuous (DE) params
    popsize: int = Field(default=80, ge=4)
    maxiter: int = Field(default=120, ge=1)
    tol_lavect: float = Field(default=1e-2, gt=0.0)
    angle_min: float = 30.0
    angle_max: float = 90.0
    top_k: int = Field(default=5, ge=1)
    random_seed: int | None = 42
    # Brute-force params
    N: int = Field(default=6, ge=1)
    ndiv: int | None = Field(default=7, ge=1)
    ndiv_strain: int | None = None
    ndiv_theta: int | None = None
    mismatch_tol: float = Field(default=0.05, gt=0.0)
    min_area: float = Field(default=1.0, ge=0.0)
    bf_top_k: int = Field(default=20, ge=1)
    # Rotation-only params
    theta_steps: int = Field(default=61, ge=2)


def _record(
    s1: float,
    s2: float,
    theta_rad: float,
    cost: float,
    v1: np.ndarray,
    v2: np.ndarray,
    area: float,
    mismatch: float,
    angle_deg: float,
    oBlat: np.ndarray,
) -> MatchingSolution:
    return MatchingSolution(
        v1=np.asarray(v1, dtype=np.float64),
        v2=np.asarray(v2, dtype=np.float64),
        s1=float(s1),
        s2=float(s2),
        theta_deg=float(np.degrees(theta_rad)),
        cost=float(cost),
        mismatch=float(mismatch),
        area=float(area),
        oBlat=np.asarray(oBlat, dtype=np.float64),
        angle_deg=float(angle_deg),
    )


def _continuous(
    Alat: np.ndarray, Blat: np.ndarray, p: LatticeMatcherParams
) -> list[MatchingSolution]:
    dim = p.dim

    def fitness(x: np.ndarray) -> float:
        if len(x) == 3:
            s1, s2, theta = x
        elif len(x) == 2:
            s1, s2 = x
            theta = 0.0
        else:
            s1 = s2 = float(x[0])
            theta = 0.0
        oBlat = S(s1, s2).dot(Blat.dot(R(theta).T))
        Bsc = supercell_points(dim, oBlat)
        rBinA = change_basis(Bsc, oBlat, Alat)
        return cost_funcion(rBinA)

    res = optimize.differential_evolution(
        fitness,
        p.bounds,
        maxiter=p.maxiter,
        popsize=p.popsize,
        polish=True,
        seed=p.random_seed,
    )

    pop = np.array(res.population, dtype=np.float64)
    energies = np.array(res.population_energies, dtype=np.float64)
    threshold = np.percentile(energies, 10)
    good_idx = np.where(energies <= threshold)[0]

    unique_x: list[np.ndarray] = []
    for idx in good_idx:
        x = pop[idx].copy()
        if not any(np.linalg.norm(x - ux) < 1e-3 for ux in unique_x):
            unique_x.append(x)
    if not any(np.linalg.norm(res.x - ux) < 1e-3 for ux in unique_x):
        unique_x.insert(0, np.array(res.x, dtype=np.float64))

    results: list[MatchingSolution] = []
    for x in unique_x:
        if len(x) == 3:
            s1, s2, theta = x
        elif len(x) == 2:
            s1, s2 = x
            theta = 0.0
        else:
            s1 = s2 = float(x[0])
            theta = 0.0
        oBlat = S(s1, s2).dot(Blat.dot(R(theta).T))
        oBsc = supercell_points(dim, oBlat)
        roBinA = change_basis(oBsc, oBlat, Alat)
        cost = cost_funcion(roBinA)
        L = get_lavects(roBinA, oBsc, tol=p.tol_lavect)
        if len(L) == 0:
            continue
        norms = np.linalg.norm(L, axis=1)
        L = L[norms > 1e-6]
        if len(L) < 2:
            continue
        pairs = find_scvect_pairs(L, p.angle_min, p.angle_max, p.top_k)
        for area, angle, v1, v2 in pairs:
            mismatch = compute_mismatch(np.array([v1, v2]), Alat, oBlat)
            results.append(_record(s1, s2, theta, cost, v1, v2, area, mismatch, angle, oBlat))

    # Deduplicate by close v1/v2 (tolerance 0.1, faithful to reference)
    deduped: list[MatchingSolution] = []
    for r in results:
        is_dup = any(
            np.linalg.norm(r.v1 - ur.v1) < 0.1 and np.linalg.norm(r.v2 - ur.v2) < 0.1
            for ur in deduped
        )
        if not is_dup:
            deduped.append(r)
    deduped.sort(key=lambda r: r.area)
    return deduped


def _enumerate_pairs(
    candidates: np.ndarray, min_area: float, angle_min: float, angle_max: float
) -> list[tuple[int, int, float, float]]:
    pairs: list[tuple[int, int, float, float]] = []
    n = len(candidates)
    for i in range(n):
        v1 = candidates[i]
        for j in range(i + 1, n):
            v2 = candidates[j]
            area = vec2d_cross(v1, v2)
            if area < min_area:
                continue
            angle = vec2d_angle_deg(v1, v2)
            if angle_min <= angle <= angle_max:
                pairs.append((i, j, float(area), float(angle)))
    return pairs


def _brute_force(
    Alat: np.ndarray, Blat: np.ndarray, p: LatticeMatcherParams
) -> list[MatchingSolution]:
    Ainv = np.linalg.inv(Alat)
    candidates = build_a_lattice_candidates(Alat, p.N)
    valid_pairs = _enumerate_pairs(candidates, p.min_area, p.angle_min, p.angle_max)

    s1_vals, s2_vals, theta_vals = param_grid_from_bounds(
        p.bounds, ndiv=p.ndiv, ndiv_strain=p.ndiv_strain, ndiv_theta=p.ndiv_theta
    )
    best_by_key: dict[tuple, MatchingSolution] = {}
    cost_cache: dict[tuple[int, int, int], float] = {}

    for i_s1, s1 in enumerate(s1_vals):
        for i_s2, s2 in enumerate(s2_vals):
            for i_th, theta in enumerate(theta_vals):
                oBlat = S(s1, s2).dot(Blat.dot(R(theta).T))
                oBinv = np.linalg.inv(oBlat)
                cache_key = (i_s1, i_s2, i_th)
                for i, j, area, angle in valid_pairs:
                    v1, v2 = candidates[i], candidates[j]
                    sc = np.array([v1, v2])
                    coeffs_B = sc.dot(oBinv)
                    res_B = float(np.max(np.abs(coeffs_B - np.round(coeffs_B))))
                    if res_B > p.mismatch_tol:
                        continue
                    iA = np.round(sc.dot(Ainv)).astype(int)
                    iB = np.round(coeffs_B).astype(int)
                    geom_key = (
                        tuple(sorted([tuple(iA[0]), tuple(iA[1])])),
                        tuple(sorted([tuple(iB[0]), tuple(iB[1])])),
                    )
                    prev = best_by_key.get(geom_key)
                    if prev is not None and prev.mismatch <= res_B:
                        continue
                    if cache_key not in cost_cache:
                        Bsc_pts = supercell_points(p.dim, oBlat)
                        rBinA = change_basis(Bsc_pts, oBlat, Alat)
                        cost_cache[cache_key] = cost_funcion(rBinA)
                    best_by_key[geom_key] = _record(
                        s1, s2, theta, cost_cache[cache_key], v1, v2, area, res_B, angle, oBlat
                    )

    results = sorted(best_by_key.values(), key=lambda r: r.area)
    return results[: p.bf_top_k]


def _rotation_only(
    Alat: np.ndarray, Blat: np.ndarray, p: LatticeMatcherParams
) -> list[MatchingSolution]:
    Ainv = np.linalg.inv(Alat)
    candidates = build_a_lattice_candidates(Alat, p.N)
    valid_pairs = _enumerate_pairs(candidates, p.min_area, p.angle_min, p.angle_max)

    thetas = np.linspace(-np.pi / 2, np.pi / 2, p.theta_steps)
    best_by_key: dict[tuple, MatchingSolution] = {}
    cost_cache: dict[int, float] = {}

    for ith, theta in enumerate(thetas):
        oBlat = Blat.dot(R(theta).T)
        oBinv = np.linalg.inv(oBlat)
        for i, j, area, angle in valid_pairs:
            v1, v2 = candidates[i], candidates[j]
            sc = np.array([v1, v2])
            coeffs_B = sc.dot(oBinv)
            res_B = float(np.max(np.abs(coeffs_B - np.round(coeffs_B))))
            if res_B > p.mismatch_tol:
                continue
            iA = np.round(sc.dot(Ainv)).astype(int)
            iB = np.round(coeffs_B).astype(int)
            key = (
                tuple(sorted([tuple(iA[0]), tuple(iA[1])])),
                tuple(sorted([tuple(iB[0]), tuple(iB[1])])),
            )
            prev = best_by_key.get(key)
            if prev is not None and prev.mismatch <= res_B:
                continue
            if ith not in cost_cache:
                Bsc_pts = supercell_points(p.dim, oBlat)
                rBinA = change_basis(Bsc_pts, oBlat, Alat)
                cost_cache[ith] = cost_funcion(rBinA)
            best_by_key[key] = _record(
                0.0, 0.0, theta, cost_cache[ith], v1, v2, area, res_B, angle, oBlat
            )

    out = sorted(best_by_key.values(), key=lambda r: r.area)
    return out[: p.bf_top_k]


class LatticeMatcherOutput(BaseModel):
    solutions: list[MatchingSolution]


@register_box
class LatticeMatcher(Box[LatticeMatcherInputs, LatticeMatcherParams, LatticeMatcherOutput]):
    name = "lattice_matcher"
    description = "Search supercell-commensurate (s1, s2, theta) solutions matching Alat and Blat."
    inputs_schema = LatticeMatcherInputs
    params_schema = LatticeMatcherParams
    outputs_schema = LatticeMatcherOutput

    def run(
        self, inputs: LatticeMatcherInputs, params: LatticeMatcherParams
    ) -> LatticeMatcherOutput:
        Alat = np.asarray(inputs.Alat, dtype=np.float64)
        Blat = np.asarray(inputs.Blat, dtype=np.float64)
        if params.method == "continuous":
            sols = _continuous(Alat, Blat, params)
        elif params.method == "brute_force":
            sols = _brute_force(Alat, Blat, params)
        elif params.method == "rotation_only":
            sols = _rotation_only(Alat, Blat, params)
        else:  # pragma: no cover - exhaustive Literal
            raise ValueError(f"Unknown method: {params.method}")
        return LatticeMatcherOutput(solutions=sols)


__all__ = [
    "LatticeMatcher",
    "LatticeMatcherInputs",
    "LatticeMatcherParams",
    "LatticeMatcherOutput",
    "Method",
]
