"""Pure 2D linear-algebra primitives used by LatticeTransformer and Validator.

Direct port of the algebra helpers from `reference/lattice_matching_02032026.py`
(lines 1213-1276, 1721-1740, 1791-1837). All functions are pure (no I/O,
no globals) and operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np


def R(theta: float) -> np.ndarray:
    """2D rotation matrix for angle `theta` in **radians**."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def S(s1: float, s2: float) -> np.ndarray:
    """Anisotropic strain matrix `diag(1+s1, 1+s2)`."""
    return np.diag([1.0 + float(s1), 1.0 + float(s2)]).astype(np.float64)


def change_basis(r: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Change of basis for vectors `r` expressed in lattice `A`, projected onto `B`."""
    r = np.asarray(r, dtype=np.float64).T
    U = np.linalg.inv(np.dot(A, B.T))
    return U.dot(A.dot(r)).T


def vec2d_dot(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(_as_2d(v1), _as_2d(v2)))


def vec2d_cross(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.abs(np.cross(_as_2d(v1), _as_2d(v2))))


def vec2d_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(_as_2d(v)))


def vec2d_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = _as_2d(v1)
    v2 = _as_2d(v2)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("Angle undefined for zero-length vectors")
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def wrap_fractional(frac: np.ndarray) -> np.ndarray:
    """Wrap fractional coords into [0, 1); snap exact 1.0 to 0.0."""
    frac = np.asarray(frac, dtype=np.float64) % 1.0
    frac[np.isclose(frac, 1.0, atol=1e-8)] = 0.0
    return frac


def wrap_basis_to_cell(basis_cart: np.ndarray, lat_mat: np.ndarray) -> np.ndarray:
    """Wrap cartesian basis atoms inside the unit cell of `lat_mat`."""
    frac = basis_cart.dot(np.linalg.inv(lat_mat))
    frac = frac % 1.0
    return frac.dot(lat_mat)


def transform_basis(basis_cart: np.ndarray, orig_lat: np.ndarray, new_lat: np.ndarray) -> np.ndarray:
    """Re-express `basis_cart` (cartesian, original lattice) inside `new_lat`."""
    frac = basis_cart.dot(np.linalg.inv(orig_lat))
    frac = frac % 1.0
    return frac.dot(new_lat)


def fractional_positions(points_cart: np.ndarray, lat_mat: np.ndarray) -> np.ndarray:
    frac = np.asarray(points_cart, dtype=np.float64).dot(np.linalg.inv(lat_mat))
    return wrap_fractional(frac)


def atoms_in_lattice_patch(
    lat_mat: np.ndarray, basis_atoms_cart: np.ndarray, reps: int = 2
) -> np.ndarray:
    """Build a `(2*reps+1)^2` tile patch of `basis_atoms_cart` under `lat_mat`."""
    atoms: list[np.ndarray] = []
    for n1 in range(-reps, reps + 1):
        for n2 in range(-reps, reps + 1):
            shift = n1 * lat_mat[0] + n2 * lat_mat[1]
            for basis in basis_atoms_cart:
                atoms.append(shift + basis)
    return np.array(atoms, dtype=np.float64) if atoms else np.empty((0, 2), dtype=np.float64)


def canonical_theta_deg(
    theta_deg: float, period_deg: float = 60.0, allow_sign_flip: bool = True
) -> float:
    """Fold angle into smallest equivalent under symmetry."""
    candidates = [float(theta_deg)]
    if allow_sign_flip:
        candidates.append(-float(theta_deg))
    canonical = []
    for value in candidates:
        wrapped = ((value + period_deg / 2.0) % period_deg) - period_deg / 2.0
        canonical.append(float(wrapped))
    return min(canonical, key=lambda x: abs(x))


def angle_error_symmetry_deg(
    recovered_deg: float,
    target_deg: float | None,
    period_deg: float = 60.0,
    allow_sign_flip: bool = True,
) -> float:
    """Minimum symmetric error (degrees) between `recovered_deg` and `target_deg`."""
    if target_deg is None:
        return float("nan")
    try:
        if np.isnan(target_deg):
            return float("nan")
    except TypeError:
        pass
    rec_candidates = [float(recovered_deg)]
    if allow_sign_flip:
        rec_candidates.append(-float(recovered_deg))
    errors = []
    for cand in rec_candidates:
        delta = ((cand - float(target_deg) + period_deg / 2.0) % period_deg) - period_deg / 2.0
        errors.append(abs(float(delta)))
    return min(errors)


def _as_2d(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (2,):
        raise ValueError(f"Expected shape (2,), got {v.shape}")
    return v


__all__ = [
    "R",
    "S",
    "change_basis",
    "vec2d_dot",
    "vec2d_cross",
    "vec2d_norm",
    "vec2d_angle_deg",
    "wrap_fractional",
    "wrap_basis_to_cell",
    "transform_basis",
    "fractional_positions",
    "atoms_in_lattice_patch",
    "canonical_theta_deg",
    "angle_error_symmetry_deg",
]
