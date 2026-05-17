"""Pure matching primitives used by the LatticeMatcher box.

Port of reference 1207-1497: supercell_points, cost_funcion, get_lavects,
find_scvect_pairs, compute_mismatch, _build_a_lattice_candidates,
param_grid_from_bounds.
"""

from __future__ import annotations

import numpy as np

from .algebra2d import change_basis, vec2d_angle_deg, vec2d_cross, vec2d_norm


def supercell_points(dims: tuple[int, int], lat_vec: np.ndarray) -> np.ndarray:
    """All `(2*nx)*(2*ny)` integer lattice points expressed in cartesian `lat_vec`."""
    nx, ny = int(dims[0]), int(dims[1])
    grid = np.mgrid[-nx:nx, -ny:ny].T.reshape(nx * 2 * ny * 2, 2)
    return grid @ lat_vec


def cost_funcion(r: np.ndarray, eta: float = 0.01) -> float:
    """Periodic-Gaussian cost: minimized when fractional coords lie near integers.

    Faithful port of reference `cost_funcion`. Returns a negative number whose
    magnitude grows with the quality of the match.
    """
    dx, dy = (r - np.floor(r)).T
    cost = 0.0
    etac = float(np.sqrt(2.0) * eta)
    n = int(np.ceil(eta))
    for nx, ny in np.mgrid[-n:n, -n:n].T.reshape(n**2 * 4, 2):
        d2 = (dx + nx) ** 2 + (dy + ny) ** 2
        cost += np.mean(np.exp(-d2 / (2.0 * etac))) / n**2
    return -float(cost)


def get_lavects(x: np.ndarray, lat: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Filter cartesian lattice vectors whose fractional coords are close to integers."""
    return lat[np.linalg.norm(np.round(x) - x, axis=1) < tol]


def find_scvect_pairs(
    L: np.ndarray,
    angle_min: float = 30.0,
    angle_max: float = 90.0,
    top_k: int = 5,
) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """Pick supercell vector pairs from candidates `L`, ranked by ascending area.

    Returns up to `top_k` tuples `(area, angle_deg, v1, v2)`.
    """
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[1] != 2:
        raise ValueError(f"L must have shape (n, 2). Got {L.shape}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    norms = np.linalg.norm(L, axis=1)
    L = L[norms > 0.0]
    if len(L) < 2:
        return []

    A = L[:, None, :]
    B = L[None, :, :]
    dot = np.sum(A * B, axis=2)
    cross = A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]
    area = np.abs(cross)
    angle_deg = np.degrees(np.arctan2(area, dot))

    iu, ju = np.triu_indices(len(L), k=1)
    area_u = area[iu, ju]
    angle_u = angle_deg[iu, ju]
    valid = (angle_u >= angle_min) & (angle_u <= angle_max)
    if not np.any(valid):
        return []

    iu = iu[valid]
    ju = ju[valid]
    area_u = area_u[valid]
    angle_u = angle_u[valid]
    k = min(top_k, area_u.size)
    sel = np.argpartition(area_u, kth=k - 1)[:k]
    sel = sel[np.argsort(area_u[sel])]

    out: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for idx in sel:
        i = int(iu[idx])
        j = int(ju[idx])
        out.append((float(area_u[idx]), float(angle_u[idx]), L[i].copy(), L[j].copy()))
    return out


def compute_mismatch(
    sc_vecs: np.ndarray, Alat: np.ndarray, Blat: np.ndarray
) -> float:
    """Max integer-distance of the supercell expressed in A and B fractional coords."""
    sc = np.asarray(sc_vecs, dtype=np.float64)
    coeffs_A = sc.dot(np.linalg.inv(Alat))
    coeffs_B = sc.dot(np.linalg.inv(Blat))
    res_A = np.max(np.abs(coeffs_A - np.round(coeffs_A)))
    res_B = np.max(np.abs(coeffs_B - np.round(coeffs_B)))
    return float(max(res_A, res_B))


def build_a_lattice_candidates(Alat: np.ndarray, N: int) -> np.ndarray:
    """All non-zero integer combinations of A-lattice vectors with `|n_i| <= N`."""
    n1, n2 = np.meshgrid(
        np.arange(-N, N + 1),
        np.arange(-N, N + 1),
        indexing="ij",
    )
    coeffs = np.stack([n1.ravel(), n2.ravel()], axis=1)
    coeffs = coeffs[~np.all(coeffs == 0, axis=1)]
    return coeffs @ Alat


def atoms_in_supercell(
    sc_vecs: np.ndarray,
    lat_mat: np.ndarray,
    basis_atoms_cart: np.ndarray,
    n_range: int | None = None,
) -> np.ndarray:
    """Fill the supercell `sc_vecs` with periodic copies of `basis_atoms_cart`.

    Faithful port of reference `atoms_in_supercell` (1700-1718): an atom is
    kept iff its fractional coordinate in the supercell falls inside
    [0, 1) (with a 1e-6 tolerance on both ends).
    """
    sc_mat = np.asarray(sc_vecs, dtype=np.float64)
    sc_inv = np.linalg.inv(sc_mat)
    lat_mat = np.asarray(lat_mat, dtype=np.float64)
    basis_atoms_cart = np.asarray(basis_atoms_cart, dtype=np.float64)
    if n_range is None:
        max_norm = max(vec2d_norm(sc_mat[0]), vec2d_norm(sc_mat[1]))
        lat_norm = min(vec2d_norm(lat_mat[0]), vec2d_norm(lat_mat[1]))
        n_range = int(np.ceil(max_norm / lat_norm)) + 2

    atoms: list[np.ndarray] = []
    for n1 in range(-n_range, n_range + 1):
        for n2 in range(-n_range, n_range + 1):
            shift = n1 * lat_mat[0] + n2 * lat_mat[1]
            for basis in basis_atoms_cart:
                pos = shift + basis
                frac = pos.dot(sc_inv)
                if (-1e-6 <= frac[0] < 1.0 - 1e-6) and (-1e-6 <= frac[1] < 1.0 - 1e-6):
                    atoms.append(pos)
    return np.array(atoms, dtype=np.float64) if atoms else np.empty((0, 2), dtype=np.float64)


def param_grid_from_bounds(
    bounds: list[tuple[float, float]],
    ndiv: int | None = None,
    ndiv_strain: int | None = None,
    ndiv_theta: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform 3D parameter grid for (s1, s2, theta) over `bounds`."""
    if len(bounds) != 3:
        raise ValueError("bounds must have three ranges")
    if ndiv is not None:
        if ndiv < 1:
            raise ValueError("ndiv must be >= 1")
        if ndiv_strain is None:
            ndiv_strain = ndiv
        if ndiv_theta is None:
            ndiv_theta = ndiv
    if ndiv_strain is None or ndiv_theta is None:
        raise ValueError("Provide either ndiv or both ndiv_strain and ndiv_theta")
    if ndiv_strain < 1 or ndiv_theta < 1:
        raise ValueError("ndiv_strain and ndiv_theta must be >= 1")
    divisions = [int(ndiv_strain), int(ndiv_strain), int(ndiv_theta)]
    grids: list[np.ndarray] = []
    for (lo, hi), nd in zip(bounds, divisions):
        lo, hi = float(lo), float(hi)
        if hi < lo:
            raise ValueError(f"Invalid bounds: {(lo, hi)}")
        if nd == 1:
            grids.append(np.array([(lo + hi) / 2.0]))
        else:
            grids.append(np.linspace(lo, hi, nd))
    return grids[0], grids[1], grids[2]


__all__ = [
    "supercell_points",
    "cost_funcion",
    "get_lavects",
    "find_scvect_pairs",
    "compute_mismatch",
    "build_a_lattice_candidates",
    "atoms_in_supercell",
    "param_grid_from_bounds",
    "change_basis",
    "vec2d_cross",
    "vec2d_angle_deg",
]
