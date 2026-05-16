"""Pure structural metrics used by the Validator box.

Port of reference functions 1840-1953:
- fractional_rmsd, pairwise_distance_signature, local_distance_signature,
  distance_signature_error, neighbor_counts, coordination_match_score,
  lattice_metrics.
"""

from __future__ import annotations

import numpy as np

from .algebra2d import vec2d_angle_deg, vec2d_norm, wrap_fractional


def periodic_fractional_distances(frac_ref: np.ndarray, frac_test: np.ndarray) -> np.ndarray:
    """Minimum-image pairwise distance in fractional coords."""
    delta = np.abs(frac_ref[:, None, :] - frac_test[None, :, :])
    delta = np.minimum(delta, 1.0 - delta)
    return np.linalg.norm(delta, axis=2)


def fractional_distance_signature(frac: np.ndarray) -> np.ndarray:
    frac = wrap_fractional(frac)
    if len(frac) < 2:
        return np.array([], dtype=np.float64)
    dists = periodic_fractional_distances(frac, frac)
    iu = np.triu_indices(len(frac), k=1)
    vals = dists[iu]
    vals = vals[vals > 1e-8]
    if len(vals) == 0:
        return np.array([], dtype=np.float64)
    return np.sort(np.round(vals, 8))


def fractional_rmsd(
    frac_ref: np.ndarray, frac_test: np.ndarray, tol_match: float = 1e-6
) -> float:
    sig_ref = fractional_distance_signature(frac_ref)
    sig_test = fractional_distance_signature(frac_test)
    n = min(len(sig_ref), len(sig_test))
    if n == 0:
        return float("nan")
    value = float(np.sqrt(np.mean((sig_ref[:n] - sig_test[:n]) ** 2)))
    return 0.0 if value < tol_match else value


def pairwise_distance_signature(points: np.ndarray, k: int = 6) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.array([], dtype=np.float64)
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices(len(points), k=1)
    vals = dists[iu]
    vals = vals[vals > 1e-6]
    if len(vals) == 0:
        return np.array([], dtype=np.float64)
    vals = np.unique(np.round(vals, 6))
    vals.sort()
    return vals[:k]


def local_distance_signature(
    points: np.ndarray, support_points: np.ndarray | None = None, k: int = 6
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.array([], dtype=np.float64)
    support = points if support_points is None else np.asarray(support_points, dtype=np.float64)
    signature: list[float] = []
    for point in points:
        dists = np.linalg.norm(support - point, axis=1)
        dists = dists[dists > 1e-8]
        if len(dists) == 0:
            continue
        dists.sort()
        signature.extend(np.round(dists[:k], 6).tolist())
    if not signature:
        return np.array([], dtype=np.float64)
    return np.array(sorted(signature), dtype=np.float64)


def distance_signature_error(
    points_ref: np.ndarray,
    points_test: np.ndarray,
    k: int = 6,
    support_points_ref: np.ndarray | None = None,
    support_points_test: np.ndarray | None = None,
) -> float:
    sig_ref = local_distance_signature(points_ref, support_points=support_points_ref, k=k)
    sig_test = local_distance_signature(points_test, support_points=support_points_test, k=k)
    n = min(len(sig_ref), len(sig_test))
    if n == 0:
        return float("nan")
    ref, test = sig_ref[:n], sig_test[:n]
    denom = max(float(np.linalg.norm(ref)), 1e-12)
    return float(np.linalg.norm(ref - test) / denom)


def neighbor_counts(
    points: np.ndarray, cutoff: float, support_points: np.ndarray | None = None
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.array([], dtype=int)
    support = points if support_points is None else np.asarray(support_points, dtype=np.float64)
    diffs = points[:, None, :] - support[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    mask = (dists > 1e-8) & (dists <= cutoff)
    return np.sum(mask, axis=1).astype(int)


def coordination_match_score(
    points_ref: np.ndarray,
    points_test: np.ndarray,
    cutoff: float,
    support_points_ref: np.ndarray | None = None,
    support_points_test: np.ndarray | None = None,
) -> float:
    counts_ref = neighbor_counts(points_ref, cutoff, support_points=support_points_ref)
    counts_test = neighbor_counts(points_test, cutoff, support_points=support_points_test)
    if len(counts_ref) == 0 or len(counts_test) == 0:
        return float("nan")
    maxbin = max(int(np.max(counts_ref)), int(np.max(counts_test)))
    hist_ref = np.bincount(counts_ref, minlength=maxbin + 1).astype(float)
    hist_test = np.bincount(counts_test, minlength=maxbin + 1).astype(float)
    hist_ref /= hist_ref.sum()
    hist_test /= hist_test.sum()
    l1 = float(np.sum(np.abs(hist_ref - hist_test)))
    return float(max(0.0, 1.0 - 0.5 * l1))


def lattice_metrics(lat: np.ndarray) -> dict[str, float]:
    a1, a2 = lat
    return {
        "a1": vec2d_norm(a1),
        "a2": vec2d_norm(a2),
        "angle_deg": vec2d_angle_deg(a1, a2),
    }


__all__ = [
    "periodic_fractional_distances",
    "fractional_distance_signature",
    "fractional_rmsd",
    "pairwise_distance_signature",
    "local_distance_signature",
    "distance_signature_error",
    "neighbor_counts",
    "coordination_match_score",
    "lattice_metrics",
]
