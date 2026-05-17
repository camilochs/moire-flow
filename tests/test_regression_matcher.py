"""Step-for-step regression of LatticeMatcher methods against the reference
brute-force / rotation-only functions on identical inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moire_flow.boxes import LatticeMatcher, LatticeMatcherInputs, LatticeMatcherParams


ALAT = np.array([[3.16, 0.0], [1.58, 2.737]])


@pytest.fixture(scope="module")
def matcher_reference():
    """Reference brute-force + rotation-only helpers, extracted via ast to
    handle multi-line def signatures correctly."""
    from .conftest import _extract_top_level_defs

    path = Path(__file__).resolve().parents[1] / "reference" / "lattice_matching_02032026.py"
    src = path.read_text()
    wanted = {
        "R", "S", "ChangeBasis",
        "vec2d_cross", "vec2d_angle_deg", "vec2d_norm", "_as_2d_vector",
        "supercell_points", "cost_funcion", "get_lavects",
        "find_scvect_pairs", "compute_mismatch", "make_solution_record",
        "_build_a_lattice_candidates", "param_grid_from_bounds",
        "run_brute_force", "run_brute_force_rotation_only_reference",
    }
    blocks = _extract_top_level_defs(src, wanted)
    ns: dict = {"np": np}
    exec("\n\n".join(blocks), ns)
    return ns


def _params(method: str, **extra) -> LatticeMatcherParams:
    base = dict(
        method=method,
        dim=(6, 6),
        N=4,
        ndiv=5,
        bf_top_k=10,
        theta_steps=13,
        random_seed=42,
        bounds=[(-0.03, 0.03), (-0.03, 0.03), (-1.0, 1.0)],
    )
    base.update(extra)
    return LatticeMatcherParams(**base)


def _solutions_to_sortable(sols):
    """Make solutions order-independent + comparable up to numeric tolerance."""
    out = []
    for s in sols:
        out.append(
            (
                round(float(s["s1"]) if isinstance(s, dict) else float(s.s1), 6),
                round(float(s["s2"]) if isinstance(s, dict) else float(s.s2), 6),
                round(float(s["theta_deg"]) if isinstance(s, dict) else float(s.theta_deg), 4),
                round(float(s["area"]) if isinstance(s, dict) else float(s.area), 6),
                round(float(s["mismatch"]) if isinstance(s, dict) else float(s.mismatch), 6),
            )
        )
    return sorted(out)


def test_brute_force_matches_reference(matcher_reference):
    """My _brute_force must produce the same (s1, s2, theta, area, mismatch)
    set as reference run_brute_force on the same inputs."""
    p = _params("brute_force", N=4, ndiv=5, mismatch_tol=0.05, min_area=1.0,
                bf_top_k=10, angle_min=30.0, angle_max=90.0)

    ref = matcher_reference["run_brute_force"](
        ALAT, ALAT, dim=list(p.dim), bounds=p.bounds,
        N=p.N, ndiv=p.ndiv, mismatch_tol=p.mismatch_tol,
        angle_min=p.angle_min, angle_max=p.angle_max,
        min_area=p.min_area, top_k=p.bf_top_k,
    )
    out = LatticeMatcher().run(LatticeMatcherInputs(Alat=ALAT, Blat=ALAT), p)
    assert _solutions_to_sortable(out.solutions) == _solutions_to_sortable(ref)


def test_rotation_only_matches_reference(matcher_reference):
    p = _params("rotation_only", N=4, theta_steps=13, mismatch_tol=0.05,
                min_area=1.0, bf_top_k=10, angle_min=30.0, angle_max=90.0)

    ref = matcher_reference["run_brute_force_rotation_only_reference"](
        ALAT, ALAT, dim=list(p.dim),
        N=p.N, mismatch_tol=p.mismatch_tol,
        angle_min=p.angle_min, angle_max=p.angle_max,
        min_area=p.min_area, top_k=p.bf_top_k, theta_steps=p.theta_steps,
    )
    out = LatticeMatcher().run(LatticeMatcherInputs(Alat=ALAT, Blat=ALAT), p)
    assert _solutions_to_sortable(out.solutions) == _solutions_to_sortable(ref)


def test_brute_force_matches_reference_on_strained_pair(matcher_reference):
    """Strain + rotation case — verify recovery solutions agree set-wise."""
    from moire_flow.core.algebra2d import R, S
    s1, s2, theta = -0.02, 0.015, 10.0
    Blat = S(s1, s2).dot(ALAT.dot(R(np.radians(theta)).T))

    p = _params("brute_force", N=5, ndiv=7, mismatch_tol=0.05, min_area=1.0,
                bf_top_k=10, angle_min=30.0, angle_max=90.0,
                bounds=[(-0.05, 0.05), (-0.05, 0.05), (-0.7, 0.7)])

    ref = matcher_reference["run_brute_force"](
        ALAT, Blat, dim=list(p.dim), bounds=p.bounds,
        N=p.N, ndiv=p.ndiv, mismatch_tol=p.mismatch_tol,
        angle_min=p.angle_min, angle_max=p.angle_max,
        min_area=p.min_area, top_k=p.bf_top_k,
    )
    out = LatticeMatcher().run(LatticeMatcherInputs(Alat=ALAT, Blat=Blat), p)
    assert _solutions_to_sortable(out.solutions) == _solutions_to_sortable(ref)
