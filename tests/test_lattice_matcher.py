"""LatticeMatcher: identity, rotation, and strain-recovery cases."""

from __future__ import annotations

import numpy as np
import pytest

from moire_flow.boxes import LatticeMatcher, LatticeMatcherInputs, LatticeMatcherParams
from moire_flow.core.algebra2d import R, S, canonical_theta_deg


ALAT = np.array([[3.16, 0.0], [1.58, 2.737]])


def _params(method: str, **overrides):
    base = dict(
        method=method,
        dim=(6, 6),
        N=4,
        ndiv=5,
        bf_top_k=5,
        top_k=5,
        theta_steps=13,
        popsize=20,
        maxiter=30,
        random_seed=42,
    )
    base.update(overrides)
    return LatticeMatcherParams(**base)


def test_brute_force_recovers_identity():
    out = LatticeMatcher().run(LatticeMatcherInputs(Alat=ALAT, Blat=ALAT), _params("brute_force"))
    assert len(out.solutions) > 0
    best = out.solutions[0]
    # hexagonal symmetry: theta = 0, ±60, ±120, 180 are all equivalent.
    assert abs(canonical_theta_deg(best.theta_deg, period_deg=60.0)) < 1e-6
    assert abs(best.s1) < 1e-9 and abs(best.s2) < 1e-9
    assert best.mismatch < 1e-9


def test_rotation_only_recovers_zero_rotation():
    out = LatticeMatcher().run(LatticeMatcherInputs(Alat=ALAT, Blat=ALAT), _params("rotation_only"))
    assert len(out.solutions) > 0
    best = out.solutions[0]
    assert abs(canonical_theta_deg(best.theta_deg, period_deg=60.0)) < 1e-6
    # mismatch carries tiny FP error from R(theta) at the discrete grid step
    assert best.mismatch < 1e-3


def test_brute_force_recovers_known_strain_pluss_rotation():
    s1, s2, theta = -0.03, 0.02, 15.0
    Blat_orig = S(s1, s2).dot(ALAT.dot(R(np.radians(theta)).T))
    # invert the transform: Blat -> recover s1≈-s1/(1+s1), s2≈-s2/(1+s2), theta≈-15
    out = LatticeMatcher().run(
        LatticeMatcherInputs(Alat=ALAT, Blat=Blat_orig),
        _params("brute_force", N=5, ndiv=9, bounds=[(-0.05, 0.05), (-0.05, 0.05), (-np.pi/6, np.pi/6)]),
    )
    assert len(out.solutions) > 0
    best = out.solutions[0]
    assert best.mismatch < 0.06


def test_continuous_recovers_identity():
    out = LatticeMatcher().run(
        LatticeMatcherInputs(Alat=ALAT, Blat=ALAT),
        _params("continuous", popsize=30, maxiter=60),
    )
    assert len(out.solutions) > 0
    best = out.solutions[0]
    # DE may converge slightly off zero; loose tolerance
    assert abs(best.s1) < 1e-2
    assert abs(best.s2) < 1e-2
    assert best.mismatch < 0.05


def test_solutions_sorted_by_area():
    out = LatticeMatcher().run(LatticeMatcherInputs(Alat=ALAT, Blat=ALAT), _params("brute_force"))
    areas = [s.area for s in out.solutions]
    assert areas == sorted(areas)


def test_solutions_are_pydantic_models():
    out = LatticeMatcher().run(LatticeMatcherInputs(Alat=ALAT, Blat=ALAT), _params("rotation_only"))
    sol = out.solutions[0]
    js = sol.model_dump_json()
    from moire_flow.core.types import MatchingSolution
    again = MatchingSolution.model_validate_json(js)
    assert again.theta_deg == pytest.approx(sol.theta_deg)
