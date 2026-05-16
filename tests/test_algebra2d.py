"""Property tests for the 2D algebra primitives."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from moire_flow.core.algebra2d import (
    R,
    S,
    angle_error_symmetry_deg,
    atoms_in_lattice_patch,
    canonical_theta_deg,
    transform_basis,
    vec2d_angle_deg,
    wrap_fractional,
)

ANGLE = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi, allow_nan=False)
SMALL = st.floats(min_value=-0.2, max_value=0.2, allow_nan=False)


@given(ANGLE)
def test_R_is_orthonormal(theta: float):
    M = R(theta)
    np.testing.assert_allclose(M @ M.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(M), 1.0, atol=1e-12)


@given(ANGLE, ANGLE)
def test_R_composition_additive(a: float, b: float):
    np.testing.assert_allclose(R(a) @ R(b), R(a + b), atol=1e-10)


@given(SMALL, SMALL)
def test_S_diagonal(s1: float, s2: float):
    M = S(s1, s2)
    np.testing.assert_allclose(M, np.diag([1 + s1, 1 + s2]))


def test_vec2d_angle_deg_known():
    assert vec2d_angle_deg([1.0, 0.0], [0.0, 1.0]) == 90.0
    np.testing.assert_allclose(vec2d_angle_deg([1.0, 0.0], [1.0, 1.0]), 45.0)


@given(st.floats(min_value=0.0, max_value=1e3))
def test_wrap_fractional_in_unit_interval(x: float):
    frac = wrap_fractional(np.array([[x, x]]))
    assert (frac >= 0).all() and (frac < 1).all()


@given(ANGLE)
def test_canonical_theta_within_period(theta_rad: float):
    out = canonical_theta_deg(np.degrees(theta_rad), period_deg=60.0)
    assert -30.0 - 1e-9 <= out <= 30.0 + 1e-9


@given(st.floats(min_value=-180, max_value=180))
def test_angle_error_zero_when_equal(t: float):
    assume(not np.isnan(t))
    err = angle_error_symmetry_deg(t, t, period_deg=60.0, allow_sign_flip=False)
    assert err < 1e-9


def test_transform_basis_roundtrips_when_lat_unchanged():
    basis = np.array([[0.0, 0.0], [1.0, 0.5]])
    lat = np.array([[3.16, 0.0], [1.58, 2.737]])
    out = transform_basis(basis, lat, lat)
    np.testing.assert_allclose(out, basis, atol=1e-12)


def test_atoms_in_lattice_patch_count():
    lat = np.eye(2)
    basis = np.array([[0.0, 0.0], [0.5, 0.5]])
    patch = atoms_in_lattice_patch(lat, basis, reps=2)
    assert patch.shape == ((2 * 2 + 1) ** 2 * 2, 2)


@given(SMALL, SMALL, st.floats(min_value=-45.0, max_value=45.0))
@settings(max_examples=30, deadline=None)
def test_strain_plus_rotation_decomposition(s1: float, s2: float, theta: float):
    Alat = np.array([[3.16, 0.0], [1.58, 2.737]])
    direct = S(s1, s2).dot(Alat.dot(R(np.radians(theta)).T))
    Astrain = S(s1, s2).dot(Alat)
    Arot = Alat.dot(R(np.radians(theta)).T)
    np.testing.assert_allclose(direct, S(s1, s2).dot(Arot), atol=1e-12)
    np.testing.assert_allclose(direct, Astrain.dot(R(np.radians(theta)).T), atol=1e-12)
