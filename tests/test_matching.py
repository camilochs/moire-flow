"""Matching primitives: regression vs reference + identity recovery."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from moire_flow.core.matching import (
    build_a_lattice_candidates,
    compute_mismatch,
    cost_funcion,
    find_scvect_pairs,
    get_lavects,
    param_grid_from_bounds,
    supercell_points,
)


@pytest.fixture(scope="module")
def reference_module():
    path = Path(__file__).resolve().parents[1] / "reference" / "lattice_matching_02032026.py"
    src = path.read_text()
    needed = [
        "def R(",
        "def S(",
        "def ChangeBasis(",
        "def supercell_points(",
        "def cost_funcion(",
        "def get_lavects(",
        "def find_scvect_pairs(",
        "def compute_mismatch(",
        "def _build_a_lattice_candidates(",
        "def param_grid_from_bounds(",
    ]
    lines = src.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if any(lines[i].startswith(n) for n in needed):
            block = [lines[i]]
            i += 1
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or lines[i] == ""):
                block.append(lines[i])
                i += 1
            blocks.append("\n".join(block))
        else:
            i += 1
    ns: dict = {"np": np}
    exec("\n\n".join(blocks), ns)
    return ns


ALAT = np.array([[3.16, 0.0], [1.58, 2.737]])


def test_supercell_points_matches_reference(reference_module):
    ours = supercell_points((4, 5), ALAT)
    theirs = reference_module["supercell_points"]((4, 5), ALAT)
    np.testing.assert_array_equal(ours, theirs)


def test_cost_funcion_matches_reference(reference_module):
    rng = np.random.default_rng(0)
    r = rng.uniform(-2, 2, size=(20, 2))
    np.testing.assert_allclose(cost_funcion(r), reference_module["cost_funcion"](r))


def test_cost_funcion_minimal_at_integer_points():
    r_int = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]])
    r_off = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    assert cost_funcion(r_int) < cost_funcion(r_off)


def test_get_lavects_filters_near_integers(reference_module):
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, size=(20, 2))
    lat = rng.uniform(-5, 5, size=(20, 2))
    ours = get_lavects(x, lat, tol=0.2)
    theirs = reference_module["get_lavects"](x, lat, tol=0.2)
    np.testing.assert_array_equal(ours, theirs)


def test_find_scvect_pairs_matches_reference(reference_module):
    L = build_a_lattice_candidates(ALAT, N=3)
    ours = find_scvect_pairs(L, angle_min=30, angle_max=90, top_k=5)
    theirs = reference_module["find_scvect_pairs"](L, angle_min=30, angle_max=90, top_k=5)
    assert len(ours) == len(theirs)
    for (a1, ang1, v1a, v2a), (a2, ang2, v1b, v2b) in zip(ours, theirs):
        assert abs(a1 - a2) < 1e-9
        assert abs(ang1 - ang2) < 1e-9
        np.testing.assert_allclose(v1a, v1b)
        np.testing.assert_allclose(v2a, v2b)


def test_compute_mismatch_zero_when_commensurate():
    sc = 2 * ALAT
    assert compute_mismatch(sc, ALAT, ALAT) < 1e-12


def test_build_a_lattice_candidates_excludes_zero(reference_module):
    ours = build_a_lattice_candidates(ALAT, N=3)
    theirs = reference_module["_build_a_lattice_candidates"](ALAT, 3)
    assert ours.shape == theirs.shape
    np.testing.assert_array_equal(ours, theirs)
    assert not np.any(np.all(ours == 0, axis=1))


def test_param_grid_from_bounds_3d():
    s1, s2, th = param_grid_from_bounds(
        [(-0.05, 0.05), (-0.05, 0.05), (-1.0, 1.0)], ndiv=5
    )
    assert s1.shape == (5,)
    assert s2.shape == (5,)
    assert th.shape == (5,)
    np.testing.assert_allclose(s1[0], -0.05)
    np.testing.assert_allclose(s1[-1], 0.05)


def test_param_grid_separate_strain_theta_div():
    s1, s2, th = param_grid_from_bounds(
        [(-0.05, 0.05), (-0.05, 0.05), (-1.0, 1.0)],
        ndiv_strain=3,
        ndiv_theta=7,
    )
    assert s1.shape == (3,) and s2.shape == (3,) and th.shape == (7,)


def test_param_grid_invalid_bounds():
    with pytest.raises(ValueError):
        param_grid_from_bounds([(0, 1), (0, 1)], ndiv=2)  # only 2 ranges
