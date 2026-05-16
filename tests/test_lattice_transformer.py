"""LatticeTransformer regression against reference module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from moire_flow.boxes import (
    LatticeTransformer,
    LatticeTransformerInputs,
    LatticeTransformerParams,
)


@pytest.fixture(scope="module")
def reference_module():
    """Load the vendored original script as a module (without executing globals
    that need Colab/network); we only need its pure helpers."""
    path = Path(__file__).resolve().parents[1] / "reference" / "lattice_matching_02032026.py"
    src = path.read_text()
    # Truncate to the pure-helpers prefix (before any Colab/network side effect).
    cutoff_marker = "# Cell 2 — Imports / setup"
    cutoff = src.find(cutoff_marker)
    src_prefix = src if cutoff < 0 else src[:cutoff]

    # We only need a couple of functions; eval just those.
    ns: dict = {"np": np}
    # Locate and exec the helper defs we need
    needed = [
        "def R(",
        "def S(",
        "def transform_basis(",
        "def build_transformed_lattice(",
        "def build_transformed_basis(",
    ]
    lines = src.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if any(line.startswith(n) for n in needed):
            block = [line]
            i += 1
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or lines[i] == ""):
                block.append(lines[i])
                i += 1
            blocks.append("\n".join(block))
        else:
            i += 1
    exec("\n\n".join(blocks), ns)
    return ns


ALAT = np.array([[3.16, 0.0], [1.58, 2.737]])
# Hexagonal MoS2 ideal: Mo at fractional (0,0), S at (1/3, 2/3) in cartesian.
BASIS = np.array([[0.0, 0.0], [2.106667, 1.824667]])
SPECIES = ["Mo", "S"]


def _run(transform_type: str, **params) -> tuple[np.ndarray, np.ndarray]:
    box = LatticeTransformer()
    out = box.run(
        LatticeTransformerInputs(Alat=ALAT, basis_A=BASIS, species_A=SPECIES),
        LatticeTransformerParams(transform_type=transform_type, **params),
    )
    return np.asarray(out.Blat), np.asarray(out.basis_B)


def test_identity_passthrough():
    Blat, basis_B = _run("identity")
    np.testing.assert_allclose(Blat, ALAT)
    np.testing.assert_allclose(basis_B, BASIS, atol=1e-12)


def test_rotation_matches_reference(reference_module):
    spec = {"type": "rotation", "theta_deg": 30.0}
    ref_Blat = reference_module["build_transformed_lattice"](ALAT, spec)
    ref_basis = reference_module["build_transformed_basis"](BASIS, ALAT, ref_Blat)
    Blat, basis_B = _run("rotation", theta_deg=30.0)
    np.testing.assert_allclose(Blat, ref_Blat, atol=1e-12)
    np.testing.assert_allclose(basis_B, ref_basis, atol=1e-12)


def test_isotropic_scale_matches_reference(reference_module):
    spec = {"type": "isotropic_scale", "scale_factor": 1.04}
    ref_Blat = reference_module["build_transformed_lattice"](ALAT, spec)
    ref_basis = reference_module["build_transformed_basis"](BASIS, ALAT, ref_Blat)
    Blat, basis_B = _run("isotropic_scale", scale=1.04)
    np.testing.assert_allclose(Blat, ref_Blat, atol=1e-12)
    np.testing.assert_allclose(basis_B, ref_basis, atol=1e-12)


def test_anisotropic_strain_matches_reference(reference_module):
    spec = {"type": "anisotropic_strain", "s1": -0.03, "s2": 0.02}
    ref_Blat = reference_module["build_transformed_lattice"](ALAT, spec)
    ref_basis = reference_module["build_transformed_basis"](BASIS, ALAT, ref_Blat)
    Blat, basis_B = _run("anisotropic_strain", s1=-0.03, s2=0.02)
    np.testing.assert_allclose(Blat, ref_Blat, atol=1e-12)
    np.testing.assert_allclose(basis_B, ref_basis, atol=1e-12)


def test_strain_plus_rotation_matches_reference(reference_module):
    spec = {"type": "strain_plus_rotation", "s1": -0.03, "s2": 0.02, "theta_deg": 15.0}
    ref_Blat = reference_module["build_transformed_lattice"](ALAT, spec)
    ref_basis = reference_module["build_transformed_basis"](BASIS, ALAT, ref_Blat)
    Blat, basis_B = _run("strain_plus_rotation", s1=-0.03, s2=0.02, theta_deg=15.0)
    np.testing.assert_allclose(Blat, ref_Blat, atol=1e-12)
    np.testing.assert_allclose(basis_B, ref_basis, atol=1e-12)


@settings(max_examples=30, deadline=None)
@given(
    s1=st.floats(min_value=-0.1, max_value=0.1),
    s2=st.floats(min_value=-0.1, max_value=0.1),
    theta=st.floats(min_value=-45.0, max_value=45.0),
)
def test_strain_plus_rotation_fuzz(reference_module, s1, s2, theta):
    spec = {"type": "strain_plus_rotation", "s1": s1, "s2": s2, "theta_deg": theta}
    ref_Blat = reference_module["build_transformed_lattice"](ALAT, spec)
    ref_basis = reference_module["build_transformed_basis"](BASIS, ALAT, ref_Blat)
    Blat, basis_B = _run("strain_plus_rotation", s1=s1, s2=s2, theta_deg=theta)
    np.testing.assert_allclose(Blat, ref_Blat, atol=1e-10)
    np.testing.assert_allclose(basis_B, ref_basis, atol=1e-10)


def test_unknown_transform_type_rejected_by_schema():
    with pytest.raises(Exception):
        LatticeTransformerParams(transform_type="nonsense")  # type: ignore[arg-type]
