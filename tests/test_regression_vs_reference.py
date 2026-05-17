"""Direct regression: every ported pure helper must match the original on
the same inputs. Inputs are drawn from a fixed RNG so failures reproduce."""

from __future__ import annotations

import numpy as np
import pytest

# ---- ours ----
from moire_flow.core.algebra2d import (
    R as my_R,
    S as my_S,
    angle_error_symmetry_deg as my_angle_error_symmetry_deg,
    atoms_in_lattice_patch as my_atoms_in_lattice_patch,
    canonical_theta_deg as my_canonical_theta_deg,
    change_basis as my_change_basis,
    fractional_positions as my_fractional_positions,
    transform_basis as my_transform_basis,
    vec2d_angle_deg as my_vec2d_angle_deg,
    vec2d_cross as my_vec2d_cross,
    vec2d_norm as my_vec2d_norm,
    wrap_basis_to_cell as my_wrap_basis_to_cell,
    wrap_fractional as my_wrap_fractional,
)
from moire_flow.core.matching import (
    atoms_in_supercell as my_atoms_in_supercell,
    build_a_lattice_candidates as my_build_a_lattice_candidates,
    compute_mismatch as my_compute_mismatch,
    cost_funcion as my_cost_funcion,
    find_scvect_pairs as my_find_scvect_pairs,
    get_lavects as my_get_lavects,
    param_grid_from_bounds as my_param_grid_from_bounds,
    supercell_points as my_supercell_points,
)
from moire_flow.core.validation_metrics import (
    coordination_match_score as my_coordination_match_score,
    distance_signature_error as my_distance_signature_error,
    fractional_distance_signature as my_fractional_distance_signature,
    fractional_rmsd as my_fractional_rmsd,
    lattice_metrics as my_lattice_metrics,
    local_distance_signature as my_local_distance_signature,
    neighbor_counts as my_neighbor_counts,
    pairwise_distance_signature as my_pairwise_distance_signature,
    periodic_fractional_distances as my_periodic_fractional_distances,
)
from moire_flow.io._mass import guess_element_from_mass as my_guess_element_from_mass
from moire_flow.io.lammps_writer import box_from_sc_vecs as my_box_from_sc_vecs


# ---------- shared inputs ----------

ALAT = np.array([[3.16, 0.0], [1.58, 2.737]])
BASIS = np.array([[0.0, 0.0], [2.106667, 1.824667]])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------- algebra2d ----------

def test_R_matches_reference(reference_helpers):
    for theta in [0.0, 0.5, 1.5707963, -1.0, 2.5]:
        np.testing.assert_allclose(my_R(theta), reference_helpers["R"](theta), atol=1e-15)


def test_S_matches_reference(reference_helpers):
    for s1, s2 in [(0, 0), (0.01, -0.02), (-0.05, 0.04)]:
        np.testing.assert_allclose(my_S(s1, s2), reference_helpers["S"](s1, s2), atol=1e-15)


def test_change_basis_matches_reference(reference_helpers, rng):
    A = ALAT
    B = my_R(0.4).dot(A)
    r = rng.uniform(-2, 2, size=(8, 2))
    ours = my_change_basis(r, A, B)
    theirs = reference_helpers["ChangeBasis"](r, A, B)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_vec2d_helpers_match(reference_helpers, rng):
    v1 = rng.uniform(-3, 3, size=2)
    v2 = rng.uniform(-3, 3, size=2)
    assert my_vec2d_dot_eq(v1, v2, reference_helpers)
    assert abs(my_vec2d_cross(v1, v2) - reference_helpers["vec2d_cross"](v1, v2)) < 1e-12
    assert abs(my_vec2d_norm(v1) - reference_helpers["vec2d_norm"](v1)) < 1e-12
    assert abs(my_vec2d_angle_deg(v1, v2) - reference_helpers["vec2d_angle_deg"](v1, v2)) < 1e-10


def my_vec2d_dot_eq(v1, v2, ref):
    from moire_flow.core.algebra2d import vec2d_dot
    return abs(vec2d_dot(v1, v2) - ref["vec2d_dot"](v1, v2)) < 1e-12


def test_wrap_basis_to_cell_matches(reference_helpers, rng):
    basis = rng.uniform(-5, 5, size=(6, 2))
    np.testing.assert_allclose(
        my_wrap_basis_to_cell(basis, ALAT),
        reference_helpers["wrap_basis_to_cell"](basis, ALAT),
        atol=1e-12,
    )


def test_transform_basis_matches(reference_helpers, rng):
    new_lat = my_R(0.3).dot(ALAT)
    basis = rng.uniform(-2, 2, size=(5, 2))
    np.testing.assert_allclose(
        my_transform_basis(basis, ALAT, new_lat),
        reference_helpers["transform_basis"](basis, ALAT, new_lat),
        atol=1e-12,
    )


def test_atoms_in_lattice_patch_matches(reference_helpers):
    np.testing.assert_allclose(
        my_atoms_in_lattice_patch(ALAT, BASIS, reps=3),
        reference_helpers["atoms_in_lattice_patch"](ALAT, BASIS, reps=3),
        atol=1e-12,
    )


def test_wrap_fractional_matches(reference_helpers, rng):
    frac = rng.uniform(-3, 3, size=(10, 2))
    np.testing.assert_allclose(
        my_wrap_fractional(frac.copy()),
        reference_helpers["wrap_fractional"](frac.copy()),
        atol=1e-12,
    )


def test_fractional_positions_matches(reference_helpers, rng):
    pts = rng.uniform(-5, 5, size=(8, 2))
    np.testing.assert_allclose(
        my_fractional_positions(pts, ALAT),
        reference_helpers["fractional_positions"](pts, ALAT),
        atol=1e-12,
    )


def test_canonical_theta_deg_matches(reference_helpers):
    for theta in [0.0, 30.0, 31.0, 59.0, 60.5, 90.0, -45.0]:
        assert (
            abs(my_canonical_theta_deg(theta, period_deg=60.0)
                - reference_helpers["canonical_theta_deg"](theta, period_deg=60.0))
            < 1e-10
        )


def test_angle_error_symmetry_deg_matches(reference_helpers):
    cases = [(15.0, 15.0), (45.0, 15.0), (-15.0, 15.0), (75.0, 15.0), (0.0, None)]
    for rec, tgt in cases:
        a = my_angle_error_symmetry_deg(rec, tgt)
        b = reference_helpers["angle_error_symmetry_deg"](rec, tgt)
        if tgt is None:
            assert np.isnan(a) and np.isnan(b)
        else:
            assert abs(a - b) < 1e-10


# ---------- matching primitives ----------

def test_supercell_points_matches(reference_helpers):
    np.testing.assert_array_equal(
        my_supercell_points((4, 5), ALAT),
        reference_helpers["supercell_points"]((4, 5), ALAT),
    )


def test_cost_funcion_matches_on_random_inputs(reference_helpers, rng):
    r = rng.uniform(-2, 2, size=(20, 2))
    assert abs(my_cost_funcion(r) - reference_helpers["cost_funcion"](r)) < 1e-14


def test_get_lavects_matches(reference_helpers, rng):
    x = rng.uniform(-3, 3, size=(20, 2))
    lat = rng.uniform(-3, 3, size=(20, 2))
    np.testing.assert_array_equal(
        my_get_lavects(x, lat, tol=0.2),
        reference_helpers["get_lavects"](x, lat, tol=0.2),
    )


def test_find_scvect_pairs_matches(reference_helpers):
    L = my_build_a_lattice_candidates(ALAT, N=3)
    ours = my_find_scvect_pairs(L)
    theirs = reference_helpers["find_scvect_pairs"](L)
    assert len(ours) == len(theirs)
    for (a, ang, v1, v2), (a2, ang2, v1b, v2b) in zip(ours, theirs):
        assert abs(a - a2) < 1e-10
        assert abs(ang - ang2) < 1e-10
        np.testing.assert_allclose(v1, v1b)
        np.testing.assert_allclose(v2, v2b)


def test_compute_mismatch_matches(reference_helpers):
    sc = 2 * ALAT
    Blat = my_R(0.2).dot(ALAT)
    assert abs(my_compute_mismatch(sc, ALAT, Blat)
               - reference_helpers["compute_mismatch"](sc, ALAT, Blat)) < 1e-12


def test_build_a_lattice_candidates_matches(reference_helpers):
    np.testing.assert_array_equal(
        my_build_a_lattice_candidates(ALAT, 4),
        reference_helpers["_build_a_lattice_candidates"](ALAT, 4),
    )


def test_param_grid_from_bounds_matches(reference_helpers):
    bounds = [(-0.05, 0.05), (-0.04, 0.04), (-1.5, 1.5)]
    ours = my_param_grid_from_bounds(bounds, ndiv=7)
    theirs = reference_helpers["param_grid_from_bounds"](bounds, ndiv=7)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b)


def test_atoms_in_supercell_matches(reference_helpers):
    """Critical: AtomAssembler relies on this."""
    sc_vecs = np.array([2.0 * ALAT[0], 2.0 * ALAT[1]])
    ours = my_atoms_in_supercell(sc_vecs, ALAT, BASIS)
    theirs = reference_helpers["atoms_in_supercell"](sc_vecs, ALAT, BASIS)
    # Order may differ — compare as sets of tuples
    ours_set = {tuple(np.round(p, 8)) for p in ours}
    theirs_set = {tuple(np.round(p, 8)) for p in theirs}
    assert ours_set == theirs_set


# ---------- synth ----------

def test_build_transformed_lattice_all_modes(reference_helpers):
    cases = [
        {"type": "identity"},
        {"type": "rotation", "theta_deg": 25.0},
        {"type": "isotropic_scale", "scale_factor": 1.07},
        {"type": "anisotropic_strain", "s1": -0.02, "s2": 0.03},
        {"type": "strain_plus_rotation", "s1": -0.02, "s2": 0.03, "theta_deg": 12.0},
    ]
    for spec in cases:
        from moire_flow.boxes.lattice_transformer import _apply_transform, LatticeTransformerParams
        # ours uses Params; replicate the mapping
        kind = spec["type"]
        params_kwargs = {"transform_type": kind}
        if kind == "rotation":
            params_kwargs["theta_deg"] = spec["theta_deg"]
        elif kind == "isotropic_scale":
            params_kwargs["scale"] = spec["scale_factor"]
        elif kind == "anisotropic_strain":
            params_kwargs.update({"s1": spec["s1"], "s2": spec["s2"]})
        elif kind == "strain_plus_rotation":
            params_kwargs.update({
                "s1": spec["s1"], "s2": spec["s2"], "theta_deg": spec["theta_deg"],
            })
        ours = _apply_transform(ALAT, LatticeTransformerParams(**params_kwargs))
        theirs = reference_helpers["build_transformed_lattice"](ALAT, spec)
        np.testing.assert_allclose(ours, theirs, atol=1e-12, err_msg=f"spec={spec}")


def test_build_transformed_basis_matches(reference_helpers):
    Blat = my_R(np.radians(15.0)).dot(ALAT)
    ours = my_transform_basis(BASIS, ALAT, Blat)
    theirs = reference_helpers["build_transformed_basis"](BASIS, ALAT, Blat)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


# ---------- validation_metrics ----------

def test_periodic_fractional_distances_matches(reference_helpers, rng):
    f1 = rng.uniform(0, 1, size=(7, 2))
    f2 = rng.uniform(0, 1, size=(5, 2))
    np.testing.assert_allclose(
        my_periodic_fractional_distances(f1, f2),
        reference_helpers["periodic_fractional_distances"](f1, f2),
        atol=1e-12,
    )


def test_fractional_distance_signature_matches(reference_helpers, rng):
    frac = rng.uniform(0, 1, size=(10, 2))
    np.testing.assert_allclose(
        my_fractional_distance_signature(frac),
        reference_helpers["fractional_distance_signature"](frac),
        atol=1e-12,
    )


def test_fractional_rmsd_matches(reference_helpers, rng):
    a = rng.uniform(0, 1, size=(10, 2))
    b = rng.uniform(0, 1, size=(10, 2))
    assert abs(my_fractional_rmsd(a, b) - reference_helpers["fractional_rmsd"](a, b)) < 1e-12


def test_pairwise_distance_signature_matches(reference_helpers, rng):
    pts = rng.uniform(-5, 5, size=(12, 2))
    np.testing.assert_allclose(
        my_pairwise_distance_signature(pts, k=6),
        reference_helpers["pairwise_distance_signature"](pts, k=6),
        atol=1e-12,
    )


def test_local_distance_signature_matches(reference_helpers, rng):
    pts = rng.uniform(-5, 5, size=(8, 2))
    support = rng.uniform(-5, 5, size=(20, 2))
    np.testing.assert_allclose(
        my_local_distance_signature(pts, support_points=support, k=5),
        reference_helpers["local_distance_signature"](pts, support_points=support, k=5),
        atol=1e-12,
    )


def test_distance_signature_error_matches(reference_helpers, rng):
    a = rng.uniform(-5, 5, size=(8, 2))
    b = rng.uniform(-5, 5, size=(8, 2))
    sa = rng.uniform(-5, 5, size=(15, 2))
    sb = rng.uniform(-5, 5, size=(15, 2))
    assert abs(
        my_distance_signature_error(a, b, k=6, support_points_ref=sa, support_points_test=sb)
        - reference_helpers["distance_signature_error"](a, b, k=6,
                                                       support_points_ref=sa,
                                                       support_points_test=sb)
    ) < 1e-12


def test_neighbor_counts_matches(reference_helpers, rng):
    pts = rng.uniform(-3, 3, size=(10, 2))
    support = rng.uniform(-3, 3, size=(20, 2))
    np.testing.assert_array_equal(
        my_neighbor_counts(pts, cutoff=2.5, support_points=support),
        reference_helpers["neighbor_counts"](pts, cutoff=2.5, support_points=support),
    )


def test_coordination_match_score_matches(reference_helpers, rng):
    a = rng.uniform(-3, 3, size=(10, 2))
    b = rng.uniform(-3, 3, size=(10, 2))
    assert abs(my_coordination_match_score(a, b, cutoff=2.5)
               - reference_helpers["coordination_match_score"](a, b, cutoff=2.5)) < 1e-12


def test_lattice_metrics_matches(reference_helpers):
    ours = my_lattice_metrics(ALAT)
    theirs = reference_helpers["lattice_metrics"](ALAT)
    for k in ours:
        assert abs(ours[k] - theirs[k]) < 1e-12


# ---------- guess_element_from_mass ----------

def test_guess_element_from_mass_matches_on_common_elements(reference_helpers):
    """The original lookup covers 25 elements; ours is wider (ase.data).
    On the 25-element subset the verdict should agree on close masses."""
    common = [("H", 1.008), ("C", 12.011), ("O", 15.999), ("Mo", 95.96),
              ("S", 32.06), ("W", 183.84), ("Se", 78.96), ("Hf", 178.49)]
    for symbol, mass in common:
        ours = my_guess_element_from_mass(mass)
        theirs = reference_helpers["guess_element_from_mass"](mass)
        assert ours == symbol, f"ours: {ours} != {symbol}"
        assert theirs == symbol, f"reference: {theirs} != {symbol}"


def test_guess_element_from_mass_handles_unknown_via_X_prefix(reference_helpers):
    # An exotic mass with no nearby element in either table
    assert my_guess_element_from_mass(1234.5).startswith("X")
    assert reference_helpers["guess_element_from_mass"](1234.5).startswith("X")


# ---------- LAMMPS writer geometry ----------

def test_box_from_sc_vecs_matches_reference(reference_helpers):
    """Our box_from_sc_vecs(sc, vacuum, slab) must produce the same
    (box, rot, area) tuple as the reference `_box`."""
    sc = np.array([[10.0, 0.0], [3.0, 8.5]])
    vacuum_z = 15.0
    slab_h = 3.5
    box_ours, rot_ours, area_ours = my_box_from_sc_vecs(sc, vacuum_z, slab_h)
    box_theirs, rot_theirs, area_theirs = reference_helpers["_box"](sc, vacuum_z, slab_h)
    np.testing.assert_allclose(box_ours, box_theirs, atol=1e-12)
    np.testing.assert_allclose(rot_ours, rot_theirs, atol=1e-12)
    assert abs(area_ours - area_theirs) < 1e-12


# ---------- AtomAssembler (build_atoms_for_best) ----------

def test_atom_assembler_matches_build_atoms_for_best(reference_helpers):
    """End-to-end equivalence of AtomAssembler.run vs reference build_atoms_for_best
    when supplied with the same matched solution."""
    from moire_flow.boxes import (
        AtomAssembler, AtomAssemblerInputs, AtomAssemblerParams,
    )
    from moire_flow.core.types import MatchingSolution

    sc_v1 = 2.0 * ALAT[0]
    sc_v2 = 2.0 * ALAT[1]
    oBlat = ALAT.copy()

    best_solution = {
        "v1": sc_v1, "v2": sc_v2,
        "s1": 0.0, "s2": 0.0, "theta_deg": 0.0,
        "cost": 0.0, "mismatch": 0.0, "area": 4 * abs(ALAT[0, 0] * ALAT[1, 1] - ALAT[0, 1] * ALAT[1, 0]),
        "oBlat": oBlat,
    }
    ref_aA, ref_aB, ref_sc_vecs, ref_oBlat = reference_helpers["build_atoms_for_best"](
        best_solution, ALAT, ALAT, BASIS, BASIS
    )
    sol = MatchingSolution(
        v1=sc_v1, v2=sc_v2, s1=0.0, s2=0.0, theta_deg=0.0,
        cost=0.0, mismatch=0.0, area=best_solution["area"], oBlat=oBlat,
    )
    out = AtomAssembler().run(
        AtomAssemblerInputs(
            solution=sol, basis_A=BASIS, basis_B=BASIS,
            Alat=ALAT, Blat=ALAT,
            species_A=["Mo", "S"], species_B=["Mo", "S"],
        ),
        AtomAssemblerParams(),
    )
    # Compare as sets
    ours_A = {tuple(np.round(p, 8)) for p in out.atoms_A}
    ours_B = {tuple(np.round(p, 8)) for p in out.atoms_B}
    theirs_A = {tuple(np.round(p, 8)) for p in ref_aA}
    theirs_B = {tuple(np.round(p, 8)) for p in ref_aB}
    assert ours_A == theirs_A
    assert ours_B == theirs_B
