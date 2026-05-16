"""Validator: structural metrics for a matched bilayer.

Port of `structure_validation_metrics` + `validate_case_physics`
(reference 2176-2285). The hardcoded short-circuit on
`name='real_pair_hfse2_mose2'` (reference 2207) is replaced by the
configurable `skip_cases` param, decoupling the box from case identity.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from moire_flow.core.algebra2d import (
    R,
    S,
    angle_error_symmetry_deg,
    atoms_in_lattice_patch,
    fractional_positions,
    transform_basis,
)
from moire_flow.core.types import (
    MatchingSolution,
    NDArray2D,
    NDArray2x2,
    ValidationMetrics,
    ValidationTarget,
)
from moire_flow.core.validation_metrics import (
    coordination_match_score,
    distance_signature_error,
    fractional_rmsd,
    pairwise_distance_signature,
)

from .base import Box, register_box


class ValidatorInputs(BaseModel):
    """Inputs required to evaluate a matching solution.

    `Blat` and `basis_B` carry the pre-deformation lattice and basis of layer B
    — needed because the metrics recompute the "target" lattice from the
    declared target strain/rotation. They will be wired automatically by the
    WorkflowEngine (M8) from the upstream `LayerPair` and `LatticeTransformer`
    outputs.
    """

    Blat: NDArray2x2
    basis_B: NDArray2D
    solution: MatchingSolution
    target: ValidationTarget | None = None
    case_name: str | None = None


class ValidatorParams(BaseModel):
    cutoff_scale: float = Field(default=1.15, gt=0.0)
    k_dist: int = Field(default=6, ge=1)
    skip_cases: list[str] = Field(default_factory=list)
    theta_tol_deg: float = 2.0
    strain_tol: float = 0.03
    mismatch_tol: float = 0.02
    fractional_rmsd_tol: float = 1e-2
    distance_signature_tol: float = 1e-2
    coordination_match_min: float = 0.9


def _nan_metrics(mismatch: float, area: float) -> ValidationMetrics:
    return ValidationMetrics(
        theta_error_sym_deg=None,
        s1_error=None,
        s2_error=None,
        fractional_rmsd=float("nan"),
        distance_signature_error=float("nan"),
        coordination_match=float("nan"),
        passes={"skipped": True, "mismatch_ok": mismatch <= 0.05},
    )


@register_box
class Validator(Box[ValidatorInputs, ValidatorParams, ValidationMetrics]):
    name = "validator"
    description = "Compute structural metrics (RMSD, signatures, coordination) for a matched bilayer."
    inputs_schema = ValidatorInputs
    params_schema = ValidatorParams
    outputs_schema = ValidationMetrics

    def run(
        self, inputs: ValidatorInputs, params: ValidatorParams
    ) -> ValidationMetrics:
        sol = inputs.solution
        target = inputs.target

        if inputs.case_name and inputs.case_name in params.skip_cases:
            return _nan_metrics(sol.mismatch, sol.area)

        theta_err: float | None = None
        s1_err: float | None = None
        s2_err: float | None = None

        if target is not None:
            theta_err = (
                None
                if target.target_theta_deg is None
                else angle_error_symmetry_deg(
                    sol.theta_deg,
                    target.target_theta_deg,
                    period_deg=target.theta_period_deg,
                    allow_sign_flip=target.allow_sign_flip,
                )
            )
            if target.target_s1 is not None:
                s1_err = abs(sol.s1 - target.target_s1)
            if target.target_s2 is not None:
                s2_err = abs(sol.s2 - target.target_s2)

        Blat = np.asarray(inputs.Blat, dtype=np.float64)
        basis_B = np.asarray(inputs.basis_B, dtype=np.float64)
        oBlat = np.asarray(sol.oBlat, dtype=np.float64)

        target_theta = 0.0 if target is None or target.target_theta_deg is None else float(target.target_theta_deg)
        target_s1 = 0.0 if target is None or target.target_s1 is None else float(target.target_s1)
        target_s2 = 0.0 if target is None or target.target_s2 is None else float(target.target_s2)

        target_oBlat = S(target_s1, target_s2).dot(Blat.dot(R(np.radians(target_theta)).T))
        basis_B_target = transform_basis(basis_B, Blat, target_oBlat)
        basis_B_recovered = transform_basis(basis_B, Blat, oBlat)

        frac_target = fractional_positions(basis_B_target, target_oBlat)
        frac_recovered = fractional_positions(basis_B_recovered, oBlat)
        frmsd = fractional_rmsd(frac_target, frac_recovered)

        target_patch = atoms_in_lattice_patch(target_oBlat, basis_B_target, reps=2)
        recovered_patch = atoms_in_lattice_patch(oBlat, basis_B_recovered, reps=2)
        dse = distance_signature_error(
            basis_B_target,
            basis_B_recovered,
            k=max(params.k_dist, 12),
            support_points_ref=target_patch,
            support_points_test=recovered_patch,
        )

        sig = pairwise_distance_signature(target_patch, k=1)
        if len(sig) > 0:
            cutoff = params.cutoff_scale * float(sig[0])
            coord = coordination_match_score(target_patch, recovered_patch, cutoff)
        else:
            coord = float("nan")

        passes: dict[str, bool] = {
            "mismatch_ok": sol.mismatch <= params.mismatch_tol,
            "fractional_rmsd_ok": (not np.isnan(frmsd)) and frmsd <= params.fractional_rmsd_tol,
            "distance_signature_ok": (not np.isnan(dse)) and dse <= params.distance_signature_tol,
            "coordination_ok": (not np.isnan(coord)) and coord >= params.coordination_match_min,
        }
        if theta_err is not None and not np.isnan(theta_err):
            passes["theta_ok"] = theta_err <= params.theta_tol_deg
        if s1_err is not None:
            passes["s1_ok"] = s1_err <= params.strain_tol
        if s2_err is not None:
            passes["s2_ok"] = s2_err <= params.strain_tol

        return ValidationMetrics(
            theta_error_sym_deg=theta_err,
            s1_error=s1_err,
            s2_error=s2_err,
            fractional_rmsd=float(frmsd),
            distance_signature_error=float(dse),
            coordination_match=float(coord),
            passes=passes,
        )


__all__ = ["Validator", "ValidatorInputs", "ValidatorParams"]
