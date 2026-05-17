"""PotentialAssigner: choose pair styles + parameters for a bilayer.

Pure logic, no file I/O. Decides:
- `intralayer_A`, `intralayer_B`: tersoff / sw / lj/cut, by element set
- `interlayer`: always lj/cut with UFF Lorentz-Berthelot mixing

The LammpsInputWriter (M6.4) consumes this plan and writes scripts.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from moire_flow.constants.potentials import SW_REGISTRY, TERSOFF_REGISTRY, UFF_LJ
from moire_flow.core.types import MDStructure, PotentialPlan

from .base import Box, register_box

IntralayerKind = Literal["auto", "tersoff", "sw", "lj"]


class PotentialAssignerInputs(BaseModel):
    md_structure: MDStructure
    species_A: list[str]
    species_B: list[str]


class PotentialAssignerParams(BaseModel):
    intralayer_kind: IntralayerKind = "auto"
    lj_cutoff: float = Field(default=8.0, gt=0.0)
    epsilon_scale: float = Field(default=1.0, gt=0.0)


def _uff_lj_pairs(elements: list[str], cutoff: float, eps_scale: float) -> dict:
    """Build UFF Lorentz-Berthelot mixed LJ parameters for `elements`."""
    missing = [e for e in elements if e not in UFF_LJ]
    if missing:
        raise KeyError(f"UFF LJ parameters missing for elements: {missing}")
    pairs: dict[str, dict[str, float]] = {}
    for i, a in enumerate(elements):
        for b in elements[i:]:
            sigma = 0.5 * (UFF_LJ[a]["sigma"] + UFF_LJ[b]["sigma"])
            epsilon = math.sqrt(UFF_LJ[a]["epsilon"] * UFF_LJ[b]["epsilon"]) * eps_scale
            pairs[f"{a}-{b}"] = {
                "sigma": float(sigma),
                "epsilon": float(epsilon),
                "cutoff": float(cutoff),
            }
    return {
        "kind": "lj",
        "pair_style": "lj/cut",
        "elements": elements,
        "pairs": pairs,
        "mixing": "Lorentz-Berthelot",
        "cutoff": float(cutoff),
        "source": "UFF",
    }


def _classical_for_elements(elements: list[str], requested: IntralayerKind) -> dict | None:
    """Return a {kind, pair_style, file, elements} dict if a classical file is known."""
    key = frozenset(elements)
    if requested in ("auto", "tersoff") and key in TERSOFF_REGISTRY:
        return {
            "kind": "tersoff",
            "pair_style": "tersoff",
            "file": TERSOFF_REGISTRY[key],
            "elements": elements,
        }
    if requested in ("auto", "sw") and key in SW_REGISTRY:
        return {
            "kind": "sw",
            "pair_style": "sw",
            "file": SW_REGISTRY[key],
            "elements": elements,
        }
    return None


def _intralayer_for(
    species: list[str], params: PotentialAssignerParams
) -> dict:
    """Pick intralayer plan for one layer."""
    elements = sorted(set(species))
    if params.intralayer_kind == "lj":
        return _uff_lj_pairs(elements, params.lj_cutoff, params.epsilon_scale)
    classical = _classical_for_elements(elements, params.intralayer_kind)
    if classical is not None:
        return classical
    # auto + no classical hit → fall back to lj
    if params.intralayer_kind == "auto":
        return _uff_lj_pairs(elements, params.lj_cutoff, params.epsilon_scale)
    raise ValueError(
        f"intralayer_kind={params.intralayer_kind} requested but no parameters known "
        f"for elements {elements}"
    )


@register_box
class PotentialAssigner(
    Box[PotentialAssignerInputs, PotentialAssignerParams, PotentialPlan]
):
    name = "potential_assigner"
    description = "Assign intralayer (tersoff/sw/lj) and interlayer (LJ) pair styles."
    inputs_schema = PotentialAssignerInputs
    params_schema = PotentialAssignerParams
    outputs_schema = PotentialPlan

    def run(
        self,
        inputs: PotentialAssignerInputs,
        params: PotentialAssignerParams,
    ) -> PotentialPlan:
        intra_A = _intralayer_for(inputs.species_A, params)
        intra_B = _intralayer_for(inputs.species_B, params)
        # Interlayer LJ over the *union* of elements across both layers
        all_elements = sorted(set(inputs.species_A) | set(inputs.species_B))
        inter = _uff_lj_pairs(all_elements, params.lj_cutoff, params.epsilon_scale)
        inter["kind"] = "lj_interlayer"
        return PotentialPlan(intralayer_A=intra_A, intralayer_B=intra_B, interlayer=inter)


__all__ = [
    "PotentialAssigner",
    "PotentialAssignerInputs",
    "PotentialAssignerParams",
    "IntralayerKind",
]
