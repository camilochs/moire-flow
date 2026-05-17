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

from moire_flow.constants.potentials import (
    GAP_REGISTRY,
    MACE_REGISTRY,
    SW_REGISTRY,
    TERSOFF_REGISTRY,
    UFF_LJ,
    atomic_number,
)
from moire_flow.core.types import MDStructure, PotentialPlan

from .base import Box, register_box

IntralayerKind = Literal["auto", "tersoff", "sw", "gap", "mace", "lj"]
MaceFlavor = Literal["mace", "mliap"]


class PotentialAssignerInputs(BaseModel):
    md_structure: MDStructure
    species_A: list[str]
    species_B: list[str]


class PotentialAssignerParams(BaseModel):
    intralayer_kind: IntralayerKind = "auto"
    lj_cutoff: float = Field(default=8.0, gt=0.0)
    epsilon_scale: float = Field(default=1.0, gt=0.0)
    # MLIP-specific
    # `mliap` is the default because it works out of the box with stable
    # upstream LAMMPS (ML-IAP + mliappy bridge). `mace` is only valid when
    # the third-party MACE-LAMMPS plugin (ACEsuit/mace_lammps_plugin) is
    # compiled into the binary — currently not part of moire-flow-runtime:full.
    mace_flavor: MaceFlavor = "mliap"
    quip_init: str = "Potential xml_label=GAP"
    # Auto-mode priority. The original notebook (resolve_intralayer_potential,
    # ref 3908) used ("tersoff", "sw", "gap", "mace", "original"). We default
    # to the same order so the foundation-model paths are reachable.
    auto_priority: list[IntralayerKind] = Field(
        default_factory=lambda: ["tersoff", "sw", "gap", "mace", "lj"]
    )


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


def _plan_for_kind(
    kind: IntralayerKind, elements: list[str], params: PotentialAssignerParams
) -> dict | None:
    """Return a plan dict for `kind` if the registry has an entry, else None.

    `kind` must be one of the concrete kinds (no "auto" here).
    """
    key = frozenset(elements)
    if kind == "tersoff" and key in TERSOFF_REGISTRY:
        return {
            "kind": "tersoff",
            "pair_style": "tersoff",
            "file": TERSOFF_REGISTRY[key],
            "elements": elements,
        }
    if kind == "sw" and key in SW_REGISTRY:
        return {
            "kind": "sw",
            "pair_style": "sw",
            "file": SW_REGISTRY[key],
            "elements": elements,
        }
    if kind == "gap" and key in GAP_REGISTRY:
        # GAP/QUIP requires atomic numbers in the pair_coeff line
        return {
            "kind": "gap",
            "pair_style": "quip",
            "file": GAP_REGISTRY[key],
            "elements": elements,
            "atomic_numbers": [atomic_number(e) for e in elements],
            "quip_init": params.quip_init,
        }
    if kind == "mace" and key in MACE_REGISTRY:
        entry = MACE_REGISTRY[key]
        flavor = params.mace_flavor
        if flavor not in entry:
            return None
        return {
            "kind": "mace",
            "pair_style": flavor,  # "mace" or "mliap"
            "flavor": flavor,
            "file": entry[flavor],
            "elements": elements,
        }
    if kind == "lj":
        # Always available as a fallback (UFF table covers ~22 elements)
        try:
            return _uff_lj_pairs(elements, params.lj_cutoff, params.epsilon_scale)
        except KeyError:
            return None
    return None


def _intralayer_for(
    species: list[str], params: PotentialAssignerParams
) -> dict:
    """Pick intralayer plan for one layer."""
    elements = sorted(set(species))
    if params.intralayer_kind == "auto":
        for k in params.auto_priority:
            if k == "auto":
                continue
            plan = _plan_for_kind(k, elements, params)
            if plan is not None:
                return plan
        raise ValueError(
            f"auto: no potential available for {elements} "
            f"(tried {params.auto_priority})"
        )
    plan = _plan_for_kind(params.intralayer_kind, elements, params)
    if plan is None:
        raise ValueError(
            f"intralayer_kind={params.intralayer_kind!r} requested but no "
            f"parameters known for elements {elements}"
        )
    return plan


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
