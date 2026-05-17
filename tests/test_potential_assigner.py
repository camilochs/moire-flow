"""PotentialAssigner: classical-vs-LJ dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from moire_flow.boxes import (
    PotentialAssigner,
    PotentialAssignerInputs,
    PotentialAssignerParams,
)
from moire_flow.core.types import MDStructure


def _md(species: list[str]) -> MDStructure:
    return MDStructure(
        atoms=np.zeros((len(species), 3)),
        species=species,
        cell=np.eye(3),
        vacuum_z=15.0,
    )


def test_mos2_gets_tersoff_intralayer():
    md = _md(["Mo", "S", "Mo", "S"])
    out = PotentialAssigner().run(
        PotentialAssignerInputs(md_structure=md, species_A=["Mo", "S"], species_B=["Mo", "S"]),
        PotentialAssignerParams(),
    )
    assert out.intralayer_A["kind"] == "tersoff"
    assert out.intralayer_A["pair_style"] == "tersoff"
    assert "MoS.tersoff" in out.intralayer_A["file"]


def test_unknown_element_falls_back_to_lj():
    out = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=_md(["C", "C"]),
            species_A=["C"],
            species_B=["C"],
        ),
        PotentialAssignerParams(),
    )
    assert out.intralayer_A["kind"] == "lj"
    assert out.intralayer_B["kind"] == "lj"


def test_interlayer_is_always_lj_with_union_of_elements():
    out = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=_md(["Mo", "S", "W", "Se"]),
            species_A=["Mo", "S"],
            species_B=["W", "Se"],
        ),
        PotentialAssignerParams(),
    )
    assert out.interlayer["kind"] == "lj_interlayer"
    assert set(out.interlayer["elements"]) == {"Mo", "S", "W", "Se"}


def test_lj_pair_uses_lorentz_berthelot_mixing():
    out = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=_md(["Mo", "S"]),
            species_A=["Mo", "S"],
            species_B=["Mo", "S"],
        ),
        PotentialAssignerParams(intralayer_kind="lj"),
    )
    pairs = out.intralayer_A["pairs"]
    # sigma_MoS = (2.719 + 3.595) / 2
    assert pairs["Mo-S"]["sigma"] == pytest.approx(0.5 * (2.719 + 3.595))


def test_explicit_tersoff_with_no_template_raises():
    with pytest.raises(ValueError, match="tersoff"):
        PotentialAssigner().run(
            PotentialAssignerInputs(
                md_structure=_md(["C", "C"]),
                species_A=["C"],
                species_B=["C"],
            ),
            PotentialAssignerParams(intralayer_kind="tersoff"),
        )
