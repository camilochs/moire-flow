"""GAP/QUIP and MACE script writers + PotentialAssigner kinds."""

from __future__ import annotations

import numpy as np
import pytest

from moire_flow.boxes import (
    LammpsInputWriter,
    LammpsInputWriterInputs,
    LammpsInputWriterParams,
    PotentialAssigner,
    PotentialAssignerInputs,
    PotentialAssignerParams,
)
from moire_flow.core.types import MDStructure
from moire_flow.io.lammps_writer import format_gap_quip_script, format_mace_script


def _md_mos2() -> MDStructure:
    return MDStructure(
        atoms=np.array(
            [[0.0, 0.0, 0.0], [1.58, 1.58, 0.0],
             [0.0, 0.0, 3.3], [1.58, 1.58, 3.3]]
        ),
        species=["Mo", "S", "Mo", "S"],
        cell=np.diag([3.16, 3.16, 20.0]),
        vacuum_z=15.0,
        interlayer_z=3.3,
    )


# ---------- format_gap_quip_script ----------

def test_gap_script_uses_pair_style_quip():
    t2e = {1: "Mo", 2: "S"}
    text = format_gap_quip_script(
        "layer.data", "MoS.gap.xml", t2e,
        quip_init="Potential xml_label=GAP_2024",
    )
    assert "pair_style      quip" in text
    assert "MoS.gap.xml" in text
    assert "Potential xml_label=GAP_2024" in text
    # Atomic numbers Mo=42, S=16 must follow the quip_init string
    assert "42" in text and "16" in text


def test_gap_script_orders_atomic_numbers_by_type_id():
    t2e = {1: "S", 2: "Mo"}  # reversed
    text = format_gap_quip_script("d.data", "g.xml", t2e)
    # The pair_coeff line should list "16 42" in that order
    coeff_line = [ln for ln in text.splitlines() if "pair_coeff" in ln][0]
    assert coeff_line.index("16") < coeff_line.index("42")


# ---------- format_mace_script ----------

def test_mace_script_native_flavor():
    text = format_mace_script(
        "layer.data", "MoS2.lammps.pt", {1: "Mo", 2: "S"}, flavor="mace"
    )
    assert "pair_style      mace" in text
    assert "MoS2.lammps.pt" in text
    assert "Mo S" in text


def test_mace_script_mliap_flavor():
    text = format_mace_script(
        "layer.data", "MoS2.mliap.pt", {1: "Mo", 2: "S"}, flavor="mliap"
    )
    assert "pair_style      mliap" in text
    assert "mliappy" in text


def test_mace_script_rejects_unknown_flavor():
    with pytest.raises(ValueError, match="flavor"):
        format_mace_script("d", "f", {1: "Mo"}, flavor="nonsense")


# ---------- PotentialAssigner kinds ----------

def test_assigner_gap_for_mos2():
    out = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=_md_mos2(), species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(intralayer_kind="gap"),
    )
    assert out.intralayer_A["kind"] == "gap"
    assert out.intralayer_A["pair_style"] == "quip"
    assert out.intralayer_A["atomic_numbers"] == [42, 16]  # Mo, S sorted
    assert out.intralayer_A["file"].endswith(".xml")


def test_assigner_mace_for_mos2():
    """Default flavor is 'mliap' (works with upstream LAMMPS)."""
    out = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=_md_mos2(), species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(intralayer_kind="mace"),
    )
    assert out.intralayer_A["kind"] == "mace"
    assert out.intralayer_A["pair_style"] == "mliap"
    assert out.intralayer_A["file"].endswith(".pt")


def test_assigner_mace_native_flavor():
    """Opt-in 'mace' flavor for the ACEsuit/mace_lammps_plugin path."""
    out = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=_md_mos2(), species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(intralayer_kind="mace", mace_flavor="mace"),
    )
    assert out.intralayer_A["pair_style"] == "mace"
    assert out.intralayer_A["flavor"] == "mace"


def test_assigner_auto_priority_picks_first_match():
    """default priority is tersoff > sw > gap > mace > lj — MoS2 hits tersoff."""
    out = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=_md_mos2(), species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(),  # auto
    )
    assert out.intralayer_A["kind"] == "tersoff"


def test_assigner_auto_can_be_steered_to_mace():
    out = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=_md_mos2(), species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(auto_priority=["mace", "tersoff", "lj"]),
    )
    assert out.intralayer_A["kind"] == "mace"


def test_assigner_gap_for_unknown_pair_falls_back_to_lj_in_auto():
    out = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=MDStructure(
                atoms=np.zeros((1, 3)), species=["C"], cell=np.eye(3),
                vacuum_z=15.0,
            ),
            species_A=["C"], species_B=["C"],
        ),
        PotentialAssignerParams(),
    )
    assert out.intralayer_A["kind"] == "lj"


def test_assigner_explicit_gap_unknown_raises():
    with pytest.raises(ValueError, match="gap"):
        PotentialAssigner().run(
            PotentialAssignerInputs(
                md_structure=MDStructure(
                    atoms=np.zeros((1, 3)), species=["C"], cell=np.eye(3),
                    vacuum_z=15.0,
                ),
                species_A=["C"], species_B=["C"],
            ),
            PotentialAssignerParams(intralayer_kind="gap"),
        )


# ---------- End-to-end: writer dispatches new kinds ----------

def test_input_writer_routes_to_gap_script(tmp_path):
    md = _md_mos2()
    plan = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=md, species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(intralayer_kind="gap"),
    )
    out = LammpsInputWriter().run(
        LammpsInputWriterInputs(
            md_structure=md, potential_plan=plan,
            species_A=["Mo", "S"], species_B=["Mo", "S"],
            n_atoms_A=2,
        ),
        LammpsInputWriterParams(output_dir=tmp_path / "run"),
    )
    text = out.layer_A_script.read_text()
    assert "pair_style      quip" in text
    assert "MoS2.gap.xml" in text


def test_input_writer_routes_to_mace_script(tmp_path):
    md = _md_mos2()
    plan = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=md, species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(intralayer_kind="mace"),  # default flavor=mliap
    )
    out = LammpsInputWriter().run(
        LammpsInputWriterInputs(
            md_structure=md, potential_plan=plan,
            species_A=["Mo", "S"], species_B=["Mo", "S"],
            n_atoms_A=2,
        ),
        LammpsInputWriterParams(output_dir=tmp_path / "run"),
    )
    text = out.layer_A_script.read_text()
    # Default flavor maps to upstream-LAMMPS-compatible mliap+mliappy
    assert "pair_style      mliap" in text
    assert "mliappy" in text
    assert ".pt" in text
