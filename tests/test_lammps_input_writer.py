"""LammpsInputWriter: writes 6 files; round-trippable via read_lammps_data."""

from __future__ import annotations

from pathlib import Path

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
from moire_flow.io import read_lammps_data


def _md_mos2() -> MDStructure:
    """Tiny MoS2 monolayer × 2 in z."""
    atoms = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.58, 1.58, 0.0],
            [0.0, 0.0, 3.3],
            [1.58, 1.58, 3.3],
        ]
    )
    cell = np.eye(3) * 1.0
    cell[0, 0] = 3.16
    cell[1, 1] = 3.16
    cell[2, 2] = 20.0
    return MDStructure(
        atoms=atoms,
        species=["Mo", "S", "Mo", "S"],
        cell=cell,
        vacuum_z=15.0,
        interlayer_z=3.3,
    )


def test_writer_creates_six_files(tmp_path: Path):
    md = _md_mos2()
    plan = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=md,
            species_A=["Mo", "S"],
            species_B=["Mo", "S"],
        ),
        PotentialAssignerParams(),
    )
    out = LammpsInputWriter().run(
        LammpsInputWriterInputs(
            md_structure=md,
            potential_plan=plan,
            species_A=["Mo", "S"],
            species_B=["Mo", "S"],
            n_atoms_A=2,
        ),
        LammpsInputWriterParams(output_dir=tmp_path / "run", nvt_steps=100),
    )
    for p in (
        out.layer_A_data, out.layer_B_data, out.bilayer_data,
        out.layer_A_script, out.layer_B_script, out.bilayer_script,
    ):
        assert p.exists()
        assert p.stat().st_size > 0


def test_layer_data_is_parseable(tmp_path: Path):
    md = _md_mos2()
    plan = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=md, species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(),
    )
    out = LammpsInputWriter().run(
        LammpsInputWriterInputs(
            md_structure=md,
            potential_plan=plan,
            species_A=["Mo", "S"],
            species_B=["Mo", "S"],
            n_atoms_A=2,
        ),
        LammpsInputWriterParams(output_dir=tmp_path / "run"),
    )
    # round-trip via our own reader (since we wrote with the same format)
    reread = read_lammps_data(out.layer_A_data)
    assert reread.atoms.shape == (2, 3)
    assert sorted(reread.species) == ["Mo", "S"]


def test_bilayer_data_has_all_four_atoms(tmp_path: Path):
    md = _md_mos2()
    plan = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=md, species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(),
    )
    out = LammpsInputWriter().run(
        LammpsInputWriterInputs(
            md_structure=md,
            potential_plan=plan,
            species_A=["Mo", "S"],
            species_B=["Mo", "S"],
            n_atoms_A=2,
        ),
        LammpsInputWriterParams(output_dir=tmp_path / "run"),
    )
    reread = read_lammps_data(out.bilayer_data)
    assert reread.atoms.shape == (4, 3)


def test_tersoff_script_references_correct_files(tmp_path: Path):
    md = _md_mos2()
    plan = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=md, species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(),
    )
    out = LammpsInputWriter().run(
        LammpsInputWriterInputs(
            md_structure=md,
            potential_plan=plan,
            species_A=["Mo", "S"],
            species_B=["Mo", "S"],
            n_atoms_A=2,
        ),
        LammpsInputWriterParams(output_dir=tmp_path / "run"),
    )
    text = out.layer_A_script.read_text()
    assert "pair_style      tersoff" in text
    assert "MoS.tersoff" in text
    assert "read_data       layer_A.data" in text


def test_bilayer_script_has_zero_intra_and_active_inter(tmp_path: Path):
    md = _md_mos2()
    plan = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=md, species_A=["Mo", "S"], species_B=["Mo", "S"]
        ),
        PotentialAssignerParams(),
    )
    out = LammpsInputWriter().run(
        LammpsInputWriterInputs(
            md_structure=md,
            potential_plan=plan,
            species_A=["Mo", "S"],
            species_B=["Mo", "S"],
            n_atoms_A=2,
        ),
        LammpsInputWriterParams(output_dir=tmp_path / "run"),
    )
    text = out.bilayer_script.read_text()
    assert "pair_style      lj/cut" in text
    assert "INTERLAYER vdW" in text
    assert "zeroed" in text  # intralayer pairs zeroed
    assert "molecule 1" in text
    assert "molecule 2" in text
