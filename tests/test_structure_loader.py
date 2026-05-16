"""StructureLoader regression vs hand-written fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moire_flow.boxes import StructureLoader, StructureLoaderInputs, StructureLoaderParams


def test_lammps_data_triclinic_cell(lammps_data_atomic: Path):
    loader = StructureLoader()
    s = loader.run(
        StructureLoaderInputs(path=lammps_data_atomic),
        StructureLoaderParams(),
    )
    assert s.atoms.shape == (2, 3)
    assert s.species == ["Mo", "S"]
    expected = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.5, 3.0, 0.0],
            [0.1, 0.2, 20.0],
        ]
    )
    np.testing.assert_allclose(s.cell, expected)


def test_lammps_data_guesses_element_from_mass(lammps_data_guessed_mass: Path):
    loader = StructureLoader()
    s = loader.run(
        StructureLoaderInputs(path=lammps_data_guessed_mass),
        StructureLoaderParams(),
    )
    assert s.species == ["C"]


def test_xyz_loader(xyz_file: Path):
    loader = StructureLoader()
    s = loader.run(StructureLoaderInputs(path=xyz_file), StructureLoaderParams())
    assert s.atoms.shape == (3, 3)
    assert s.species == ["Mo", "S", "S"]
    np.testing.assert_allclose(np.diag(s.cell), [5.0, 5.0, 10.0])


def test_format_auto_dispatch_by_extension(lammps_data_atomic: Path, xyz_file: Path):
    loader = StructureLoader()
    s_data = loader.run(StructureLoaderInputs(path=lammps_data_atomic), StructureLoaderParams())
    s_xyz = loader.run(StructureLoaderInputs(path=xyz_file), StructureLoaderParams())
    assert s_data.atoms.shape == (2, 3)
    assert s_xyz.atoms.shape == (3, 3)


def test_format_unknown_extension_raises(tmp_path: Path):
    weird = tmp_path / "x.foo"
    weird.write_text("hello")
    loader = StructureLoader()
    with pytest.raises(ValueError, match="auto-detect"):
        loader.run(StructureLoaderInputs(path=weird), StructureLoaderParams())


def test_missing_file_raises(tmp_path: Path):
    loader = StructureLoader()
    with pytest.raises(FileNotFoundError):
        loader.run(
            StructureLoaderInputs(path=tmp_path / "missing.cif"),
            StructureLoaderParams(),
        )


def test_dedup_removes_close_duplicates(tmp_path: Path):
    p = tmp_path / "dup.xyz"
    p.write_text(
        "3\n"
        'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3\n'
        "Mo 0.0 0.0 0.0\n"
        "Mo 0.0001 0.0 0.0\n"
        "Mo 5.0 0.0 0.0\n"
    )
    loader = StructureLoader()
    s = loader.run(
        StructureLoaderInputs(path=p),
        StructureLoaderParams(duplicate_tolerance=1e-3),
    )
    assert s.atoms.shape == (2, 3)


def test_strict_elements_rejects_unknown_mass(tmp_path: Path):
    p = tmp_path / "weird.data"
    p.write_text(
        "x\n\n1 atoms\n1 atom types\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n1 1234.5\n\nAtoms # atomic\n\n1 1 0 0 0\n"
    )
    loader = StructureLoader()
    with pytest.raises(ValueError, match="strict_elements"):
        loader.run(
            StructureLoaderInputs(path=p),
            StructureLoaderParams(strict_elements=True),
        )
