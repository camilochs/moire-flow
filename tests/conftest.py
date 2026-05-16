"""Test fixtures: minimal, file-format-correct samples generated at runtime."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def lammps_data_atomic(tmp_path: Path) -> Path:
    """Minimal LAMMPS `.data` with `atomic` style, triclinic box, 2 atom types."""
    text = """# 2-atom MoS bilayer fixture

2 atoms
2 atom types

0.000000 3.000000 xlo xhi
0.000000 3.000000 ylo yhi
0.000000 20.000000 zlo zhi
0.500000 0.100000 0.200000 xy xz yz

Masses

1 95.96    # Mo
2 32.06    # S

Atoms # atomic

1 1 0.0 0.0 5.0
2 2 1.5 1.5 8.0
"""
    p = tmp_path / "frame.data"
    p.write_text(text)
    return p


@pytest.fixture
def lammps_data_guessed_mass(tmp_path: Path) -> Path:
    """LAMMPS data with no element comment — exercises mass→element guess."""
    text = """LAMMPS data no comment

1 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 12.011

Atoms # atomic

1 1 0.0 0.0 0.0
"""
    p = tmp_path / "carbon.data"
    p.write_text(text)
    return p


@pytest.fixture
def xyz_file(tmp_path: Path) -> Path:
    """Minimal extended XYZ with a cell."""
    text = (
        "3\n"
        'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3\n'
        "Mo 0.0 0.0 5.0\n"
        "S  1.5 0.0 6.5\n"
        "S  1.5 0.0 3.5\n"
    )
    p = tmp_path / "frame.xyz"
    p.write_text(text)
    return p
