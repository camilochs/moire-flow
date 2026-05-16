"""LAMMPS `.data` reader → `Structure`.

Decomposed port of `parse_lammps_data_file` (reference 557–740).
Differences from the original:
- Returns a triclinic 3×3 cell (preserves xz, yz; ANALYSIS.md §8.5).
- Element lookup uses ase.data, not a 25-element hardcoded table.
- Each parsing concern lives in its own helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from moire_flow.core.types import Structure

from ._mass import guess_element_from_mass

AtomStyle = Literal["atomic", "charge", "full"]


def _parse_box(lines: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    """Extract triclinic cell (3×3) and box metadata from header lines."""
    xlo = xhi = ylo = yhi = zlo = zhi = None
    xy = xz = yz = 0.0
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
            xlo, xhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
            ylo, yhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
            zlo, zhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 6 and parts[-3:] == ["xy", "xz", "yz"]:
            xy, xz, yz = float(parts[0]), float(parts[1]), float(parts[2])
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError("Could not parse box bounds (xlo/xhi, ylo/yhi, zlo/zhi)")
    lx, ly, lz = float(xhi - xlo), float(yhi - ylo), float(zhi - zlo)
    cell = np.array(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ],
        dtype=np.float64,
    )
    box = {
        "xlo": float(xlo), "xhi": float(xhi),
        "ylo": float(ylo), "yhi": float(yhi),
        "zlo": float(zlo), "zhi": float(zhi),
        "lx": lx, "ly": ly, "lz": lz,
        "xy": float(xy), "xz": float(xz), "yz": float(yz),
    }
    return cell, box


def _parse_masses(lines: list[str]) -> tuple[dict[int, str], dict[int, float]]:
    """Parse `Masses` section → {type_id: element}, {type_id: mass}."""
    type_to_element: dict[int, str] = {}
    type_to_mass: dict[int, float] = {}
    in_masses = False
    section_starts = ("Atoms", "Velocities", "Bonds", "Angles", "Pair Coeffs")
    for line in lines:
        s = line.strip()
        if s.startswith("Masses"):
            in_masses = True
            continue
        if not in_masses:
            continue
        if not s:
            continue
        if s.startswith(section_starts):
            break
        if not s[0].isdigit():
            continue
        raw = s.split("#", 1)
        left = raw[0].split()
        if len(left) < 2:
            continue
        type_id = int(left[0])
        mass = float(left[1])
        type_to_mass[type_id] = mass
        if len(raw) > 1 and raw[1].strip():
            element = raw[1].strip().split()[0]
        else:
            element = guess_element_from_mass(mass)
        type_to_element[type_id] = element
    if not type_to_element:
        raise ValueError("Could not parse Masses section")
    return type_to_element, type_to_mass


def _parse_atom_line(parts: list[str], style: AtomStyle | None) -> tuple[int, float, float, float]:
    """Return (atom_type, x, y, z) for one line in `Atoms` section."""
    if style == "atomic":
        return int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    if style == "charge":
        return int(parts[1]), float(parts[3]), float(parts[4]), float(parts[5])
    if style == "full":
        return int(parts[2]), float(parts[4]), float(parts[5]), float(parts[6])
    n = len(parts)
    if n == 5:
        return int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    if n == 6:
        return int(parts[1]), float(parts[3]), float(parts[4]), float(parts[5])
    if n >= 7:
        return int(parts[2]), float(parts[4]), float(parts[5]), float(parts[6])
    raise ValueError(f"Cannot parse atom line: {' '.join(parts)}")


def _parse_atoms(
    lines: list[str], type_to_element: dict[int, str]
) -> tuple[np.ndarray, list[str]]:
    """Parse `Atoms` section into (positions, species)."""
    style: AtomStyle | None = None
    in_atoms = False
    positions: list[list[float]] = []
    species: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("Atoms"):
            in_atoms = True
            if "#" in s:
                tag = s.split("#", 1)[1].strip().lower()
                if tag in ("atomic", "charge", "full"):
                    style = tag  # type: ignore[assignment]
            continue
        if not in_atoms:
            continue
        if not s or s.startswith("#"):
            continue
        if not s[0].isdigit():
            break
        parts = s.split()
        atom_type, x, y, z = _parse_atom_line(parts, style)
        positions.append([x, y, z])
        species.append(type_to_element.get(atom_type, f"TYPE_{atom_type}"))
    if not positions:
        raise ValueError("No atoms parsed from Atoms section")
    return np.asarray(positions, dtype=np.float64), species


def read_lammps_data(path: Path | str) -> Structure:
    """Read a LAMMPS `.data` file and return a `Structure` with triclinic 3×3 cell."""
    text = Path(path).read_text(errors="ignore").splitlines()
    cell, _box = _parse_box(text)
    type_to_element, _type_to_mass = _parse_masses(text)
    atoms, species = _parse_atoms(text, type_to_element)
    return Structure(atoms=atoms, species=species, cell=cell)


__all__ = ["read_lammps_data"]
