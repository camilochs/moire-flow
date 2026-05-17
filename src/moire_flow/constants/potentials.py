"""Potential parameter tables.

- `UFF_LJ`: UFF-derived LJ (sigma in Å, epsilon in eV) — for **interlayer
  vdW only**, never intralayer covalent bonding. Port of reference 3620-3643.
- `TERSOFF_REGISTRY`, `SW_REGISTRY`: pair-of-elements → potential file.
  Port of reference 3648-3660.
- `atomic_mass`, `atomic_number`: thin wrappers over ase.data so we don't
  re-import the 22-element hardcoded table from the original.
"""

from __future__ import annotations

from typing import Final

import ase.data

UFF_LJ: Final[dict[str, dict[str, float]]] = {
    "H":  {"sigma": 2.886, "epsilon": 0.0020},
    "B":  {"sigma": 3.637, "epsilon": 0.0040},
    "C":  {"sigma": 3.431, "epsilon": 0.002416},
    "N":  {"sigma": 3.261, "epsilon": 0.002897},
    "O":  {"sigma": 3.118, "epsilon": 0.0035},
    "P":  {"sigma": 4.147, "epsilon": 0.0050},
    "S":  {"sigma": 3.595, "epsilon": 0.2740},
    "Cl": {"sigma": 3.947, "epsilon": 0.0050},
    "Ti": {"sigma": 2.829, "epsilon": 0.0170},
    "Zn": {"sigma": 2.462, "epsilon": 0.0003},
    "As": {"sigma": 3.768, "epsilon": 0.0040},
    "Se": {"sigma": 3.746, "epsilon": 0.2910},
    "Zr": {"sigma": 2.783, "epsilon": 0.0690},
    "Mo": {"sigma": 2.719, "epsilon": 0.0560},
    "Cd": {"sigma": 2.537, "epsilon": 0.0003},
    "In": {"sigma": 2.834, "epsilon": 0.0003},
    "Sb": {"sigma": 3.937, "epsilon": 0.0040},
    "Te": {"sigma": 4.009, "epsilon": 0.3980},
    "Hf": {"sigma": 2.798, "epsilon": 0.0720},
    "W":  {"sigma": 2.735, "epsilon": 0.0670},
    "Pt": {"sigma": 2.754, "epsilon": 0.0005},
    "Pb": {"sigma": 2.846, "epsilon": 0.0004},
}

TERSOFF_REGISTRY: Final[dict[frozenset[str], str]] = {
    frozenset({"Mo", "S"}): "MoS.tersoff",
    frozenset({"Mo", "Se"}): "MoSe2.tersoff",
    frozenset({"W", "S"}): "WS2.tersoff",
    frozenset({"W", "Se"}): "WSe2.tersoff",
    frozenset({"Hf", "Se"}): "HfSe2.tersoff",
}

SW_REGISTRY: Final[dict[frozenset[str], str]] = {
    frozenset({"Hf", "Se"}): "HfSe2.sw",
    frozenset({"Mo", "S"}): "MoS2.sw",
    frozenset({"W", "S"}): "WS2.sw",
}


def atomic_mass(symbol: str) -> float:
    """Atomic mass for `symbol` (AMU). Raises KeyError if unknown."""
    try:
        Z = ase.data.atomic_numbers[symbol]
    except KeyError as err:
        raise KeyError(f"Unknown element symbol: {symbol!r}") from err
    return float(ase.data.atomic_masses[Z])


def atomic_number(symbol: str) -> int:
    """Atomic number for `symbol`. Raises KeyError if unknown."""
    try:
        return int(ase.data.atomic_numbers[symbol])
    except KeyError as err:
        raise KeyError(f"Unknown element symbol: {symbol!r}") from err


__all__ = ["UFF_LJ", "TERSOFF_REGISTRY", "SW_REGISTRY", "atomic_mass", "atomic_number"]
