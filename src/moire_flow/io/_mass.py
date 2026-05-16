"""Mass → element lookup, populated from `ase.data.atomic_masses`.

Replaces the 25-element `ATOMIC_MASS_LOOKUP` of the original
`lattice_matching_02032026.py` (line 490) — see ANALYSIS.md §8.6.
"""

from __future__ import annotations

import ase.data

_BY_MASS: list[tuple[float, str]] = sorted(
    (float(ase.data.atomic_masses[Z]), ase.data.chemical_symbols[Z])
    for Z in range(1, len(ase.data.atomic_masses))
    if ase.data.atomic_masses[Z] > 0
)


def guess_element_from_mass(mass: float, tol: float = 0.35) -> str:
    """Closest-match lookup of element symbol given an atomic mass.

    Falls back to `X<mass>` if no element is within `tol` AMU. Mirrors the
    original `guess_element_from_mass` semantics with a wider table.
    """
    m = float(mass)
    best_el = "X"
    best_diff = float("inf")
    for ref_mass, sym in _BY_MASS:
        diff = abs(m - ref_mass)
        if diff < best_diff:
            best_diff = diff
            best_el = sym
    if best_diff <= tol:
        return best_el
    return f"X{m:.2f}"


__all__ = ["guess_element_from_mass"]
