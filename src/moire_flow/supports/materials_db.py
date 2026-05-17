"""MaterialsDB: read-only client for the 2D materials JSON snapshot.

Support A (not a box — it is a service, not a transformation). Reads the
JSON snapshot served from a Gist or a local file. Port of `load_materials_db`,
`get_material`, `list_materials`, `extract_lattice_2d`, `extract_basis_2d`
(reference 1159-1202, 1179-1197).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

from moire_flow.constants.potentials import atomic_number

DEFAULT_GIST_URL = (
    "https://gist.githubusercontent.com/camilochs/8a172df57f1b81f902307961f8850d0b/raw/"
    "9ddc4b7c65e66ddf6571ca3839298a9364288061/gistfile1.txt"
)


class MaterialNotFound(KeyError):
    """Raised when get_material can't find the requested identifier."""


class MaterialsDB:
    """Read-only client over the 2D materials JSON snapshot."""

    def __init__(self, entries: list[dict[str, Any]]):
        self._entries: list[dict[str, Any]] = list(entries)

    @classmethod
    def from_url(cls, url: str = DEFAULT_GIST_URL, timeout: float = 30.0) -> "MaterialsDB":
        """Load from an HTTP(S) URL — the production path."""
        with urlopen(url, timeout=timeout) as response:
            entries = json.load(response)
        return cls(entries)

    @classmethod
    def from_path(cls, path: str | Path) -> "MaterialsDB":
        """Load from a local JSON file — preferred for tests and offline runs."""
        with open(path) as fh:
            entries = json.load(fh)
        return cls(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def list_formulas(self) -> list[str]:
        return [str(e.get("formula", "")) for e in self._entries]

    def get_material(self, identifier: str, deep_copy: bool = True) -> dict[str, Any]:
        for entry in self._entries:
            if entry.get("formula") == identifier or entry.get("id") == identifier:
                return copy.deepcopy(entry) if deep_copy else entry
        raise MaterialNotFound(f"Material {identifier!r} not in database")

    @staticmethod
    def extract_lattice_2d(entry: dict[str, Any], source: str = "SIESTA") -> np.ndarray:
        si = entry[source]["structure_info"]
        if source == "C2DB":
            lv = np.asarray(si["lattice_vectors"][0], dtype=np.float64).reshape(3, 3)
        else:
            lv = np.asarray(si["lattice_vectors"], dtype=np.float64)
        return lv[:2, :2]

    @staticmethod
    def extract_basis_2d(
        entry: dict[str, Any], source: str = "SIESTA"
    ) -> tuple[np.ndarray, list[str]]:
        si = entry[source]["structure_info"]
        pos = np.asarray(si["positions"], dtype=np.float64)[:, :2]
        if source == "SIESTA":
            elements = list(si["atomic_elements"])
        else:
            # C2DB stores atomic numbers; convert to symbols via ase.data.
            from ase.data import chemical_symbols
            elements = [chemical_symbols[int(z)] for z in si["atomic_numbers"]]
            # round-trip sanity (ensures atomic_number works on each)
            for e in elements:
                atomic_number(e)
        return pos, elements


__all__ = ["MaterialsDB", "MaterialNotFound", "DEFAULT_GIST_URL"]
