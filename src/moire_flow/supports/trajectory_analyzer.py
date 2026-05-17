"""TrajectoryAnalyzer: parse LAMMPS log files into per-thermo arrays + metrics.

Support C. Port of `parse_lammps_log` (reference 11601-11653) with an
added Pydantic-typed summary (interlayer gap, vdW energy, strain %).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class TrajectoryAnalysis(BaseModel):
    """Summary of a LAMMPS log file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    log_path: Path
    n_blocks: int
    columns: list[str]
    n_rows: int
    interlayer_gap_ang: float | None = None
    vdW_eV: float | None = None
    vdW_per_area_eV_A2: float | None = None
    final_pe_eV: float | None = None
    final_temp_K: float | None = None
    raw_last_block: dict[str, list[float]]


def _parse_blocks(text: str) -> list[tuple[list[str], np.ndarray]]:
    """Split a LAMMPS log into (header, array) thermo blocks."""
    blocks: list[tuple[list[str], np.ndarray]] = []
    header: list[str] | None = None
    rows: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Step"):
            if header is not None and rows:
                blocks.append((header, np.array(rows, dtype=np.float64)))
            header = s.split()
            rows = []
            continue
        if header and s:
            first = s.split()[0]
            is_numeric = first.replace(".", "", 1).replace("-", "", 1).isdigit()
            if is_numeric:
                try:
                    row = [float(x) for x in s.split()]
                    if len(row) >= len(header):
                        rows.append(row[: len(header)])
                except ValueError:
                    continue
        if header and ("Loop time" in s or "--- " in s):
            if rows:
                blocks.append((header, np.array(rows, dtype=np.float64)))
            header = None
            rows = []
    if header is not None and rows:
        blocks.append((header, np.array(rows, dtype=np.float64)))
    return blocks


class TrajectoryAnalyzer:
    """Read a LAMMPS log file and produce a `TrajectoryAnalysis`."""

    @staticmethod
    def analyze(log_path: str | Path) -> TrajectoryAnalysis:
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        blocks = _parse_blocks(log_path.read_text(errors="ignore"))
        if not blocks:
            return TrajectoryAnalysis(
                log_path=log_path,
                n_blocks=0,
                columns=[],
                n_rows=0,
                raw_last_block={},
            )
        header, arr = blocks[-1]
        cols: dict[str, np.ndarray] = {h: arr[:, i] for i, h in enumerate(header) if i < arr.shape[1]}

        def _final(col: str) -> float | None:
            data = cols.get(col)
            if data is None or len(data) == 0:
                return None
            return float(data[-1])

        # Default LAMMPS thermo aliases used in the bilayer LJ script
        final_pe = _final("PotEng") or _final("pe")
        final_temp = _final("Temp") or _final("temp")
        dz = _final("v_dz")
        vdW = _final("c_vdW")
        vdW_per_A = _final("v_vdW_per_A")

        return TrajectoryAnalysis(
            log_path=log_path,
            n_blocks=len(blocks),
            columns=list(header),
            n_rows=int(arr.shape[0]),
            interlayer_gap_ang=dz,
            vdW_eV=vdW,
            vdW_per_area_eV_A2=vdW_per_A,
            final_pe_eV=final_pe,
            final_temp_K=final_temp,
            raw_last_block={k: v.tolist() for k, v in cols.items()},
        )


__all__ = ["TrajectoryAnalyzer", "TrajectoryAnalysis"]
