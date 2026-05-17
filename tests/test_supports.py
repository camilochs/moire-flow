"""Supports: MaterialsDB (local JSON), LammpsExecutor (cmd-build only),
TrajectoryAnalyzer (synthetic log)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from moire_flow.supports import (
    LammpsExecutor,
    MaterialNotFound,
    MaterialsDB,
    TrajectoryAnalyzer,
)


# ---------- MaterialsDB ----------

def test_materials_db_loads_from_local_json(tmp_path: Path):
    entries = [
        {
            "formula": "MoSe2",
            "id": "test_mose2",
            "SIESTA": {
                "structure_info": {
                    "lattice_vectors": [[3.32, 0.0, 0.0], [1.66, 2.876, 0.0], [0.0, 0.0, 20.0]],
                    "positions": [[0.0, 0.0, 0.0], [1.66, 0.96, 1.65], [1.66, 0.96, -1.65]],
                    "atomic_elements": ["Mo", "Se", "Se"],
                },
            },
        }
    ]
    p = tmp_path / "db.json"
    p.write_text(json.dumps(entries))
    db = MaterialsDB.from_path(p)
    assert len(db) == 1
    assert db.list_formulas() == ["MoSe2"]
    mat = db.get_material("MoSe2")
    assert mat["formula"] == "MoSe2"


def test_materials_db_extract_lattice_2d(tmp_path: Path):
    entry = {
        "SIESTA": {
            "structure_info": {
                "lattice_vectors": [[3.32, 0.0, 0.0], [1.66, 2.876, 0.0], [0.0, 0.0, 20.0]],
                "positions": [],
                "atomic_elements": [],
            }
        }
    }
    lat = MaterialsDB.extract_lattice_2d(entry, source="SIESTA")
    assert lat.shape == (2, 2)
    np.testing.assert_allclose(lat, [[3.32, 0.0], [1.66, 2.876]])


def test_materials_db_extract_basis_2d_siesta():
    entry = {
        "SIESTA": {
            "structure_info": {
                "lattice_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "positions": [[0.0, 0.0, 0.0], [1.66, 0.96, 1.65]],
                "atomic_elements": ["Mo", "Se"],
            }
        }
    }
    pos, els = MaterialsDB.extract_basis_2d(entry)
    assert pos.shape == (2, 2)
    assert els == ["Mo", "Se"]


def test_materials_db_get_material_missing_raises():
    db = MaterialsDB([])
    with pytest.raises(MaterialNotFound):
        db.get_material("NotInDB")


# ---------- LammpsExecutor ----------

def test_lammps_executor_capabilities_when_docker_absent():
    ex = LammpsExecutor(docker_bin="this_binary_does_not_exist_xyz")
    caps = ex.capabilities()
    assert caps["docker_available"] is False


def test_lammps_executor_rejects_missing_script(tmp_path: Path):
    ex = LammpsExecutor()
    with pytest.raises(FileNotFoundError):
        ex.run(tmp_path / "nope.in")


# ---------- TrajectoryAnalyzer ----------

def test_trajectory_analyzer_empty_log(tmp_path: Path):
    log = tmp_path / "empty.log"
    log.write_text("# nothing\n")
    out = TrajectoryAnalyzer.analyze(log)
    assert out.n_blocks == 0
    assert out.interlayer_gap_ang is None


def test_trajectory_analyzer_parses_one_block(tmp_path: Path):
    log = tmp_path / "ok.log"
    log.write_text(
        "Some preamble\n"
        "Step Temp PotEng v_dz c_vdW v_vdW_per_A\n"
        "0    300.0 -10.0 3.30 -0.50 -0.02\n"
        "100  301.5 -10.2 3.28 -0.55 -0.022\n"
        "200  299.8 -10.3 3.27 -0.57 -0.023\n"
        "Loop time of 1.0 on 1 procs\n"
    )
    out = TrajectoryAnalyzer.analyze(log)
    assert out.n_blocks == 1
    assert out.n_rows == 3
    assert "PotEng" in out.columns
    assert out.interlayer_gap_ang == pytest.approx(3.27)
    assert out.vdW_eV == pytest.approx(-0.57)
    assert out.final_temp_K == pytest.approx(299.8)


def test_trajectory_analyzer_picks_last_block(tmp_path: Path):
    log = tmp_path / "two.log"
    log.write_text(
        "Step Temp\n"
        "0  100\n"
        "100  150\n"
        "Loop time of 0.5\n"
        "Step Temp\n"
        "0  200\n"
        "100  250\n"
        "Loop time of 1.0\n"
    )
    out = TrajectoryAnalyzer.analyze(log)
    assert out.n_blocks == 2
    assert out.final_temp_K == pytest.approx(250.0)
