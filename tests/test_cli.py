"""CLI: typer commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from moire_flow.cli import app

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    from moire_flow import __version__
    assert __version__ in result.stdout


def test_list_boxes_lists_all_nine():
    result = runner.invoke(app, ["list-boxes"])
    assert result.exit_code == 0
    for name in (
        "structure_loader", "bilayer_splitter", "lattice_transformer",
        "lattice_matcher", "atom_assembler", "validator",
        "md_supercell_builder", "potential_assigner", "lammps_input_writer",
    ):
        assert name in result.stdout


def test_describe_known_box_emits_valid_json():
    result = runner.invoke(app, ["describe", "structure_loader"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["name"] == "structure_loader"
    assert "inputs_schema" in parsed
    assert "params_schema" in parsed


def test_describe_unknown_box_exits_nonzero():
    result = runner.invoke(app, ["describe", "nonsense"])
    assert result.exit_code != 0


def test_run_executes_a_spec(tmp_path: Path):
    spec = {
        "nodes": [
            {
                "id": "t",
                "box_name": "lattice_transformer",
                "params": {"transform_type": "identity"},
                "inputs": {
                    "Alat": [[3.16, 0.0], [0.0, 3.16]],
                    "basis_A": [[0.0, 0.0], [1.58, 1.58]],
                    "species_A": ["Mo", "S"],
                },
            }
        ],
        "edges": [],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    out_dir = tmp_path / "results"
    result = runner.invoke(app, ["run", str(spec_path), "--out", str(out_dir)])
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "t.json").exists()
