"""FastAPI backend: catalog + run endpoints."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from web.backend.server import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_boxes_lists_all_nine():
    r = client.get("/api/boxes")
    assert r.status_code == 200
    payload = r.json()
    assert len(payload) == 9
    names = {b["name"] for b in payload}
    assert names == {
        "atom_assembler", "bilayer_splitter", "lammps_input_writer",
        "lattice_matcher", "lattice_transformer", "md_supercell_builder",
        "potential_assigner", "structure_loader", "validator",
    }
    for box in payload:
        assert "inputs_schema" in box
        assert "params_schema" in box
        assert "outputs_schema" in box
        assert isinstance(box["inputs"], list)
        assert isinstance(box["outputs"], list)


def test_describe_known_box():
    r = client.get("/api/boxes/lattice_transformer")
    assert r.status_code == 200
    box = r.json()
    assert box["name"] == "lattice_transformer"
    assert "transform_type" in box["params"]


def test_describe_unknown_box_404():
    r = client.get("/api/boxes/not_a_real_box")
    assert r.status_code == 404


def test_validate_workflow_accepts_valid_spec():
    spec = {
        "nodes": [
            {
                "id": "t",
                "box_name": "lattice_transformer",
                "params": {"transform_type": "identity"},
                "inputs": {},
            }
        ],
        "edges": [],
    }
    r = client.post("/api/workflows/validate", json=spec)
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_validate_workflow_rejects_invalid_spec():
    r = client.post(
        "/api/workflows/validate",
        json={"nodes": [{"id": "a", "box_name": "nonexistent_box"}]},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert r.json()["errors"]


def test_run_workflow_end_to_end():
    spec = {
        "nodes": [
            {
                "id": "t",
                "box_name": "lattice_transformer",
                "params": {"transform_type": "rotation", "theta_deg": 30.0},
                "inputs": {
                    "Alat": [[3.16, 0.0], [0.0, 3.16]],
                    "basis_A": [[0.0, 0.0], [1.58, 1.58]],
                    "species_A": ["Mo", "S"],
                },
            }
        ],
        "edges": [],
    }
    r = client.post("/api/workflows/run", json={"spec": spec})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "t" in body["results"]
    assert "Blat" in body["results"]["t"]


def test_run_workflow_reports_error_for_bad_inputs():
    spec = {
        "nodes": [
            {
                "id": "t",
                "box_name": "lattice_transformer",
                "params": {"transform_type": "identity"},
                "inputs": {},  # missing required fields
            }
        ],
        "edges": [],
    }
    r = client.post("/api/workflows/run", json={"spec": spec})
    body = r.json()
    assert body["ok"] is False
    assert body["error"]
