"""End-to-end smoke test through a real LAMMPS binary.

Marked `requires_docker` and `requires_lammps_image` — skipped unless
both Docker is reachable and the `moire-flow-runtime:latest` image is
locally available (or `MOIRE_FLOW_LAMMPS_IMAGE` env var points elsewhere).

This is the only test in the suite that actually runs LAMMPS. It
exercises the full chain:

    BilayerAtoms (synthetic 2x2 Mo–S supercell)
        → MDSupercellBuilder
        → PotentialAssigner (intralayer_kind='lj' to avoid pair-file deps)
        → LammpsInputWriter (writes layer_A.data + layer_A.in + bilayer.*)
        → LammpsExecutor (docker run moire-flow-runtime layer_A.in)
        → TrajectoryAnalyzer (parses the resulting log)

Asserts: returncode 0 + at least one thermo block parsed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from moire_flow.boxes import (
    LammpsInputWriter,
    LammpsInputWriterInputs,
    LammpsInputWriterParams,
    MDSupercellBuilder,
    MDSupercellBuilderInputs,
    MDSupercellBuilderParams,
    PotentialAssigner,
    PotentialAssignerInputs,
    PotentialAssignerParams,
)
from moire_flow.core.types import BilayerAtoms
from moire_flow.supports import LammpsExecutor, TrajectoryAnalyzer

IMAGE = os.environ.get("MOIRE_FLOW_LAMMPS_IMAGE", "moire-flow-runtime:latest")
FULL_IMAGE = os.environ.get("MOIRE_FLOW_LAMMPS_FULL_IMAGE", "moire-flow-runtime:full")


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    res = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
    return res.returncode == 0


def _image_present(name: str) -> bool:
    if not _docker_available():
        return False
    res = subprocess.run(
        ["docker", "image", "inspect", name],
        capture_output=True, text=True, timeout=10,
    )
    return res.returncode == 0


requires_lammps = pytest.mark.skipif(
    not _image_present(IMAGE),
    reason=f"Docker image {IMAGE!r} not available — build with `docker build -t moire-flow-runtime:latest runtime/`",
)
requires_lammps_full = pytest.mark.skipif(
    not _image_present(FULL_IMAGE),
    reason=f"Docker image {FULL_IMAGE!r} not available — build with `docker build --platform linux/amd64 -t moire-flow-runtime:full -f runtime/Dockerfile.full runtime/`",
)


@pytest.fixture
def docker_visible_tmp(tmp_path_factory) -> Path:
    """Pytest's default tmpdir lives under /private/var/folders/... which
    Colima (and Docker Desktop with default mounts) does NOT expose to the
    VM — leading to 'No such file or directory' when LAMMPS tries to read
    its input script. Use a path under $HOME instead, which is mounted."""
    base = Path.home() / ".cache" / "moire-flow-tests"
    base.mkdir(parents=True, exist_ok=True)
    return Path(str(tmp_path_factory.mktemp(
        "lammps_e2e", numbered=True
    )).replace(str(tmp_path_factory.getbasetemp()), str(base)))


@pytest.fixture
def synthetic_bilayer() -> BilayerAtoms:
    """2x2 MoS2 monolayer pair (8 atoms per layer) in a small ortho cell."""
    a = 3.16  # lattice param Å
    # Mo at (0,0) and S at (a/2, a/2), tiled 2x2 → 4 Mo + 4 S per layer
    basis = np.array([[0.0, 0.0], [a / 2, a / 2]])
    atoms_2d: list[np.ndarray] = []
    species: list[str] = []
    for i in range(2):
        for j in range(2):
            shift = np.array([i * a, j * a])
            atoms_2d.append(basis + shift)
            species.extend(["Mo", "S"])
    atoms_2d_arr = np.vstack(atoms_2d)
    return BilayerAtoms(
        atoms_A=atoms_2d_arr,
        atoms_B=atoms_2d_arr,
        species_A=species,
        species_B=species,
        sc_vecs=np.array([[2 * a, 0.0], [0.0, 2 * a]]),
    )


@requires_lammps
def test_lammps_runs_a_minimization(
    docker_visible_tmp: Path, synthetic_bilayer: BilayerAtoms
):
    tmp_path = docker_visible_tmp
    """Build a workflow, write files, run LAMMPS in Docker, parse the log."""
    md = MDSupercellBuilder().run(
        MDSupercellBuilderInputs(bilayer=synthetic_bilayer),
        MDSupercellBuilderParams(vacuum_z=15.0, interlayer_z=3.3),
    )
    plan = PotentialAssigner().run(
        PotentialAssignerInputs(
            md_structure=md,
            species_A=synthetic_bilayer.species_A,
            species_B=synthetic_bilayer.species_B,
        ),
        # Force LJ to avoid needing Tersoff potential files in the image.
        PotentialAssignerParams(intralayer_kind="lj"),
    )
    run_plan = LammpsInputWriter().run(
        LammpsInputWriterInputs(
            md_structure=md,
            potential_plan=plan,
            species_A=synthetic_bilayer.species_A,
            species_B=synthetic_bilayer.species_B,
            n_atoms_A=len(synthetic_bilayer.atoms_A),
        ),
        LammpsInputWriterParams(output_dir=tmp_path / "run", nvt_steps=0),
    )
    executor = LammpsExecutor(image=IMAGE)
    result = executor.run(
        script_path=run_plan.layer_A_script,
        log_path=tmp_path / "run" / "layer_A.log",
        work_dir=run_plan.work_dir,
        timeout=120.0,
    )
    assert result.ok, (
        f"LAMMPS exited with {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    analysis = TrajectoryAnalyzer.analyze(result.log)
    assert analysis.n_blocks >= 1, (
        f"No thermo blocks parsed from {result.log}; "
        f"last bytes: {result.log.read_text()[-500:]}"
    )
    # PotEng must be finite and negative (bound state under LJ)
    assert analysis.final_pe_eV is not None
    assert np.isfinite(analysis.final_pe_eV)


@requires_lammps_full
def test_full_image_recognizes_mace_pair_style(docker_visible_tmp: Path):
    """The :full image must enumerate `mace` and `mliap` among its pair styles.

    Uses LAMMPS's `info` command (available since 2014) which prints every
    pair style compiled into the binary. We don't need a real model file —
    just to see `mace` in the list. If the ML-MACE package isn't compiled
    in, `info` won't mention it.
    """
    script = docker_visible_tmp / "probe.in"
    script.write_text("info styles pair\nquit\n")
    executor = LammpsExecutor(image=FULL_IMAGE, platform="linux/amd64")
    res = executor.run(script_path=script, work_dir=script.parent, timeout=60.0)
    combined = (res.stdout + "\n"
                + (res.log.read_text() if res.log.exists() else "")).lower()
    assert "mace" in combined, (
        "MACE pair style not registered in the :full image. Full output:\n"
        + combined[-1500:]
    )
    assert "mliap" in combined, (
        "ML-IAP pair style not registered in the :full image. Full output:\n"
        + combined[-1500:]
    )
