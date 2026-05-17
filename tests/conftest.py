"""Test fixtures: minimal, file-format-correct samples generated at runtime."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest


# ---------- LAMMPS / structural fixtures ----------

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


# ---------- Reference-module loader ----------

@pytest.fixture(scope="session")
def reference_helpers():
    """Load the vendored original script and extract its pure helpers as
    a dict of {name: callable}. Cached for the session — no Colab side
    effects (we never run the cells that hit `google.colab` or `urlopen`).
    """
    path = Path(__file__).resolve().parents[1] / "reference" / "lattice_matching_02032026.py"
    src = path.read_text()
    wanted = {
        # algebra2d
        "ChangeBasis", "R", "S",
        "vec2d_dot", "vec2d_cross", "vec2d_norm", "vec2d_angle_deg", "_as_2d_vector",
        "wrap_basis_to_cell", "transform_basis", "tile_points",
        "wrap_fractional", "fractional_positions", "atoms_in_lattice_patch",
        "canonical_theta_deg", "angle_error_symmetry_deg",
        # matching
        "supercell_points", "cost_funcion", "get_lavects",
        "find_scvect_pairs", "compute_mismatch",
        "_build_a_lattice_candidates", "param_grid_from_bounds",
        "atoms_in_supercell",
        # synth
        "build_transformed_lattice", "build_transformed_basis",
        # validation_metrics
        "periodic_fractional_distances", "fractional_distance_signature",
        "fractional_rmsd", "pairwise_distance_signature",
        "local_distance_signature", "distance_signature_error",
        "neighbor_counts", "coordination_signature", "coordination_match_score",
        "lattice_metrics",
        # assembly
        "build_atoms_for_best",
        # writer geometry
        "_v2d", "_box",
        # mass guess
        "guess_element_from_mass",
    }
    blocks = _extract_top_level_defs(src, wanted)
    ns: dict = {"np": np}
    # Pull a few top-level constants the helpers reference globally.
    # ATOMIC_MASS_LOOKUP is needed by `guess_element_from_mass`.
    lines = src.splitlines()
    const_pattern = "ATOMIC_MASS_LOOKUP = {"
    for j, line in enumerate(lines):
        if line.startswith(const_pattern):
            k = j
            depth = 0
            while k < len(lines):
                depth += lines[k].count("{") - lines[k].count("}")
                k += 1
                if depth == 0:
                    break
            const_block = "\n".join(lines[j:k])
            exec(const_block, ns)
            break
    # Some helpers reference these top-level names from elsewhere in the file
    import math
    ns["math"] = math
    # The reference's `_box` uses `_v2d` which we also extract; need to also
    # include `ATOMIC_MASS` / `ATOMIC_MASS_LOOKUP` and `UFF_LJ` if any helper
    # touches them indirectly — but the wanted set above is closed under deps.
    exec("\n\n".join(blocks), ns)
    return ns


def _starts_def(line: str) -> str | None:
    if not line.startswith("def "):
        return None
    rest = line[4:]
    end = rest.find("(")
    if end < 0:
        return None
    return rest[:end]


def _scrub_colab(src: str) -> str:
    """Remove lines that the Python parser would reject (Colab `!pip install`
    shell escapes). Preserves line count so AST line numbers stay aligned."""
    cleaned: list[str] = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!"):
            cleaned.append("# " + line)  # comment it out but keep alignment
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def _extract_top_level_defs(src: str, wanted: set[str]) -> list[str]:
    """Use ast to extract the full source of every top-level FunctionDef
    whose name is in `wanted`. Skips redefinitions silently (keeps the first
    occurrence per name — matches `exec` semantics where last-wins, but in
    the reference the first definition is generally the canonical one)."""
    cleaned = _scrub_colab(src)
    tree = ast.parse(cleaned)
    lines = cleaned.splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted and node.name not in seen:
            seen.add(node.name)
            start = node.lineno - 1
            end = node.end_lineno  # inclusive — Python lineno is 1-based; slice is exclusive
            out.append("\n".join(lines[start:end]))
    return out
