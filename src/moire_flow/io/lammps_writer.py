"""LAMMPS `.data` + `.in` script generation.

All functions produce strings. The single file write is at the caller's
boundary so we can unit-test the output without touching the filesystem.
Port of reference 9193-9742 with the following decompositions:

- `box_from_sc_vecs`: pure geometry (replaces `_box`)
- `format_layer_data`, `format_bilayer_data`: data-file string builders
- `format_tersoff_script`, `format_sw_script`, `format_bilayer_lj_script`:
  script string builders
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np

from moire_flow.constants.potentials import UFF_LJ, atomic_mass, atomic_number


def box_from_sc_vecs(
    sc_vecs: np.ndarray, vacuum_z: float, slab_h: float
) -> tuple[tuple[float, ...], np.ndarray, float]:
    """Triclinic LAMMPS box from 2D supercell vectors.

    Returns `((xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz), rot2x2, area)`.
    """
    v1 = np.asarray(sc_vecs[0], dtype=np.float64)
    v2 = np.asarray(sc_vecs[1], dtype=np.float64)
    ax = float(np.linalg.norm(v1))
    if ax <= 1e-12:
        raise ValueError("First supercell vector has zero length")
    ex = v1 / ax
    bx = float(np.dot(v2, ex))
    by = float(np.sqrt(max(float(np.dot(v2, v2)) - bx**2, 0.0)))
    rot = np.array([[ex[0], ex[1]], [-ex[1], ex[0]]], dtype=np.float64)
    xlo = min(0.0, bx)
    xhi = ax + max(0.0, bx)
    box = (
        xlo, xhi,
        0.0, by,
        -vacuum_z / 2.0,
        slab_h + vacuum_z / 2.0,
        bx, 0.0, 0.0,
    )
    return box, rot, ax * by


def _format_box_block(box: tuple[float, ...]) -> list[str]:
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = box
    lines = [
        f"{xlo:.8f} {xhi:.8f} xlo xhi",
        f"{ylo:.8f} {yhi:.8f} ylo yhi",
        f"{zlo:.8f} {zhi:.8f} zlo zhi",
    ]
    if abs(xy) > 1e-10 or abs(xz) > 1e-10 or abs(yz) > 1e-10:
        lines.append(f"{xy:.8f} {xz:.8f} {yz:.8f} xy xz yz")
    return lines


def format_layer_data(
    atoms2d: np.ndarray,
    species: list[str],
    sc_vecs: np.ndarray,
    vacuum_z: float,
    mol_id: int = 1,
    z: float = 0.0,
    title: str = "Layer",
) -> tuple[str, dict[int, str], np.ndarray]:
    """Build the text of a LAMMPS `.data` file for a single layer.

    Returns `(text, type_to_element, rot2x2)`.
    """
    atoms2d = np.asarray(atoms2d, dtype=np.float64)
    if len(atoms2d) != len(species):
        raise ValueError(f"atoms/species mismatch: {len(atoms2d)} vs {len(species)}")
    unique_elements = sorted(set(species))
    e2t = {e: i + 1 for i, e in enumerate(unique_elements)}
    type_to_element = {v: k for k, v in e2t.items()}

    box, rot, _ = box_from_sc_vecs(sc_vecs, vacuum_z, slab_h=3.5)
    rotated = atoms2d @ rot.T

    lines = [
        f"# {title}",
        "",
        f"{len(atoms2d)} atoms",
        f"{len(unique_elements)} atom types",
        "",
    ]
    lines.extend(_format_box_block(box))
    lines.extend(["", "Masses", ""])
    for t in sorted(type_to_element):
        el = type_to_element[t]
        lines.append(f"  {t}  {atomic_mass(el):.4f}  # {el}")
    lines.extend(["", "Atoms  # full", ""])
    for idx, (xy2, sp) in enumerate(zip(rotated, species), 1):
        tid = e2t[sp]
        lines.append(
            f"  {idx}  {mol_id}  {tid}  0.0  "
            f"{xy2[0]:.8f}  {xy2[1]:.8f}  {z:.8f}"
        )
    return "\n".join(lines) + "\n", type_to_element, rot


def format_bilayer_data(
    atoms_A_2d: np.ndarray,
    species_A: list[str],
    atoms_B_2d: np.ndarray,
    species_B: list[str],
    sc_vecs: np.ndarray,
    vacuum_z: float,
    dz: float,
    title: str = "Bilayer",
) -> tuple[str, dict[int, str], dict[int, int], np.ndarray, float]:
    """Build the text of a bilayer `.data` file.

    Returns `(text, type_to_element, type_to_layer, rot2x2, area)`.
    """
    atoms_A_2d = np.asarray(atoms_A_2d, dtype=np.float64)
    atoms_B_2d = np.asarray(atoms_B_2d, dtype=np.float64)
    if len(atoms_A_2d) != len(species_A):
        raise ValueError(f"Layer A atoms/species mismatch: {len(atoms_A_2d)} vs {len(species_A)}")
    if len(atoms_B_2d) != len(species_B):
        raise ValueError(f"Layer B atoms/species mismatch: {len(atoms_B_2d)} vs {len(species_B)}")

    uA = sorted(set(species_A))
    uB = sorted(set(species_B))
    tmap: dict[tuple[str, int], int] = {}
    tid = 1
    for e in uA:
        tmap[(e, 1)] = tid
        tid += 1
    for e in uB:
        tmap[(e, 2)] = tid
        tid += 1
    type_to_element = {v: k[0] for k, v in tmap.items()}
    type_to_layer = {v: k[1] for k, v in tmap.items()}

    box, rot, area = box_from_sc_vecs(sc_vecs, vacuum_z, slab_h=dz + 3.5)

    pos_A_3d = np.column_stack([atoms_A_2d, np.zeros(len(atoms_A_2d))])
    pos_B_3d = np.column_stack([atoms_B_2d, np.full(len(atoms_B_2d), dz)])
    pos = np.vstack([pos_A_3d, pos_B_3d])
    pos_r = pos.copy()
    pos_r[:, :2] = pos[:, :2] @ rot.T

    tids = [tmap[(s, 1)] for s in species_A] + [tmap[(s, 2)] for s in species_B]
    lids = [1] * len(atoms_A_2d) + [2] * len(atoms_B_2d)

    lines = [
        f"# {title}",
        "",
        f"{len(pos)} atoms",
        f"{len(type_to_element)} atom types",
        "",
    ]
    lines.extend(_format_box_block(box))
    lines.extend(["", "Masses", ""])
    for t in sorted(type_to_element):
        el = type_to_element[t]
        lines.append(
            f"  {t}  {atomic_mass(el):.4f}  # {el} (L{type_to_layer[t]})"
        )
    lines.extend(["", "Atoms  # full", ""])
    for idx, (p, t, l) in enumerate(zip(pos_r, tids, lids), 1):
        lines.append(
            f"  {idx}  {l}  {t}  0.0  "
            f"{p[0]:.8f}  {p[1]:.8f}  {p[2]:.8f}"
        )
    return "\n".join(lines) + "\n", type_to_element, type_to_layer, rot, area


def format_tersoff_script(
    data_file: str,
    tersoff_file: str,
    type_to_element: dict[int, str],
    *,
    temperature_K: float = 300.0,
    nvt_steps: int = 0,
    timestep_ps: float = 0.0005,
    dump_every: int = 100,
    prefix: str = "out",
) -> str:
    elems = " ".join(type_to_element[i] for i in range(1, len(type_to_element) + 1))
    nvt_block = "" if nvt_steps <= 0 else textwrap.dedent(f"""
        reset_timestep  0
        velocity        all create {temperature_K:.0f} 42 dist gaussian mom yes rot yes
        fix             nvt all nvt temp {temperature_K} {temperature_K} 0.1
        timestep        {timestep_ps}
        dump            traj all custom {dump_every} {prefix}_traj.lammpstrj id type mol x y z
        run             {nvt_steps}
        unfix           nvt
        undump          traj
    """)
    return textwrap.dedent(f"""\
        units           metal
        atom_style      full
        boundary        p p f

        read_data       {data_file}

        pair_style      tersoff
        pair_coeff      * * {tersoff_file} {elems}

        neighbor        2.0 bin
        neigh_modify    every 1 delay 0 check yes

        thermo          {dump_every}
        thermo_style    custom step temp pe ke etotal press

        min_style       cg
        minimize        1e-4 1e-6 500 2000

        write_data      {prefix}_minimised.data
    """) + nvt_block


def format_sw_script(
    data_file: str,
    sw_file: str,
    type_to_element: dict[int, str],
    *,
    temperature_K: float = 300.0,
    nvt_steps: int = 0,
    timestep_ps: float = 0.0005,
    dump_every: int = 100,
    prefix: str = "out",
) -> str:
    elems = " ".join(type_to_element[i] for i in range(1, len(type_to_element) + 1))
    nvt_block = "" if nvt_steps <= 0 else textwrap.dedent(f"""
        reset_timestep  0
        velocity        all create {temperature_K:.0f} 42 dist gaussian mom yes rot yes
        fix             nvt all nvt temp {temperature_K} {temperature_K} 0.1
        timestep        {timestep_ps}
        dump            traj all custom {dump_every} {prefix}_traj.lammpstrj id type mol x y z
        run             {nvt_steps}
        unfix           nvt
        undump          traj
    """)
    return textwrap.dedent(f"""\
        units           metal
        atom_style      full
        boundary        p p f

        read_data       {data_file}

        pair_style      sw
        pair_coeff      * * {sw_file} {elems}

        neighbor        2.0 bin
        neigh_modify    every 1 delay 0 check yes

        thermo          {dump_every}
        thermo_style    custom step temp pe ke etotal press

        min_style       cg
        minimize        1e-4 1e-6 500 2000

        write_data      {prefix}_minimised.data
    """) + nvt_block


def format_gap_quip_script(
    data_file: str,
    gap_file: str,
    type_to_element: dict[int, str],
    *,
    quip_init: str = "Potential xml_label=GAP",
    temperature_K: float = 300.0,
    nvt_steps: int = 0,
    timestep_ps: float = 0.0005,
    dump_every: int = 100,
    prefix: str = "out",
) -> str:
    """GAP/QUIP intralayer script.

    LAMMPS syntax:
        pair_style quip
        pair_coeff * * gap.xml "Potential xml_label=..." Z1 Z2 ...

    Port of reference `_write_gap_quip_script` (9556-9611).
    """
    zmap = [str(atomic_number(type_to_element[i])) for i in range(1, len(type_to_element) + 1)]
    nvt_block = "" if nvt_steps <= 0 else textwrap.dedent(f"""
        reset_timestep  0
        velocity        all create {temperature_K:.0f} 42 dist gaussian mom yes rot yes
        fix             nvt all nvt temp {temperature_K} {temperature_K} 0.1
        timestep        {timestep_ps}
        dump            traj all custom {dump_every} {prefix}_traj.lammpstrj id type mol x y z
        run             {nvt_steps}
        unfix           nvt
        undump          traj
    """)
    return textwrap.dedent(f"""\
        units           metal
        atom_style      full
        boundary        p p f

        read_data       {data_file}

        pair_style      quip
        pair_coeff      * * {gap_file} "{quip_init}" {' '.join(zmap)}

        neighbor        2.0 bin
        neigh_modify    every 1 delay 0 check yes

        thermo          {dump_every}
        thermo_style    custom step temp pe ke etotal press

        min_style       cg
        minimize        1e-4 1e-6 500 2000

        write_data      {prefix}_minimised.data
    """) + nvt_block


def format_mace_script(
    data_file: str,
    mace_file: str,
    type_to_element: dict[int, str],
    *,
    flavor: str = "mliap",  # "mliap" → pair_style mliap mliappy (default, works with upstream LAMMPS)
    temperature_K: float = 300.0,
    nvt_steps: int = 0,
    timestep_ps: float = 0.0005,
    dump_every: int = 100,
    prefix: str = "out",
) -> str:
    """MACE intralayer script.

    Two flavors map to two LAMMPS pair styles:
        flavor="mliap" → pair_style mliap model mliappy <model> descriptor mliappy <model>
                         (default; works with upstream LAMMPS + ML-IAP + mliappy)
        flavor="mace"  → pair_style mace; pair_coeff * * model.lammps.pt Mo S
                         (requires the third-party ACEsuit/mace_lammps_plugin)

    Port of the reference MACE flow (ensure_mace 3862-3905). The original
    notebook does NOT have a dedicated _write_mace_script — the MACE branch
    reused `_write_tersoff_script` with a swapped pair_style line. We
    separate it here for clarity.
    """
    elems = " ".join(type_to_element[i] for i in range(1, len(type_to_element) + 1))
    if flavor == "mace":
        pair_lines = (
            f"pair_style      mace\n"
            f"        pair_coeff      * * {mace_file} {elems}"
        )
    elif flavor == "mliap":
        pair_lines = (
            f"pair_style      mliap model mliappy {mace_file} descriptor mliappy {mace_file}\n"
            f"        pair_coeff      * * {elems}"
        )
    else:
        raise ValueError(f"Unknown MACE flavor: {flavor!r} (expected 'mace' or 'mliap')")

    nvt_block = "" if nvt_steps <= 0 else textwrap.dedent(f"""
        reset_timestep  0
        velocity        all create {temperature_K:.0f} 42 dist gaussian mom yes rot yes
        fix             nvt all nvt temp {temperature_K} {temperature_K} 0.1
        timestep        {timestep_ps}
        dump            traj all custom {dump_every} {prefix}_traj.lammpstrj id type mol x y z
        run             {nvt_steps}
        unfix           nvt
        undump          traj
    """)
    return textwrap.dedent(f"""\
        units           metal
        atom_style      full
        boundary        p p f

        read_data       {data_file}

        {pair_lines}

        neighbor        2.0 bin
        neigh_modify    every 1 delay 0 check yes

        thermo          {dump_every}
        thermo_style    custom step temp pe ke etotal press

        min_style       cg
        minimize        1e-4 1e-6 500 2000

        write_data      {prefix}_minimised.data
    """) + nvt_block


def format_bilayer_lj_script(
    data_file: str,
    type_to_element: dict[int, str],
    type_to_layer: dict[int, int],
    *,
    cutoff: float = 8.0,
    temperature_K: float = 300.0,
    nvt_steps: int = 1000,
    timestep_ps: float = 0.00025,
    dump_every: int = 100,
    epsilon_scale: float = 1.0,
    prefix: str = "out",
    area_ang2: float = 1.0,
) -> str:
    """Bilayer LJ script with intralayer pairs zeroed and interlayer vdW active."""
    pairs: list[str] = []
    n_types = len(type_to_element)
    missing = sorted({type_to_element[i] for i in range(1, n_types + 1) if type_to_element[i] not in UFF_LJ})
    for i in range(1, n_types + 1):
        for j in range(i, n_types + 1):
            ei, ej = type_to_element[i], type_to_element[j]
            li, lj = type_to_layer[i], type_to_layer[j]
            if li != lj and ei in UFF_LJ and ej in UFF_LJ:
                eps = math.sqrt(UFF_LJ[ei]["epsilon"] * UFF_LJ[ej]["epsilon"]) * epsilon_scale
                sig = 0.5 * (UFF_LJ[ei]["sigma"] + UFF_LJ[ej]["sigma"])
                pairs.append(
                    f"pair_coeff  {i}  {j}  {eps:.8f}  {sig:.4f}  {cutoff:.1f}"
                    f"  # {ei}(L{li})-{ej}(L{lj}) INTERLAYER vdW scale={epsilon_scale}"
                )
            else:
                pairs.append(
                    f"pair_coeff  {i}  {j}  0.0  1.0  {cutoff:.1f}"
                    f"  # zeroed — {ei}(L{li})-{ej}(L{lj})"
                )
    pair_block = "\n".join(pairs)
    warn = ""
    if missing:
        warn = f"\n# WARNING: missing LJ parameters for: {sorted(missing)}\n"
    return textwrap.dedent(f"""\
        units           metal
        atom_style      full
        boundary        p p f
        {warn}
        read_data       {data_file}

        pair_style      lj/cut {cutoff:.1f}
        {pair_block}

        neighbor        2.0 bin
        neigh_modify    every 1 delay 0 check yes

        group           layer1 molecule 1
        group           layer2 molecule 2

        compute         z1        layer1 reduce ave z
        compute         z2        layer2 reduce ave z
        variable        dz        equal c_z2-c_z1
        compute         vdW       all pair lj/cut
        variable        vdW_per_A equal c_vdW/{area_ang2:.6f}

        thermo          {dump_every}
        thermo_style    custom step temp pe ke etotal v_dz c_vdW v_vdW_per_A press
        thermo_modify   norm no

        min_style       cg
        minimize        1e-4 1e-6 500 2000

        reset_timestep  0
        velocity        all create {temperature_K:.0f} 42 dist gaussian mom yes rot yes
        fix             nvt all nvt temp {temperature_K} {temperature_K} 0.1
        timestep        {timestep_ps}

        dump            traj all custom {dump_every} {prefix}_traj.lammpstrj id type mol x y z
        run             {nvt_steps}
        unfix           nvt
        undump          traj

        write_data      {prefix}_final.data
    """)


def format_intralayer_lj_script(
    data_file: str,
    type_to_element: dict[int, str],
    lj_plan: dict,
    *,
    cutoff: float = 8.0,
    temperature_K: float = 300.0,
    nvt_steps: int = 0,
    timestep_ps: float = 0.0005,
    dump_every: int = 100,
    prefix: str = "out",
) -> str:
    """Single-layer LJ script (fallback when no Tersoff/SW available)."""
    pairs: list[str] = []
    n_types = len(type_to_element)
    for i in range(1, n_types + 1):
        for j in range(i, n_types + 1):
            ei, ej = type_to_element[i], type_to_element[j]
            key = f"{ei}-{ej}"
            if key not in lj_plan["pairs"]:
                key = f"{ej}-{ei}"
            params = lj_plan["pairs"].get(key)
            if params is None:
                pairs.append(f"pair_coeff  {i}  {j}  0.0  1.0  {cutoff:.1f}  # missing")
                continue
            pairs.append(
                f"pair_coeff  {i}  {j}  {params['epsilon']:.8f}  {params['sigma']:.4f}  {cutoff:.1f}  # {ei}-{ej}"
            )
    pair_block = "\n".join(pairs)
    nvt_block = "" if nvt_steps <= 0 else textwrap.dedent(f"""
        reset_timestep  0
        velocity        all create {temperature_K:.0f} 42 dist gaussian mom yes rot yes
        fix             nvt all nvt temp {temperature_K} {temperature_K} 0.1
        timestep        {timestep_ps}
        dump            traj all custom {dump_every} {prefix}_traj.lammpstrj id type mol x y z
        run             {nvt_steps}
        unfix           nvt
        undump          traj
    """)
    return textwrap.dedent(f"""\
        units           metal
        atom_style      full
        boundary        p p f

        read_data       {data_file}

        pair_style      lj/cut {cutoff:.1f}
        {pair_block}

        neighbor        2.0 bin
        neigh_modify    every 1 delay 0 check yes

        thermo          {dump_every}
        thermo_style    custom step temp pe ke etotal press

        min_style       cg
        minimize        1e-4 1e-6 500 2000

        write_data      {prefix}_minimised.data
    """) + nvt_block


def write_text_atomic(path: Path, text: str) -> Path:
    """Write text to `path`, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


__all__ = [
    "box_from_sc_vecs",
    "format_layer_data",
    "format_bilayer_data",
    "format_tersoff_script",
    "format_sw_script",
    "format_gap_quip_script",
    "format_mace_script",
    "format_bilayer_lj_script",
    "format_intralayer_lj_script",
    "write_text_atomic",
]
