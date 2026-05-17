"""LammpsInputWriter: write the 6-file LAMMPS run bundle.

Takes an MDStructure (already 3D with vacuum) + a PotentialPlan (chosen
pair styles + params) and writes:
- `<work_dir>/layer_A.data`, `<work_dir>/layer_B.data`, `<work_dir>/bilayer.data`
- `<work_dir>/layer_A.in`, `<work_dir>/layer_B.in`, `<work_dir>/bilayer.in`

Returns a `LammpsRunPlan` with all paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from moire_flow.core.types import LammpsRunPlan, MDStructure, PotentialPlan
from moire_flow.io.lammps_writer import (
    format_bilayer_data,
    format_bilayer_lj_script,
    format_intralayer_lj_script,
    format_layer_data,
    format_sw_script,
    format_tersoff_script,
    write_text_atomic,
)

from .base import Box, register_box


class LammpsInputWriterInputs(BaseModel):
    md_structure: MDStructure
    potential_plan: PotentialPlan
    species_A: list[str]
    species_B: list[str]
    n_atoms_A: int


class LammpsInputWriterParams(BaseModel):
    output_dir: Path
    temperature_K: float = 300.0
    nvt_steps: int = Field(default=0, ge=0)
    timestep_ps: float = Field(default=0.0005, gt=0.0)
    dump_every: int = Field(default=100, ge=1)
    minimize_first: bool = True


def _script_for_intralayer(
    plan: dict,
    data_file_name: str,
    type_to_element: dict[int, str],
    params: LammpsInputWriterParams,
    prefix: str,
) -> str:
    kind = plan["kind"]
    if kind == "tersoff":
        return format_tersoff_script(
            data_file_name,
            plan["file"],
            type_to_element,
            temperature_K=params.temperature_K,
            nvt_steps=params.nvt_steps,
            timestep_ps=params.timestep_ps,
            dump_every=params.dump_every,
            prefix=prefix,
        )
    if kind == "sw":
        return format_sw_script(
            data_file_name,
            plan["file"],
            type_to_element,
            temperature_K=params.temperature_K,
            nvt_steps=params.nvt_steps,
            timestep_ps=params.timestep_ps,
            dump_every=params.dump_every,
            prefix=prefix,
        )
    if kind == "lj":
        return format_intralayer_lj_script(
            data_file_name,
            type_to_element,
            plan,
            cutoff=plan.get("cutoff", 8.0),
            temperature_K=params.temperature_K,
            nvt_steps=params.nvt_steps,
            timestep_ps=params.timestep_ps,
            dump_every=params.dump_every,
            prefix=prefix,
        )
    raise ValueError(f"Unknown intralayer kind: {kind}")


@register_box
class LammpsInputWriter(
    Box[LammpsInputWriterInputs, LammpsInputWriterParams, LammpsRunPlan]
):
    name = "lammps_input_writer"
    description = "Write LAMMPS .data and .in files for layer_A, layer_B, and bilayer."
    inputs_schema = LammpsInputWriterInputs
    params_schema = LammpsInputWriterParams
    outputs_schema = LammpsRunPlan

    def run(
        self,
        inputs: LammpsInputWriterInputs,
        params: LammpsInputWriterParams,
    ) -> LammpsRunPlan:
        md = inputs.md_structure
        plan = inputs.potential_plan
        atoms = np.asarray(md.atoms, dtype=np.float64)
        n_A = inputs.n_atoms_A
        atoms_A_2d = atoms[:n_A, :2]
        atoms_B_2d = atoms[n_A:, :2]

        sc_vecs = md.cell[:2, :2]
        vacuum_z = md.vacuum_z
        dz = float(md.interlayer_z or 3.3)

        work_dir = Path(params.output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # --- Layer A
        layer_A_text, type_to_element_A, _ = format_layer_data(
            atoms_A_2d, list(inputs.species_A), sc_vecs, vacuum_z,
            mol_id=1, z=0.0, title="Layer A",
        )
        layer_A_data = write_text_atomic(work_dir / "layer_A.data", layer_A_text)
        layer_A_script_text = _script_for_intralayer(
            plan.intralayer_A, layer_A_data.name, type_to_element_A, params, prefix="layer_A"
        )
        layer_A_script = write_text_atomic(work_dir / "layer_A.in", layer_A_script_text)

        # --- Layer B
        layer_B_text, type_to_element_B, _ = format_layer_data(
            atoms_B_2d, list(inputs.species_B), sc_vecs, vacuum_z,
            mol_id=1, z=0.0, title="Layer B",
        )
        layer_B_data = write_text_atomic(work_dir / "layer_B.data", layer_B_text)
        layer_B_script_text = _script_for_intralayer(
            plan.intralayer_B, layer_B_data.name, type_to_element_B, params, prefix="layer_B"
        )
        layer_B_script = write_text_atomic(work_dir / "layer_B.in", layer_B_script_text)

        # --- Bilayer
        bilayer_text, type_to_element_bi, type_to_layer_bi, _, area = format_bilayer_data(
            atoms_A_2d, list(inputs.species_A),
            atoms_B_2d, list(inputs.species_B),
            sc_vecs, vacuum_z, dz, title="Bilayer",
        )
        bilayer_data = write_text_atomic(work_dir / "bilayer.data", bilayer_text)
        bilayer_script_text = format_bilayer_lj_script(
            bilayer_data.name,
            type_to_element_bi,
            type_to_layer_bi,
            cutoff=plan.interlayer.get("cutoff", 8.0),
            temperature_K=params.temperature_K,
            nvt_steps=params.nvt_steps,
            timestep_ps=params.timestep_ps,
            dump_every=params.dump_every,
            prefix="bilayer",
            area_ang2=float(area),
        )
        bilayer_script = write_text_atomic(work_dir / "bilayer.in", bilayer_script_text)

        return LammpsRunPlan(
            layer_A_data=layer_A_data,
            layer_B_data=layer_B_data,
            bilayer_data=bilayer_data,
            layer_A_script=layer_A_script,
            layer_B_script=layer_B_script,
            bilayer_script=bilayer_script,
            work_dir=work_dir,
        )


__all__ = [
    "LammpsInputWriter",
    "LammpsInputWriterInputs",
    "LammpsInputWriterParams",
]
