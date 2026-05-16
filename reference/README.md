# `reference/` — script original vendored

Contiene el script `.py` original que se está refactorizando. **Frozen**: no se modifica. Sirve como fuente de verdad durante el refactor y permite git blame línea por línea.

- `lattice_matching_02032026.py` — 12 164 líneas, auto-convertido desde un notebook Colab por Camilo el 2026-03-02

## Mapeo de cajas a líneas fuente

| Caja del refactor | Líneas en el original | Funciones clave |
|---|---|---|
| `StructureLoader` | 87–740, 518–555 | `extract_sc_vecs`, `split_layers_from_z`, `parse_lammps_data_file`, `guess_element_from_mass` |
| `BilayerSplitter` | 97–229 | `split_layers_from_z`, `tile_prebuilt_layer`, `normalize_prebuilt_bilayer` |
| `LatticeTransformer` | 1213–1788 | `R`, `S`, `ChangeBasis`, `build_transformed_lattice`, `build_transformed_basis` |
| `LatticeMatcher` | 1207–1700 | `cost_funcion`, `find_scvect_pairs`, `run_continuous_optimization`, `run_brute_force` |
| `AtomAssembler` | 1700–1789, 2166 | `build_atoms_for_best`, `atoms_in_supercell`, `wrap_basis_to_cell` |
| `Validator` | 1823–2287 | `fractional_*`, `coordination_match_score`, `structure_validation_metrics` |
| `MDSupercellBuilder` | 6419–6641, 5897–5948 | `prepare_md_supercells`, `build_layer_supercell`, `_tile` |
| `PotentialAssigner` | 7116–8112 | `_prepare_one_potential`, `_guess_classical_plan_from_chemistry`, `_build_uff_lj_parameters` |
| `LammpsInputWriter` | 9287–9748 | `_write_*_data`, `_write_*_script` |
| Support A — `MaterialsDB` | 984–1200 | `load_materials_db`, `get_material` |
| Support B — `LammpsExecutor` | 2912–3050, 9750 | `run_cmd` (v4 canónica), `_lmp`, `validate_lammps_capabilities` |
| Support C — `TrajectoryAnalyzer` | 11225–12120 | `parse_lammps_log`, `get_bilayer_log`, `parse_last_thermo_block` |

## Forma canónica de funciones duplicadas

Funciones definidas más de una vez en el script — la **última definición** gana al cargar el módulo. Estas son las versiones que se portan:

| Función | Línea canónica | Otras versiones | Notas |
|---|---|---|---|
| `run_cmd` | **4950** | 2912, 4173, 4613 | La variante v2 (4173) con `shell=True` es código muerto; descartar |
| `_as_list` | **8114** | 5322, 7248 | Las tres son idénticas |
| `_is_numeric_label` | **8124** | 7828 | Verificar antes de portar |
| `infer_mode` | **11656** | 11277 | Implementaciones distintas — el bloque 11656+ es el canónico |
| `infer_eq_cut` | **11660** | 11281 | Idem |
| `infer_chemistry_from_species` | **11684** | 11285 | Idem |
| `clean_value` | **11713** | 11314 | Idem |

`_detect_style` (2967) y `detect_style` (4767) tienen **nombres distintos** pero hacen lo mismo — usar la versión con underscore.

Detalle completo en `spike/M0_dedup.md`.

## Exclusión del build

Este directorio se excluye del wheel (`pyproject.toml [tool.hatch.build.targets.sdist] exclude`) y del lint (`pyproject.toml [tool.ruff] exclude`). El contenido está disponible para desarrollo y trazabilidad, no se distribuye con el package.
