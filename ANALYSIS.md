# Análisis del script original `lattice_matching_02032026.py`

> **Documento canónico** que guía la modularización de moire-flow. Consolida el análisis del Explore agent + los spikes M0.2 (dataflow), M0.3 (dedup) y M0.4 (Docker). Las referencias `file:line` apuntan a `reference/lattice_matching_02032026.py`.

## 1. Arquitectura top-level

- **Origen**: notebook Google Colab auto-convertido a `.py` (12 164 líneas)
- **Modelo de ejecución**: secuencial, top-down, sin `if __name__ == "__main__"`. Cada `# Cell N` se ejecuta in-order
- **Dependencias core**: ASE, NumPy, SciPy, Matplotlib, pandas, SQLite, subprocess→LAMMPS
- **Dependencias Colab** (a eliminar en el refactor):
  - `from google.colab import files` — líneas 24, 313, 1026
  - `!{sys.executable} -m pip install` — línea 29
  - Paths `/content/...` hardcoded — líneas 65, 338, 427–431

### Estado global

| Variable | Inicio | Mutado en |
|---|---|---|
| `validation_results_md` | línea 71 | 30+ sitios, todas las fases |
| `wf` (workflow config) | línea 8848 | recargado de JSON en 11594 |
| `pot_cfg` (potential config) | línea 8863 | escrito y recargado |
| `LAMMPS_EXE`, `LAMMPS_INFO` | líneas 3450–3462 | re-discoverado en 8852, 3524 |

**Decisión del refactor**: todo este estado global desaparece. Las cajas reciben/devuelven objetos Pydantic tipados. El estado del workflow vive en el `WorkflowEngine` (M8), no en globals.

## 2. Las 7 fases del pipeline

| Fase | Propósito | Líneas | Salida principal |
|---|---|---|---|
| 0 | Importar CIFs de HetDB y splittear bilayers | 53–306 | `validation_results_md[case]` con atoms_A/B, sc_vecs |
| 1 | Indexar casos LAMMPS prebuilt | 308–399 | `prebuilt_lammps_cases.json` |
| 2 | Parsear archivos LAMMPS `.data` | 401–911 | `imported_lammps_data_cases` |
| 3a | Cargar materiales de C2DB | 984–1143 | `db` (dict en memoria) |
| 3b | Construir casos sintéticos vía transformación | 2099–2164 | `case_spec` con Alat, Blat transformados |
| 4 | Lattice matching (DE + brute force) | 2365–2456 | `methods[m].best_solution` + atoms |
| 5 | MD supercell prep (5 paths por `source_mode`) | 6419–6641 | `md_status` + cell 3D |
| 6 | Asignación de potenciales | 7116–9108 | `pot_cfg`, archivos `.data` y `.in` |
| 7 | Ejecución LAMMPS + análisis de trayectoria | 9109–12164 | logs, CSVs de gap/vdW/strain |

## 3. Inventario de funciones top-level

213 definiciones `def` a nivel de módulo. Agrupadas por dominio (referencia detallada en notas internas):

- **Geometría**: `extract_sc_vecs` (87), `split_layers_from_z` (97), `tile_*`, `wrap_basis_to_cell` (1731), `transform_basis` (1737), `build_transformed_lattice` (1767), `build_transformed_basis` (1785)
- **Álgebra 2D**: `R` (1219), `S` (1226), `ChangeBasis` (1213), `vec2d_*` (1253–1276)
- **Matching core**: `cost_funcion` (1230), `get_lavects` (1241), `find_scvect_pairs` (1278), `compute_mismatch` (1321), `make_solution_record` (1330), `run_continuous_optimization` (1345), `run_brute_force` (1500), `run_brute_force_rotation_only_reference` (1589)
- **Validación**: `fractional_*` (1823–1854), `pairwise_distance_signature` (1868), `coordination_match_score` (1933), `lattice_metrics` (1947), `structure_validation_metrics` (2176), `validate_case_physics` (2242)
- **I/O**: `parse_lammps_data_file` (557), `load_c2db` (984), `load_materials_db` (1159), `get_material` (1171)
- **MD prep**: `prepare_md_supercells` (6423), `build_layer_supercell` (5897), `_tile` (9220)
- **Potenciales**: `infer_required_elements_from_validation_results_md` (7992), `_prepare_one_potential` (7530), `_build_uff_lj_parameters` (7472)
- **LAMMPS scripts**: `_write_layer_data` (9287), `_write_bilayer_data` (9330), `_write_*_script` (9466–9748)
- **Ejecución LAMMPS**: `run_cmd` (×4: 2912/4173/4613/4950), `_lmp` (9750), `validate_lammps_capabilities` (2995), `run_staged_workflow` (10002)
- **Análisis trayectoria**: `parse_lammps_log` (11601), `get_bilayer_log` (11824), `parse_last_thermo_block` (11225)
- **Plotting**: 5 funciones `plot_*` (2573–2910) — quedan **fuera del scope de cajas**, se agrupan en Support D (Visualization) opcional

## 4. Duplicación (M0.3)

Ver `spike/M0_dedup.md` para el catálogo completo. Resumen:

- `run_cmd` redefinido **4 veces** (2912/4173/4613/4950) — v4 gana. La variante con `shell=True` (v2) es código muerto.
- `_as_list` redefinido **3 veces** (5322/7248/8114) — idénticas
- `_detect_style` (2967) vs `detect_style` (4767) — nombres distintos por error de copy-paste
- Bloque completo de análisis de trayectoria duplicado: las versiones de la línea 11656+ ganan al cargar el módulo y son las que se portan
- Tablas globales (`UFF_LJ`, `TERSOFF_REGISTRY`, `SW_REGISTRY`) inicializadas en múltiples cells

## 5. Configuración hardcoded

Parámetros con valores en línea:

- **Estructura**: `IMPORT_TILE_FACTOR=1.2` (83), `MAX_IMPORT_ATOMS=2000` (84), `DEFAULT_PADDING_CELLS` (5292)
- **Optimización DE**: `popsize=1000`, `maxiter=1000`, `tol_lavect=1e-2`, `angle_min=30.0`, `angle_max=90.0`, `top_k=5` (líneas 1350–1356)
- **Brute force**: `N=8`, `ndiv=21`, `mismatch_tol=0.05`, `min_area=1.0` (líneas 1506–1514)
- **Validación**: `cutoff_scale=1.15`, `k_dist=6` (línea 2176)
- **MD**: `vacuum_z`, `interlayer_z`, `lj_cutoff`, `nvt.{temperature, run_steps, dump_every, timestep}` (todas en `wf` JSON)
- **Tablas**: `UFF_LJ` (30 elementos), `TERSOFF_REGISTRY` (6 pares), `SW_REGISTRY` (4 pares), `ATOMIC_MASS_LOOKUP` (25 elementos — insuficiente, ver §7)

**Decisión del refactor**: cada uno de estos pasa a ser campo de un Pydantic Params model con default + docstring + bounds. Las tablas pasan a constantes en `src/moire_flow/constants/`.

## 6. Estructuras de datos

Ver `spike/M0_dataflow.md` para el detalle por campo. Resumen:

- `validation_results_md[case_name]` es un dict con ~15 claves de nivel superior + nested `methods[m]` con ~10 sub-claves
- Lattices son `ndarray(2,2)` (2D, post-conversión desde celda 3D)
- Basis atoms son `ndarray(N, 2)` en cartesianas
- `MatchingSolution` (en `methods[m].results`) contiene: `s1, s2, theta_deg, cost, mismatch, area, v1, v2, oBlat, angle_deg`
- En la fase MD, las atoms y cell pasan a 3D (`ndarray(N,3)` y `ndarray(3,3)`)

## 7. Side effects identificados

- **File reading**: ASE `read()`, `zipfile.extractall()`, `urlopen()` (C2DB), `json.load()`
- **File writing**: JSON state files, LAMMPS `.data` y `.in`, plots PNG vía `plt.savefig()`
- **Subprocess**: LAMMPS executable, pip install (Colab), QUIP/MACE build attempts
- **Plotting**: `plt.show()` en celdas (líneas 2789–2811, 12046)
- **Network**: `urlopen(DATABASE_URL)` línea 1161 — descarga el snapshot de C2DB de un Gist

**Decisión del refactor**: side effects van a:
- `io/` para reads (cif.py, lammps_data.py, materials_db.py)
- `boxes/lammps_input_writer.py` para writes de archivos LAMMPS
- `executors/lammps.py` para subprocess
- `supports/visualization.py` para plotting (opcional, dep extra)

## 8. Obstáculos para el refactor

1. **Estado global mutable**: `validation_results_md` se muta en 30+ lugares. Solución: cada caja devuelve un objeto Pydantic nuevo; el `WorkflowEngine` (M8) acumula el estado fuera de las cajas.

2. **Funciones monolíticas**:
   - `parse_lammps_data_file` (557–740) — ~180 líneas, parsea header + atoms + masses + velocities + forces en una sola función. Refactor: 4–5 helpers en `io/lammps_data.py`.
   - `prepare_md_supercells` (6423–6641) — 5 ramas según `source_mode`. Refactor: el dispatch lo hace el WorkflowEngine; cada caja maneja UN tipo de input.
   - `run_staged_workflow` (10002–10276) — triple loop. Refactor: el engine itera; cada caja se ejecuta sobre un único case.

3. **Magic numbers sin justificar** (ver §5): pasan a defaults documentados en Pydantic Params.

4. **Parsing frágil** de output LAMMPS (regex en `parse_lammps_log`): mantener pero con tests de regresión contra logs grabados.

5. **Cell geometry 2D→3D**: `parse_lammps_data_file` devuelve `sc_vecs` 2×2, dropping `xz/yz/lz`. El refactor devuelve la matriz 3×3 triclínica completa (ver `core/types.py`).

6. **Element lookup table insuficiente**: `ATOMIC_MASS_LOOKUP` cubre sólo 25 elementos. Faltan Fe, Co, Ni, Cr, V, Bi, Ga, Ge, I, Br, F, Al, K, Li, Na, Ca, Cu, Ag, Au, Sn, Ru, Rh, Pd, Os, Ir, Re, Ta, Nb, Y, Sc, La. Refactor: poblar desde `ase.data.atomic_masses` (~100 elementos).

7. **Workflow JSON no portable**: `wf.mlip_backends.gap_quip.available` se serializa según `HAS_QUIP` del entorno. El refactor separa **WorkflowSpec** (declarativo, portable) de **RuntimeCapabilities** (descubierto al iniciar el executor).

## 9. Descomposición en cajas (final)

Confirmada por M0.2. **9 cajas + 3 supports**.

| # | Caja | Responsabilidad | Fuente original |
|---|---|---|---|
| 1 | `StructureLoader` | CIF / LAMMPS .data / XYZ → `Structure` | 87–740, 518–555 |
| 2 | `BilayerSplitter` | Bilayer → 2 layers + tiling al cutoff LJ | 97–229 |
| 3 | `LatticeTransformer` | Strain + rotación sobre lattice + basis | 1213–1788 |
| 4 | `LatticeMatcher` | DE + brute force search | 1207–1700 |
| 5 | `AtomAssembler` | Solución + basis → BilayerAtoms | 1700–1789, 2166 |
| 6 | `Validator` | Métricas estructurales (RMSD, signatures, etc.) | 1823–2287 |
| 7 | `MDSupercellBuilder` | 2D → 3D con vacuum Z | 6419–6641, 5897–5948 |
| 8 | `PotentialAssigner` | Asignar Tersoff / SW / LJ / GAP / MACE | 7116–8112 |
| 9 | `LammpsInputWriter` | Escribir .data y .in para LAMMPS | 9287–9748 |
| A | `MaterialsDB` | Cliente C2DB (URL o JSON local) | 984–1200 |
| B | `LammpsExecutor` | Subprocess + capability probing | 2912–3050, 9750 |
| C | `TrajectoryAnalyzer` | Parse log + métricas de gap/energía/strain | 11225–12120 |

Detalle de inputs/outputs/params en `ARCHITECTURE.md`.

## 10. Decisiones derivadas de los agentes

- **No fusionar cajas** (skeptic propuso colapsar PotentialAssigner+LammpsInputWriter; descartado en M0.2)
- **Linux/amd64 only** para el Docker runtime (engineer verificó imposibilidad de arm64 con QUIP+MACE)
- **Validator necesita `LayerPair` + `MatchingSolution` + `target`** (no sólo `MatchingSolution`)
- **El short-circuit por `name='real_pair_hfse2_mose2'`** se parametriza como `skip_cases: list[str]` (no hardcoded)
- **Plotting fuera del scope del 9-caja**: va a Support D (opcional, matplotlib como dep extra)
- **Re-discovery de capabilities LAMMPS no se serializa** en el WorkflowSpec — separación entre spec portable y runtime
