# M0.2 — Dataflow audit a nivel de campo

Auditoría de los campos de `case_result` (y su contenedor `validation_results_md`) que cada caja propuesta lee y escribe. Objetivo: confirmar o refutar que el grafo de dependencias entre cajas es un **DAG a nivel de campo**, no sólo a nivel de caja.

El script original muta un único dict `validation_results_md[case_name]` (alias `case_result`) durante todo el pipeline (167 accesos a `case_result`, distribuidos en 7 fases). En el refactor, ese estado se descompone en outputs tipados que viajan entre cajas.

## Esquema completo de `case_result`

Reconstruido por inspección de fuente:

```python
case_result = {
    # Metadata (escrito por las fases de carga/construcción)
    "name": str,
    "A_formula": str,
    "B_formula": str,
    "transform_type": str,
    "transform_spec": dict,
    "target": {"target_theta_deg": float, "target_s1": float, "target_s2": float, ...},
    "source_mode": Literal["prebuilt", "prebuilt_lammps", "imported_lammps_data", "continuous", ...],

    # Inputs algorítmicos (escrito por la fase de construcción del caso)
    "inputs": {
        "Alat": ndarray(2,2),
        "Blat": ndarray(2,2),
        "basis_A": ndarray(N_A, 2),
        "basis_B": ndarray(N_B, 2),
        "elems_A": list[str],
        "bounds": list[tuple[float, float]],
        "dim": int,
        "cont_kwargs": dict,
        "bf_kwargs": dict,
    },

    # Salidas del matcher (por método)
    "methods": {
        "continuous": {
            "results": list[dict],
            "best_solution": {"v1", "v2", "s1", "s2", "theta_deg", "cost", "mismatch", "area", "oBlat"},
            "atoms_A": ndarray(N_sc_A, 2),
            "atoms_B": ndarray(N_sc_B, 2),
            "species_A": list[str],
            "species_B": list[str],
            "sc_vecs": list[list[float]],
            "oBlat": ndarray(2,2),
            "runtime_s": float,
            "validation_metrics": dict,
            "validation": dict,
            # añadido por la fase MD:
            "data_file": str | None,
            "cell": ndarray(3,3),
            "source_mode": str,
        },
        "brute_force": { ... idem ... },
    },
    "summary_rows": list[dict],

    # Añadido por la fase MD-prep
    "md_status": Literal["prebuilt_run_directly", "md_ready", "generic_structure_ready_for_supercell_builder"],
    "data_file": str | None,
    "cell": ndarray(3,3),

    # Añadido por la fase de asignación de potencial
    "potential_assignment": dict,
    "potential_plan": dict,
}
```

## Tabla de lectura/escritura por caja

Para cada caja, los campos de `case_result` que lee y escribe (citando líneas del original):

### Box 1 — StructureLoader
- **Lee**: ninguno (entrada externa: path a archivo CIF / .data)
- **Escribe**: `name`, `methods[continuous].{atoms_A, atoms_B, species_A, species_B, sc_vecs}`, `source_mode = "prebuilt"` (líneas 267–284)
- **Observación**: en el original, escribe **directamente** dentro de `methods[continuous]` aunque sea una carga de estructura cruda. Es un misuso del namespace de métodos. En el refactor, escribe en `Structure` (output Pydantic) y la siguiente caja decide cómo organizarlo.

### Box 2 — BilayerSplitter
- **Lee**: el `Structure` salido de Box 1 (o de un import directo de bilayer)
- **Escribe**: `LayerPair { layer_A, layer_B, sc_vecs, ... }` (output Pydantic)
- **Líneas fuente**: 97–229
- **No toca** `case_result` directamente; opera sobre el output de la Box anterior.

### Box 3 — LatticeTransformer
- **Lee**: `Alat`, `basis_A` (de `case_result.inputs`)
- **Escribe**: `Blat`, `basis_B` (en `case_result.inputs`)
- **Líneas fuente**: 1767–1788, 1785–1788
- **Observación**: en el original esto se hace al **construir el case_spec** (build_synthetic_pair_case, línea 2155), no durante el run. En el refactor, lo separamos como caja independiente para que la UI pueda componer "tomá esta estructura y aplicale rotación + strain".

### Box 4 — LatticeMatcher
- **Lee**: `inputs.{Alat, Blat, bounds, dim, cont_kwargs, bf_kwargs}` (líneas 2210, 2243, 2290, 2401)
- **Escribe**: `methods[m].{results, best_solution, oBlat, runtime_s}` (líneas 2401–2411)
- **Atómico**: read-only sobre `inputs`, escribe en una sola subkey `methods[m]`.

### Box 5 — AtomAssembler
- **Lee**: `inputs.{basis_A, basis_B, Alat, Blat}`, `methods[m].best_solution`, `methods[m].oBlat`
- **Escribe**: `methods[m].{atoms_A, atoms_B, species_A, species_B, sc_vecs}` (líneas 2402–2405)
- **Líneas fuente**: 2166 (build_atoms_for_best), 1700–1789
- **Observación**: en el original, este paso está **fusionado dentro de `run_validation_case`** junto con el matcher (línea 2365). El refactor los separa para componibilidad.

### Box 6 — Validator
- **Lee** (líneas 2176–2240 de `structure_validation_metrics`):
  - `inputs.Blat`
  - `inputs.basis_B`
  - `methods[m].oBlat`
  - `methods[m].runtime_s`
  - `methods[m].best_solution`
  - **`name`** — caso especial line 2207: `if case_result['name'] == 'real_pair_hfse2_mose2': return base`
  - `target.{target_theta_deg, target_s1, target_s2, theta_period_deg, allow_sign_flip}` (líneas 2179, 2243)
- **Escribe**: `methods[m].validation_metrics`, `methods[m].validation` (líneas 2413–2414)
- **Hallazgo crítico**: el Validator **NO es función pura del MatchingSolution**. Requiere:
  - `inputs.basis_B` (del input del matcher, no de su output)
  - `target` (del case_spec original)
  - `name` (para un short-circuit por caso especial)

### Box 7 — MDSupercellBuilder
- **Lee** (líneas 6423–6641 de `prepare_md_supercells`):
  - `source_mode` para decidir branch (5 paths)
  - Para path "native bilayer": `methods[m].{atoms_A, atoms_B, species_A, species_B, sc_vecs}`
  - Para path "prebuilt LAMMPS": el `data_file` original
  - Para path "generic structure": `inputs.basis_A` / atoms del case
  - **`md_status` existente** (puede ya estar pre-set por Box 1 si la carga era prebuilt)
- **Escribe**:
  - `md_status`
  - `methods[m].{cell, data_file, source_mode}` (claves nuevas dentro del namespace de método)
  - A veces reescribe `methods[m].{atoms_A, atoms_B}` si re-tile (línea 6502)
- **Hallazgo**: Box 7 lee de `methods[m]` y **escribe sobre los mismos namespaces** — pero claves diferentes (añade `cell`, `data_file`; modifica `atoms_A/B` por re-tile). Esto es una sub-fusión: lectura del output del AtomAssembler + transformación 2D→3D.

### Box 8 — PotentialAssigner
- **Lee** (líneas 7116–8112):
  - `methods[m].species_A`, `species_B` (qué elementos hay)
  - Tablas globales `UFF_LJ`, `TERSOFF_REGISTRY`, `SW_REGISTRY` (estado externo)
  - `LAMMPS_INFO` (capabilities — sólo determina qué planes son viables)
- **Escribe**:
  - `potential_assignment` (a nivel del case, no por método)
  - `potential_plan` (idem) — líneas 8727–8728

### Box 9 — LammpsInputWriter
- **Lee**:
  - `methods[m].{atoms_A, atoms_B, species_A, species_B, cell, data_file}`
  - `potential_assignment` o `potential_plan` (del Box 8)
  - MD settings de `wf.experiments.nvt` (config externo)
- **Escribe**: archivos en disco (.data, .in). Devuelve `LammpsRunPlan` con paths.

## Resumen: ¿es un DAG a nivel de campo?

```
StructureLoader ─→ Structure ─→ BilayerSplitter ─→ LayerPair
                          ↓                              ↓
                          ↓                       LatticeTransformer ─→ (Alat,Blat,basis_A,basis_B)
                          ↓                                                       ↓
                          └────────────────────→ LatticeMatcher ←──────────────────┘
                                                       ↓
                                              MatchingSolution
                                                       ↓
                          ┌────────────────────────────┤
                          ↓                            ↓
                  AtomAssembler              (parallel branch to Validator)
                          ↓                            ↑
                   BilayerAtoms ─────────────→ Validator (also needs basis_B, target, name)
                          ↓
                  MDSupercellBuilder ─→ MDStructure
                                          ↓
                          ┌───────────────┤
                          ↓               ↓
                  PotentialAssigner    (input to LammpsInputWriter)
                          ↓               ↑
                   PotentialConfig ─→ LammpsInputWriter
                                          ↓
                                     LammpsRunPlan
```

**Es un DAG**, pero con tres aristas "hacia atrás" que el plan original no capturaba:

1. **Validator necesita `basis_B`**: que es input del Matcher, no su output. Hay que pasar `LayerPair` (o un subset) a Validator además del `MatchingSolution`.
2. **Validator necesita `target`**: el "ground truth" del case_spec. En el refactor, Validator acepta `target: ValidationTarget | None`. Si `target is None`, sólo computa métricas absolutas (RMSD, signatures); si está presente, computa también `theta_error_sym_deg`, `s1_error`, `s2_error`.
3. **Validator tiene un short-circuit por nombre** (`if name == 'real_pair_hfse2_mose2'`). En el refactor: parametrizar como `skip_validation_for: list[str] = []` (NO en el case_spec — en los params de la caja).

## Hallazgo sobre acoplamientos sospechosos

### ❓ ¿PotentialAssigner + LammpsInputWriter debe fusionarse?

**No.** El plan/config de potencial es información estructurada (qué pair style, qué archivo, qué cutoff) que puede ser:
- Inspeccionada por el usuario en la UI antes de escribir nada
- Persistida como configuración reusable entre cases
- Modificada (override) antes de pasar al writer

Mantener cajas separadas. **El acoplamiento es de tipo "el output de X es el input de Y" — eso ES DAG**. Lo que sería problemático es si el Writer modificara el config plan, lo cual no hace.

### ❓ ¿BilayerSplitter + AtomAssembler debe fusionarse?

**No.** Operan sobre dominios distintos:
- BilayerSplitter: bilayer cargado → 2 monolayers (decisión sobre Z midpoint)
- AtomAssembler: una `MatchingSolution` + basis → poblar el supercell

Estos son procesos completamente distintos en el original (los caracteriza distinta data: BilayerSplitter trata coordenadas 3D, AtomAssembler trata lattice + basis). El plan tenía razón en separarlos.

### ❓ ¿AtomAssembler + Validator debe fusionarse?

**Probablemente no, pero hay que matizar.** En el original están fusionados dentro de `run_validation_case` (línea 2365), pero conceptualmente son distintos: AtomAssembler **construye** la bilayer, Validator la **evalúa**. Mantener separados permite a la UI conectar "Matcher → Assembler → Mi-Custom-Validator" en vez del Validator default.

## Decisión final sobre count de cajas

**Se mantienen 9 cajas + 3 supports.** El audit no encontró ciclos ni acoplamientos que justifiquen fusión. Las tres dependencias "hacia atrás" del Validator se resuelven extendiendo su input contract (acepta `LayerPair` + `MatchingSolution` + `target` + `params.skip_for: list[str]`), no fusionando cajas.

## Implicaciones para M1 (concretas)

1. **`Validator.inputs_schema`** debe declarar `layer_pair: LayerPair`, `solution: MatchingSolution`, `target: ValidationTarget | None`. NO sólo `MatchingSolution`.
2. **`Validator.params_schema`** debe incluir `skip_cases: list[str] = []` para reemplazar el short-circuit hardcoded.
3. **`StructureLoader` output debe ser `Structure`** (estructura cruda), NO `LayerPair`. El BilayerSplitter es quien decide si es bicapa o monocapa.
4. **El campo `name` deja de ser propiedad del case_result global**. Cada caja recibe los inputs necesarios sin saber el "nombre del caso" — esa información vive en el WorkflowEngine (M8).
5. **`source_mode` desaparece** del modelo de datos. Es un discriminador de Colab; en el refactor cada caja recibe un tipo concreto (Structure vs LayerPair vs MDStructure) y el dispatch es por tipo, no por flag.
6. **El short-circuit `validate_case_physics`** (línea 2207) se documenta como código heredado: ¿por qué `real_pair_hfse2_mose2` se saltaba la validación? Hay que preguntar al autor original (Camilo) o asumir que es un caso patológico del fixture. En el refactor el comportamiento se hace explícito vía params, no oculto en el código.

## Veredicto del Gate M0.2

✅ **Pasa.** El dataflow es DAG a nivel de campo. Las tres dependencias del Validator se modelan explícitamente como inputs adicionales en su schema. Las 9 cajas se mantienen como propuestas. Se procede a M1 sin reagrupamientos.
