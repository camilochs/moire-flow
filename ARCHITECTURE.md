# Arquitectura — moire-flow

> Diseño de las 9 cajas + 3 supports que conforman moire-flow. Confirmado por el dataflow audit `spike/M0_dataflow.md`. Cada caja es una unidad ejecutable independiente con I/O tipada (Pydantic v2 + numpydantic), serializable a JSON para el futuro web UI (M9).

## Principios

1. **Cajas puras**: la función `run(inputs, params) -> output` no muta nada externo. Sin `print`, sin `plt.show`, sin globals.
2. **Side effects en adapters**: I/O en `src/moire_flow/io/`, subprocess en `src/moire_flow/executors/`. Las cajas reciben los objetos ya cargados; los adapters se ejecutan **fuera** del `run()`.
3. **Schemas serializables**: `inputs_schema`, `params_schema`, `outputs_schema` son `BaseModel` válidos para `model_json_schema()`. La web UI los consume vía `BOX_REGISTRY`.
4. **Composabilidad por tipo**: si la caja A devuelve `Structure` y la caja B requiere `Structure`, son compatibles. Validación en tiempo de diseño en la UI.
5. **Sin estado de caso**: ninguna caja sabe el "nombre del case" ni el `source_mode`. El `WorkflowEngine` (M8) orquesta múltiples cases pero las cajas son single-shot.

## Tipos de datos compartidos (`src/moire_flow/core/types.py`)

```python
# Tipos numpy con shape validation vía numpydantic
NDArray2D       # (N, 2) float64
NDArray3D       # (N, 3) float64
NDArray2x2      # (2, 2) float64
NDArray3x3      # (3, 3) float64

class Structure(BaseModel):
    """Carga cruda de un archivo estructural. Puede ser monolayer o bilayer."""
    atoms: NDArray3D
    species: list[str]
    cell: NDArray3x3
    energy: float | None = None
    forces: NDArray3D | None = None

class LayerPair(BaseModel):
    layer_A: Structure
    layer_B: Structure
    sc_vecs: NDArray2x2          # supercell común compartido por ambas
    replication: tuple[int, int]
    z_mean_A: float
    z_mean_B: float

class MatchingSolution(BaseModel):
    v1: NDArray2x2               # supercell vec 1 (2D)
    v2: NDArray2x2
    s1: float                    # strain factor 1
    s2: float
    theta_deg: float             # rotación
    cost: float
    mismatch: float              # mismatch residual del supercell
    area: float                  # |v1 x v2|
    oBlat: NDArray2x2            # B lattice rotated + strained

class BilayerAtoms(BaseModel):
    atoms_A: NDArray2D           # 2D coords en el supercell
    atoms_B: NDArray2D
    species_A: list[str]
    species_B: list[str]
    sc_vecs: NDArray2x2

class ValidationTarget(BaseModel):
    """Ground truth opcional para casos sintéticos."""
    target_theta_deg: float | None = None
    target_s1: float | None = None
    target_s2: float | None = None
    theta_period_deg: float = 60.0
    allow_sign_flip: bool = True

class ValidationMetrics(BaseModel):
    theta_error_sym_deg: float | None = None
    s1_error: float | None = None
    s2_error: float | None = None
    fractional_rmsd: float
    distance_signature_error: float
    coordination_match: float
    passes: dict[str, bool]     # e.g., {"theta_ok": True, "rmsd_ok": True}

class MDStructure(BaseModel):
    """Estructura 2D promovida a 3D con vacuum Z para LAMMPS."""
    atoms: NDArray3D
    species: list[str]
    cell: NDArray3x3            # triclínico completo
    vacuum_z: float
    interlayer_z: float | None = None
    data_file: Path | None = None  # path si ya existe LAMMPS .data preservada

class PotentialPlan(BaseModel):
    intralayer_A: dict          # kind, file, params
    intralayer_B: dict
    interlayer: dict            # always LJ for vdW

class LammpsRunPlan(BaseModel):
    """Output de Box 9 — paths a archivos generados."""
    layer_A_data: Path
    layer_B_data: Path
    bilayer_data: Path
    layer_A_script: Path
    layer_B_script: Path
    bilayer_script: Path
    work_dir: Path
```

## Box ABC (`src/moire_flow/boxes/base.py`)

```python
from typing import ClassVar, Generic, TypeVar
from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
ParamsT = TypeVar("ParamsT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

class Box(Generic[InputT, ParamsT, OutputT]):
    name: ClassVar[str]                          # e.g., "structure_loader"
    inputs_schema: ClassVar[type[BaseModel]]
    params_schema: ClassVar[type[BaseModel]]
    outputs_schema: ClassVar[type[BaseModel]]
    description: ClassVar[str]

    def run(self, inputs: InputT, params: ParamsT) -> OutputT: ...

BOX_REGISTRY: dict[str, type[Box]] = {}

def register_box(cls):
    BOX_REGISTRY[cls.name] = cls
    return cls
```

## Diagrama de flujo

```
        ┌─────────────┐
        │  CIF file   │
        │  .data file │
        └──────┬──────┘
               │
       ┌───────▼────────┐
       │ StructureLoader│ ──► Structure
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ BilayerSplitter│ ──► LayerPair (si era bilayer)
       └───────┬────────┘
               │
   ┌───────────┴─────────────────────┐
   │                                 │
┌──▼────────────┐               ┌────▼──────────┐
│LatticeTransfo │               │ MaterialsDB   │
│-rmer (opc.)   │               │ (Support A)   │
└──┬────────────┘               └───────────────┘
   │
   ▼ (Alat, Blat, basis_A, basis_B)
┌───────────────┐
│ LatticeMatcher│ ──► MatchingSolution
└──┬────────────┘
   │
   ├──────────────────────────────────┐
   │                                  │
┌──▼──────────┐                    ┌──▼──────────────────┐
│AtomAssembler│ ──► BilayerAtoms ──┤ Validator           │
└──┬──────────┘                    │ (also reads LayerPair│
   │                               │  + ValidationTarget) │
   │                               └──────────────────────┘
┌──▼──────────────────┐
│ MDSupercellBuilder  │ ──► MDStructure
└──┬──────────────────┘
   │
   ├──────────────────────────┐
   │                          │
┌──▼─────────────────┐    ┌──▼─────────────────────┐
│ PotentialAssigner  │ ─► │ LammpsInputWriter      │ ─► LammpsRunPlan
└────────────────────┘    └──┬─────────────────────┘
                             │
                       ┌─────▼──────────────┐
                       │ LammpsExecutor     │ (Support B)
                       └─────┬──────────────┘
                             │ logs / dumps
                       ┌─────▼──────────────┐
                       │ TrajectoryAnalyzer │ (Support C) ─► metrics CSV
                       └────────────────────┘
```

## Especificación por caja

### 1. `StructureLoader` — caja piloto de M1

```python
class StructureLoaderInputs(BaseModel):
    path: Path

class StructureLoaderParams(BaseModel):
    format: Literal["auto", "cif", "lammps_data", "xyz"] = "auto"
    wrap_to_cell: bool = True
    duplicate_tolerance: float = 1e-3
    strict_elements: bool = False  # si True, falla si no encuentra elemento por masa

# Output: Structure
```

Implementación: dispatch por extensión a `io/cif.py`, `io/lammps_data.py`, o `io/xyz.py`. Normaliza a `Structure` con cell 3×3 nativa (incluyendo tilts para LAMMPS triclínicas).

### 2. `BilayerSplitter`

```python
class BilayerSplitterInputs(BaseModel):
    structure: Structure

class BilayerSplitterParams(BaseModel):
    lj_cutoff: float = 8.0
    tile_factor: float = 1.2     # box ≥ factor × cutoff
    max_atoms: int = 2000
    z_axis_index: int = 2        # qué eje es Z (típicamente 2)

# Output: LayerPair
```

### 3. `LatticeTransformer`

```python
class LatticeTransformerInputs(BaseModel):
    Alat: NDArray2x2
    basis_A: NDArray2D
    species_A: list[str]

class LatticeTransformerParams(BaseModel):
    transform_type: Literal["identity", "rotation", "isotropic_scale",
                            "anisotropic_strain", "strain_plus_rotation"]
    theta_deg: float = 0.0
    s1: float = 1.0
    s2: float = 1.0
    scale: float = 1.0

# Output: dict {Blat: NDArray2x2, basis_B: NDArray2D}
```

### 4. `LatticeMatcher`

```python
class LatticeMatcherInputs(BaseModel):
    Alat: NDArray2x2
    Blat: NDArray2x2

class LatticeMatcherParams(BaseModel):
    method: Literal["continuous", "brute_force", "rotation_only"] = "continuous"
    dim: int = 3                 # supercell search dimension
    bounds: list[tuple[float, float]] = [(0.95, 1.05), (0.95, 1.05), (-30.0, 30.0)]
    # DE params
    popsize: int = 1000
    maxiter: int = 1000
    tol_lavect: float = 1e-2
    angle_min: float = 30.0
    angle_max: float = 90.0
    top_k: int = 5
    random_seed: int | None = 42
    # Brute force params
    N: int = 8
    ndiv: int = 21
    mismatch_tol: float = 0.05
    min_area: float = 1.0

# Output: list[MatchingSolution]
```

### 5. `AtomAssembler`

```python
class AtomAssemblerInputs(BaseModel):
    solution: MatchingSolution
    basis_A: NDArray2D
    basis_B: NDArray2D
    Alat: NDArray2x2
    Blat: NDArray2x2
    species_A: list[str]
    species_B: list[str]

class AtomAssemblerParams(BaseModel):
    pass  # ningún parámetro

# Output: BilayerAtoms
```

### 6. `Validator`

```python
class ValidatorInputs(BaseModel):
    bilayer: BilayerAtoms
    solution: MatchingSolution
    layer_pair: LayerPair        # necesario para basis_B (descubierto en M0.2)
    target: ValidationTarget | None = None

class ValidatorParams(BaseModel):
    cutoff_scale: float = 1.15
    k_dist: int = 6
    skip_cases: list[str] = []   # reemplaza el hardcode 'real_pair_hfse2_mose2'

# Output: ValidationMetrics
```

### 7. `MDSupercellBuilder`

```python
class MDSupercellBuilderInputs(BaseModel):
    bilayer: BilayerAtoms

class MDSupercellBuilderParams(BaseModel):
    vacuum_z: float = 15.0
    interlayer_z: float = 3.3
    padding_cells: int = 0
    lj_cutoff: float = 8.0       # para tiling adicional si hace falta

# Output: MDStructure
```

### 8. `PotentialAssigner`

```python
class PotentialAssignerInputs(BaseModel):
    md_structure: MDStructure

class PotentialAssignerParams(BaseModel):
    intralayer_kind: Literal["auto", "tersoff", "sw", "lj", "gap", "mace"] = "auto"
    interlayer_kind: Literal["lj"] = "lj"
    lj_cutoff: float = 8.0
    epsilon_scale: float = 1.0
    # Si "auto", consulta tablas y disponibilidad

# Output: PotentialPlan
```

### 9. `LammpsInputWriter`

```python
class LammpsInputWriterInputs(BaseModel):
    md_structure: MDStructure
    potential_plan: PotentialPlan

class LammpsInputWriterParams(BaseModel):
    output_dir: Path
    temperature_K: float = 300.0
    nvt_steps: int = 100_000
    timestep_ps: float = 0.001
    dump_every: int = 1_000
    minimize_first: bool = True

# Output: LammpsRunPlan
```

## Supports

### A. `MaterialsDB`
Cliente de C2DB (URL Gist en el original). Métodos: `get_material(formula)`, `list_materials()`. No es una caja porque es un servicio, no una transformación de datos.

### B. `LammpsExecutor`
Wrapper sobre `docker run ghcr.io/camilochs/moire-flow-runtime`. Métodos: `run(script, log_out)`, `capabilities()`. Implementado en M7.

### C. `TrajectoryAnalyzer`
Lee LAMMPS log files y dumps, produce métricas: interlayer gap (Å), vdW energy (eV), strain (%). Output: `TrajectoryAnalysis` Pydantic model.

## Roadmap por milestone

| Milestone | Entrega | Tests |
|---|---|---|
| **M1** (este) | `StructureLoader` + tipos core + Box ABC + registry | round-trip JSON, contrato Box, regresión vs original |
| M2 | `LatticeTransformer` + `Validator` | hypothesis para fuzz de ángulos |
| M3 | `LatticeMatcher` (corazón algorítmico) | benchmarks vs casos conocidos |
| M4 | `BilayerSplitter` + `AtomAssembler` | structural equivalence vs original |
| M5 | `MDSupercellBuilder` | golden file de cell triclínica |
| M6 | `PotentialAssigner` + `LammpsInputWriter` | regresión de `.data` y `.in` |
| M7 | Supports A/B/C + **runtime Docker construido** | smoke test con LAMMPS real |
| M8 | `WorkflowEngine` (DAG executor) | pipelines end-to-end |
| M9 | **Web UI** bajo `web/` (FastAPI + React Flow + RJSF + Ajv) | E2E tests con Playwright |

## Decisiones derivadas del M0 audit

- `Validator` necesita `layer_pair` además del `solution` (descubierto en M0.2)
- El campo `name` no es propiedad de las cajas — la identidad del caso vive en el `WorkflowEngine`
- `source_mode` desaparece — el dispatch es por tipo Python (Structure vs LayerPair vs MDStructure)
- Tablas de potenciales pasan a constantes inmutables
- Plotting va a Support D opcional (matplotlib como dep extra)
