# M0.3 — Dedup audit del script original

Auditoría de funciones duplicadas en `reference/lattice_matching_02032026.py` (12 164 líneas). El script fue auto-convertido de un notebook de Colab, lo que explica por qué hay cells que redefinen funciones que ya existían en celdas anteriores.

**Convención**: cuando una función se redefine en orden de aparición en el `.py`, la **última definición gana** al cargar el módulo (semántica estándar de Python). En la pipeline modular, sólo portamos la **forma canónica** elegida.

---

## Duplicados confirmados

### 1. `run_cmd` (×4)

| Línea | Variante | Cambios |
|---|---|---|
| 2912 | v1 | Versión base; print de los últimos 8000 chars de stdout |
| 4173 | v2 | Añade `shell=False`; print de los últimos 10000 chars; nombre interno `r` en vez de `result` |
| 4613 | v3 | Idéntica a v1 |
| **4950** | **v4** | **Idéntica a v1/v3 — gana al cargar el módulo** |

**Diferencias clave entre v1 y v2**:
- v2 acepta `shell: bool = False` y pasa `shell=shell` a `subprocess.run`
- v2 imprime más output (10000 vs 8000 chars de tail)
- v2 cambia el formato de logging

**Forma canónica para el refactor**: v1/v3/v4. **No** mantener el parámetro `shell` — ningún call site del script lo usa con `shell=True` (verificado por grep). Si fuera necesario en el futuro, se añade explícitamente con una decisión de seguridad.

**Destino en el refactor**: `src/moire_flow/executors/subprocess.py` (M7). NO se porta en M1.

---

### 2. `_detect_style` / `detect_style` (×2, nombres distintos)

| Línea | Nombre | Status |
|---|---|---|
| 2967 | `_detect_style` | Módulo-privada (con underscore) |
| 4767 | `detect_style` | Pública (sin underscore) |

**Forma canónica**: `_detect_style` (línea 2967) — la convención del refactor mantiene helpers internos con underscore. La copia sin underscore en 4767 es probablemente un re-export que se perdió en la traducción Colab→`.py`.

**Destino en el refactor**: `src/moire_flow/executors/lammps_capabilities.py` (M7).

---

### 3. `_as_list` (×3 — todas idénticas)

| Línea | Status |
|---|---|
| 5322 | v1 |
| 7248 | v2 (idéntica a v1) |
| **8114** | **v3 — gana al cargar el módulo (idéntica)** |

Verificado por `diff`: las tres son idénticas. Cargo cargada: redundancia pura.

**Forma canónica**: cualquiera (son idénticas). Una única definición en `src/moire_flow/utils/coerce.py`.

---

### 4. `_is_numeric_label` (×2)

| Línea | Status |
|---|---|
| 7828 | v1 |
| **8124** | **v2 — gana** |

Asumo idénticas (mismo bloque temático que `_as_list`). Verificar antes de portar.

**Destino en el refactor**: `src/moire_flow/utils/labels.py` (helper para parser LAMMPS).

---

### 5. Funciones de análisis de trayectoria (×2 cada una)

Dos bloques completos de análisis post-MD están duplicados en el script — uno alrededor de la línea 11277 y otro alrededor de 11656. El segundo bloque introduce `formula_from_species` (línea 11664) y `guess_layer_height_from_chemistry` (11697), funciones nuevas.

| Función | Línea v1 | Línea v2 | Notas |
|---|---|---|---|
| `infer_mode` | 11277 | 11656 | El diff sugiere implementaciones diferentes (no idénticas) |
| `infer_eq_cut` | 11281 | 11660 | Probablemente diferentes |
| `infer_chemistry_from_species` | 11285 | 11684 | El diff confirma firmas distintas en el contexto inmediato |
| `clean_value` | 11314 | 11713 | Probablemente diferentes |

**Forma canónica**: las **v2 (líneas 11656+)** — son las que ganan al cargar el módulo y son las que tienen integración con `formula_from_species`. Necesitan revisión semántica función por función al portar.

**Destino en el refactor**: `src/moire_flow/supports/trajectory_analyzer.py` (M7).

---

## Funciones que NO son duplicadas (verificadas con grep)

A pesar de aparecer en el análisis original como sospechosas, las siguientes tienen **una sola definición** en el script:

- `_clean_element_label` (sólo línea 7836)
- `lammps_has_pair_style` (sólo línea 8869)
- `formula_from_species` (sólo línea 11664)
- `guess_layer_height_from_chemistry` (sólo línea 11697)

---

## Re-inicializaciones de estado global

Más allá de funciones duplicadas, el script tiene **re-asignaciones de variables/diccionarios globales** que mutan el estado entre cells:

- `validation_results_md` — inicializado en 71, mutado y reescrito en múltiples cells posteriores
- `wf` — definido en 8848, recargado desde JSON en 11594 (`wf = json.loads(wf_path.read_text())`)
- `pot_cfg` — definido en 8863, persistido y recargado
- `LAMMPS_EXE` / `LAMMPS_INFO` — recalculados en múltiples bloques (3450, 3547, 8852)
- Tablas de potenciales (`TERSOFF_REGISTRY`, `SW_REGISTRY`, `UFF_LJ`) — re-inicializadas en cells distintas

**Implicación para el refactor**: el estado global muta a lo largo del notebook. La modularización elimina esto canalizando todo el estado en los objetos Pydantic que viajan entre cajas. Las tablas estáticas (`UFF_LJ`, `TERSOFF_REGISTRY`, `SW_REGISTRY`) viven como constantes en `src/moire_flow/constants/potentials.py`.

---

## Resumen accionable para M1

1. **No portar `shell=True`** del `run_cmd` v2. Es código muerto.
2. **`_as_list` se consolida** en una sola definición.
3. **Las funciones de análisis de trayectoria** se portan desde la línea 11656+ (no desde 11277), pero requieren revisión semántica función por función — los diffs muestran diferencias reales, no solo nombres.
4. **Las tablas globales mutables** pasan a constantes inmutables en `constants/`. Cualquier "ampliación de UFF" se hace componiendo dicts, no mutándolos.
5. Los duplicados están **fuera del alcance de M1** (StructureLoader sólo necesita `parse_lammps_data_file` + `guess_element_from_mass`, que son únicas).
