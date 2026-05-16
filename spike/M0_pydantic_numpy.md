# M0.1 — Pydantic v2 + numpydantic round-trip benchmark

**Veredicto: ✅ Pasa todos los gates con margen amplio.** Pydantic v2 + numpydantic es viable para el caso de uso del web UI.

## Setup

- numpy 2.4.4
- pydantic 2.13.4
- numpydantic 1.8.1
- Hardware: Apple Silicon (laptop de desarrollo)
- Modelo:
  ```python
  class Structure(BaseModel):
      atoms: NDArray[Shape["* x, 3 y"], np.float64]
      species: list[str]
      cell: NDArray[Shape["3 x, 3 y"], np.float64]
  ```
- Operación medida: `model_dump_json()` → `model_validate_json()`
- `n_repeats = 5`, reporto el **mínimo** de cada lado (estima `t_ideal` sin ruido de GC/OS)

## Resultados

| N átomos | JSON (KB) | Serialización (ms) | Deserialización (ms) | Round-trip (ms) | dtype preservado |
|---:|---:|---:|---:|---:|:---:|
|     100 |     6.3 |  0.07 |  0.16 |   0.23 | ✅ |
|   1 000 |    62.4 |  0.61 |  1.26 |   1.86 | ✅ |
|  10 000 |   623.1 |  3.32 |  4.26 |   7.59 | ✅ |
| 100 000 | 6 231.9 | 22.91 | 33.60 |  56.51 | ✅ |

## Gates pasados (objetivo: N=10000)

- **Round-trip < 100 ms**: 7.59 ms (13× de margen) ✅
- **JSON < 5 MB**: 0.61 MB (8× de margen) ✅
- **dtype preservado tras round-trip**: `float64` antes y después ✅

## Conclusiones

1. **Pydantic v2 + numpydantic es viable** para el patrón "cajas conectadas con cables JSON" del web UI. La latencia es despreciable comparada con el coste de cualquier cálculo real (matcher = segundos; LAMMPS MD = minutos).

2. **Escalabilidad lineal**: el tiempo y el tamaño escalan linealmente con N. Incluso a N=100 000 (10× el target), round-trip < 60 ms — sigue siendo aceptable.

3. **`numpydantic` no expone `__version__`**: usar `importlib.metadata.version('numpydantic')` para introspección.

4. **Sintaxis Shape**: `NDArray[Shape["* x, 3 y"], np.float64]` (no `Shape["* x 3"]` como sugería el plan). Las dimensiones se separan por coma y cada una tiene un nombre. Una dim wildcard se escribe `* nombre`.

5. **El `model_config = ConfigDict(arbitrary_types_allowed=True)` no es estrictamente necesario** — numpydantic registra los validators directamente en el core schema de Pydantic. Lo dejo por claridad pero se puede quitar.

## Implicación para M1

- **Mantener Pydantic v2 + numpydantic** en `core/types.py`. No hace falta `msgspec` ni blobs base64.
- **Los tipos `NDArray2D`, `NDArray3x3`, `NDArray2x2`** del plan se materializan como aliases de `NDArray[Shape[...], np.float64]`. Definidos una sola vez en `core/types.py`.
- **El web UI (M9) puede recibir Structures completas como JSON** sin penalización notable. Para estructuras grandes (>50k átomos) podríamos en el futuro implementar paginación o blobs binarios, pero no es prioritario.

## Artefactos

- `spike/m0_bench.py` — script throwaway
- `spike/m0_bench_results.json` — resultados raw

Ambos se commitean para trazabilidad.
