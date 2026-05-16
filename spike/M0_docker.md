# M0.4 — Decisión Docker

## Verdicto

**Plataforma: `linux/amd64` only.** Sin promesas multi-arch.

## Razones

El engineer verificó vía Docker Hub API y búsquedas en hubs de imágenes científicas:

1. **`lammps/lammps`** — imagen oficial. Todos los tags publicados son `amd64`. Último tag: `stable_29Sep2021` / `patch_7Jan2022` (4+ años de antigüedad, **pre-MACE**). No sirve como base actualizada para el stack que necesitamos.

2. **`libatomsquip/quip`** — imagen mantenida por el grupo libAtoms (autores de QUIP/GAP). Incluye LAMMPS+QUIP+GAP integrado. **También `amd64` only**. No publican manifests `arm64`.

3. **MACE-torch en `arm64`** — no existe imagen pública con LAMMPS + MACE bundled. El paquete `mace-torch` se instala vía pip y depende de PyTorch. PyTorch tiene wheels arm64 nativos para macOS pero no para Linux/arm64 con CUDA. Dentro de un container Linux/arm64 (ej. corriendo en Docker Desktop sobre Apple Silicon), no hay aceleración GPU disponible.

4. **QUIP/GAP en `arm64`** es un dolor histórico:
   - Codebase Fortran con flags de arquitectura hardcoded (`linux_x86_64_gfortran` vs `linux_arm64_gfortran`)
   - Builds CI oficiales no publican artefactos `arm64`
   - Cross-compilation requiere parchear el sistema de build, no es una flag de `buildx`

## Stack target (a construir en M7)

```dockerfile
FROM ubuntu:22.04

# Base toolchain
RUN apt-get update && apt-get install -y \
    build-essential gfortran cmake git wget \
    libopenmpi-dev openmpi-bin \
    libfftw3-dev libopenblas-dev liblapack-dev \
    python3.11 python3-pip

# LAMMPS desde source con paquetes ML
# Configurar con: -D PKG_MANYBODY -D PKG_KOKKOS -D PKG_ML-IAP -D PKG_ML-PACE -D PKG_QUIP
# Build estimado: ~30-60 min en runner CI con 4 cores

# QUIP/GAP desde source
# Requiere QUIP_ARCH=linux_x86_64_gfortran y configuración manual
# Build estimado: ~60-90 min

# MACE-torch via pip
# RUN pip install mace-torch
# Wheels disponibles para amd64

# Volúmenes esperados al ejecutar:
# -v ./runs:/workspace/runs           (output del MD)
# -v ./potentials:/opt/potentials     (modelos GAP, MACE checkpoints, params Tersoff/SW)
```

**Tiempo total de build estimado (primera vez)**: 2–4 horas en un runner CI estándar (GitHub Actions ubuntu-latest, 4 cores, 16 GB RAM).

**Tamaño esperado de la imagen**: 4–6 GB. Mitigación: multi-stage build, mover modelos pesados (MACE checkpoints) a volumen montable en vez de bundle en la imagen.

## Para desarrollo local en Apple Silicon (M-series)

Docker Desktop con Rosetta 2 + QEMU permite correr imágenes `linux/amd64` sobre arm64:

- Habilitar "Use Rosetta for x86_64/amd64 emulation" en Docker Desktop settings
- Ejecutar con `docker run --platform linux/amd64 ghcr.io/camilochs/moire-flow-runtime`
- **Coste de rendimiento**: 5–10× más lento que ejecución nativa. Aceptable para iterar sobre fixtures pequeñas, **no aceptable para producción**.

## Para producción (servidor de la web UI, M9)

El backend FastAPI correrá en un servidor Linux `x86_64` nativo (probablemente cluster ICN2 o VPS estándar). Ahí la imagen `linux/amd64` corre nativamente sin penalización.

## Qué NO se construye en M1

Sólo el stub: `runtime/Dockerfile` con la cabecera `FROM ubuntu:22.04` y comentarios. El build real es M7.

## Decisión pendiente para M7

¿Publicar a **GitHub Container Registry** (`ghcr.io/camilochs/moire-flow-runtime`) o a **Docker Hub** (`camilochs/moire-flow-runtime`)? Recomendación inicial: GHCR — integración nativa con el repo, control de permisos por organización GitHub, sin rate limits para pull desde GitHub Actions.

## Reproducibilidad real (no solo Docker)

El skeptic señaló: Docker resuelve **distribución**, no **reproducibilidad**. Para que un workflow sea genuinamente reproducible necesitamos también:

1. **Seeds explícitos** en `LatticeMatcher` (scipy.optimize.differential_evolution acepta `seed=`)
2. **Hashes de archivos de potencial** (GAP, MACE checkpoints) registrados en el workflow JSON
3. **Hash de la imagen Docker** (`docker inspect --format='{{.Id}}' moire-flow-runtime`)

Estos tres elementos van en `core/types.py` como parte del `WorkflowSpec` (M8), pero el `core/types.py` de M1 ya deja espacio para `random_seed: int | None` en los modelos de parámetros relevantes.
