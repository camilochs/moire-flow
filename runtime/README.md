# runtime/

Docker image that bundles a working LAMMPS for `moire_flow.supports.LammpsExecutor`.

## What's inside

| File | Image | Pair styles |
|---|---|---|
| [`Dockerfile`](Dockerfile) | `ghcr.io/camilochs/moire-flow-runtime:latest` | tersoff, sw, lj/cut, KSPACE, MANYBODY |
| `Dockerfile.full` (TODO) | `ghcr.io/camilochs/moire-flow-runtime:full` | + QUIP/GAP, MACE, ML-IAP |

Both are **linux/amd64 only**. QUIP and MACE-LAMMPS cannot be built on
arm64 today (the QUIP Fortran modules require GCC + linker flags that
break on Apple Silicon; MACE-LAMMPS depends on a libtorch wheel that
ships only x86_64). See ANALYSIS.md §10.

## Build locally

```bash
docker build --platform linux/amd64 \
    -t ghcr.io/camilochs/moire-flow-runtime:latest \
    runtime/
```

The build is ~2-3 minutes on Apple Silicon with Rosetta or Colima
(`colima start --arch x86_64`).

## Smoke check

```bash
docker run --platform linux/amd64 --rm \
    ghcr.io/camilochs/moire-flow-runtime:latest lmp -help | head -5
```

Should print `LAMMPS (...)` then a usage banner. If you see a "Could not
find this pair style" message, the image is missing a package — file an
issue.

## Run a workflow output

After `LammpsInputWriter` produces files in `work_dir`:

```bash
docker run --platform linux/amd64 --rm \
    -v "$PWD/work_dir:/work" -w /work \
    ghcr.io/camilochs/moire-flow-runtime:latest \
    -in bilayer.in -log bilayer.log
```

…or just call `LammpsExecutor.run(script_path)` from Python; it shells
out to `docker run` with the right flags.
