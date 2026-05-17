# runtime/

Docker image that bundles a working LAMMPS for `moire_flow.supports.LammpsExecutor`.

## What's inside

| File | Image | Pair styles |
|---|---|---|
| [`Dockerfile`](Dockerfile) | `moire-flow-runtime:latest` | tersoff, sw, lj/cut, KSPACE, MANYBODY |
| [`Dockerfile.full`](Dockerfile.full) | `moire-flow-runtime:full` | + MACE (`pair_style mace`), ML-IAP (`pair_style mliap`), SNAP |

The minimal image is multi-arch. The `:full` image is **linux/amd64 only**
because libtorch ships only x86_64 wheels and the MACE-LAMMPS package
links against it. On Apple Silicon the `:full` build runs under Rosetta
or QEMU emulation via Colima (`colima start --memory 8`), so expect
60-120 minutes for the source compile.

QUIP/GAP is intentionally out of the `:full` image — it requires a
separate `libquip` Fortran build with gfortran-specific flags and is
largely superseded by MACE for foundation-model workflows. The
`format_gap_quip_script` writer is still present in `io/lammps_writer.py`
so a future `Dockerfile.gap` recipe can drop in without code changes.

## Build locally

Minimal (Tersoff/SW/LJ — multi-arch, ~2-3 min):

```bash
docker build -t moire-flow-runtime:latest runtime/
```

Full (+ MACE + ML-IAP — **linux/amd64 only**, CI recommended):

```bash
docker build --platform linux/amd64 \
    -t moire-flow-runtime:full \
    -f runtime/Dockerfile.full runtime/
```

The `:full` build downloads libtorch (~1 GB), compiles LAMMPS from source
(`-DPKG_ML-MACE=on -DPKG_ML-IAP=on -DMLIAP_ENABLE_PYTHON=on
-DDOWNLOAD_LIBTORCH=on`), and installs `mace-torch` for the mliappy
bridge. Total image size ~6 GB.

> **⚠ Apple Silicon (M-series) caveat.** Building `:full` locally on
> aarch64 hosts via Rosetta/QEMU emulation is **not reliable** —
> `g++ 11` segfaults at random C++ units (we observed crashes in
> `atom_vec_body.cpp` and `ntopo_angle_all.cpp` with both `-j1` and
> `-j2`, with 4 GiB and 8 GiB VMs). This is an emulation-layer bug,
> not a memory issue.
>
> **Use a native amd64 host or CI instead.** The repo ships
> [`.github/workflows/build-runtime.yml`](../.github/workflows/build-runtime.yml)
> which builds and pushes both images to GHCR on `ubuntu-latest`
> (native amd64). Trigger it from the *Actions* tab of the GitHub
> UI with `workflow_dispatch` (choose `full` or `both`). After it
> finishes:
>
> ```bash
> docker pull ghcr.io/camilochs/moire-flow-runtime:full
> docker tag ghcr.io/camilochs/moire-flow-runtime:full moire-flow-runtime:full
> ```

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
