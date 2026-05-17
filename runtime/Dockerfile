# moire-flow LAMMPS runtime (minimal)
#
# Based on the official lammps/lammps multi-arch image: ships with the
# stable LAMMPS binary + Tersoff + SW + LJ + many other pair styles.
# Enough to run our format_tersoff_script, format_sw_script, and
# format_bilayer_lj_script outputs end-to-end.
#
# Multi-arch: the upstream image builds for linux/amd64 and linux/arm64.
#
# QUIP/GAP and MACE/ML-IAP are NOT included here — they require source
# builds with PyTorch and libquip and roughly double the image size.
# They live in `Dockerfile.full` (TODO) which is linux/amd64-only.
#
# Build:    docker build -t ghcr.io/camilochs/moire-flow-runtime:latest runtime/
# Test:     docker run --rm ghcr.io/camilochs/moire-flow-runtime:latest -help | head -5

FROM lammps/lammps:latest

LABEL org.opencontainers.image.source="https://github.com/camilochs/moire-flow"
LABEL org.opencontainers.image.description="LAMMPS runtime for moire-flow (tersoff/sw/lj minimal)"

# Upstream ships `lmp_serial` and `lmp_mpi` but no plain `lmp`. Our
# LammpsExecutor invokes `lmp` — add a stable alias.
USER root
RUN ln -s "$(command -v lmp_serial)" /usr/local/bin/lmp

# Smoke check: print pair-style banner
RUN lmp -help 2>&1 | head -3 || (echo "[warn] lmp probe failed at build time"; true)

WORKDIR /work
ENTRYPOINT ["lmp"]
CMD ["-help"]
