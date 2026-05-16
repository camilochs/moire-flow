"""M0.1 spike — Pydantic v2 + numpydantic round-trip benchmark.

Throwaway script. Output written to M0_pydantic_numpy.md.
"""
from __future__ import annotations

import json
import time

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict


class Structure(BaseModel):
    """Minimal Structure model for the benchmark — N atoms in 3D."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    atoms: NDArray[Shape["* x, 3 y"], np.float64]
    species: list[str]
    cell: NDArray[Shape["3 x, 3 y"], np.float64]


def make_structure(n: int) -> Structure:
    rng = np.random.default_rng(seed=0)
    return Structure(
        atoms=rng.random((n, 3)),
        species=["C"] * n,
        cell=np.eye(3, dtype=np.float64) * 10.0,
    )


def bench(n: int, n_repeats: int = 5) -> dict:
    s = make_structure(n)

    # Serialization
    times_ser = []
    payload = ""
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        payload = s.model_dump_json()
        times_ser.append(time.perf_counter() - t0)

    # Deserialization
    times_de = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        Structure.model_validate_json(payload)
        times_de.append(time.perf_counter() - t0)

    # Dtype check after round-trip
    s2 = Structure.model_validate_json(payload)
    dtype_ok = s2.atoms.dtype == np.float64 and s2.cell.dtype == np.float64

    # Round-trip total
    rt_total_ms = (min(times_ser) + min(times_de)) * 1000

    return {
        "n_atoms": n,
        "json_size_bytes": len(payload),
        "ser_min_ms": min(times_ser) * 1000,
        "de_min_ms": min(times_de) * 1000,
        "round_trip_ms": rt_total_ms,
        "dtype_preserved": dtype_ok,
    }


def main() -> None:
    print(f"numpy={np.__version__}")
    import pydantic
    from importlib.metadata import version as _v
    print(f"pydantic={pydantic.VERSION}")
    print(f"numpydantic={_v('numpydantic')}")
    print()

    sizes = [100, 1_000, 10_000, 100_000]
    results = [bench(n) for n in sizes]

    # Print table
    print(f"{'N':>8} | {'JSON KB':>10} | {'ser ms':>8} | {'de ms':>8} | {'round-trip ms':>14} | {'dtype OK':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['n_atoms']:>8} | "
            f"{r['json_size_bytes'] / 1024:>10.1f} | "
            f"{r['ser_min_ms']:>8.2f} | "
            f"{r['de_min_ms']:>8.2f} | "
            f"{r['round_trip_ms']:>14.2f} | "
            f"{str(r['dtype_preserved']):>8}"
        )

    # Gate check
    print()
    target = next(r for r in results if r["n_atoms"] == 10_000)
    rt_ok = target["round_trip_ms"] < 100
    size_ok = target["json_size_bytes"] < 5 * 1024 * 1024
    print(f"Gate (N=10000): round-trip < 100ms: {rt_ok} ({target['round_trip_ms']:.2f}ms)")
    print(f"Gate (N=10000): JSON size < 5 MB:    {size_ok} ({target['json_size_bytes'] / 1024 / 1024:.2f} MB)")
    print(f"Gate (N=10000): dtype preserved:     {target['dtype_preserved']}")

    # Write JSON for the markdown report to pick up
    with open("spike/m0_bench_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote spike/m0_bench_results.json")


if __name__ == "__main__":
    main()
