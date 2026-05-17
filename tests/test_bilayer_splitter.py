"""BilayerSplitter: synthetic bilayer split + tiling logic."""

from __future__ import annotations

import numpy as np
import pytest

from moire_flow.boxes import BilayerSplitter, BilayerSplitterInputs, BilayerSplitterParams
from moire_flow.core.types import Structure


def _make_bilayer(z_A: float = 0.0, z_B: float = 3.3) -> Structure:
    """Orthogonal in-plane cell so the skew-correction never kicks in."""
    cell = np.array([[3.16, 0.0, 0.0], [0.0, 3.16, 0.0], [0.0, 0.0, 20.0]])
    atoms = np.array(
        [
            [0.0, 0.0, z_A],
            [1.58, 1.58, z_A],
            [0.0, 0.0, z_B],
            [1.58, 1.58, z_B],
        ]
    )
    species = ["Mo", "S", "Mo", "S"]
    return Structure(atoms=atoms, species=species, cell=cell)


def test_split_lower_and_upper():
    # lj_cutoff=2.0 keeps tiling at 1x1 so we can assert per-layer atom counts
    lp = BilayerSplitter().run(
        BilayerSplitterInputs(structure=_make_bilayer()),
        BilayerSplitterParams(lj_cutoff=2.0),
    )
    assert lp.z_mean_A == pytest.approx(0.0)
    assert lp.z_mean_B == pytest.approx(3.3)
    assert lp.layer_A.atoms.shape[0] == 2
    assert lp.layer_B.atoms.shape[0] == 2
    assert lp.layer_A.species == ["Mo", "S"]
    assert lp.layer_B.species == ["Mo", "S"]


def test_split_always_lower_is_A():
    """Inverting z must still produce A=lower."""
    lp = BilayerSplitter().run(
        BilayerSplitterInputs(structure=_make_bilayer(z_A=5.0, z_B=0.5)),
        BilayerSplitterParams(),
    )
    assert lp.z_mean_A < lp.z_mean_B


def test_sc_vecs_match_cell_2d_block():
    s = _make_bilayer()
    lp = BilayerSplitter().run(
        BilayerSplitterInputs(structure=s),
        BilayerSplitterParams(lj_cutoff=2.0),
    )
    # Small cutoff → no tiling → original sc_vecs
    np.testing.assert_allclose(lp.sc_vecs, s.cell[:2, :2])
    assert lp.replication == (1, 1)


def test_tiling_triggered_when_cell_smaller_than_cutoff():
    """A 3.16 Å in-plane cell needs ~3× tiling to reach 8 Å × 1.2 = 9.6 Å."""
    lp = BilayerSplitter().run(
        BilayerSplitterInputs(structure=_make_bilayer()),
        BilayerSplitterParams(lj_cutoff=8.0, tile_factor=1.2, max_atoms=200),
    )
    assert lp.replication != (1, 1)
    # 4 atoms × nx × ny / 2 layers = atoms per layer
    assert lp.layer_A.atoms.shape[0] == 2 * lp.replication[0] * lp.replication[1]


def test_tiling_skipped_when_exceeds_max_atoms():
    """If tiling would exceed max_atoms, keep the original cell."""
    lp = BilayerSplitter().run(
        BilayerSplitterInputs(structure=_make_bilayer()),
        BilayerSplitterParams(lj_cutoff=100.0, max_atoms=10),
    )
    assert lp.replication == (1, 1)


def test_layer_pair_round_trips_to_json():
    lp = BilayerSplitter().run(
        BilayerSplitterInputs(structure=_make_bilayer()),
        BilayerSplitterParams(lj_cutoff=2.0),
    )
    from moire_flow.core.types import LayerPair
    again = LayerPair.model_validate_json(lp.model_dump_json())
    assert again.replication == lp.replication
