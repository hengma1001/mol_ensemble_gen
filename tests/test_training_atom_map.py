"""Offline unit tests for the mdCATH→ESMFold2 atom mapping (no esm/GPU).

The heavy-atom table is injected, so these run anywhere and pin down the
correctness keystone: slot layout, matched fraction, masking of missing atoms,
and the scatter that places MD coordinates into model slots.
"""

from __future__ import annotations

import numpy as np
import pytest

from mol_ensemble_gen.training.atom_map import (
    build_atom_map,
    build_model_atom_layout,
    canonical_resname,
    is_heavy_atom,
)

# A tiny stand-in for PROTEIN_HEAVY_ATOMS covering the residues used below.
FAKE_TABLE = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "GLY": ["N", "CA", "C", "O"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "UNK": ["N", "CA", "C", "O"],
}


@pytest.mark.unit
def test_layout_order_and_length():
    layout = build_model_atom_layout("AGS", heavy_atoms=FAKE_TABLE)
    assert len(layout) == 5 + 4 + 6
    # residue indices are contiguous and atoms follow the table order
    assert layout[:5] == [(0, "N"), (0, "CA"), (0, "C"), (0, "O"), (0, "CB")]
    assert layout[5] == (1, "N")
    assert layout[-1] == (2, "OG")


@pytest.mark.unit
def test_unknown_residue_falls_back_to_backbone():
    layout = build_model_atom_layout("X", heavy_atoms=FAKE_TABLE)
    assert layout == [(0, "N"), (0, "CA"), (0, "C"), (0, "O")]


@pytest.mark.unit
def test_full_match_fraction_and_scatter():
    seq = "AG"
    md = [(0, "N"), (0, "CA"), (0, "C"), (0, "O"), (0, "CB"),
          (1, "N"), (1, "CA"), (1, "C"), (1, "O")]
    amap = build_atom_map(seq, md, heavy_atoms=FAKE_TABLE)
    assert amap.num_slots == 9
    assert amap.matched_fraction == 1.0
    # each MD row carries its row index as coordinates → gt slot i == its md row
    frame = (np.arange(len(md))[:, None] * np.ones((1, 3))).astype(np.float32)
    gt, mask = amap.scatter(frame)
    assert gt.shape == (9, 3) and mask.all()
    np.testing.assert_allclose(gt, frame)


@pytest.mark.unit
def test_missing_atom_is_masked_not_zero_filled_into_loss():
    seq = "A"
    md = [(0, "N"), (0, "CA"), (0, "C"), (0, "O")]  # CB absent from MD
    amap = build_atom_map(seq, md, heavy_atoms=FAKE_TABLE)
    assert amap.num_slots == 5
    assert amap.matched_fraction == pytest.approx(4 / 5)
    _, mask = amap.scatter(np.zeros((4, 3), np.float32))
    assert mask.tolist() == [True, True, True, True, False]  # CB slot masked


@pytest.mark.unit
def test_scatter_batch_matches_per_frame():
    seq = "AG"
    md = [(0, "N"), (0, "CA"), (0, "C"), (0, "O"), (0, "CB"),
          (1, "N"), (1, "CA"), (1, "C"), (1, "O")]
    amap = build_atom_map(seq, md, heavy_atoms=FAKE_TABLE)
    frames = np.random.default_rng(0).normal(size=(3, len(md), 3)).astype(np.float32)
    batched = amap.scatter_batch(frames)
    for i in range(3):
        gt, _ = amap.scatter(frames[i])
        np.testing.assert_allclose(batched[i], gt)


@pytest.mark.unit
def test_scatter_rejects_wrong_atom_count():
    amap = build_atom_map("A", [(0, "N"), (0, "CA"), (0, "C"), (0, "O")], heavy_atoms=FAKE_TABLE)
    with pytest.raises(ValueError):
        amap.scatter(np.zeros((3, 3), np.float32))


@pytest.mark.unit
def test_canonical_resname_and_heavy_detection():
    assert canonical_resname("MSE") == "MET"
    assert canonical_resname("HSD") == "HIS"
    assert canonical_resname("ala") == "ALA"
    assert is_heavy_atom("CA", "C")
    assert not is_heavy_atom("H", "H")
    assert not is_heavy_atom("HB2")          # inferred from name
    assert not is_heavy_atom("OXT")          # terminal oxygen skipped
    assert is_heavy_atom("N")
