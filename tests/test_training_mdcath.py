"""Tests for mdCATH topology parsing and the frame-streaming dataset.

``test_parse_topology_*`` are pure (no h5py/torch/esm). The dataset test builds a
synthetic HDF5 file plus a matching atom-map cache and is skipped where h5py or
torch are unavailable.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mol_ensemble_gen.training import mdcath
from mol_ensemble_gen.training.atom_map import build_atom_map
from mol_ensemble_gen.training.config import TrainConfig, _build
from mol_ensemble_gen.training.featurize import _save_atom_map, domain_cache_dir

FAKE_TABLE = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "GLY": ["N", "CA", "C", "O"],
    "UNK": ["N", "CA", "C", "O"],
}

# A 2-residue (ALA-GLY) PDB with hydrogens and a terminal OXT to be stripped.
PDB_TEXT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       4.000   0.000   0.000  1.00  0.00           C
ATOM      6  HA  ALA A   1       5.000   0.000   0.000  1.00  0.00           H
ATOM      7  N   GLY A   2       6.000   0.000   0.000  1.00  0.00           N
ATOM      8  CA  GLY A   2       7.000   0.000   0.000  1.00  0.00           C
ATOM      9  C   GLY A   2       8.000   0.000   0.000  1.00  0.00           C
ATOM     10  O   GLY A   2       9.000   0.000   0.000  1.00  0.00           O
ATOM     11  OXT GLY A   2      10.000   0.000   0.000  1.00  0.00           O
END
"""


@pytest.mark.unit
def test_parse_topology_strips_h_and_terminal():
    topo = mdcath.parse_topology(PDB_TEXT)
    assert topo.sequence == "AG"
    assert topo.n_atoms == 11
    # heavy atoms only: HA (row 5) and OXT (row 10) dropped
    assert topo.md_records == [
        (0, "N"), (0, "CA"), (0, "C"), (0, "O"), (0, "CB"),
        (1, "N"), (1, "CA"), (1, "C"), (1, "O"),
    ]
    assert topo.heavy_indices.tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9]


@pytest.mark.unit
def test_ca_atom_indices():
    assert mdcath.ca_atom_indices(PDB_TEXT).tolist() == [1, 7]


@pytest.mark.unit
def test_map_from_parsed_topology_full_match():
    topo = mdcath.parse_topology(PDB_TEXT)
    amap = build_atom_map(topo.sequence, topo.md_records, heavy_atoms=FAKE_TABLE)
    assert amap.num_slots == 9
    assert amap.matched_fraction == 1.0


def _write_synthetic_h5(path, domain, coords):
    import h5py

    with h5py.File(path, "w") as f:
        g = f.create_group(domain)
        g.attrs["numProteinAtoms"] = coords.shape[1]
        rg = g.create_group("320").create_group("0")
        rg.create_dataset("coords", data=coords)


@pytest.mark.unit
def test_dataset_streams_scattered_frames(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("torch")

    domain = "aA00"
    n_frames, n_atoms = 10, 11
    # coords[f, a] = f*100 + a, broadcast over xyz — makes provenance checkable.
    coords = (np.arange(n_frames)[:, None] * 100 + np.arange(n_atoms)[None, :]).astype(np.float32)
    coords = np.repeat(coords[:, :, None], 3, axis=2)

    mdcath_dir = tmp_path / "md"
    mdcath_dir.mkdir()
    _write_synthetic_h5(mdcath.domain_path(mdcath_dir, domain), domain, coords)

    # Build + persist a matching atom map; heavy atoms are the first 9 rows.
    md_records = [
        (0, "N"), (0, "CA"), (0, "C"), (0, "O"), (0, "CB"),
        (1, "N"), (1, "CA"), (1, "C"), (1, "O"),
    ]
    amap = build_atom_map("AG", md_records, heavy_atoms=FAKE_TABLE)
    amap = dataclasses.replace(amap, heavy_indices=np.arange(9, dtype=np.int64))
    cache_dir = tmp_path / "cache"
    cache = domain_cache_dir(cache_dir, domain)
    cache.mkdir(parents=True)
    _save_atom_map(cache, amap)

    ds = mdcath.MDCathDataset(
        mdcath_dir=str(mdcath_dir),
        cache_dir=str(cache_dir),
        domains=[domain],
        temperatures=[320],
        replicas=[0],
        skip_frames=1,
        frames_per_step=4,
        rank=0,
        world_size=1,
    )
    batches = list(iter(ds))
    assert [b.gt_coords.shape[0] for b in batches] == [4, 4, 2]
    for b in batches:
        assert b.domain == domain
        assert b.temperature == 320.0
        assert b.gt_coords.shape[1:] == (9, 3)
        assert b.atom_mask.all()
    # Row order is identity here, so each slot holds its source atom's coords.
    stacked = np.concatenate([b.gt_coords for b in batches], axis=0)
    np.testing.assert_allclose(stacked, coords[:, :9, :])


@pytest.mark.unit
def test_make_dataset_builds_the_lazy_class(tmp_path):
    """``make_dataset`` is the trainer's entry point and must resolve the lazy class.

    Regression: it referenced the bare global ``MDCathDataset``, which only
    exists via the module ``__getattr__`` — never consulted for global-name
    lookup inside the module — so every ``finetune`` run died with a NameError
    while the attribute-access tests above kept passing.
    """
    pytest.importorskip("torch")

    cfg = _build(
        TrainConfig,
        {
            "data": {
                "mdcath_dir": str(tmp_path / "md"),
                "cache_dir": str(tmp_path / "cache"),
                "domains": ["aA00"],
                "temperatures": [320],
                "frames_per_step": 4,
            }
        }
    )
    ds = mdcath.make_dataset(cfg, rank=0, world_size=1)
    assert ds.domains == ["aA00"]
    assert ds.temperatures == [320]
    assert ds.frames_per_step == 4


@pytest.mark.unit
def test_dataset_skips_domain_without_cache(tmp_path, capsys):
    pytest.importorskip("h5py")
    pytest.importorskip("torch")

    ds = mdcath.MDCathDataset(
        mdcath_dir=str(tmp_path / "md"),
        cache_dir=str(tmp_path / "cache"),
        domains=["missing"],
        temperatures=[320],
        replicas=[0],
        skip_frames=1,
        frames_per_step=4,
        rank=0,
        world_size=1,
    )
    assert list(iter(ds)) == []
    assert "no cache for missing" in capsys.readouterr().out
