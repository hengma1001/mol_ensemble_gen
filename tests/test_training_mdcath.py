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
        (0, "N"),
        (0, "CA"),
        (0, "C"),
        (0, "O"),
        (0, "CB"),
        (1, "N"),
        (1, "CA"),
        (1, "C"),
        (1, "O"),
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
        (0, "N"),
        (0, "CA"),
        (0, "C"),
        (0, "O"),
        (0, "CB"),
        (1, "N"),
        (1, "CA"),
        (1, "C"),
        (1, "O"),
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
        # This test is about frame scattering and chunk arithmetic, so the order is
        # pinned; with the default shuffle the short trailing chunk lands anywhere.
        shuffle=False,
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
        },
    )
    ds = mdcath.make_dataset(cfg, rank=0, world_size=1)
    assert ds.domains == ["aA00"]
    assert ds.temperatures == [320]
    assert ds.frames_per_step == 4


@pytest.mark.unit
def test_seed_offset_shifts_the_shuffle_seed(tmp_path):
    """A resumed run must not replay the domain order it already trained on.

    The dataset's epoch counter starts at 0 on every construction and the shuffle
    is seeded on (seed, epoch, rank), so a resume rebuilt the identical order --
    and, with the trainer's matching torch reseed, the identical sigma sequence.
    Measured on a 4-GPU interrupt test: every logged sigma matched the pre-kill run
    at an offset of exactly the checkpointed step. The trainer passes the resumed
    global step as ``seed_offset``; validation must NOT get one (it relies on a
    fixed order), so the default stays 0.
    """
    pytest.importorskip("torch")

    def build(**kw):
        cfg = _build(
            TrainConfig,
            {
                "data": {
                    "mdcath_dir": str(tmp_path / "md"),
                    "cache_dir": str(tmp_path / "cache"),
                    "domains": ["aA00"],
                    "temperatures": [320],
                    "shuffle_seed": 1000,
                }
            },
        )
        return mdcath.make_dataset(cfg, rank=0, world_size=1, **kw)

    assert build().shuffle_seed == 1000, "default must leave the seed alone"
    assert build(seed_offset=0).shuffle_seed == 1000
    assert build(seed_offset=26600).shuffle_seed == 27600
    # Distinct resume points must not collide onto one order.
    seeds = {build(seed_offset=n).shuffle_seed for n in (0, 100, 26600, 532000)}
    assert len(seeds) == 4


@pytest.mark.unit
def test_shuffled_order_actually_changes_with_the_offset(tmp_path, monkeypatch):
    """The offset has to change the *order*, not merely a stored integer."""
    names = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"]
    base = _stream(tmp_path, names, shuffle=True, seed=1000, monkeypatch=monkeypatch)
    moved = _stream(tmp_path, names, shuffle=True, seed=1000 + 26600, monkeypatch=monkeypatch)
    assert base != moved, "offsetting the seed left the stream order unchanged"
    assert sorted(base) == sorted(moved), "the offset changed the data, not just the order"


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


# ---------------------------------------------------------------------------
# stream shuffling
# ---------------------------------------------------------------------------


def _fake_domain(tmp_path, name, n_frames=48, n_atoms=6, temps=(320, 450), reps=(0, 1)):
    """Minimal mdCATH-shaped h5 plus the atom-map cache the stream requires."""
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    md = tmp_path / "md"
    md.mkdir(exist_ok=True)
    with h5py.File(md / f"mdcath_dataset_{name}.h5", "w") as f:
        g = f.create_group(name)
        for T in temps:
            tg = g.create_group(str(T))
            for r in reps:
                rg = tg.create_group(str(r))
                rg.create_dataset("coords", data=np.zeros((n_frames, n_atoms, 3), "f4"))
    return md


def _stream(tmp_path, names, shuffle, seed=1, monkeypatch=None):
    """Collect one full pass of (domain, temperature) pairs from the stream."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from mol_ensemble_gen.training import mdcath as M

    md = None
    for n in names:
        md = _fake_domain(tmp_path, n)

    class FakeMap:
        heavy_indices = np.arange(6)
        present_mask = np.ones(6, bool)

        def scatter_batch(self, c):
            return c

    monkeypatch.setattr(M, "domain_path", lambda d, dom: md / f"mdcath_dataset_{dom}.h5")
    import mol_ensemble_gen.training.featurize as F

    monkeypatch.setattr(F, "load_atom_map", lambda *a, **k: FakeMap())

    ds = M._dataset_cls()(
        mdcath_dir=str(md),
        cache_dir=str(tmp_path),
        domains=list(names),
        temperatures=[320, 450],
        replicas=None,
        skip_frames=8,
        frames_per_step=2,
        rank=0,
        world_size=1,
        shuffle=shuffle,
        shuffle_seed=seed,
    )
    return [(b.domain, b.temperature) for b in ds]


@pytest.mark.unit
def test_shuffle_preserves_the_data_exactly(tmp_path, monkeypatch):
    """Shuffling must reorder, never drop or duplicate.

    The stream is the only thing standing between a config's domain list and what
    the model actually sees, so a shuffle bug that silently lost batches would be
    invisible in the loss curve.
    """
    names = ["aaa", "bbb", "ccc"]
    plain = _stream(tmp_path, names, shuffle=False, monkeypatch=monkeypatch)
    mixed = _stream(tmp_path, names, shuffle=True, monkeypatch=monkeypatch)
    from collections import Counter

    assert Counter(plain) == Counter(mixed), "shuffling changed the multiset of batches"
    assert plain != mixed, "shuffle=True produced the unshuffled order"


@pytest.mark.unit
def test_unshuffled_walks_domains_and_temperatures_in_blocks(tmp_path, monkeypatch):
    """Documents the old behaviour, which is why shuffling was needed.

    Unshuffled, a run consumes domains strictly in list order and each domain's
    temperatures in blocks — so `max_steps` alone decides which domains are ever
    reached, and consecutive gradients share a temperature.
    """
    order = _stream(tmp_path, ["aaa", "bbb"], shuffle=False, monkeypatch=monkeypatch)
    doms = [d for d, _ in order]
    assert doms == sorted(doms), "unshuffled stream should be in list order"
    first_dom = [t for d, t in order if d == "aaa"]
    assert first_dom == sorted(first_dom), "unshuffled temperatures should come in blocks"


@pytest.mark.unit
def test_shuffle_is_reproducible_and_varies_by_epoch(tmp_path, monkeypatch):
    """Same seed -> same order; successive epochs -> different orders."""
    names = ["aaa", "bbb", "ccc"]
    a = _stream(tmp_path, names, shuffle=True, seed=7, monkeypatch=monkeypatch)
    b = _stream(tmp_path, names, shuffle=True, seed=7, monkeypatch=monkeypatch)
    assert a == b, "same seed must give the same order"
    c = _stream(tmp_path, names, shuffle=True, seed=99, monkeypatch=monkeypatch)
    assert a != c, "a different seed must give a different order"

    # a second pass over the *same* dataset object must differ from the first
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from mol_ensemble_gen.training import mdcath as M

    md = None
    for n in names:
        md = _fake_domain(tmp_path, n)

    class FakeMap:
        heavy_indices = np.arange(6)
        present_mask = np.ones(6, bool)

        def scatter_batch(self, c):
            return c

    monkeypatch.setattr(M, "domain_path", lambda d, dom: md / f"mdcath_dataset_{dom}.h5")
    import mol_ensemble_gen.training.featurize as F

    monkeypatch.setattr(F, "load_atom_map", lambda *a, **k: FakeMap())
    ds = M._dataset_cls()(
        mdcath_dir=str(md),
        cache_dir=str(tmp_path),
        domains=names,
        temperatures=[320, 450],
        replicas=None,
        skip_frames=8,
        frames_per_step=2,
        rank=0,
        world_size=1,
        shuffle=True,
        shuffle_seed=3,
    )
    e1 = [(b.domain, b.temperature) for b in ds]
    e2 = [(b.domain, b.temperature) for b in ds]
    from collections import Counter

    assert Counter(e1) == Counter(e2), "epochs must contain the same data"
    assert e1 != e2, "consecutive epochs must not repeat the same order"
