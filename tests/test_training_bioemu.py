"""Tests for the BioEmu-format MD reader (topology.pdb + trajs/*.xtc).

Metadata/temperature handling is pure. The trajectory tests build a synthetic
BioEmu system with mdtraj and are skipped where mdtraj is unavailable; they exist
mainly to pin the two things that would fail *silently* rather than loudly: the
nanometre-to-Ångström conversion, and that a strided chunked read returns the
frames it claims to.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mol_ensemble_gen.training import bioemu

mdtraj = pytest.importorskip("mdtraj", reason="mdtraj is needed to write/read xtc")

# Two residues (ALA-GLY), heavy atoms only -- enough for a Cα selection of size 2.
PDB_TEXT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       4.000   0.000   0.000  1.00  0.00           C
ATOM      6  N   GLY A   2       6.000   0.000   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       7.000   0.000   0.000  1.00  0.00           C
ATOM      8  C   GLY A   2       8.000   0.000   0.000  1.00  0.00           C
ATOM      9  O   GLY A   2       9.000   0.000   0.000  1.00  0.00           O
END
"""


def _make_system(root, name, *, n_runs=2, n_frames=50, temperature=300.0):
    """Write a BioEmu-layout system whose frame f of run r has x = r*1000 + f (nm)."""
    sysdir = root / name
    (sysdir / "trajs").mkdir(parents=True)
    top_path = sysdir / "topology.pdb"
    top_path.write_text(PDB_TEXT)
    (sysdir / "dataset.json").write_text(json.dumps({"temperature": temperature, "forcefield": "amber99sb-ildn"}))

    top = mdtraj.load_topology(str(top_path))
    for run in range(n_runs):
        xyz = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
        xyz[:, :, 0] = (np.arange(n_frames) + run * 1000)[:, None]
        mdtraj.Trajectory(xyz, top).save_xtc(str(sysdir / "trajs" / f"run{run:03d}_protein.cmprsd.xtc"))
    return sysdir


@pytest.mark.unit
def test_find_temperature_handles_spellings_and_nesting():
    assert bioemu._find_temperature({"temperature": 300}) == 300.0
    assert bioemu._find_temperature({"md": {"temperature_K": 310}}) == 310.0
    assert bioemu._find_temperature({"sim_temp_setpoint": "295.5"}) == 295.5
    assert bioemu._find_temperature({"forcefield": "amber99sb-ildn"}) is None


@pytest.mark.unit
def test_system_temperature_defaults_to_300(tmp_path):
    (tmp_path / "sysA").mkdir()
    assert bioemu.system_temperature(tmp_path, "sysA") == bioemu.DEFAULT_TEMPERATURE


@pytest.mark.unit
def test_available_systems_descends_one_nesting_level(tmp_path):
    _make_system(tmp_path / "ONE_cath1", "sysA", n_runs=1, n_frames=4)
    assert bioemu.available_systems(tmp_path) == ["sysA"]
    assert bioemu.resolve_root(tmp_path) == tmp_path / "ONE_cath1"


@pytest.mark.unit
def test_read_reference_ca_converts_nm_to_angstrom(tmp_path):
    _make_system(tmp_path, "sysA", n_runs=1, n_frames=4)
    ca = bioemu.read_reference_ca(tmp_path, "sysA", skip=1)
    assert ca.shape == (4, 2, 3)
    # frame f had x = f nm on every atom, so Å must be 10x that.
    assert np.allclose(ca[:, 0, 0], [0.0, 10.0, 20.0, 30.0], atol=1e-3)


@pytest.mark.unit
def test_read_reference_ca_strides_and_concatenates_runs(tmp_path):
    _make_system(tmp_path, "sysA", n_runs=2, n_frames=50)
    ca = bioemu.read_reference_ca(tmp_path, "sysA", skip=10)
    assert ca.shape[0] == 10  # 5 kept frames per run, 2 runs
    xs = ca[:, 0, 0] / 10.0
    assert np.allclose(xs[:5], [0, 10, 20, 30, 40], atol=1e-3)
    assert np.allclose(xs[5:], [1000, 1010, 1020, 1030, 1040], atol=1e-3)


@pytest.mark.unit
def test_read_reference_ca_replica_selection(tmp_path):
    _make_system(tmp_path, "sysA", n_runs=2, n_frames=10)
    ca = bioemu.read_reference_ca(tmp_path, "sysA", skip=1, replicas=[1])
    assert ca.shape[0] == 10
    assert ca[0, 0, 0] == pytest.approx(10000.0, abs=1e-2)  # run 1, frame 0


@pytest.mark.unit
def test_max_frames_thins_evenly_not_from_the_head(tmp_path):
    _make_system(tmp_path, "sysA", n_runs=1, n_frames=100)
    ca = bioemu.read_reference_ca(tmp_path, "sysA", skip=1, max_frames=10)
    assert ca.shape[0] == 10
    # A head slice would end at frame 9; even thinning must reach the last frame.
    assert ca[-1, 0, 0] == pytest.approx(990.0, abs=1e-2)


@pytest.mark.unit
def test_wrong_temperature_raises_rather_than_scoring_the_wrong_state(tmp_path):
    _make_system(tmp_path, "sysA", n_runs=1, n_frames=4, temperature=300.0)
    with pytest.raises(KeyError, match="single-temperature"):
        bioemu.read_reference_ca(tmp_path, "sysA", 450, skip=1)
    bioemu.read_reference_ca(tmp_path, "sysA", 300, skip=1)  # matching T is fine


@pytest.mark.unit
def test_write_fasta_matches_the_reference_residue_count(tmp_path):
    _make_system(tmp_path, "sysA", n_runs=1, n_frames=4)
    (path,) = bioemu.write_fasta(tmp_path, ["sysA"], tmp_path / "fasta")
    seq = path.read_text().splitlines()[1]
    assert seq == "AG"
    assert len(seq) == bioemu.read_reference_ca(tmp_path, "sysA", skip=1).shape[1]


# ---------------------------------------------------------------------------
# Atom-name canonicalization (BioEmu path only -- mdCATH's cached maps predate it)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_canonical_md_atom_name():
    from mol_ensemble_gen.training.atom_map import canonical_md_atom_name

    assert canonical_md_atom_name("ILE", "CD") == "CD1"  # the big one: 104/159 mdCATH misses
    assert canonical_md_atom_name("LEU", "CD") == "CD"  # LEU has no CD slot; must not be renamed
    assert canonical_md_atom_name("ARG", "CD") == "CD"  # ARG's CD is a real slot
    assert canonical_md_atom_name("GLU", "OC1") == "O"  # GROMACS charged C-terminus
    assert canonical_md_atom_name("GLU", "OC2") is None  # ...its second oxygen is dropped
    assert canonical_md_atom_name("HIE", "CD") == "CD"  # residue name folded first


@pytest.mark.unit
def test_read_topology_keeps_records_and_indices_in_step(tmp_path):
    """A dropped atom must leave both parallel arrays, or every later coord shifts."""
    pdb = (
        "ATOM      1  N   ILE A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ILE A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CD  ILE A   1       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  OC1 ILE A   1       3.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      5  OC2 ILE A   1       4.000   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    )
    (tmp_path / "sysA").mkdir()
    (tmp_path / "sysA" / "topology.pdb").write_text(pdb)
    topo = bioemu.read_topology(tmp_path, "sysA")
    assert [nm for _, nm in topo.md_records] == ["N", "CA", "CD1", "O"]
    assert topo.heavy_indices.tolist() == [0, 1, 2, 3]  # OC2 (index 4) dropped from both
    assert len(topo.md_records) == len(topo.heavy_indices)
    assert topo.n_atoms == 5  # the full atom axis is unchanged


@pytest.mark.unit
def test_canonicalization_lifts_matched_fraction_on_real_topologies():
    """Regression guard on the real release: every heavy atom should find a slot."""
    from mol_ensemble_gen.training.atom_map import build_atom_map

    root = Path("/nfs/lambda_stor_01/homes/heng.ma/dataset/bioemu_cath/ONE_cath1")
    if not root.is_dir():
        pytest.skip("BioEmu ONE_cath1 not present")
    for system in bioemu.available_systems(root)[:4]:
        topo = bioemu.read_topology(root, system)
        amap = build_atom_map(topo.sequence, topo.md_records)
        assert amap.matched_fraction == pytest.approx(1.0), system


# ---------------------------------------------------------------------------
# Training stream
# ---------------------------------------------------------------------------


class _FakeMap:
    """Identity atom map over the 2 heavy atoms of the synthetic ALA-GLY topology."""

    heavy_indices = np.arange(9)
    present_mask = np.ones(9, bool)

    def scatter_batch(self, c):
        return c


def _dataset(tmp_path, monkeypatch, **kw):
    import mol_ensemble_gen.training.featurize as F

    monkeypatch.setattr(F, "load_atom_map", lambda *a, **k: _FakeMap())
    defaults = dict(
        root=bioemu.resolve_root(tmp_path),
        cache_dir=str(tmp_path),
        domains=["sysA", "sysB"],
        temperature=None,
        skip_frames=1,
        frames_per_step=8,
        rank=0,
        world_size=1,
        shuffle=False,
        units_per_domain=None,
    )
    return bioemu._dataset_cls()(**{**defaults, **kw})


@pytest.mark.unit
def test_stream_yields_frame_batches_at_the_system_temperature(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _make_system(tmp_path, "sysA", n_runs=1, n_frames=16, temperature=300.0)
    _make_system(tmp_path, "sysB", n_runs=1, n_frames=16, temperature=300.0)
    batches = list(_dataset(tmp_path, monkeypatch))
    assert len(batches) == 4  # 16 frames / 8 per step, two systems
    assert {b.domain for b in batches} == {"sysA", "sysB"}
    assert all(b.temperature == 300.0 for b in batches)
    assert all(b.gt_coords.shape == (8, 9, 3) for b in batches)
    # nm -> Å applies on the training path too, not just the eval path.
    first = next(b for b in batches if b.domain == "sysA")
    assert first.gt_coords[:, 0, 0].min() >= 0.0 and first.gt_coords[:, 0, 0].max() <= 150.0


@pytest.mark.unit
def test_stream_temperature_override(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _make_system(tmp_path, "sysA", n_runs=1, n_frames=8, temperature=300.0)
    (b,) = list(_dataset(tmp_path, monkeypatch, domains=["sysA"], temperature=320.0))
    assert b.temperature == 320.0


@pytest.mark.unit
def test_stream_skips_domains_without_a_cache(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    import mol_ensemble_gen.training.featurize as F

    _make_system(tmp_path, "sysA", n_runs=1, n_frames=8)
    monkeypatch.setattr(F, "load_atom_map", lambda c, d: (_ for _ in ()).throw(FileNotFoundError(d)))
    ds = bioemu._dataset_cls()(
        root=tmp_path,
        cache_dir=str(tmp_path),
        domains=["sysA"],
        temperature=None,
        skip_frames=1,
        frames_per_step=8,
        rank=0,
        world_size=1,
        shuffle=False,
    )
    assert list(ds) == []
    assert "no cache for sysA" in capsys.readouterr().out


@pytest.mark.unit
def test_stream_shards_disjointly_across_ranks(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    for name in ("sysA", "sysB", "sysC", "sysD"):
        _make_system(tmp_path, name, n_runs=1, n_frames=8)
    doms = ["sysA", "sysB", "sysC", "sysD"]
    seen = [{b.domain for b in _dataset(tmp_path, monkeypatch, domains=doms, rank=r, world_size=2)} for r in (0, 1)]
    assert seen[0] & seen[1] == set()
    assert seen[0] | seen[1] == set(doms)


@pytest.mark.unit
def test_units_per_domain_caps_the_validation_stream(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _make_system(tmp_path, "sysA", n_runs=1, n_frames=80)
    batches = list(_dataset(tmp_path, monkeypatch, domains=["sysA"], units_per_domain=3))
    assert len(batches) == 3  # not 10


@pytest.mark.unit
def test_unshuffled_stream_is_reproducible(tmp_path, monkeypatch):
    """_run_validation re-iterates the loader expecting identical frames."""
    pytest.importorskip("torch")
    _make_system(tmp_path, "sysA", n_runs=2, n_frames=24)
    ds = _dataset(tmp_path, monkeypatch, domains=["sysA"])
    first = [b.gt_coords.copy() for b in ds]
    second = [b.gt_coords.copy() for b in ds]
    assert len(first) == len(second)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


# ---------------------------------------------------------------------------
# Systems that ship without trajectories (real in MSR_cath2: 3 of 1,043)
# ---------------------------------------------------------------------------


def _make_topology_only_system(root, name):
    """A system directory with a topology and metadata but no trajs/ at all."""
    d = root / name
    d.mkdir(parents=True)
    (d / "topology.pdb").write_text(PDB_TEXT)
    (d / "dataset.json").write_text(json.dumps({"temperature_K": 300.0}))
    return d


@pytest.mark.unit
def test_available_systems_excludes_systems_without_trajectories(tmp_path):
    _make_system(tmp_path, "good", n_runs=1, n_frames=4)
    _make_topology_only_system(tmp_path, "bad")
    assert bioemu.available_systems(tmp_path) == ["good"]


@pytest.mark.unit
def test_has_trajectories_needs_an_actual_xtc(tmp_path):
    d = _make_topology_only_system(tmp_path, "bad")
    assert not bioemu.has_trajectories(d)
    (d / "trajs").mkdir()
    assert not bioemu.has_trajectories(d)  # empty trajs/ is still unusable
    (d / "trajs" / "run000_protein.cmprsd.xtc").write_bytes(b"")
    assert bioemu.has_trajectories(d)


@pytest.mark.unit
def test_stream_skips_trajectoryless_domains_instead_of_raising(tmp_path, monkeypatch, capsys):
    """One bad domain must not kill every DDP rank -- it took down a 4-GPU run."""
    pytest.importorskip("torch")
    _make_system(tmp_path, "good", n_runs=1, n_frames=8)
    _make_topology_only_system(tmp_path, "bad")
    batches = list(_dataset(tmp_path, monkeypatch, domains=["bad", "good"]))
    assert [b.domain for b in batches] == ["good"]
    assert "no trajectories for bad" in capsys.readouterr().out
