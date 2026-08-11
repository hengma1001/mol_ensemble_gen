"""Tests for PDB-seeded ensemble runs.

The parser and the end-to-end wiring are covered offline (fake model). A real
GPU fold is provided as an opt-in integration test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mol_ensemble_gen.ensemble import EnsembleSpec, ESMFold2Ensemble, SamplingParams
from mol_ensemble_gen.pdb import parse_pdb_sequences

# chignolin GYDPETGTWG, plus a second chain and noise to exercise the parser
_PDB = """\
REMARK test
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA ATYR A   2       4.800   0.000   0.000  0.60  0.00           C
ATOM      4  CA BTYR A   2       4.900   0.000   0.000  0.40  0.00           C
ATOM      5  CA  ASP A   3       8.600   0.000   0.000  1.00  0.00           C
HETATM    6  O   HOH A 101       9.000   0.000   0.000  1.00  0.00           O
ATOM      7  CA  MSE B   1       0.000   5.000   0.000  1.00  0.00           C
ATOM      8  CA  TRP B   2       3.800   5.000   0.000  1.00  0.00           C
ENDMDL
ATOM      9  CA  ALA A   4      12.000   0.000   0.000  1.00  0.00           C
END
"""


@pytest.mark.unit
def test_parse_pdb_sequences(tmp_path):
    pdb = tmp_path / "s.pdb"
    pdb.write_text(_PDB)

    seqs = parse_pdb_sequences(pdb)

    assert list(seqs) == ["A", "B"]        # chain order preserved
    assert seqs["A"] == "GYD"              # CA-only, altloc A kept once, HETATM & 2nd model ignored
    assert seqs["B"] == "MW"               # MSE mapped to M


@pytest.mark.unit
def test_parse_pdb_empty_raises(tmp_path):
    pdb = tmp_path / "empty.pdb"
    pdb.write_text("REMARK nothing here\nEND\n")
    with pytest.raises(ValueError, match="no protein"):
        parse_pdb_sequences(pdb)


@pytest.mark.unit
def test_generate_from_pdb_end_to_end(tmp_path, monkeypatch):
    """PDB -> parse -> fold (fake) -> per-member .cif + metadata, no GPU."""

    class _Arr:
        def mean(self):
            return 0.88

    class _Cplx:
        def to_mmcif(self):
            return "data_x\n"

    class _Res:
        complex = _Cplx()
        plddt = _Arr()
        ptm = 0.7
        iptm = None

    pdb = tmp_path / "prot.pdb"
    pdb.write_text(_PDB)

    ens = ESMFold2Ensemble.__new__(ESMFold2Ensemble)
    ens.spec = EnsembleSpec(members=3, sampling=SamplingParams(num_diffusion_samples=1))
    ens.device = "cpu"
    ens.model = object()
    ens._builder = type("B", (), {"fold": staticmethod(lambda *a, seed, **k: _Res())})()
    # avoid importing esm: stub the protein-SPI builder with the parsed sequences
    monkeypatch.setattr(ESMFold2Ensemble, "_protein_spi", staticmethod(lambda seqs: seqs))

    members = ens.generate_from_pdb(pdb, tmp_path / "out")

    assert len(members) == 3
    out = tmp_path / "out"
    assert sorted(p.name for p in out.glob("*.cif")) == [
        "prot_m0000_s00.cif", "prot_m0001_s00.cif", "prot_m0002_s00.cif"
    ]
    assert (out / "metadata.csv").exists()


@pytest.mark.integration
@pytest.mark.gpu
def test_real_pdb_fold_run(tmp_path):
    """Opt-in real fold on GPU (downloads weights). Run with: pytest -m gpu"""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU available")

    example = Path(__file__).resolve().parents[1] / "examples" / "example.pdb"
    spec = EnsembleSpec(
        members=2, base_seed=1,
        sampling=SamplingParams(num_loops=4, num_sampling_steps=20, num_diffusion_samples=1),
    )
    ens = ESMFold2Ensemble(spec, device="cuda")
    members = ens.generate_from_pdb(example, tmp_path / "run")

    assert len(members) == 2
    assert all(Path(m.cif_path).exists() for m in members)
    assert all(0.0 <= m.plddt <= 1.0 for m in members)
