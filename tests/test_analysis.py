"""Offline unit tests for the analysis layer (numpy/scipy, no GPU)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mol_ensemble_gen.analysis import (
    analyze_run,
    cluster,
    confidence_filter,
    pairwise_rmsd,
    pca,
    representatives,
    rmsd,
    rmsf,
)
from mol_ensemble_gen.cif import ca_coords, parse_cif_atoms

_FIELDS = ["group_PDB", "label_atom_id", "label_comp_id", "label_asym_id",
           "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "B_iso_or_equiv"]


def _write_cif(path, coords, chain="A", plddt=88.0):
    """Write a minimal CA-only mmCIF whose atom_site loop the parser can read."""
    lines = ["data_test", "loop_"] + [f"_atom_site.{f}" for f in _FIELDS]
    for i, (x, y, z) in enumerate(coords, 1):
        lines.append(f"ATOM CA GLY {chain} {i} {x:.3f} {y:.3f} {z:.3f} {plddt:.2f}")
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


# -- cif parsing -----------------------------------------------------------


@pytest.mark.unit
def test_cif_roundtrip(tmp_path):
    coords = np.array([[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]])
    cif = tmp_path / "x.cif"
    _write_cif(cif, coords)

    atoms = parse_cif_atoms(cif)
    assert len(atoms) == 3
    res, ca = ca_coords(atoms)
    assert list(res) == [1, 2, 3]
    np.testing.assert_allclose(ca, coords)


# -- geometry --------------------------------------------------------------


@pytest.mark.unit
def test_rmsd_invariant_to_rigid_motion():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 3))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    b = a @ rot.T + np.array([10.0, -5.0, 3.0])   # rotate + translate
    assert rmsd(a, b) < 1e-6                        # superposition removes it
    assert rmsd(a, b, superpose=False) > 1.0        # ...but only if we superpose


@pytest.mark.unit
def test_pairwise_rmsd_matrix_properties():
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(4, 6, 3))
    m = pairwise_rmsd(coords)
    assert m.shape == (4, 4)
    np.testing.assert_allclose(np.diag(m), 0, atol=1e-9)
    np.testing.assert_allclose(m, m.T)


@pytest.mark.unit
def test_rmsf_zero_for_identical_ensemble():
    base = np.array([[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [11.4, 0, 0]])
    coords = np.stack([base, base, base])
    np.testing.assert_allclose(rmsf(coords), 0, atol=1e-9)


@pytest.mark.unit
def test_pca_shapes():
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(5, 6, 3))
    proj, var = pca(coords, n_components=2)
    assert proj.shape == (5, 2)
    assert len(var) == 2 and 0 <= var.sum() <= 1.0 + 1e-9


# -- clustering + confidence ----------------------------------------------


def _two_conformer_ensemble():
    """Two distinct conformations (a hinge move), 2 noisy members each."""
    rng = np.random.default_rng(3)
    base = np.array([[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0],
                     [11.4, 0, 0], [15.2, 0, 0], [19.0, 0, 0]])
    bent = base.copy()
    bent[3:] += np.array([0.0, 8.0, 0.0])          # displace the C-terminal half
    members = [base + rng.normal(scale=0.05, size=base.shape) for _ in range(2)]
    members += [bent + rng.normal(scale=0.05, size=bent.shape) for _ in range(2)]
    return np.stack(members)


@pytest.mark.unit
def test_cluster_separates_two_conformers():
    coords = _two_conformer_ensemble()
    mat = pairwise_rmsd(coords)
    labels = cluster(mat, cutoff=2.0)
    assert len(np.unique(labels)) == 2
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]

    reps = representatives(mat, labels)
    assert len(reps) == 2
    assert set(reps.values()).issubset(range(4))


@pytest.mark.unit
def test_confidence_filter():
    df = pd.DataFrame({"plddt": [0.9, 0.5, 0.8], "ptm": [0.7, 0.7, 0.3], "iptm": [0.6, 0.6, 0.6]})
    assert len(confidence_filter(df, min_plddt=0.7)) == 2
    assert len(confidence_filter(df, min_ptm=0.5)) == 2
    assert len(confidence_filter(df, min_plddt=0.7, min_ptm=0.5)) == 1


# -- end to end ------------------------------------------------------------


@pytest.mark.unit
def test_analyze_run_writes_outputs(tmp_path):
    coords = _two_conformer_ensemble()
    rows = []
    for i, c in enumerate(coords):
        cif = tmp_path / f"m{i}.cif"
        _write_cif(cif, c)
        rows.append({"member_idx": i, "sample_idx": 0, "seed": i,
                     "cif_path": str(cif), "plddt": 0.9, "ptm": 0.8, "iptm": ""})
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)

    result = analyze_run(tmp_path, cluster_cutoff=2.0)

    assert result.summary["n_members"] == 4
    assert result.summary["n_clusters"] == 2
    assert result.summary["max_pairwise_rmsd"] > 2.0
    assert "cluster" in result.metadata.columns
    for name in ("rmsd_matrix.npy", "analysis_metadata.csv", "pca.csv", "rmsf.csv", "analysis_summary.json"):
        assert (tmp_path / name).exists()


@pytest.mark.unit
def test_analyze_run_confidence_filter_can_empty(tmp_path):
    _write_cif(tmp_path / "m0.cif", np.zeros((3, 3)))
    pd.DataFrame([{"cif_path": str(tmp_path / "m0.cif"), "plddt": 0.5, "ptm": 0.5, "iptm": ""}]).to_csv(
        tmp_path / "metadata.csv", index=False)
    with pytest.raises(ValueError, match="no members pass"):
        analyze_run(tmp_path, min_plddt=0.9)
