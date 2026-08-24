"""Ensemble analysis: RMSD/RMSF, clustering, PCA, confidence filtering.

Reads an ensemble produced by :class:`~mol_ensemble_gen.ensemble.ESMFold2Ensemble`
(``metadata.csv`` + per-member ``.cif``) and turns it into a conformational
landscape: which structures cluster together, how much each residue moves, and
where members sit in coordinate PCA space. numpy + scipy only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cif import ca_coords, parse_cif_atoms

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def kabsch(mobile: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Superpose ``mobile`` (L,3) onto ``ref`` (L,3); return the aligned copy."""
    mc = mobile - mobile.mean(0)
    rc = ref - ref.mean(0)
    u, _, vt = np.linalg.svd(mc.T @ rc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return mc @ rot.T + ref.mean(0)


def rmsd(a: np.ndarray, b: np.ndarray, *, superpose: bool = True) -> float:
    """RMSD between two (L,3) coordinate sets, optimally superposed by default."""
    if superpose:
        a = kabsch(a, b)
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def pairwise_rmsd(coords: np.ndarray) -> np.ndarray:
    """(M,M) matrix of superposed pairwise RMSDs for coords stacked (M,L,3)."""
    m = len(coords)
    out = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            out[i, j] = out[j, i] = rmsd(coords[i], coords[j])
    return out


def _align_all(coords: np.ndarray) -> np.ndarray:
    """Iteratively align all members to their running mean (2 passes)."""
    ref = coords[0]
    aligned = coords
    for _ in range(2):
        aligned = np.stack([kabsch(c, ref) for c in coords])
        ref = aligned.mean(0)
    return aligned


def rmsf(coords: np.ndarray) -> np.ndarray:
    """Per-residue RMS fluctuation (L,) about the aligned mean structure."""
    aligned = _align_all(coords)
    mean = aligned.mean(0)
    return np.sqrt(((aligned - mean) ** 2).sum(-1).mean(0))


def pca(coords: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """PCA of aligned coordinates.

    Returns ``(projections (M,k), explained_variance_ratio (k,))``.
    """
    k = min(n_components, len(coords) - 1)
    flat = _align_all(coords).reshape(len(coords), -1)
    centered = flat - flat.mean(0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    proj = u[:, :k] * s[:k]
    var = (s**2) / (s**2).sum() if s.sum() else np.zeros_like(s)
    return proj, var[:k]


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def cluster(rmsd_matrix: np.ndarray, *, cutoff: float | None = 2.0, n_clusters: int | None = None) -> np.ndarray:
    """Average-linkage hierarchical clustering on an RMSD matrix.

    Provide either ``cutoff`` (RMSD in Å) or ``n_clusters``. Returns 1-based
    cluster labels (M,).
    """
    m = len(rmsd_matrix)
    if m < 2:
        return np.ones(m, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    z = linkage(squareform(rmsd_matrix, checks=False), method="average")
    if n_clusters is not None:
        return fcluster(z, t=n_clusters, criterion="maxclust")
    return fcluster(z, t=cutoff, criterion="distance")


def representatives(rmsd_matrix: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """Medoid index of each cluster (member with min mean RMSD to its cluster)."""
    reps: dict[int, int] = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        sub = rmsd_matrix[np.ix_(members, members)]
        reps[int(c)] = int(members[sub.mean(1).argmin()])
    return reps


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------


def confidence_filter(
    df: pd.DataFrame, *, min_plddt: float = 0.0, min_ptm: float = 0.0, min_iptm: float | None = None
) -> pd.DataFrame:
    """Keep only members passing confidence thresholds."""
    keep = df["plddt"] >= min_plddt
    if "ptm" in df:
        keep &= df["ptm"].fillna(0) >= min_ptm
    if min_iptm is not None and "iptm" in df:
        keep &= df["iptm"].fillna(0) >= min_iptm
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# high-level driver
# ---------------------------------------------------------------------------


@dataclass
class EnsembleAnalysis:
    metadata: pd.DataFrame  # with a 'cluster' column added
    rmsd_matrix: np.ndarray
    rmsf: np.ndarray
    pca_proj: np.ndarray
    pca_var: np.ndarray
    representatives: dict[int, int]

    @property
    def summary(self) -> dict:
        m = self.rmsd_matrix
        off = m[np.triu_indices(len(m), k=1)] if len(m) > 1 else np.array([0.0])
        return {
            "n_members": int(len(self.metadata)),
            "n_clusters": int(self.metadata["cluster"].nunique()),
            "mean_pairwise_rmsd": float(off.mean()),
            "max_pairwise_rmsd": float(off.max()),
            "mean_plddt": float(self.metadata["plddt"].mean()),
            "max_rmsf": float(self.rmsf.max()),
            "pca_explained_variance": [float(v) for v in self.pca_var],
            "representatives": {str(k): self.metadata.iloc[v]["cif_path"] for k, v in self.representatives.items()},
        }


def load_coords(metadata: pd.DataFrame, chain: str | None = None) -> np.ndarray:
    """Stack Cα coordinates (M,L,3) for every member in ``metadata``."""
    coords = [ca_coords(parse_cif_atoms(p), chain)[1] for p in metadata["cif_path"]]
    lengths = {len(c) for c in coords}
    if len(lengths) != 1:
        raise ValueError(
            f"members have differing Cα counts {sorted(lengths)}; " "pass chain= or ensure a consistent sequence"
        )
    return np.stack(coords)


def analyze_run(
    out_dir: str | Path,
    *,
    min_plddt: float = 0.0,
    min_ptm: float = 0.0,
    min_iptm: float | None = None,
    cluster_cutoff: float = 2.0,
    n_clusters: int | None = None,
    chain: str | None = None,
    write: bool = True,
) -> EnsembleAnalysis:
    """Analyze an ensemble directory and (optionally) write result files."""
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "metadata.csv")
    df = confidence_filter(df, min_plddt=min_plddt, min_ptm=min_ptm, min_iptm=min_iptm)
    if len(df) == 0:
        raise ValueError("no members pass the confidence filter")

    coords = load_coords(df, chain)
    mat = pairwise_rmsd(coords)
    flu = rmsf(coords)
    proj, var = pca(coords)
    labels = cluster(mat, cutoff=None if n_clusters else cluster_cutoff, n_clusters=n_clusters)
    reps = representatives(mat, labels)
    df = df.assign(cluster=labels)

    result = EnsembleAnalysis(df, mat, flu, proj, var, reps)

    if write:
        np.save(out_dir / "rmsd_matrix.npy", mat)
        df.to_csv(out_dir / "analysis_metadata.csv", index=False)
        pd.DataFrame(proj, columns=[f"PC{i + 1}" for i in range(proj.shape[1])]).assign(cluster=labels).to_csv(
            out_dir / "pca.csv", index=False
        )
        pd.DataFrame({"residue": np.arange(1, len(flu) + 1), "rmsf": flu}).to_csv(out_dir / "rmsf.csv", index=False)
        (out_dir / "analysis_summary.json").write_text(json.dumps(result.summary, indent=2))

    return result
