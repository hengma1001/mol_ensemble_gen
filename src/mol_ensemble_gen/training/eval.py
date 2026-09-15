"""Evaluate how well a sampled ensemble reproduces reference MD ensembles.

Per temperature, compares a sampled ensemble (a ``T<K>/`` directory of CIFs from
:mod:`.sample`) against held-out reference Cα frames on distributional metrics that
do not assume calibrated populations — the appropriate frame for a
model-distribution sampler (see ``DESIGN.md`` §9):

* **RMSF correlation** (primary): per-residue fluctuation, Pearson + Spearman.
* **Radius-of-gyration KS**: global size/compactness distribution overlap.
* **RMSD-to-MD-mean KS**: spread-around-the-native distribution overlap.
* **PCA spread ratio**: sample vs MD std along MD-fit PC1/PC2.
* **Temperature monotonicity**: mean pairwise RMSD should rise with T.

Reuses :mod:`mol_ensemble_gen.analysis` for geometry (``kabsch``, ``rmsf``,
``pairwise_rmsd``) so the sampled- and MD-side metrics are computed identically.

Two reference sources are supported, selected by the ``reference`` argument:

* ``"mdcath"`` (default) — :mod:`.mdcath`, five temperatures per domain. The
  training distribution, so scores against it are in-distribution by construction.
* ``"bioemu"`` — :mod:`.bioemu`, the BioEmu CATH release. Single-temperature
  (~300 K) and a different force field (amber99sb-ildn vs mdCATH's CHARMM22*), so
  it is an *out-of-distribution* benchmark on both axes. Its ONE_cath1 subset is
  the reason to care: >100 µs per system means the distributional metrics below
  are measured against a converged ensemble rather than a 500 ns sample of one.
  ``spread_monotonic`` is undefined there — a single temperature is one point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import analysis
from .mdcath import read_reference_ca


def _mdcath_reference(md_dir, target, temperature, *, skip, max_frames):
    return read_reference_ca(md_dir, target, int(temperature), skip=skip, max_frames=max_frames)


def _bioemu_reference(md_dir, target, temperature, *, skip, max_frames):
    from .bioemu import read_reference_ca as read_bioemu_ca

    return read_bioemu_ca(md_dir, target, temperature, skip=skip, max_frames=max_frames)


#: Reference loaders by name; each returns Cα frames ``(F, L, 3)`` in Ångström.
REFERENCE_READERS = {"mdcath": _mdcath_reference, "bioemu": _bioemu_reference}

#: Default frame stride per reference. These are *not* interchangeable: mdCATH
#: writes frames every 1 ns, so a stride of 10 costs nothing but redundancy, while
#: BioEmu already saves every 10 ns (``save_traj_ns`` in its ``dataset.json``) and
#: the same stride would silently throw away 90% of an ensemble that is coarse to
#: begin with — ~2.5k frames per system rather than ~25k.
DEFAULT_SKIP = {"mdcath": 10, "bioemu": 1}


def get_reference_reader(name: str):
    """Look up a reference loader by name, erroring on an unknown one."""
    try:
        return REFERENCE_READERS[name]
    except KeyError:
        raise ValueError(f"unknown reference {name!r}; expected one of {sorted(REFERENCE_READERS)}") from None


def radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    """Rg per frame for ``(M, L, 3)`` coordinates."""
    centered = coords - coords.mean(axis=1, keepdims=True)
    return np.sqrt((centered**2).sum(-1).mean(-1))


def _align_to_ref(coords: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.stack([analysis.kabsch(c, ref) for c in coords])


def _rmsd_to_ref(coords: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.asarray([analysis.rmsd(c, ref) for c in coords])


def _load_sampled_ca(temp_dir: str | Path) -> np.ndarray:
    """Stack Cα coords ``(M, L, 3)`` from a sampled temperature directory."""
    import pandas as pd

    df = pd.read_csv(Path(temp_dir) / "metadata.csv")
    return analysis.load_coords(df)


def _pca_spread_ratio(sample_ca: np.ndarray, md_ca: np.ndarray) -> list[float]:
    """Std of sample vs MD along the top-2 PCs fit on the MD ensemble."""
    md_aligned = analysis._align_all(md_ca)
    ref = md_aligned.mean(0)
    md_flat = md_aligned.reshape(len(md_ca), -1)
    mean = md_flat.mean(0)
    _, _, vt = np.linalg.svd(md_flat - mean, full_matrices=False)
    pcs = vt[:2]  # (2, L*3)
    samp_flat = _align_to_ref(sample_ca, ref).reshape(len(sample_ca), -1)
    md_proj = (md_flat - mean) @ pcs.T
    samp_proj = (samp_flat - mean) @ pcs.T
    md_std = md_proj.std(0)
    samp_std = samp_proj.std(0)
    return [float(s / m) if m > 0 else 0.0 for s, m in zip(samp_std, md_std)]


def evaluate_temperature(
    sampled_dir: str | Path,
    mdcath_dir: str | Path,
    domain: str,
    temperature: int,
    *,
    skip: int = 10,
    max_md_frames: int | None = 2000,
    reference: str = "mdcath",
    md_temperature: float | int | None = None,
) -> dict:
    """Compute MD-match metrics for one temperature.

    ``md_temperature`` decouples the reference state from the sampled one: normally
    both are ``temperature``, but a model trained outside the reference's range has
    no honest single answer. Scoring mdCATH-trained weights against BioEmu's 300 K
    ensemble needs both readings — the model asked for 300 K (extrapolating below
    its 320-450 K training range) and asked for 320 K (its nearest trained state,
    which will overshoot a 300 K target). Reporting one alone flatters or penalises
    it by choice of protocol, so the result records which was used.
    """
    from scipy.stats import ks_2samp, pearsonr, spearmanr

    md_temp = temperature if md_temperature is None else md_temperature
    sample_ca = _load_sampled_ca(Path(sampled_dir) / f"T{int(temperature)}")
    md_ca = get_reference_reader(reference)(mdcath_dir, domain, md_temp, skip=skip, max_frames=max_md_frames)
    if sample_ca.shape[1] != md_ca.shape[1]:
        raise ValueError(f"residue count mismatch: sampled {sample_ca.shape[1]} vs MD {md_ca.shape[1]}")

    md_rmsf = analysis.rmsf(md_ca)
    samp_rmsf = analysis.rmsf(sample_ca)
    md_ref = analysis._align_all(md_ca).mean(0)

    md_rg, samp_rg = radius_of_gyration(md_ca), radius_of_gyration(sample_ca)
    md_rmsd = _rmsd_to_ref(_align_to_ref(md_ca, md_ref), md_ref)
    samp_rmsd = _rmsd_to_ref(_align_to_ref(sample_ca, md_ref), md_ref)

    return {
        "temperature": int(temperature),
        "md_temperature": float(md_temp),
        "reference": reference,
        "n_md_frames": int(md_ca.shape[0]),
        "n_samples": int(sample_ca.shape[0]),
        "n_res": int(md_ca.shape[1]),
        "rmsf_pearson": float(pearsonr(md_rmsf, samp_rmsf)[0]),
        "rmsf_spearman": float(spearmanr(md_rmsf, samp_rmsf)[0]),
        "rg_md_mean": float(md_rg.mean()),
        "rg_sample_mean": float(samp_rg.mean()),
        "rg_ks": float(ks_2samp(md_rg, samp_rg).statistic),
        "rmsd_ks": float(ks_2samp(md_rmsd, samp_rmsd).statistic),
        "sample_mean_pairwise_rmsd": float(
            analysis.pairwise_rmsd(sample_ca)[np.triu_indices(len(sample_ca), 1)].mean()
        ),
        "pca_spread_ratio": _pca_spread_ratio(sample_ca, md_ca),
    }


def evaluate_run(
    sampled_dir: str | Path,
    mdcath_dir: str | Path,
    domain: str,
    temperatures: list[int],
    *,
    skip: int = 10,
    write: bool = True,
    max_md_frames: int | None = 2000,
    reference: str = "mdcath",
    md_temperature: float | int | None = None,
) -> dict:
    """Evaluate every temperature and check temperature-monotonic spread.

    ``spread_monotonic`` is the Spearman correlation between temperature and the
    sampled ensemble's mean pairwise RMSD — positive means spread grows with T. It
    stays ``None`` for a single-temperature reference such as ``"bioemu"``, where
    there is no temperature axis to correlate against.
    """
    from scipy.stats import spearmanr

    per_temp = []
    for temp in temperatures:
        try:
            per_temp.append(
                evaluate_temperature(
                    sampled_dir,
                    mdcath_dir,
                    domain,
                    temp,
                    skip=skip,
                    max_md_frames=max_md_frames,
                    reference=reference,
                    md_temperature=md_temperature,
                )
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            per_temp.append({"temperature": int(temp), "error": repr(exc)})

    ok = [r for r in per_temp if "error" not in r]
    monotonic = None
    if len(ok) >= 2:
        temps = [r["temperature"] for r in ok]
        spread = [r["sample_mean_pairwise_rmsd"] for r in ok]
        monotonic = float(spearmanr(temps, spread)[0])

    summary = {
        "domain": domain,
        "reference": reference,
        "per_temperature": per_temp,
        "spread_monotonic": monotonic,
        "mean_rmsf_pearson": float(np.mean([r["rmsf_pearson"] for r in ok])) if ok else None,
    }
    if write:
        out = Path(sampled_dir) / "md_eval_summary.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"[eval] wrote {out}", flush=True)
    return summary
