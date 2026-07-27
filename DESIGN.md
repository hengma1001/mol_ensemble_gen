# `mol_ensemble_gen` — Design

Generate biomolecular **configuration ensembles** (conformational / complex
ensembles) with **ESMFold2**, a diffusion-based all-atom structure predictor.
Because ESMFold2 is a *diffusion* model, ensembles come almost for free: vary the
seed and the sampling knobs and collect the draws.

> **What ESMFold2 is** (verified against the installed `esm` Biohub fork):
> a diffusion-based, all-atom, **multi-entity** predictor — protein / DNA / RNA /
> ligand complexes — in the AF3 / Boltz family (reports `plddt`, `ptm`, `iptm`,
> `pae`, `pde`). Loaded via `transformers` `ESMFold2Model` (`biohub/ESMFold2`)
> and driven through `esm.models.esmfold2.ESMFold2InputBuilder().fold(...)`.
> It takes a Boltz-style input (`StructurePredictionInput`) — this is *not* the
> old single-sequence ESMFold.

---

## 1. Where diversity comes from

The `fold()` call already exposes every sampling dial as a keyword argument, so
**no PyTorch hook surgery is needed** — the sampler just sweeps these:

```
fold(model, input, *, num_loops=20, num_sampling_steps=200, num_diffusion_samples=1,
     seed=None, noise_scale=None, step_scale=None, max_inference_sigma=None,
     lm_mask_pct=None, early_exit=False, lm_dropout=0.3,
     msa_max_depth=1024, msa_column_mask_rate=0.1, complex_id='pred')
  -> MolecularComplexResult | list[MolecularComplexResult]
```

| Diversity mechanism | kwarg | Notes |
|---|---|---|
| **Independent diffusion trajectories** | `seed` | Primary knob. One seed → one member. Reproducible. |
| **Multiple samples per forward** | `num_diffusion_samples` | Returns a **list**; amortizes the trunk. Ensemble size = `members × num_diffusion_samples`. |
| **Diffusion temperature** | `noise_scale`, `step_scale`, `max_inference_sigma` | Higher = more spread, lower plausibility. Main diversity/quality dial. |
| **LM dropout (MC dropout)** | `lm_dropout` (default 0.3) | Perturbs the language-model features. |
| **Sequence masking** | `lm_mask_pct` | Single-sequence analog of MSA subsampling. |
| **MSA subsampling** | `msa_max_depth`, `msa_column_mask_rate` | When an MSA is provided per entity. |
| **Convergence** | `num_loops`, `num_sampling_steps` | Quality vs speed; low values = fast previews. |

> The three strategies originally requested (MC dropout, sequence masking,
> embedding noise) map onto `lm_dropout`, `lm_mask_pct`, and
> `noise_scale`/`max_inference_sigma` respectively — all native. The
> forward-hook `perturbations/` module from the first draft is **dropped**.

---

## 2. Architecture

```
input (FASTA | Boltz YAML) ──▶ EnsembleGenerator ──▶ Executor ──▶ per-member .cif + confidence
                                   │  (backend + SamplingParams + per-member seeds)
                                   ▼
                              Ensemble ──▶ I/O (multi-model / trajectory + metadata)
                                   │
                                   ▼
                              Analysis (RMSD/RMSF, clustering, PCA, confidence filtering)
```

### Module layout
```
src/mol_ensemble_gen/
  __init__.py
  esmfold2.py            # ESMFold2Model_API — backend wrapper (model load + single fold)   [exists]
  utils.py               # FASTA / Boltz-schema parsing                                      [exists]
  ensemble.py            # SamplingParams, EnsembleSpec, EnsembleMember, ESMFold2Ensemble    [NEW]
  execution/
    base.py              # Executor protocol + WorkItem/Result
    local.py             # LocalGPUExecutor (torch.multiprocessing work queue, model per GPU)
    slurm.py             # SlurmExecutor (job-array template + submission)
    manifest.py          # checkpoint / restart (completed WorkItems)
  io/
    trajectory.py        # stack per-member .cif -> .xtc/.dcd + topology (MDAnalysis/mdtraj)
    metadata.py          # ensemble metadata table (json/csv now; parquet when pandas added)
  analysis/
    rmsd.py              # superposition, pairwise RMSD matrix, RMSF
    cluster.py           # hierarchical clustering on RMSD
    reduce.py            # PCA of Cα / all-atom coordinates
    confidence.py        # plddt / ptm / iptm / pae summaries + filtering
  cli.py                 # typer app: run | analyze | slurm-submit | inspect
```

### Key abstractions

**Backend** — `esmfold2.py:ESMFold2Model_API` already wraps model load + a single
`fold`. The ensemble layer reuses a *preloaded* model so the executor can load
once per GPU and stream members through it (load dominates cost).

**`SamplingParams`** — dataclass of the `fold()` diversity kwargs. `None` means
"use the library default", so only explicitly-set knobs are passed. This is the
sweep definition.

**`EnsembleSpec`** — `members` (number of independent seeds), `base_seed`, and a
`SamplingParams`. Per-member seed is derived deterministically:
`seed = blake2b(base_seed | protein_id | member_idx)` → reproducible ensembles.

**`ESMFold2Ensemble`** — orchestrates: for each member, derive seed → `fold` →
**normalize list vs single result** → write one `.cif` per (member, diffusion
sample) with a unique name → collect confidence + provenance into a manifest.

**`Executor`** (phase 2) — consumes `WorkItem(input_id, member_idx, seed)`:
- `LocalGPUExecutor`: `torch.multiprocessing`, one worker/model per visible GPU.
- `SlurmExecutor`: renders an `sbatch --array=0-N%K` script; tasks shard the manifest.
- `Manifest`: idempotent restart — skip members whose `.cif` + metadata row exist.

---

## 3. Data & outputs

- **`EnsembleMember`**: `member_idx`, `sample_idx`, `seed`, `cif_path`, `plddt`
  (mean), `ptm`, `iptm` (None for monomers), `params` (full provenance).
- **`MolecularComplexResult`** fields available for richer analysis: `complex`,
  `plddt`, `ptm`, `iptm`, `pae`, `pde`, `distogram`, `pair_chains_iptm`,
  `residue_index`, `entity_id`, `num_tokens`.
- **On disk**, per input:
  - `{id}_m{member:04d}_s{sample:02d}.cif` — one structure per draw, unique names.
  - `manifest.json` — resolved `EnsembleSpec` + list of members (provenance).
  - `metadata.csv` — one row per member: seed, sampling params, plddt/ptm/iptm, path.
  - `ensemble.xtc` + `topology.(pdb|cif)` — trajectory form (phase: io/trajectory).

## 4. Analysis (post-hoc, reads `metadata.csv` + `.cif`)

- Superposition + pairwise Cα (or all-atom / per-entity) **RMSD** matrix, **RMSF**.
- **Clustering** on RMSD → representative per cluster.
- **PCA** of coordinates → conformational landscape colored by pLDDT / cluster.
- **Confidence filtering**: drop low-`ptm` / low-`iptm` / high-`pae` members before
  analysis — diffusion + high `noise_scale` can produce implausible members.
- Complex-aware: per-chain metrics + interface (`iptm`, `pair_chains_iptm`).

## 5. Config (dataclasses + YAML — matches existing style)

```yaml
input: complex.yaml            # Boltz-style YAML, or a .fasta
out_dir: runs/exp1
ensemble:
  members: 50                  # independent seeds
  base_seed: 20260722
sampling:
  num_loops: 20
  num_sampling_steps: 200
  num_diffusion_samples: 2     # -> 100 structures total
  lm_dropout: 0.3
  lm_mask_pct: 0.15
  noise_scale: 1.0             # diffusion temperature
  msa_max_depth: 1024
execution:
  backend: local               # local | slurm
  gpus: [0,1,2,3,4,5,6,7]
  slurm: {partition: gpu, time: "04:00:00", array_throttle: 8}
```

## 6. CLI

```
mol-ensemble-gen run     config.yaml              # local or slurm per config
mol-ensemble-gen slurm-submit config.yaml         # render + sbatch the array
mol-ensemble-gen analyze runs/exp1 --cluster --pca
mol-ensemble-gen inspect runs/exp1/metadata.csv
```

## 7. Dependencies

Core: `esm@git+https://github.com/Biohub/esm.git` (installed; pulls `torch` +
`transformers` with the `esmfold2` model). Add for I/O + analysis:
`MDAnalysis` (trajectory), `biotite` (structure RMSD), `numpy`, `scikit-learn`,
`pandas`/`pyarrow` (metadata), `typer`+`pyyaml`+`tqdm` (CLI). No OpenFold build.

## 8. HPC / V100-32GB notes

- Model load dominates → **load once per worker**, stream members through it.
- `num_diffusion_samples` amortizes the trunk vs re-folding per seed — cheaper
  spread, but samples from one forward are correlated; mix with seed variation.
- Lower `num_loops`/`num_sampling_steps` for fast previews; raise for production.
- Manifest-based restart makes SLURM array preemption safe.
- Per-member seeds → reproducible inputs (CUDA kernels add minor nondeterminism).

## 9. Caveats

Diffusion ensembles are samples from the model's learned distribution, **not**
Boltzmann-weighted populations. Confidence-filter members and read them as
*plausible alternative configurations*. `noise_scale` / `lm_mask_pct` trade
diversity against plausibility and should be calibrated per system.

## 10. Phased implementation

1. **Sampler** (`ensemble.py`): seeds, list-vs-single handling, per-member `.cif`,
   manifest + metadata. ✅ done
2. **PDB input** (`pdb.py`): seed an ensemble from an experimental structure's
   sequence. ✅ done
3. **Analysis** (`analysis.py`): RMSD/RMSF, clustering, PCA, confidence filtering. ✅ done
4. **Local multi-GPU executor** (`execution.py`) + per-member checkpoint restart. ✅ done
5. **I/O**: trajectory export (`.xtc`/`.dcd` + topology). ← *remaining*
6. **SLURM executor**: array template + `slurm-submit`. ← *remaining*
7. **CLI + example configs + `@pytest.mark.gpu` integration test.** ✅ (SLURM subcommand pending)

### Multi-GPU executor (phase 4, implemented)

`execution.py` fans an ensemble across local GPUs with `torch.multiprocessing.spawn`:
one worker/model per GPU, member indices sharded round-robin (`plan_shards`). Each
completed member writes an atomic `.member_XXXX.json` checkpoint; a restart skips
members already on disk (`pending_members`) and the parent merges all checkpoints
into `metadata.csv` / `manifest.json` (`merge_sidecars`). Workers rebuild the input
per process via `ESMFold2Ensemble.build_spi(path)` — no model objects are pickled
across the spawn boundary. Only the parent writes shared metadata; workers write
only their own uniquely-named `.cif`s and checkpoints, so there is no contention.

Driven from the CLI by an `execution:` config block or `--gpus 0,1,2,3`:

```yaml
execution:
  backend: local            # local (fan across gpus) | inline (single device)
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
```
