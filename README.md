# mol_ensemble_gen

Generate biomolecular **configuration/conformational ensembles** with
[ESMFold2](https://github.com/Biohub/esm) — the diffusion-based, all-atom,
multi-entity (protein / DNA / RNA / ligand) structure predictor from Biohub.

A single ESMFold2 `fold` is stochastic, so an *ensemble* is produced by folding
the same input under many independent seeds while sweeping the sampling knobs
that control diversity. Every structure is attributed to the exact
`(seed, sampling params)` that produced it, so runs are reproducible and members
are traceable. The package fans members across local GPUs and ships an analysis
layer (RMSD/RMSF, clustering, PCA, confidence filtering).

- **Input**: FASTA, experimental PDB (uses its sequence), or a Boltz-style YAML complex.
- **Diversity**: native `fold` kwargs — no PyTorch hook surgery.
  - MC dropout → `lm_dropout`
  - sequence masking → `lm_mask_pct`
  - embedding/diffusion noise → `noise_scale` / `max_inference_sigma`
- **Scale**: single device, or multi-GPU fan-out with per-member checkpoint/restart.
- **Analysis**: pairwise RMSD, RMSF, PCA, hierarchical clustering + medoid
  representatives, and confidence (pLDDT / pTM / ipTM) filtering.

## Install

```bash
pip install -e .          # installs esm (from git), numpy, scipy, pandas, pyyaml
```

ESMFold2 weights (`biohub/ESMFold2`) are downloaded on first use and require a
CUDA GPU to run.

## Quick start

### CLI

```bash
# Generate an ensemble from a config
mol-ensemble-gen run examples/config.yaml

# Fan members across local GPUs (restart-safe)
mol-ensemble-gen run examples/config_multigpu.yaml
mol-ensemble-gen run examples/config.yaml --gpus 0,1,2,3,4,5,6,7

# Override config fields on the command line
mol-ensemble-gen run examples/config.yaml --members 50 --out-dir runs/exp2

# Analyze a finished run
mol-ensemble-gen analyze runs/example --cluster-cutoff 2.0 --min-plddt 0.7
```

A run writes, into `out_dir`:

- `<input>_m<member>_s<sample>.cif` — one structure per (seed, diffusion sample)
- `manifest.json` — spec + full per-member provenance
- `metadata.csv` — flat table (member, seed, pLDDT, pTM, ipTM, sampling params)

`analyze` adds `rmsd_matrix.npy`, `analysis_metadata.csv` (with cluster labels),
`pca.csv`, `rmsf.csv`, and `analysis_summary.json`.

### Python API

```python
from mol_ensemble_gen import ESMFold2Ensemble, EnsembleSpec, SamplingParams, analyze_run

spec = EnsembleSpec(
    members=32,
    base_seed=20260722,
    sampling=SamplingParams(
        num_diffusion_samples=1,
        noise_scale=1.0,     # diffusion temperature (diversity dial)
        lm_dropout=0.3,      # MC dropout on LM features
    ),
)

ens = ESMFold2Ensemble(spec, device="cuda")
members = ens.generate_from_pdb("examples/example.pdb", "runs/example")   # or _from_fasta / _from_yaml

result = analyze_run("runs/example", cluster_cutoff=2.0)
print(result.summary)   # n_members, n_clusters, mean/max pairwise RMSD, max RMSF, PCA variance
```

## Configuration

```yaml
# examples/config_multigpu.yaml
input: examples/example.pdb          # .fasta | .pdb | Boltz-style .yaml
out_dir: runs/example_multigpu
model_name: biohub/ESMFold2

execution:
  backend: local                     # local (fan across gpus) | inline (single device)
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]     # one worker/model per GPU

ensemble:
  members: 64                        # independent seeds
  base_seed: 20260722                # reproducible per-member seeds

sampling:
  num_loops: 20
  num_sampling_steps: 200
  num_diffusion_samples: 1           # structures per seed
  noise_scale: 1.0                   # diffusion temperature
  lm_dropout: 0.3                    # MC dropout on LM features
  lm_mask_pct: 0.15                  # sequence masking
  # step_scale, max_inference_sigma, msa_max_depth, msa_column_mask_rate ...
```

Per-member seeds are derived deterministically via
`blake2b(base_seed:input_id:member_idx)`, so an ensemble is reproducible and
shardable across workers without coordination.

## Multi-GPU execution

`LocalGPUExecutor` (`mol_ensemble_gen.execution`) fans members across GPUs with
`torch.multiprocessing.spawn` — one worker/model per GPU, member indices sharded
round-robin. Each completed member writes an atomic `.member_XXXX.json`
checkpoint, so an interrupted run resumes by skipping members already on disk and
the parent merges everything into `metadata.csv` / `manifest.json`. Workers
rebuild the input per process (no model objects pickled across the spawn
boundary), and only the parent writes shared metadata — no contention.

## Project layout

```
src/mol_ensemble_gen/
  ensemble.py      # ESMFold2Ensemble, EnsembleSpec, SamplingParams, seed derivation
  execution.py     # LocalGPUExecutor: multi-GPU fan-out + checkpoint/restart
  analysis.py      # RMSD/RMSF, PCA, clustering, confidence filtering
  cif.py / pdb.py  # mmCIF atom_site reader; PDB sequence parser
  cli.py           # `mol-ensemble-gen run | analyze`
DESIGN.md          # architecture + phased implementation status
examples/          # config.yaml, config_multigpu.yaml, example.fasta/.pdb
tests/             # offline unit tests + @pytest.mark.gpu integration tests
```

See [DESIGN.md](DESIGN.md) for architecture and roadmap (trajectory export and a
SLURM executor are the remaining phases).

## Testing

```bash
# Fast offline suite (no GPU, no model load)
env PYTHONPATH=src python -m pytest tests/ -q -m "not integration"

# Opt-in real GPU folds (downloads weights)
env PYTHONPATH=src python -m pytest tests/ -q -m gpu
```
