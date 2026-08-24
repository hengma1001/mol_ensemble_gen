# mol_ensemble_gen

Generate biomolecular **conformational ensembles** with
[ESMFold2](https://github.com/Biohub/esm) — the diffusion-based, all-atom,
multi-entity structure predictor from Biohub — and **finetune its denoiser to
reproduce molecular-dynamics ensembles**, conditioned on temperature.

The package covers two related jobs:

1. **Sampling ensembles from the pretrained model** (`run`, `analyze`). A single
   ESMFold2 fold is stochastic, so an ensemble comes from folding one input under
   many independent seeds while sweeping the knobs that control diversity. Every
   structure is attributed to the exact `(seed, sampling params)` that produced it.
2. **Finetuning on mdCATH** (`featurize-cache`, `finetune`, `sample-md`,
   `eval-md`). mdCATH samples each CATH domain at five temperatures
   (320/348/379/413/450 K) across five replicas; one finetuned denoiser should
   reproduce all of them, so that a temperature becomes a *dial* on conformational
   spread.

Both paths share the analysis layer (RMSD/RMSF, clustering, PCA, confidence
filtering) and the multi-GPU fan-out.

## Install

```bash
pip install -e .          # installs esm (from git), numpy, scipy, pandas, pyyaml
```

ESMFold2 weights (`biohub/ESMFold2`) download on first use and need a CUDA GPU.

If the package is not installed into the active environment, every command below
also works as `env PYTHONPATH=src python -m mol_ensemble_gen.cli ...`.

## CLI

```
mol-ensemble-gen {run,analyze,featurize-cache,finetune,sample-md,eval-md,slurm-train}
```

| command | what it does |
| --- | --- |
| `run` | generate an ensemble from a YAML config |
| `analyze` | RMSD/RMSF/PCA/clustering over a finished run |
| `featurize-cache` | cache frozen-trunk conditioning + atom maps per mdCATH domain |
| `finetune` | train the diffusion denoiser on mdCATH (flow matching or EDM) |
| `sample-md` | sample a temperature-conditioned ensemble from a checkpoint |
| `eval-md` | compare a sampled ensemble against the MD reference |
| `slurm-train` | submit a finetuning job to SLURM |

## Sampling from the pretrained model

```bash
mol-ensemble-gen run examples/config.yaml                       # single device
mol-ensemble-gen run examples/config_multigpu.yaml              # fan across GPUs
mol-ensemble-gen run examples/config.yaml --gpus 0,1,2,3        # or override
mol-ensemble-gen run examples/config.yaml --members 50 --out-dir runs/exp2
mol-ensemble-gen analyze runs/example --cluster-cutoff 6.0 --min-plddt 0.7
```

Diversity comes from native `fold` kwargs — no hook surgery: MC dropout
(`lm_dropout`), sequence masking (`lm_mask_pct`), and embedding/diffusion noise
(`noise_scale`, `max_inference_sigma`).

A run writes into `out_dir`:

- `<input>_m<member>_s<sample>.cif` — one structure per (seed, diffusion sample)
- `manifest.json` — spec plus full per-member provenance
- `metadata.csv` — flat table (member, seed, pLDDT, pTM, ipTM, sampling params)

`analyze` adds `rmsd_matrix.npy`, `analysis_metadata.csv` (with cluster labels),
`pca.csv`, `rmsf.csv`, `analysis_summary.json`.

### Python API

```python
from mol_ensemble_gen import ESMFold2Ensemble, EnsembleSpec, SamplingParams, analyze_run

spec = EnsembleSpec(
    members=32,
    base_seed=20260722,
    sampling=SamplingParams(num_diffusion_samples=1, noise_scale=1.0, lm_dropout=0.3),
)
ens = ESMFold2Ensemble(spec, device="cuda")
members = ens.generate_from_pdb("examples/example.pdb", "runs/example")

result = analyze_run("runs/example", cluster_cutoff=6.0)
print(result.summary)
```

Per-member seeds derive deterministically from
`blake2b(base_seed:input_id:member_idx)`, so ensembles are reproducible and
shardable across workers without coordination.

## Finetuning on MD ensembles (mdCATH)

```bash
# 1. cache the frozen-trunk conditioning once per domain (GPU, parallelizable by
#    sharding the domain list across configs)
mol-ensemble-gen featurize-cache examples/finetune_mdcath_bal1200.yaml

# 2. train the denoiser (single GPU, or torchrun for DDP)
mol-ensemble-gen finetune examples/finetune_mdcath_bal1200.yaml
torchrun --standalone --nproc_per_node=8 -m mol_ensemble_gen.cli finetune <config>

# 3. sample a held-out domain at every temperature
mol-ensemble-gen sample-md <config> --input domain.fasta --members 30 \
    --temperatures 320,348,379,413,450 --out-dir runs/eval/dom

# 4. score against MD
mol-ensemble-gen eval-md <config> --sampled-dir runs/eval/dom --domain 1abcA00
```

### How it works

Only the **diffusion denoiser** is trained. The trunk's output is captured once
per domain by `featurize-cache` and replayed, so at train and sample time the
sole part of ESMFold2 that runs is `structure_head.diffusion_module` plus two
stateless geometry helpers. That makes training cheap and lets the package use
its own reimplementation of the denoiser
(`model/denoiser.py`, `backend: ours`) — module and parameter names mirror the
reference exactly, so pretrained tensors load with a strict
`load_state_dict` (345 tensors totalling 131,501,446 elements — of which
131,500,934 are trainable, the rest two fixed Fourier buffers; 350 tensors with
the native flow-time head). `tests/test_model_denoiser.py` asserts the structural signature
offline and numerical parity against the real module on GPU.

**Two training schemes** share one implementation, because the pretrained network
is an EDM x₀-predictor `D(x; σ)`:

- `scheme: edm` — Karras noise, `σ = σ_d·exp(p_mean + p_std·N(0,1))`.
- `scheme: flow` (default) — rectified flow on the linear path
  `x_t = (1−t)x₀ + t·ε`, reusing the same weights via `σ = t/(1−t)`. Velocity
  matching reduces to a `1/t²`-weighted coordinate MSE. **The `ln σ_d` shift in
  the time draw is load-bearing**: without it the model trains at `1/σ_d` of the
  intended noise level, disjoint from where the sampler starts.

**Temperature** enters through the single representation: a Fourier embedding of
the normalised temperature feeds an MLP producing a 451-dim bias added to every
token's `s_inputs`. The output layer is zero-initialised, so at step 0 the model
reproduces the pretrained one exactly and temperature is learned from that
identity start. `temperature.film: true` adds a multiplicative gain alongside the
bias (also identity at init).

Atom mapping (`training/atom_map.py`) matches mdCATH heavy atoms to model slots by
`(res_idx, atom_name)`; unmatched slots are **masked, never zero-filled**, and
domains below `min_matched_fraction` are dropped rather than trained on.

Example configs in `examples/` are named for the experiment they encode
(`finetune_mdcath_bal1200.yaml` is the current best: 1,196 length-balanced
domains, flow, `p_std: 2.2`, one epoch).

## Evaluation protocol

Getting this right took several wrong turns, so the conventions are worth stating
explicitly. They are not defaults — the defaults are wrong for this task.

**Metric.** Mean `|log(model_spread / MD_spread)|` over all
(domain, temperature) cells, where spread is the mean pairwise Cα RMSD of the
ensemble. Zero is perfect. A plain "% of MD spread" average is **misleading**: it
rewards over-shooting, so a model at 161% of MD's spread at 320 K scores as if it
beat one sitting at 100%.

**Clustering cutoff is 6.0 Å**, not the 2.0 Å default in `analysis.py`. At 2.0 Å
every ensemble looks fully fragmented (≈30 clusters, ≈3% in the largest) and the
numbers are incomparable with any earlier table.

**MD reference.** 30 frames per (domain, temperature): six evenly spaced frames
from each of the five replicas. Drawing 30 frames from one replica badly
understates spread at low temperature. Check the per-replica breakdown before
trusting a cell — some are dominated by a single partial-unfolding excursion
(`16pkA02`/379 K has one replica at 22 Å against 2.6–3.3 Å for the other four),
which is a rare event rather than an equilibrium ensemble.

**Use all seven held-out domains.** The three-domain protocol has a seed-to-seed
standard deviation of ~0.027 on the aggregate error, which cannot resolve the
0.02–0.04 differences that training changes actually produce. Seven domains
(35 cells) brings it to **~0.007**. Quote effects as multiples of that.

**Validation loss is not a proxy for ensemble quality.** It has now missed four
separate real effects, and until 2026-08-14 it also carried a 2–4% measurement
noise floor of its own (an unseeded rigid augmentation in the loss path, since
fixed — see `augment_with_generator`). Decide with the sampling metric.

## Multi-GPU execution

`LocalGPUExecutor` (`mol_ensemble_gen.execution`) fans members across GPUs with
`torch.multiprocessing.spawn` — one worker/model per GPU, member indices sharded
round-robin. Each completed member writes an atomic `.member_XXXX.json`
checkpoint, so an interrupted run resumes by skipping members already on disk,
and the parent merges everything into `metadata.csv` / `manifest.json`. Workers
rebuild the input per process (no model objects crossing the spawn boundary), and
only the parent writes shared metadata.

Finetuning uses DDP instead (`torchrun`); the dataset shards domains by rank, and
validation is built with `world_size=1` so every rank scores identical frames and
needs no collective.

## Project layout

```
src/mol_ensemble_gen/
  ensemble.py        # ESMFold2Ensemble, EnsembleSpec, SamplingParams, seed derivation
  execution.py       # LocalGPUExecutor: multi-GPU fan-out + checkpoint/restart
  analysis.py        # RMSD/RMSF, PCA, clustering, confidence filtering
  esmfold2.py        # thin wrapper over the reference model
  cif.py / pdb.py    # mmCIF atom_site reader; PDB sequence parser
  utils.py
  cli.py             # every subcommand above
  model/
    denoiser.py      # weight-compatible reimplementation of ESMFold2's denoiser
    flow.py          # flow-matching interface + the one probability-flow ODE sampler
  training/
    config.py        # dataclass config tree + validation
    mdcath.py        # streaming mdCATH dataset
    atom_map.py      # mdCATH heavy atoms -> model atom slots
    featurize.py     # frozen-trunk conditioning cache
    conditioning.py  # temperature embedder
    loss.py          # EDM + flow-matching losses, spread term
    trainer.py       # training loop, DDP, validation, checkpoints
    sample.py        # temperature-conditioned sampling from a checkpoint
    eval.py          # comparison against MD references
    slurm.py         # SLURM submission
DESIGN.md            # architecture, rationale, and the record of what was measured
examples/            # ensemble configs + one config per finetuning experiment
tests/               # offline unit tests + @pytest.mark.gpu integration tests
```

See [DESIGN.md](DESIGN.md) for architecture and the experiment record — including
the levers that turned out **not** to work, which is most of them.

## Testing

```bash
# Fast offline suite: no GPU, no model load (133 tests)
env PYTHONPATH=src python -m pytest tests/ -q

# Opt-in real GPU parity + integration tests (downloads weights)
env PYTHONPATH=src python -m pytest tests/ -q -m gpu
```

The offline suite covers the denoiser's structural signature, the flow/EDM time
draw, the sampling path, config validation, and the reproducibility invariants
that past bugs violated (seeded augmentation, validation determinism, atom-map
masking).
