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

As built — the subpackages this section originally sketched (`execution/`, `io/`,
`analysis/`) collapsed into single modules, since each stayed small enough that a
package added nesting without adding structure.

```
src/mol_ensemble_gen/
  __init__.py            # public API: ESMFold2Ensemble, EnsembleSpec, SamplingParams, analyze_run
  ensemble.py            # SamplingParams, EnsembleSpec, EnsembleMember, ESMFold2Ensemble, seeds
  execution.py           # LocalGPUExecutor: multi-GPU fan-out + per-member checkpoint/restart
  analysis.py            # RMSD/RMSF, PCA, hierarchical clustering, confidence filtering
  cif.py                 # header-driven mmCIF _atom_site reader (stdlib + numpy)
  pdb.py                 # PDB -> per-chain sequence (Cα records)
  utils.py               # FASTA / Boltz-schema parsing
  esmfold2.py            # ESMFold2Model_API — original single-fold wrapper (superseded by ensemble.py)
  cli.py                 # argparse app (see §6)
  model/
    denoiser.py          # our weight-compatible reimplementation of the denoiser (§12)
  training/              # mdCATH finetuning (§11)
    config.py            # dataclass configs + YAML loader (import-light, torch-free)
    atom_map.py          # mdCATH heavy atom -> model atom slot mapping
    mdcath.py            # HDF5 topology parsing + streaming IterableDataset
    featurize.py         # offline capture of frozen-trunk conditioning per domain
    conditioning.py      # TemperatureEmbedder (zero-init output head)
    loss.py              # EDM + rectified-flow objectives over one shared core
    trainer.py           # DDP loop, AMP, LRU conditioning cache, atomic ckpt/resume
    sample.py            # temperature-conditioned sampling + flow ODE sampler
    eval.py              # MD-match metrics (RMSF correlation, KS, PCA overlap)
    slurm.py             # sbatch rendering/submission for multi-node finetuning
```

Still unbuilt: trajectory export (`.xtc`/`.dcd` + topology) and a SLURM executor
for *ensemble generation* (the SLURM support that exists covers finetuning only).

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

The subcommands that exist today:

```
mol-ensemble-gen run     config.yaml              # inline or multi-GPU per config
mol-ensemble-gen run     config.yaml --gpus 0,1,2,3
mol-ensemble-gen analyze runs/exp1 --cluster-cutoff 2.0 --min-plddt 0.7

# finetuning (§11)
mol-ensemble-gen featurize-cache examples/finetune_mdcath.yaml
mol-ensemble-gen finetune        examples/finetune_mdcath.yaml   # under torchrun for DDP
mol-ensemble-gen sample-md       examples/finetune_mdcath.yaml --input some.fasta
mol-ensemble-gen eval-md         examples/finetune_mdcath.yaml --sampled-dir ... --domain 1abcA00
mol-ensemble-gen slurm-train     examples/finetune_mdcath.yaml [--submit]
```

Not implemented: `slurm-submit` (a SLURM *ensemble* array) and `inspect`.

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

## 11. Training / finetuning (match mdCATH ensembles, temperature-conditioned)

Everything above **samples a frozen model** — draws from ESMFold2's *folding*
distribution, not a physical Boltzmann ensemble. The `training/` subpackage
finetunes the diffusion module so sampling reproduces the conformational
ensembles in **mdCATH** (all-atom classical-FF MD of 5,398 CATH domains, 5
replicas × 5 temperatures 320–450 K), **conditioned on temperature** — one model
reproduces the spread at every temperature. Runs only in the fork env (`genAI`).

### Why only the diffusion module is trained

`ESMFold2Model.forward` and the sampler are inference-gated, but the denoiser
`DiffusionModule.forward` is ungated and differentiable, reachable as
`model.structure_head.diffusion_module`. Crucially, **every tensor the denoiser
consumes is a pure function of sequence and the idealized reference conformer —
independent of the diffusion coordinate `x` and of temperature** (proven by
`sample()`: only `x_noisy`/`t_hat` change in the loop, and `s_trunk` is `None`).
So we **freeze the trunk, cache its conditioning once per domain, and train only
`diffusion_module` + a small temperature embedder.** The training job never loads
the trunk or the PLM.

### Stage A — offline featurization (`featurize.py`, `mdcath.py`, `atom_map.py`)

Per domain, run the real forward **once** and capture — by monkeypatching
`structure_head.sample` and aborting before the first diffusion step — exactly the
kwargs it forwards to the denoiser (`s_inputs`, `z_trunk`,
`relative_position_encoding`, `ref_*`, token features, atom mask). This is
byte-identical to inference conditioning with zero risk of trunk-replay drift.
Floats are stored fp16; the capture is idempotent and resumable.

**Atom mapping is the correctness keystone.** The model lays atoms out
residue-by-residue in `PROTEIN_HEAVY_ATOMS[resname]` order, contiguously, real
atoms first — so a slot is fully determined by `(res_idx, atom_name)`. We parse
the domain's embedded PDB blob, strip hydrogens / terminal `OXT`, canonicalize
force-field residue names (`MSE→MET`, CHARMM/AMBER HIS states, `CYX→CYS`, …), and
match each mdCATH heavy atom to its slot. Unmatched slots are **masked out of the
loss, never zero-filled**; domains below `min_matched_fraction` (default 0.98) are
dropped. The map is then lifted into the model's *padded* atom axis using the
captured `ref_mask`, so the streaming dataset scatters straight into the
denoiser's coordinate layout. A byte-identical map-then-check on diverse
sequences is a unit test (injectable heavy-atom table, no `esm`/GPU).

`MDCathDataset` is an `IterableDataset` streaming strided frames from
`f[domain]/{temp}/{repl}/coords`, sharded across DDP ranks **and** DataLoader
workers. One micro-batch is `B` frames of a single `(domain, temperature)`.

### Stage B — temperature-conditioned finetuning (`conditioning.py`, `loss.py`, `trainer.py`)

**Temperature embedder.** `TemperatureEmbedder` maps fixed Fourier features of the
normalized T through a 2-layer MLP to dim 451 (`c_s_inputs`), with its **output
layer zero-initialized** so at step 0 the added bias is exactly zero and the
denoiser reproduces the pretrained model — the temperature signal is learned from
that identity start. The bias is added to `s_inputs` at every real token before the
diffusion module; the *same* injection point is reused at sampling time.

**One denoising core, two schemes.** Because the pretrained `DiffusionModule` is an
EDM x₀-predictor `D(x;σ)` with all preconditioning built in, both objectives share
a single core (`_denoise_and_weighted_mse`): draw a per-frame noise level `σ`, form
`x_noisy = center_aug(x0) + σ·ε`, denoise, align **GT→prediction** with the model's
own fp32 Kabsch (`_weighted_rigid_align`), and take a per-frame-weighted coordinate
MSE over matched atoms only. The `B` frames of a `(domain, T)` micro-batch share
batch-1 conditioning, broadcast via the denoiser's `num_diffusion_samples`
(`repeat_interleave`) — the large `z_trunk` is never duplicated. The schemes differ
*only* in how `σ` is drawn and how each frame is weighted:

| scheme | `σ` per frame | weight | sampler |
|---|---|---|---|
| **edm** (default) | `σ = σ_d·exp(p_mean + p_std·N)`, `σ_d=16`, `p=(-1.2, 1.5)` | `(σ²+σ_d²)/(σ·σ_d)²` | Karras SDE (`head.sample`) |
| **flow** (`optim.scheme: flow`) | `t` logit-normal (`lnσ = lnσ_d + N(p_mean,p_std)`) or uniform → `σ = t/(1−t)` | velocity `1/t²` (or `1` for `data`) | rectified-flow ODE (`flow_ode_sample`) |

**Why flow reuses the pretrained weights unchanged.** The rectified-flow path
`x_t = (1−t)·x0 + t·ε` (`ε~N(0,I)`, `t: 0=data → 1=noise`) divides to
`x_t/(1−t) = x0 + σ·ε` with **`σ = t/(1−t)`** — exactly the EDM input `D(x;σ)`
already consumes, so there is *no architecture or weight change*. Because `x0` is at
raw coordinate scale, that `σ` is a **raw** noise level in the denoiser's own units,
so matching the EDM branch's coverage requires drawing
`ln σ = ln σ_d + p`, `p ~ N(-1.2, 1.5)` — i.e. logit-normal
**`t = sigmoid(ln σ_d + p)`**, median `t ≈ 0.83`.

> ⚠️ The `ln σ_d` shift is load-bearing and was **missing** in the first
> implementation: `t = sigmoid(p)` gives `σ = exp(p)`, training the denoiser at
> `1/σ_d` (1/16) of the intended noise level. The first flow pilot (2026-08-05)
> ran this way — observed median σ 0.68 against EDM's 10.98, ratio 16.06 — which
> both invalidated the cross-scheme loss comparison and left the model untrained
> across the upper half of the ODE sampler's trajectory (it starts at `σ_max=256`).
> A unit test now pins flow's σ draw to EDM's elementwise.
> `time_dist: uniform` is deliberately *not* EDM-matched (median σ = 1).
>
> **Measured after the fix** (`runs/finetune_pilot_flow_v2`, 5000 steps): σ median
> 10.94 vs EDM's 10.98 ✓, and the unweighted MSE over the first 25 windows is 5.494
> vs EDM's 5.499 — the two schemes now start from the same point, as the zero-init
> temperature head implies they must. But the correction did **not** relieve the
> gradient clip: it went from 94% of steps to **100%**, median grad norm 1.91 → 8.94.
> Smaller weights (`1/t²` ≈ 1.5 instead of ≈19) are more than offset by the ~13×
> larger MSE at the higher noise band. Flow's MSE also *rose* over training
> (5.494 → 6.328) while EDM's fell (5.499 → 4.806). So with these
> hyper-parameters flow is clip-bound and loses to EDM on the comparable metric;
> `grad_clip` / `lr` need retuning for the flow branch before the two schemes can
> be judged on modelling merit. Velocity matching is just that shared MSE reweighted: with
`x̂₀ = D(x_noisy;σ)` the residual is `v_θ − v* = (x̂₀ − x0)/t`, so
`‖v_θ − v*‖² = ‖x̂₀ − x0‖²/t²`; `t` is clamped to `[t_min, 1−t_min]` to tame the
`1/t²`. `diffusion_loss(scheme, …, flow=cfg.flow)` dispatches, and **EDM stays the
default**; trainer, cache, CLI, and checkpoint format are identical across schemes.

> **Two subtleties the GPU test pinned down.** `s_trunk=None` is a **required**
> positional of the denoiser (forward it, don't drop it); and the Kabsch align must
> run with **autocast disabled** (its SVD/`det` have no bf16 kernel) even though the
> trainer wraps the step in autocast.

**Sampling** honours the checkpoint's scheme automatically — both paths go through
the `structure_head.sample` monkeypatch that injects the temperature bias. EDM
keeps the model's Karras SDE; flow instead routes to `flow_ode_sample`, a
**deterministic** probability-flow ODE integrator on a **uniform-in-t** grid
(`t: σ_max/(1+σ_max) → t_min`, `σ_k = t_k/(1−t_k)`) taking the score step
`x += (σ'−σ)·(x−x̂₀)/σ` (Euler, or 2nd-order Heun), with a per-step fp32 align and
*no* stochastic churn. Everything upstream (SPI/trunk/conditioning) and downstream
(CIF decode/manifest via `ESMFold2Ensemble`) is reused verbatim.

**Trainer.** DDP over a thin module holding only `{diffusion_module, temp_embedder}`
(all other params `requires_grad_(False)`); bf16 AMP (fp16 path uses a
`GradScaler`), grad accumulation with `no_sync`, warmup+cosine LR, an LRU
conditioning cache, and atomic checkpoint + idempotent resume (mirrors
`execution.py`'s sidecar convention). With `wandb.enabled`, rank 0 lazily imports
`wandb` and logs `loss`/`mse`/`sigma_mean` (plus `t_mean` for flow)/`lr`/`grad_norm`
every `log_every` steps with the full config; the run id is a slug of `out_dir`
under `resume="allow"`, so resuming a checkpoint continues the *same* run. `wandb`
is only required when enabled (it lives in the `training` extra).

### Stage C — sampling + MD-match evaluation (`sample.py`, `eval.py`)

`sample.py` loads a checkpoint, injects `s_inputs += temp_emb(T)` via the same
monkeypatch, and reuses `ESMFold2Ensemble`'s output/decoding to emit per-member
`.cif` at each temperature. `eval.py` reuses `analysis.py` to compare sampled vs
held-out mdCATH: per-residue **RMSF correlation** (primary), Rg + RMSD-to-native
KS distance, PCA-landscape overlap, and a **temperature-monotonicity** check
(spread ↑ with T).

### De-risking order

1. Atom mapping → byte-identical validation + matched-fraction gating + unit tests.
2. Loss frame consistency → align GT→pred in fp32; prove via single-domain overfit.
3. Trunk-fidelity → conditioning is *captured*, not re-implemented, so it can't drift.
4. Pretrained-behavior preservation → zero-init temp head (step 0 ≈ stock model).
5. Semantics → samples are model-distribution draws; temperature teaches *relative*
   spread, not calibrated free energies (§9). Frame eval as distributional overlap.

### Pipeline (CLI; see `examples/finetune_mdcath.yaml`)

```
mol-ensemble-gen featurize-cache examples/finetune_mdcath.yaml
torchrun --standalone --nproc_per_node=8 -m mol_ensemble_gen.cli finetune examples/finetune_mdcath.yaml
mol-ensemble-gen slurm-train  examples/finetune_mdcath.yaml     # multi-node full run
mol-ensemble-gen sample-md    examples/finetune_mdcath.yaml --input some.fasta
mol-ensemble-gen eval-md      examples/finetune_mdcath.yaml --sampled-dir ... --domain 1abcA00
```

Start with the **pilot** (a handful of short domains, single GPU) to validate the
end-to-end path, then swap in the full-run / SLURM block. Needs the `training`
extra (`pip install -e '.[training]'` — adds `h5py`, plus optional `mdtraj` for
DSSP and `wandb` for tracking).

## 12. Our own denoiser (`model/denoiser.py`)

Everything in §11 trains and samples **only** the diffusion denoiser: the trunk's
output is captured once per domain and replayed, so at train/sample time the sole
piece of ESMFold2 that executes is `structure_head.diffusion_module` plus two
stateless geometry helpers. `model/denoiser.py` reimplements exactly that surface
from scratch — no `transformers`/`esm` import, no 1.5 GB model load.

**Weight-compatible, not merely equivalent.** Module and parameter names mirror the
reference, so pretrained tensors load with `strict=True`: **345 tensors,
131,501,446 parameters**, zero missing/extra keys, zero shape mismatches (asserted
offline in `tests/test_model_denoiser.py`). `load_denoiser()` accepts either an
ESMFold2 state dict or one of our training checkpoints.

**Parity is measured against the reference's own nondeterminism.** The reference
does not reproduce itself bitwise — its bf16 attention reductions vary run to run
by ~3e-3 max-abs on `x_denoised`. Our implementation differs from it by the same
magnitude, so the GPU test asserts `|ref − ours| ≤ 4·|ref − ref|` rather than a
hard-coded tolerance, which would be arbitrary.

The fidelity details that matter (all deliberate, all load-bearing for parity):
sliding-window atom attention reproduces the reference's **non-flash** fallback,
including its bf16 promotion of q/k/v, its window measured in *valid-atom rank*
(so padding consumes no window budget), the always-attendable diagonal, and the
zeroing of padded rows; 3D RoPE tables are built fp32 then cast to bf16;
`AttentionPairBias` implements the standard kernel path (the fused/cuEq kernels are
inference-only optimizations gated behind `set_kernel_backend` upstream);
`_weighted_rigid_align` casts the covariance with `H.float()`, so the recovered
rotation is fp32-accurate *even for float64 input*.

**What this buys.** `GeometryOps` substitutes for the reference `structure_head`
wherever the losses and samplers use it, so the full EDM **and** flow objectives —
forward, backward, and the gradient into the temperature head — now run on CPU in
the offline suite instead of only under `@pytest.mark.gpu`. That closes the gap
that let the `make_dataset` NameError reach a real run with a green suite.

**Not reimplemented:** the pairformer/MSA trunk and the ESMC language model.
Folding a *new* sequence still needs the reference model once, to produce the
conditioning that `featurize-cache` stores. Training and sampling from a cached
domain need none of it.

## 13. Flow matching as the primary scheme (`model/flow.py`)

**Flow matching is now the default** (`optim.scheme: flow`), and `model/flow.py`
makes it a first-class interface on our own denoiser rather than something bolted
onto the EDM path.

### The three pieces

**`sample_flow_time` is the single source of truth for drawing `t`.** Training and
sampling both call it, so they cannot drift — which is exactly how the `σ_d` bug in
§11 survived: the draw was written twice. It includes the load-bearing
`ln σ_d` shift, and a unit test pins its median σ to `σ_d·e^{p_mean}`.

**Velocity is native.** `FlowDenoiser.velocity(x_t, t)` returns
`v̂ = (x_t − x̂₀)/t`, and `predict_x0` performs the `(1−t)` rescaling that the EDM
denoiser needs (`D` consumes `x_t/(1−t)`, **not** `x_t`). That division is easy to
omit: an integrator working in EDM σ-space skips it legitimately, one working in
flow space must not. `FlowDenoiser` holds no parameters and never appears in a
`state_dict`.

**The ODE sampler steps in flow space.** `flow_ode_sample` integrates
`dx/dt = v̂(x, t)` on a uniform-in-`t` grid from `t_max = σ_max/(1+σ_max)` down to
`t_min`, Euler or Heun, with a per-step fp32 Kabsch re-alignment and **no
stochastic churn** — the probability-flow ODE, not the Karras SDE. Same seed gives
a bit-identical trajectory (unit-tested).

### Native flow-time conditioning

The pretrained model is told where it is on the path via `0.25·ln(σ/σ_d)`.
`flow.t_conditioning` adds a genuinely `t`-native input:

| mode | behaviour |
|---|---|
| `off` | pretrained conditioning only; `state_dict` byte-identical to the release |
| **`add`** (default) | embeds `t` alongside the log-σ features, output **zero-init** |
| `replace` | drops the log-σ path entirely — purest flow parameterization |

`add` turned out strictly better than the trade-off it was meant to make: because
the output projection is zero-init, step 0 is **bitwise** the pretrained model
(unit-tested with `torch.equal`), so native-`t` is learned from an exact identity
start rather than costing identity-at-init. `replace` still needs real retraining.

The head adds 5 tensors (`conditioning.t_{fourier.w,fourier.b,norm.weight,norm.bias,proj.weight}`),
which a pretrained checkpoint does not contain. `load_denoiser` and the trainer's
resume path allow **exactly those** keys to be missing and nothing else, so an
older checkpoint can be continued into a `t`-conditioned model while a genuine
mismatch still raises.

Native `t` requires `model.backend: ours` — the reference denoiser has no `flow_t`
input, and the config rejects the combination rather than silently ignoring it.

### Retuned optimizer (this is not cosmetic)

Flow's velocity objective produces far larger gradients than EDM's at the same
noise band. Measured over the corrected pilot's 250 windows: **median grad norm
8.9, p90 24.3, p99 120, max 186** — against `grad_clip: 1.0`, so *every* step
clipped, every update was a unit vector, and the warmup+cosine LR schedule never
actually applied. The defaults now scale with the scheme:

```
grad_clip: 35.0    # ≈p85 → clips the top ~14%, so magnitudes carry information again
lr:        1.0e-5  # 1e-4 / 8.9 → preserves the effective step size the clipped run took
```

The clip was calibrated over successive pilots, each time from the *measured* p85
rather than a guess:

| clip | median / p85 grad norm | clipped | measured in |
|---|---|---|---|
| 1.0 | 8.9 / 24.3 | 100% | `flow_v2` |
| 20.0 | 14.6 / 32.6 | 33.6% | `flow_v3` |
| **35.0** | — | ~14% expected | `flow_v4` |

The median rose from 8.9 to 14.6 once the native flow-time head was added (more
parameters, more gradient), which is why the first estimate undershot. Note the
median norm stays *below* every candidate clip, so raising it does not change the
median step — only the clipped tail gets larger steps, which is the point. `lr`
therefore does not move with the clip.

**`scheme: edm` must override both** (`lr: 1.0e-4`, `grad_clip: 1.0`); EDM's norms
peaked at 0.75 and never clipped.

> Note the default flip is a *policy* choice, not an evidence-backed win. As of the
> last measured pilots EDM still fits better on the comparable metric (§11): flow's
> unweighted MSE rose over training while EDM's fell. Flow is now the default
> because it is where the work is going; whether it beats EDM is still open and
> wants `sample-md` + `eval-md` on both checkpoints to settle.
