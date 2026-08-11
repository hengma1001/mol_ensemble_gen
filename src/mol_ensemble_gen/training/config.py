"""Dataclass configs + YAML loader for the finetuning pipeline.

Kept import-light (stdlib only) so config can be validated in a unit test without
a GPU, torch, or the model — mirroring ``cli.py``'s split of config parsing from
the heavy run path. Nested dataclasses are built recursively from a plain dict so
one YAML file drives featurization, training, sampling, and evaluation.

mdCATH samples five temperatures; ``TEMPERATURES`` is the canonical set.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# mdCATH replica temperatures (K); the HDF5 groups are keyed by these as strings.
TEMPERATURES: tuple[int, ...] = (320, 348, 379, 413, 450)

# EDM / Karras preconditioning + training-noise constants, read from the fork's
# configuration_esmfold2.py (sigma_data=16.0; train_noise_log_mean/std=-1.2/1.5).
SIGMA_DATA: float = 16.0
TRAIN_NOISE_LOG_MEAN: float = -1.2
TRAIN_NOISE_LOG_STD: float = 1.5


@dataclass
class DataConfig:
    """Where the mdCATH HDF5 files live and how to stream them."""

    mdcath_dir: str = "mdcath"               # dir of mdcath_dataset_<domain>.h5
    cache_dir: str = "cache/featurized"      # per-domain conditioning caches
    domains: list[str] = field(default_factory=list)   # explicit list, or...
    domains_file: str | None = None          # ...a text file, one domain id per line
    temperatures: list[int] = field(default_factory=lambda: list(TEMPERATURES))
    replicas: list[int] | None = None        # None = all 5 replicas
    skip_frames: int = 10                     # stride when streaming trajectory frames
    frames_per_step: int = 8                  # frames per optimizer micro-step (one domain+T)
    max_len: int | None = None                # skip domains longer than this (residues)
    min_matched_fraction: float = 0.98        # drop domains whose atom map is worse
    val_domains: list[str] = field(default_factory=list)  # held out from training


@dataclass
class ModelConfig:
    """ESMFold2 backbone identity + trunk-cache depth."""

    model_name: str = "biohub/ESMFold2"
    num_loops: int = 20                       # trunk recycles when caching conditioning
    #: Which denoiser implementation to train.
    #:
    #: * ``"ours"`` (default) — :mod:`mol_ensemble_gen.model.denoiser`, loaded with
    #:   the pretrained weights. Required for native flow-time conditioning, and
    #:   avoids loading the trunk/PLM at all.
    #: * ``"reference"`` — pull ``structure_head`` out of the ``transformers``
    #:   model, as the original trainer did. No native-``t`` support.
    backend: str = "ours"
    #: Load the pretrained ESMFold2 denoiser weights. ``False`` trains the
    #: diffusion module from **random initialization** — a from-scratch experiment
    #: rather than a finetune, so ESMFold2 is never loaded at all. Note the
    #: optimizer defaults are tuned for finetuning and are far too conservative
    #: here; ``lr`` in particular needs raising. Requires ``backend: ours``.
    pretrained: bool = True


@dataclass
class TemperatureConfig:
    """Normalization for the temperature embedder input."""

    ref: float = 379.0                        # centering temperature (K), mdCATH mid-point
    scale: float = 65.0                       # ~std of the five temperatures (K)
    embed_dim: int = 451                      # must equal s_inputs channel dim (c_s_inputs)
    hidden_dim: int = 256
    num_fourier: int = 32                     # Fourier features of normalized T


@dataclass
class OptimConfig:
    """Optimizer settings. **Defaults are tuned for the default scheme (flow).**

    Flow's velocity objective produces much larger gradients than EDM's at the same
    noise band. With EDM's original ``grad_clip: 1.0`` every single flow step
    clipped, which throws away all gradient-magnitude information and makes the LR
    schedule fictional. ``grad_clip`` is therefore set near the measured p85 so
    clipping is the exception, and ``lr`` is scaled by the median norm to preserve
    the effective step size (1e-4 / 8.9 ≈ 1.1e-5).

    Calibration history, each from a full 5000-step pilot:

    ======  ==============  =========  =============  ==============
    clip    median / p85    clipped    next clip      measured in
    ======  ==============  =========  =============  ==============
    1.0     8.9 / 24.3      100%       20.0           flow_v2
    20.0    14.6 / 32.6     33.6%      **35.0**       flow_v3
    ======  ==============  =========  =============  ==============

    The median rose from 8.9 to 14.6 once the native flow-time head was added
    (extra parameters, extra gradient), which is why the first estimate undershot.
    Note the median norm sits *below* every candidate clip, so raising it does not
    change the median step — only the clipped tail gets larger steps. ``lr`` stays.

    **Running ``scheme: edm`` means overriding these**: EDM wants ``lr: 1.0e-4``
    and ``grad_clip: 1.0`` (its grad norms peaked at 0.75, never clipping).
    """

    lr: float = 1e-5                          # flow-tuned; use 1e-4 for edm
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 35.0                   # flow-tuned (≈p85); use 1.0 for edm
    warmup_steps: int = 500
    max_steps: int = 50_000
    grad_accum: int = 4
    lr_min_ratio: float = 0.05                # cosine floor as a fraction of lr
    scheme: str = "flow"                      # training/sampling scheme: "flow" | "edm"


@dataclass
class FlowConfig:
    """Rectified-flow (flow-matching) hyper-parameters (used when scheme=="flow").

    The pretrained EDM denoiser is reused unchanged via the change of variables
    ``σ = t/(1−t)`` (see DESIGN.md §11). ``p_mean``/``p_std`` default to the EDM
    training-noise values so ``ln σ`` coverage is identical; ``t`` is then
    ``σ/(1+σ)`` (logit-normal). Velocity matching weights the aligned coordinate
    MSE by ``1/t²``.
    """

    time_dist: str = "logitnormal"            # "logitnormal" (lnσ~N) | "uniform" (t~U)
    p_mean: float = TRAIN_NOISE_LOG_MEAN      # logit-normal mean of ln σ
    p_std: float = TRAIN_NOISE_LOG_STD        # logit-normal std of ln σ
    weighting: str = "velocity"               # "velocity" (1/t²) | "data" (unit)
    t_min: float = 1e-3                        # clamp t∈[t_min,1−t_min] (tames 1/t²)
    num_sampling_steps: int = 50              # ODE integration steps
    sampler: str = "euler"                    # "euler" | "heun" (2nd-order)
    sigma_max: float = 256.0                   # start noise level (t_max=σ/(1+σ))
    #: Native flow-time conditioning in our denoiser: "off" | "add" | "replace".
    #: "add" embeds t directly alongside the pretrained log-σ features with a
    #: zero-init output, so step 0 is bitwise the pretrained model and native-t is
    #: learned from there. "replace" drops the log-σ path entirely (no
    #: identity-at-init). Requires ``model.backend: ours``.
    t_conditioning: str = "add"


@dataclass
class WandbConfig:
    """Weights & Biases experiment tracking (rank 0 only; off by default).

    Lazily imported in the trainer, so ``wandb`` is only required when
    ``enabled`` is true (``pip install -e '.[training]'`` provides it). The run
    resumes in place when training resumes from a checkpoint, keyed by ``id`` (or
    a deterministic id derived from ``out_dir`` when unset).
    """

    enabled: bool = False
    project: str = "mol-ensemble-gen"
    entity: str | None = None                 # W&B team/user; None = default
    run_name: str | None = None               # display name; None = W&B auto-name
    id: str | None = None                     # resume key; None = derived from out_dir
    tags: list[str] = field(default_factory=list)
    mode: str = "online"                      # online | offline | disabled


@dataclass
class TrainConfig:
    """Top-level finetuning run configuration."""

    out_dir: str = "runs/finetune"
    seed: int = 20260728
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)  # used when optim.scheme=="flow"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    amp_dtype: str = "bfloat16"               # bfloat16 | float16 | float32
    log_every: int = 20
    ckpt_every: int = 1000
    #: Run the loss over ``val_batches`` micro-batches of ``data.val_domains``
    #: every N optimizer steps (0 disables). Without this there is no signal that
    #: distinguishes "still learning" from "overfitting three domains".
    val_every: int = 500
    val_batches: int = 16
    resume: bool = True                       # continue from out_dir checkpoint if present
    slurm: dict[str, Any] = field(default_factory=dict)   # SLURM resources (see slurm.py)


def _build(dc_type: type, raw: Any) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Rejects unknown keys with a helpful message (as ``cli.load_config`` does) and
    coerces nested dataclass fields from their sub-dicts. ``from __future__ import
    annotations`` stores field types as strings, so hints are resolved against
    this module's namespace before checking for nested dataclasses.
    """
    import typing

    if not is_dataclass(dc_type):
        return raw
    if raw is None:
        return dc_type()
    if not isinstance(raw, dict):
        raise ValueError(f"expected a mapping for {dc_type.__name__}, got {type(raw).__name__}")
    known = {f.name for f in fields(dc_type)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown {dc_type.__name__} keys: {unknown} (expected {sorted(known)})")
    hints = typing.get_type_hints(dc_type)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in raw:
            continue
        ftype = hints.get(name)
        if is_dataclass(ftype):
            kwargs[name] = _build(ftype, raw[name])
        else:
            kwargs[name] = raw[name]
    return dc_type(**kwargs)


def load_train_config(path: str | Path) -> TrainConfig:
    """Load a finetuning config from YAML, validating keys at every level."""
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(TrainConfig, raw)
    _normalize(cfg)
    _validate(cfg)
    return cfg


def _normalize(cfg: TrainConfig) -> None:
    """Repair values that YAML's type rules mangle before validation sees them.

    YAML 1.1 treats bare ``off``/``no`` as boolean ``False`` (and ``on``/``yes`` as
    ``True``), so ``t_conditioning: off`` arrives as ``False`` rather than the
    string. Accept that spelling rather than making people remember quotes; the
    ``True`` side is genuinely ambiguous, so leave it to fail in ``_validate``.
    """
    if cfg.flow.t_conditioning is False:
        cfg.flow.t_conditioning = "off"


def _validate(cfg: TrainConfig) -> None:
    """Cheap semantic checks beyond key/type validation."""
    if cfg.optim.scheme not in ("edm", "flow"):
        raise ValueError(f"optim.scheme must be 'edm' or 'flow', got {cfg.optim.scheme!r}")
    if cfg.flow.time_dist not in ("logitnormal", "uniform"):
        raise ValueError(f"flow.time_dist must be 'logitnormal' or 'uniform', got {cfg.flow.time_dist!r}")
    if cfg.flow.weighting not in ("velocity", "data"):
        raise ValueError(f"flow.weighting must be 'velocity' or 'data', got {cfg.flow.weighting!r}")
    if cfg.flow.sampler not in ("euler", "heun"):
        raise ValueError(f"flow.sampler must be 'euler' or 'heun', got {cfg.flow.sampler!r}")
    if cfg.flow.t_conditioning not in ("off", "add", "replace"):
        raise ValueError(
            "flow.t_conditioning must be 'off', 'add' or 'replace', "
            f"got {cfg.flow.t_conditioning!r}"
        )
    if cfg.model.backend not in ("ours", "reference"):
        raise ValueError(f"model.backend must be 'ours' or 'reference', got {cfg.model.backend!r}")
    if not cfg.model.pretrained and cfg.model.backend != "ours":
        raise ValueError(
            "model.pretrained: false needs model.backend: ours — the reference "
            "backend has no way to build a randomly-initialized denoiser."
        )
    if cfg.model.backend == "reference" and cfg.flow.t_conditioning != "off":
        raise ValueError(
            "native flow-time conditioning needs model.backend: ours "
            f"(got backend={cfg.model.backend!r}, flow.t_conditioning="
            f"{cfg.flow.t_conditioning!r}). The reference denoiser has no flow_t input."
        )
    if cfg.wandb.mode not in ("online", "offline", "disabled"):
        raise ValueError(
            f"wandb.mode must be 'online', 'offline' or 'disabled', got {cfg.wandb.mode!r}"
        )


def resolve_domains(data: DataConfig) -> list[str]:
    """Resolve the training domain list from ``domains`` and/or ``domains_file``.

    ``val_domains`` are excluded so a config can list every domain once and carve
    out the validation split declaratively.
    """
    ids: list[str] = list(data.domains)
    if data.domains_file:
        text = Path(data.domains_file).read_text()
        ids += [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    seen: set[str] = set()
    val = set(data.val_domains)
    out: list[str] = []
    for d in ids:
        if d in seen or d in val:
            continue
        seen.add(d)
        out.append(d)
    return out


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Serialize a (nested) dataclass config to a plain dict for checkpoints/logs."""
    return dataclasses.asdict(cfg)
