"""Command-line entry point for ensemble generation.

    mol-ensemble-gen run config.yaml
    mol-ensemble-gen run config.yaml --members 50 --out-dir runs/exp2

Config parsing (:func:`load_config`, :func:`build_spec`) is kept separate from
the heavy run path so it can be unit-tested without a GPU or model load.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ensemble import EnsembleSpec, ESMFold2Ensemble, SamplingParams


@dataclass
class RunConfig:
    """A full ensemble run, as loaded from YAML."""

    input: str                              # path to a .fasta or Boltz-style .yaml
    out_dir: str = "runs/ensemble"
    model_name: str = "biohub/ESMFold2"
    device: str = "cuda"
    ensemble: dict[str, Any] = field(default_factory=dict)   # members, base_seed
    sampling: dict[str, Any] = field(default_factory=dict)   # SamplingParams knobs
    execution: dict[str, Any] = field(default_factory=dict)  # backend, gpus (multi-GPU)


def load_config(path: str | Path) -> RunConfig:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in RunConfig.__dataclass_fields__.values()}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown} (expected {sorted(known)})")
    if "input" not in raw:
        raise ValueError("config must define 'input' (a .fasta or Boltz .yaml path)")
    return RunConfig(**raw)


def build_spec(cfg: RunConfig) -> EnsembleSpec:
    """Turn a RunConfig into an EnsembleSpec, validating knob names early."""
    valid = {f.name for f in SamplingParams.__dataclass_fields__.values()}
    bad = sorted(set(cfg.sampling) - valid)
    if bad:
        raise ValueError(f"unknown sampling knobs: {bad} (valid: {sorted(valid)})")
    sampling = SamplingParams(**cfg.sampling)
    return EnsembleSpec(
        members=cfg.ensemble.get("members", 10),
        base_seed=cfg.ensemble.get("base_seed", 0),
        sampling=sampling,
    )


def run(cfg: RunConfig) -> list:
    """Load the model and generate the ensemble (heavy: needs GPU + weights).

    With ``execution.backend == "local"`` (or any ``execution.gpus`` given) the
    members are fanned across the listed GPUs, one worker/model each, with
    checkpoint/restart. Otherwise a single model runs on ``device``.
    """
    spec = build_spec(cfg)
    gpus = cfg.execution.get("gpus")
    backend = cfg.execution.get("backend", "inline")

    if backend == "local" or gpus:
        from .execution import LocalGPUExecutor

        gpus = gpus or [0]
        members = LocalGPUExecutor(gpus=gpus, model_name=cfg.model_name).run(
            cfg.input, cfg.out_dir, spec
        )
    else:
        ens = ESMFold2Ensemble(spec, device=cfg.device, model_name=cfg.model_name)
        spi, input_id = ens.build_spi(cfg.input)
        members = ens.generate(spi, input_id, cfg.out_dir)

    print(f"[ensemble] wrote {len(members)} structures to {cfg.out_dir}")
    return members


def parse_gpus(spec: str) -> list[int]:
    """Parse a ``--gpus`` value like ``"0,1,2,3"`` into ``[0, 1, 2, 3]``."""
    return [int(g) for g in spec.split(",") if g.strip() != ""]


def _apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    if args.input is not None:
        cfg.input = args.input
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.members is not None:
        cfg.ensemble["members"] = args.members
    if args.base_seed is not None:
        cfg.ensemble["base_seed"] = args.base_seed
    if args.device is not None:
        cfg.device = args.device
    if args.gpus is not None:
        cfg.execution["backend"] = "local"
        cfg.execution["gpus"] = parse_gpus(args.gpus)
    return cfg


def analyze(args: argparse.Namespace) -> None:
    """Analyze an ensemble directory: RMSD/RMSF, clustering, PCA, confidence."""
    from .analysis import analyze_run

    result = analyze_run(
        args.out_dir,
        min_plddt=args.min_plddt,
        min_ptm=args.min_ptm,
        min_iptm=args.min_iptm,
        cluster_cutoff=args.cluster_cutoff,
        n_clusters=args.n_clusters,
        chain=args.chain,
    )
    summary = result.summary
    print(f"[analyze] {summary['n_members']} members -> {summary['n_clusters']} clusters")
    print(f"[analyze] mean/max pairwise RMSD: {summary['mean_pairwise_rmsd']:.2f} / "
          f"{summary['max_pairwise_rmsd']:.2f} Å   max RMSF: {summary['max_rmsf']:.2f} Å")
    print(f"[analyze] PC variance: {[round(v, 3) for v in summary['pca_explained_variance']]}")
    print(f"[analyze] wrote analysis_metadata.csv, rmsd_matrix.npy, pca.csv, rmsf.csv, "
          f"analysis_summary.json to {args.out_dir}")


def _parse_int_list(spec: str | None) -> list[int] | None:
    """Parse ``"320,450"`` → ``[320, 450]`` (None passes through)."""
    if spec is None:
        return None
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def featurize_cache(args: argparse.Namespace) -> None:
    """Cache the frozen trunk's conditioning + atom map for every domain."""
    from .training.config import load_train_config
    from .training.featurize import featurize_all

    cfg = load_train_config(args.config)
    results = featurize_all(cfg, overwrite=args.overwrite)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"[featurize] {dict(sorted(by_status.items()))}")


def finetune(args: argparse.Namespace) -> None:
    """Finetune the diffusion module + temperature embedder (DDP under torchrun)."""
    from .training.config import load_train_config
    from .training.trainer import train

    train(load_train_config(args.config))


def sample_md(args: argparse.Namespace) -> None:
    """Sample temperature-conditioned ensembles from a finetuned checkpoint."""
    from .training.config import load_train_config
    from .training.sample import sample_temperatures

    cfg = load_train_config(args.config)
    temps = _parse_int_list(args.temperatures) or cfg.data.temperatures
    out_dir = args.out_dir or str(Path(cfg.out_dir) / "samples")
    ckpt = args.checkpoint or str(Path(cfg.out_dir) / "checkpoint.pt")
    sample_temperatures(
        ckpt, args.input, [float(t) for t in temps], out_dir,
        members=args.members, base_seed=args.base_seed, device=args.device or "cuda",
    )


def eval_md(args: argparse.Namespace) -> None:
    """Score a sampled ensemble against mdCATH MD (per-temperature + monotonicity)."""
    from .training.config import load_train_config
    from .training.eval import evaluate_run

    cfg = load_train_config(args.config)
    temps = _parse_int_list(args.temperatures) or cfg.data.temperatures
    summary = evaluate_run(args.sampled_dir, cfg.data.mdcath_dir, args.domain, temps, skip=args.skip)
    print(f"[eval] {args.domain}: mean RMSF Pearson {summary['mean_rmsf_pearson']}, "
          f"spread monotonic {summary['spread_monotonic']}")


def slurm_train(args: argparse.Namespace) -> None:
    """Render (and optionally submit) an sbatch script for finetuning."""
    from .training.config import load_train_config
    from .training.slurm import slurm_config_from_dict, submit, write_script

    cfg = load_train_config(args.config)
    slurm = slurm_config_from_dict(cfg.slurm)
    if args.submit:
        submit(args.config, slurm, args.script_path)
    else:
        path = write_script(args.config, slurm, args.script_path)
        print(f"[slurm] wrote {path} (submit with: sbatch {path})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mol-ensemble-gen", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="generate an ensemble from a YAML config")
    run_p.add_argument("config", type=str, help="path to the run config YAML")
    run_p.add_argument("--input", type=str, default=None, help="override input .fasta/.pdb/.yaml")
    run_p.add_argument("--out-dir", type=str, default=None, help="override output directory")
    run_p.add_argument("--members", type=int, default=None, help="override number of seeds")
    run_p.add_argument("--base-seed", type=int, default=None, help="override base seed")
    run_p.add_argument("--device", type=str, default=None, help="override device (e.g. cuda:0)")
    run_p.add_argument("--gpus", type=str, default=None,
                       help="fan members across local GPUs, e.g. '0,1,2,3,4,5,6,7'")

    an_p = sub.add_parser("analyze", help="analyze a generated ensemble directory")
    an_p.add_argument("out_dir", type=str, help="ensemble directory (containing metadata.csv)")
    an_p.add_argument("--min-plddt", type=float, default=0.0, help="drop members below this pLDDT")
    an_p.add_argument("--min-ptm", type=float, default=0.0, help="drop members below this pTM")
    an_p.add_argument("--min-iptm", type=float, default=None, help="drop members below this ipTM (complexes)")
    an_p.add_argument("--cluster-cutoff", type=float, default=2.0, help="RMSD cutoff (Å) for clustering")
    an_p.add_argument("--n-clusters", type=int, default=None, help="fixed cluster count (overrides cutoff)")
    an_p.add_argument("--chain", type=str, default=None, help="restrict RMSD/PCA to one chain id")

    # -- training / finetuning subcommands ---------------------------------
    fc_p = sub.add_parser("featurize-cache", help="cache frozen-trunk conditioning per domain")
    fc_p.add_argument("config", type=str, help="finetuning config YAML")
    fc_p.add_argument("--overwrite", action="store_true", help="re-featurize cached domains")

    ft_p = sub.add_parser("finetune", help="finetune the diffusion module (run under torchrun for DDP)")
    ft_p.add_argument("config", type=str, help="finetuning config YAML")

    sm_p = sub.add_parser("sample-md", help="sample temperature-conditioned ensembles from a checkpoint")
    sm_p.add_argument("config", type=str, help="finetuning config YAML")
    sm_p.add_argument("--input", type=str, required=True, help="input .fasta/.pdb/.yaml to fold")
    sm_p.add_argument("--checkpoint", type=str, default=None, help="checkpoint (default out_dir/checkpoint.pt)")
    sm_p.add_argument("--temperatures", type=str, default=None, help="e.g. '320,450' (default: config temps)")
    sm_p.add_argument("--members", type=int, default=50, help="seeds per temperature")
    sm_p.add_argument("--base-seed", type=int, default=0, help="base seed for member derivation")
    sm_p.add_argument("--out-dir", type=str, default=None, help="output dir (default out_dir/samples)")
    sm_p.add_argument("--device", type=str, default=None, help="device (default cuda)")

    em_p = sub.add_parser("eval-md", help="score a sampled ensemble against mdCATH MD")
    em_p.add_argument("config", type=str, help="finetuning config YAML")
    em_p.add_argument("--sampled-dir", type=str, required=True, help="dir of T<K>/ sampled ensembles")
    em_p.add_argument("--domain", type=str, required=True, help="mdCATH domain id to compare against")
    em_p.add_argument("--temperatures", type=str, default=None, help="e.g. '320,450' (default: config temps)")
    em_p.add_argument("--skip", type=int, default=10, help="MD frame stride when reading references")

    st_p = sub.add_parser("slurm-train", help="render/submit an sbatch script for finetuning")
    st_p.add_argument("config", type=str, help="finetuning config YAML")
    st_p.add_argument("--submit", action="store_true", help="sbatch the script instead of just writing it")
    st_p.add_argument("--script-path", type=str, default="train.slurm", help="where to write the sbatch script")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        cfg = _apply_overrides(load_config(args.config), args)
        run(cfg)
    elif args.command == "analyze":
        analyze(args)
    elif args.command == "featurize-cache":
        featurize_cache(args)
    elif args.command == "finetune":
        finetune(args)
    elif args.command == "sample-md":
        sample_md(args)
    elif args.command == "eval-md":
        eval_md(args)
    elif args.command == "slurm-train":
        slurm_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
