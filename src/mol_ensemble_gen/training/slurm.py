"""Render + submit a SLURM job for multi-GPU / multi-node finetuning.

The template drives ``torchrun`` (one task per node, one process per GPU) against
``mol_ensemble_gen.cli finetune`` — the same entry point used locally, so the
distributed path is exactly the single-node path scaled out. Rendering is plain
``str.format`` (no jinja2 dependency); ``submit`` shells out to ``sbatch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_TEMPLATE = Path(__file__).parent / "templates" / "train.slurm.j2"


@dataclass
class SlurmConfig:
    """SLURM resources for a finetuning job."""

    job_name: str = "esmfold2-finetune"
    partition: str = "gpu"
    nodes: int = 1
    gpus_per_node: int = 8
    cpus_per_task: int = 32
    time: str = "24:00:00"
    master_port: int = 29500
    log_dir: str = "runs/finetune/logs"
    env_setup: str = ""  # e.g. "source ~/mamba/bin/activate genAI"
    extra_sbatch: list[str] = field(default_factory=list)  # extra #SBATCH lines


def render_script(config_path: str | Path, slurm: SlurmConfig) -> str:
    """Render the sbatch script text for ``finetune <config_path>``."""
    template = _TEMPLATE.read_text()
    extra = "\n".join(f"#SBATCH {line}" for line in slurm.extra_sbatch)
    return template.format(
        job_name=slurm.job_name,
        partition=slurm.partition,
        nodes=slurm.nodes,
        gpus_per_node=slurm.gpus_per_node,
        cpus_per_task=slurm.cpus_per_task,
        time=slurm.time,
        master_port=slurm.master_port,
        log_dir=slurm.log_dir,
        env_setup=slurm.env_setup,
        extra_sbatch=extra,
        config=str(config_path),
    )


def write_script(config_path: str | Path, slurm: SlurmConfig, out_path: str | Path) -> Path:
    """Render and write the sbatch script; ensures the log dir exists."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(slurm.log_dir).mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_script(config_path, slurm))
    return out_path


def submit(config_path: str | Path, slurm: SlurmConfig, out_path: str | Path = "train.slurm") -> str:
    """Write the script and ``sbatch`` it; returns sbatch's stdout (job id line)."""
    import subprocess

    script = write_script(config_path, slurm, out_path)
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["sbatch", str(script)], check=True, capture_output=True, text=True
    )
    print(result.stdout.strip(), flush=True)
    return result.stdout.strip()


def slurm_config_from_dict(raw: dict | None) -> SlurmConfig:
    """Build a :class:`SlurmConfig` from a config ``slurm:`` block."""
    if not raw:
        return SlurmConfig()
    known = {f for f in SlurmConfig.__dataclass_fields__}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown slurm keys: {unknown} (expected {sorted(known)})")
    return SlurmConfig(**raw)
