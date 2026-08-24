"""Local multi-GPU execution: fan ensemble members across GPUs.

One worker process per GPU, each loading ESMFold2 once and folding its shard of
members. Every completed member drops a ``.member_XXXX.json`` checkpoint, so an
interrupted run resumes by skipping members already on disk. The parent process
merges the checkpoints into the usual ``metadata.csv`` / ``manifest.json``.

Only the parent writes the shared metadata; workers write only their own
uniquely-named CIFs and checkpoints, so there is no cross-process contention.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .ensemble import EnsembleMember, EnsembleSpec, write_manifest


def _sidecar(out_dir: Path, member_idx: int) -> Path:
    return out_dir / f".member_{member_idx:04d}.json"


def plan_shards(todo: list[int], n_workers: int) -> list[list[int]]:
    """Round-robin split of member indices across workers (balanced by count)."""
    return [todo[r::n_workers] for r in range(n_workers)]


def pending_members(out_dir: str | Path, members: int) -> list[int]:
    """Member indices with no checkpoint yet (i.e. still to fold)."""
    out_dir = Path(out_dir)
    return [m for m in range(members) if not _sidecar(out_dir, m).exists()]


def merge_sidecars(out_dir: str | Path, input_id: str, spec: EnsembleSpec) -> list[EnsembleMember]:
    """Collect all member checkpoints into metadata.csv + manifest.json."""
    out_dir = Path(out_dir)
    members: list[EnsembleMember] = []
    for path in sorted(out_dir.glob(".member_*.json")):
        for record in json.loads(path.read_text()):
            members.append(EnsembleMember(**record))
    write_manifest(out_dir, input_id, spec, members)
    return members


def _write_sidecar(out_dir: Path, member_idx: int, records: list[EnsembleMember]) -> None:
    """Atomically write a member's checkpoint (tmp + rename)."""
    tmp = out_dir / f".member_{member_idx:04d}.json.tmp"
    tmp.write_text(json.dumps([asdict(r) for r in records]))
    tmp.replace(_sidecar(out_dir, member_idx))


def _worker(
    rank: int,
    gpus: list[int],
    input_path: str,
    out_dir: str,
    spec: EnsembleSpec,
    shards: list[list[int]],
    model_name: str,
) -> None:
    """Subprocess entry point: fold the members assigned to this GPU."""
    import torch

    from .ensemble import ESMFold2Ensemble

    gpu = gpus[rank]
    torch.cuda.set_device(gpu)
    ens = ESMFold2Ensemble(spec, device=f"cuda:{gpu}", model_name=model_name)
    spi, input_id = ens.build_spi(input_path)

    out = Path(out_dir)
    for member_idx in shards[rank]:
        if _sidecar(out, member_idx).exists():
            continue  # finished on a previous run
        records = ens.fold_member(spi, input_id, member_idx, out)
        _write_sidecar(out, member_idx, records)
        print(f"[gpu {gpu}] member {member_idx} done ({len(records)} structures)", flush=True)


@dataclass
class LocalGPUExecutor:
    """Fan an ensemble across local GPUs with checkpoint/restart."""

    gpus: list[int]
    model_name: str = "biohub/ESMFold2"

    def run(self, input_path: str | Path, out_dir: str | Path, spec: EnsembleSpec) -> list[EnsembleMember]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        input_id = Path(input_path).stem

        todo = pending_members(out_dir, spec.members)
        if not todo:
            print(f"[executor] all {spec.members} members already complete; merging")
            return merge_sidecars(out_dir, input_id, spec)

        n = min(len(self.gpus), len(todo))
        shards = plan_shards(todo, n)
        print(
            f"[executor] {len(todo)} members over {n} GPU(s): "
            + ", ".join(f"gpu{self.gpus[r]}:{len(shards[r])}" for r in range(n))
        )

        import torch.multiprocessing as mp

        mp.spawn(
            _worker,
            args=(self.gpus[:n], str(input_path), str(out_dir), spec, shards, self.model_name),
            nprocs=n,
            join=True,
        )
        return merge_sidecars(out_dir, input_id, spec)
