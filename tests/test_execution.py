"""Offline tests for the multi-GPU executor's pure logic (no torch / no GPU).

The sharding, checkpoint/restart, and merge helpers are torch-free, so they run
without a GPU. The ``spawn`` path is exercised with a fake ``torch.multiprocessing``
so the orchestration (plan -> workers -> merge) is covered without loading a model.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from python_package import execution
from python_package.ensemble import EnsembleMember, EnsembleSpec


def _member(m: int) -> EnsembleMember:
    return EnsembleMember(m, 0, m * 10, f"m{m}.cif", 90.0, 0.8, None, {"seed": m * 10})


# -- pure helpers ----------------------------------------------------------


@pytest.mark.unit
def test_plan_shards_round_robin_balanced():
    shards = execution.plan_shards([0, 1, 2, 3, 4, 5, 6], 3)
    assert shards == [[0, 3, 6], [1, 4], [2, 5]]
    assert sorted(m for s in shards for m in s) == [0, 1, 2, 3, 4, 5, 6]
    # counts differ by at most one
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


@pytest.mark.unit
def test_pending_members_skips_checkpointed(tmp_path):
    execution._write_sidecar(tmp_path, 1, [_member(1)])
    execution._write_sidecar(tmp_path, 3, [_member(3)])
    assert execution.pending_members(tmp_path, 5) == [0, 2, 4]


@pytest.mark.unit
def test_sidecar_roundtrip(tmp_path):
    execution._write_sidecar(tmp_path, 2, [_member(2)])
    [recovered] = execution.merge_sidecars(tmp_path, "prot", EnsembleSpec(members=1))
    assert recovered == _member(2)
    # no leftover .tmp file
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.unit
def test_merge_sidecars_sorts_and_writes(tmp_path):
    for m in (2, 0, 1):                      # written out of order
        execution._write_sidecar(tmp_path, m, [_member(m)])
    members = execution.merge_sidecars(tmp_path, "prot", EnsembleSpec(members=3))
    assert [m.member_idx for m in members] == [0, 1, 2]
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metadata.csv").exists()


# -- orchestration (fake spawn) -------------------------------------------


def _install_fake_torch(monkeypatch):
    """Replace torch.multiprocessing.spawn with one that runs workers inline."""
    def fake_spawn(fn, args, nprocs, join):
        gpus, input_path, out_dir, spec, shards, model_name = args
        out = Path(out_dir)
        for rank in range(nprocs):
            for m in shards[rank]:
                execution._write_sidecar(out, m, [_member(m)])

    fake_mp = types.ModuleType("torch.multiprocessing")
    fake_mp.spawn = fake_spawn
    fake_torch = types.ModuleType("torch")
    fake_torch.multiprocessing = fake_mp
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", fake_mp)


@pytest.mark.unit
def test_executor_run_fans_and_merges(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    spec = EnsembleSpec(members=6)
    ex = execution.LocalGPUExecutor(gpus=[0, 1, 2])
    members = ex.run(tmp_path / "prot.fasta", tmp_path, spec)

    assert len(members) == 6
    assert [m.member_idx for m in members] == [0, 1, 2, 3, 4, 5]
    assert (tmp_path / "metadata.csv").exists()
    assert (tmp_path / "manifest.json").exists()


@pytest.mark.unit
def test_executor_restart_only_folds_pending(tmp_path, monkeypatch):
    _install_fake_torch(monkeypatch)
    spec = EnsembleSpec(members=6)
    # simulate a crashed run that finished members 0,1,2
    for m in (0, 1, 2):
        execution._write_sidecar(tmp_path, m, [_member(m)])

    seen = {}
    real_spawn = sys.modules["torch.multiprocessing"].spawn

    def tracking_spawn(fn, args, nprocs, join):
        seen["shards"] = args[4]
        return real_spawn(fn, args, nprocs, join)

    sys.modules["torch.multiprocessing"].spawn = tracking_spawn

    ex = execution.LocalGPUExecutor(gpus=[0, 1, 2])
    members = ex.run(tmp_path / "prot.fasta", tmp_path, spec)

    assert sorted(m for s in seen["shards"] for m in s) == [3, 4, 5]  # only pending
    assert len(members) == 6                                          # merged with prior


@pytest.mark.unit
def test_executor_all_complete_skips_spawn(tmp_path, monkeypatch):
    for m in range(4):
        execution._write_sidecar(tmp_path, m, [_member(m)])

    def boom(*a, **k):
        raise AssertionError("spawn must not run when nothing is pending")

    fake_mp = types.ModuleType("torch.multiprocessing")
    fake_mp.spawn = boom
    fake_torch = types.ModuleType("torch")
    fake_torch.multiprocessing = fake_mp
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", fake_mp)

    ex = execution.LocalGPUExecutor(gpus=[0, 1])
    members = ex.run(tmp_path / "prot.fasta", tmp_path, EnsembleSpec(members=4))
    assert len(members) == 4


@pytest.mark.unit
def test_executor_caps_workers_at_pending(tmp_path, monkeypatch):
    """More GPUs than members -> only as many workers as members."""
    captured = {}
    _install_fake_torch(monkeypatch)
    real_spawn = sys.modules["torch.multiprocessing"].spawn

    def capturing_spawn(fn, args, nprocs, join):
        captured["nprocs"] = nprocs
        return real_spawn(fn, args, nprocs, join)

    sys.modules["torch.multiprocessing"].spawn = capturing_spawn

    ex = execution.LocalGPUExecutor(gpus=[0, 1, 2, 3, 4, 5, 6, 7])
    members = ex.run(tmp_path / "prot.fasta", tmp_path, EnsembleSpec(members=2))
    assert captured["nprocs"] == 2
    assert len(members) == 2


# -- real multi-GPU fold ---------------------------------------------------


@pytest.mark.integration
@pytest.mark.gpu
def test_real_multi_gpu_fold(tmp_path):
    """Opt-in real fold fanned across >=2 GPUs. Run with: pytest -m gpu"""
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("need >=2 GPUs")

    example = Path(__file__).resolve().parents[1] / "examples" / "example.pdb"
    from python_package.ensemble import SamplingParams

    spec = EnsembleSpec(
        members=4, base_seed=1,
        sampling=SamplingParams(num_loops=4, num_sampling_steps=20, num_diffusion_samples=1),
    )
    members = execution.LocalGPUExecutor(gpus=[0, 1]).run(example, tmp_path / "run", spec)

    assert len(members) == 4
    assert all(Path(m.cif_path).exists() for m in members)
    assert (tmp_path / "run" / "metadata.csv").exists()
