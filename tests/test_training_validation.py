"""Tests for the held-out validation pass.

Five 5000-step pilots ran with no validation signal at all, so a rising training
MSE in the last quarter could not be told apart from overfitting. These cover the
mechanics of the pass rather than the numbers: that it averages the right thing,
leaves the module's training mode alone, and is reproducible across calls.
"""

from __future__ import annotations

import pytest

from mol_ensemble_gen.training.trainer import _run_validation


class _FakeBatch:
    def __init__(self, domain, temperature, gt, mask):
        self.domain = domain
        self.temperature = temperature
        self.gt_coords = gt
        self.atom_mask = mask


class _FakeCache:
    def get(self, domain):
        return {"domain": domain}


class _FakeModule:
    """Records calls; returns a loss that depends on the generator seed."""

    def __init__(self):
        import torch

        self.training = True
        self.calls = []
        self._torch = torch

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, cond, gt, mask, temperature, generator=None):
        torch = self._torch
        # Draw from the generator so a fixed seed gives a reproducible "loss".
        val = torch.rand(1, generator=generator, device=gt.device).item()
        self.calls.append((cond["domain"], temperature, self.training, val))
        return torch.tensor(val), {"mse": val * 2.0}


def _loader(n, torch, temps=(320.0, 450.0)):
    """A re-iterable batch source — what _run_validation requires.

    Deliberately a list, not a generator: the whole point of the contract is that
    iterating it twice yields the same batches twice.
    """
    import numpy as np

    return [
        _FakeBatch(
            f"d{i % 2}", temps[i % len(temps)],
            np.zeros((2, 4, 3), dtype="float32"),
            np.ones(4, dtype=bool),
        )
        for i in range(n)
    ]


@pytest.mark.unit
def test_validation_averages_and_restores_training_mode():
    torch = pytest.importorskip("torch")

    m = _FakeModule()
    m.train()
    out = _run_validation(
        m, _FakeCache(), _loader(8, torch), 4, torch.float32, torch.device("cpu"), seed=0
    )
    assert set(out) == {"loss", "mse"}
    assert len(m.calls) == 4
    # eval() during the pass...
    assert all(not training for *_, training, _ in ((c[0], c[1], c[2], c[3]) for c in m.calls))
    # ...train() restored after.
    assert m.training
    assert out["mse"] == pytest.approx(out["loss"] * 2.0)


@pytest.mark.unit
def test_validation_is_reproducible_across_calls():
    """A fixed seed means the curve moves because the model moved, not the σ draw."""
    torch = pytest.importorskip("torch")

    a = _run_validation(_FakeModule(), _FakeCache(), _loader(4, torch), 4,
                        torch.float32, torch.device("cpu"), seed=7)
    b = _run_validation(_FakeModule(), _FakeCache(), _loader(4, torch), 4,
                        torch.float32, torch.device("cpu"), seed=7)
    assert a == b

    c = _run_validation(_FakeModule(), _FakeCache(), _loader(4, torch), 4,
                        torch.float32, torch.device("cpu"), seed=8)
    assert c != a


@pytest.mark.unit
def test_repeated_calls_score_the_same_batches():
    """Regression: a shared advancing iterator scored *different* frames each call.

    Domains and temperatures differ enough in difficulty to swamp the training
    signal — observed live as val_mse 3.3 -> 6.3 between two evaluations of a
    barely-changed model. The loader must be re-iterated from the start.
    """
    torch = pytest.importorskip("torch")

    loader = _loader(20, torch)          # more batches than we consume per call
    m = _FakeModule()
    _run_validation(m, _FakeCache(), loader, 4, torch.float32, torch.device("cpu"), seed=1)
    first = [(d, t) for d, t, _, _ in m.calls]
    m.calls.clear()
    _run_validation(m, _FakeCache(), loader, 4, torch.float32, torch.device("cpu"), seed=1)
    assert [(d, t) for d, t, _, _ in m.calls] == first


@pytest.mark.unit
def test_validation_stops_early_when_the_stream_runs_out():
    torch = pytest.importorskip("torch")

    m = _FakeModule()
    out = _run_validation(m, _FakeCache(), _loader(2, torch), 10,
                          torch.float32, torch.device("cpu"), seed=0)
    assert len(m.calls) == 2 and out


@pytest.mark.unit
@pytest.mark.parametrize("val_iter,n", [(None, 4), ("iter", 0)])
def test_validation_is_a_no_op_when_disabled(val_iter, n):
    torch = pytest.importorskip("torch")

    it = None if val_iter is None else _loader(4, torch)
    m = _FakeModule()
    assert _run_validation(m, _FakeCache(), it, n, torch.float32,
                           torch.device("cpu"), seed=0) == {}
    assert m.calls == []
    assert m.training, "a disabled validation pass must not touch training mode"


@pytest.mark.unit
def test_val_config_defaults_and_dataset_override():
    """val_every/val_batches exist, and make_dataset can build the held-out split."""
    pytest.importorskip("torch")
    from mol_ensemble_gen.training.config import TrainConfig, _build
    from mol_ensemble_gen.training.mdcath import make_dataset

    cfg = TrainConfig()
    assert cfg.val_every == 500 and cfg.val_batches == 16

    cfg = _build(TrainConfig, {"data": {"domains": ["a"], "val_domains": ["v1", "v2"]}})
    train_ds = make_dataset(cfg, rank=0, world_size=1)
    val_ds = make_dataset(cfg, rank=0, world_size=1, domains=cfg.data.val_domains)
    assert train_ds.domains == ["a"], "val domains must stay out of training"
    assert val_ds.domains == ["v1", "v2"]
