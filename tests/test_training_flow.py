"""Offline unit tests for the flow-matching (rectified-flow) scheme.

No GPU/torch model needed: these cover the σ↔t change of variables, the
``diffusion_loss`` dispatcher, and config load/validation of ``scheme``/``flow``.
The end-to-end loss + ODE sampler are exercised in ``test_training_int_gpu.py``.
"""

from __future__ import annotations

import math
import textwrap

import pytest

from mol_ensemble_gen.training import loss as loss_mod
from mol_ensemble_gen.training.config import (
    FlowConfig,
    TrainConfig,
    _build,
    load_train_config,
)


@pytest.mark.unit
@pytest.mark.parametrize("sigma", [1e-3, 0.1, 1.0, 16.0, 256.0])
def test_sigma_t_change_of_variables_roundtrips(sigma):
    # t = σ/(1+σ);  σ = t/(1−t)  — the mapping the flow scheme relies on.
    t = sigma / (1.0 + sigma)
    back = t / (1.0 - t)
    assert back == pytest.approx(sigma, rel=1e-9, abs=1e-12)
    assert 0.0 < t < 1.0


@pytest.mark.unit
def test_logitnormal_t_is_sigmoid_of_log_sigma():
    import math

    # ln σ ~ N(p_mean,p_std) ⇒ t = σ/(1+σ) = sigmoid(ln σ).
    for log_sigma in (-3.0, -1.2, 0.0, 2.0):
        sigma = math.exp(log_sigma)
        assert 1.0 / (1.0 + math.exp(-log_sigma)) == pytest.approx(sigma / (1.0 + sigma))


@pytest.mark.unit
def test_flow_sigma_coverage_matches_edm():
    """The flow σ draw must land on the EDM branch's noise band, not σ_d× below it.

    Regression: the logit-normal time was drawn as ``sigmoid(p)``, which yields
    ``σ = exp(p)`` — a factor ``σ_d`` (16×) below EDM's ``σ = σ_d·exp(p)``. The
    two branches must sample the *same* σ distribution, since that identity is
    the whole reason flow can reuse the pretrained EDM weights.
    """
    torch = pytest.importorskip("torch")

    from mol_ensemble_gen.training.config import (
        SIGMA_DATA,
        TRAIN_NOISE_LOG_MEAN,
        TRAIN_NOISE_LOG_STD,
    )

    g = torch.Generator().manual_seed(0)
    n = 200_000
    p = TRAIN_NOISE_LOG_MEAN + TRAIN_NOISE_LOG_STD * torch.randn(n, generator=g)

    edm_sigma = SIGMA_DATA * torch.exp(p)
    # The flow path: t = sigmoid(ln σ_d + p), then σ = t/(1−t).
    t = torch.sigmoid(math.log(SIGMA_DATA) + p)
    flow_sigma = t / (1.0 - t)

    # Same draw of p ⇒ the mapping is exact, not merely distributional. Compared
    # in log space: recovering σ from t loses float32 precision at the extreme
    # tail (σ ~ 5e3) through cancellation in (1−t), which is conditioning, not a
    # logic error — the same check in float64 agrees to ~4e-13.
    torch.testing.assert_close(flow_sigma.log(), edm_sigma.log(), rtol=0, atol=1e-3)
    assert flow_sigma.median().item() == pytest.approx(
        SIGMA_DATA * math.exp(TRAIN_NOISE_LOG_MEAN), rel=0.02
    )


@pytest.mark.unit
def test_flow_median_time_sits_on_the_noise_side():
    """The EDM-matched band puts most mass at high t, shrinking the 1/t² weight.

    Median t ≈ 0.83 gives a velocity weight near 1.5, against ≈ 19 for the
    unshifted draw's median t ≈ 0.23. Note this does *not* by itself relieve
    gradient-clip pressure — measured on the corrected pilot, clip saturation rose
    to 100%, because the MSE at the higher noise band is ~13× larger. This test
    pins the weight scale only.
    """
    from mol_ensemble_gen.training.config import SIGMA_DATA, TRAIN_NOISE_LOG_MEAN

    sigma_med = SIGMA_DATA * math.exp(TRAIN_NOISE_LOG_MEAN)
    t_med = sigma_med / (1.0 + sigma_med)
    assert t_med == pytest.approx(0.828, abs=0.01)
    assert 1.0 / t_med**2 < 2.0


@pytest.mark.unit
def test_dispatch_routes_by_scheme(monkeypatch):
    calls = {}

    def fake_edm(*args, **kwargs):
        calls["edm"] = (args, kwargs)
        return "edm-result"

    def fake_flow(*args, **kwargs):
        calls["flow"] = (args, kwargs)
        return "flow-result"

    monkeypatch.setattr(loss_mod, "edm_diffusion_loss", fake_edm)
    monkeypatch.setattr(loss_mod, "flow_matching_loss", fake_flow)

    assert loss_mod.diffusion_loss("edm", 1, 2, 3) == "edm-result"
    assert calls["edm"][0] == (1, 2, 3)

    flow = FlowConfig(p_mean=-0.5, p_std=2.0, time_dist="uniform", weighting="data", t_min=5e-3)
    assert loss_mod.diffusion_loss("flow", 1, flow=flow) == "flow-result"
    fkw = calls["flow"][1]
    assert fkw["p_mean"] == -0.5 and fkw["p_std"] == 2.0
    assert fkw["time_dist"] == "uniform" and fkw["weighting"] == "data"
    assert fkw["t_min"] == 5e-3


@pytest.mark.unit
def test_dispatch_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown scheme"):
        loss_mod.diffusion_loss("brownian")


@pytest.mark.unit
def test_dispatch_flow_without_config_uses_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(loss_mod, "flow_matching_loss", lambda *a, **k: seen.update(k) or "ok")
    assert loss_mod.diffusion_loss("flow") == "ok"
    # No flow config → flow_matching_loss falls back to its own defaults.
    assert seen == {}


@pytest.mark.unit
def test_defaults_scheme_is_flow():
    """Flow is the default scheme (was EDM before the σ_d fix landed).

    The optimizer defaults move with it — see
    ``test_model_flow.test_flow_is_the_default_scheme_with_flow_tuned_optimizer``
    for why ``grad_clip``/``lr`` are what they are.
    """
    cfg = TrainConfig()
    assert cfg.optim.scheme == "flow"
    assert cfg.flow.time_dist == "logitnormal"
    assert cfg.flow.weighting == "velocity"


@pytest.mark.unit
def test_config_loads_flow_scheme(tmp_path):
    yaml_text = textwrap.dedent(
        """
        optim:
          scheme: flow
        flow:
          time_dist: uniform
          weighting: data
          num_sampling_steps: 30
          sampler: heun
          sigma_max: 128.0
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)
    cfg = load_train_config(path)
    assert cfg.optim.scheme == "flow"
    assert cfg.flow.time_dist == "uniform"
    assert cfg.flow.weighting == "data"
    assert cfg.flow.num_sampling_steps == 30
    assert cfg.flow.sampler == "heun"
    assert cfg.flow.sigma_max == 128.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "yaml_text,match",
    [
        ("optim:\n  scheme: sde\n", "scheme"),
        ("flow:\n  time_dist: cosine\n", "time_dist"),
        ("flow:\n  weighting: snr\n", "weighting"),
        ("flow:\n  sampler: rk4\n", "sampler"),
    ],
)
def test_config_rejects_bad_flow_values(tmp_path, yaml_text, match):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match=match):
        load_train_config(path)


@pytest.mark.unit
def test_config_rejects_unknown_flow_key():
    with pytest.raises(ValueError, match="FlowConfig"):
        _build(TrainConfig, {"flow": {"bogus_knob": 1}})


# ---------------------------------------------------------------------------
# temperature conditioning: FiLM gain + spread-matching term
# ---------------------------------------------------------------------------


def test_film_gain_is_identity_at_init():
    """Enabling ``film`` must not perturb a pretrained model at step 0.

    The gain head is zero-initialized, so the multiplier is exactly 1.0 and
    ``inject_temperature`` reduces to the additive-bias behaviour it had before.
    """
    import torch

    from mol_ensemble_gen.training.conditioning import build_temperature_embedder
    from mol_ensemble_gen.training.config import TemperatureConfig
    from mol_ensemble_gen.training.loss import inject_temperature

    cond = {
        "s_inputs": torch.randn(1, 7, 451),
        "token_attention_mask": torch.tensor([[1.0] * 5 + [0.0, 0.0]]),
    }
    plain = build_temperature_embedder(TemperatureConfig(film=False))
    film = build_temperature_embedder(TemperatureConfig(film=True))
    for T in (320.0, 450.0):
        a = inject_temperature(cond, plain, T, torch.float32)
        b = inject_temperature(cond, film, T, torch.float32)
        assert torch.allclose(a, b, atol=0, rtol=0), "film must be exactly identity at init"
        # and at init the injection is a no-op on s_inputs entirely
        assert torch.allclose(a, cond["s_inputs"], atol=1e-6)

    assert torch.equal(film.scale(torch.tensor([320.0, 450.0])),
                       torch.ones(2, 451)), "gain must start at exactly 1.0"


def test_film_gain_is_multiplicative_once_trained():
    """A non-zero gain scales s_inputs and leaves padding tokens alone."""
    import torch

    from mol_ensemble_gen.training.conditioning import build_temperature_embedder
    from mol_ensemble_gen.training.config import TemperatureConfig
    from mol_ensemble_gen.training.loss import inject_temperature

    te = build_temperature_embedder(TemperatureConfig(film=True))
    with torch.no_grad():                      # pretend it trained: constant gain of +0.5
        te.gain[-1].bias.fill_(0.5)
    si = torch.randn(1, 4, 451)
    cond = {"s_inputs": si, "token_attention_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]])}
    out = inject_temperature(cond, te, 450.0, torch.float32)
    assert torch.allclose(out[:, :2], si[:, :2] * 1.5, atol=1e-5), "real tokens scale by the gain"
    assert torch.allclose(out[:, 2:], si[:, 2:], atol=1e-6), "padding tokens are untouched"


def test_internal_spread_is_rigid_invariant_and_scales():
    """The spread probe must see conformational change, not pose."""
    import torch

    from mol_ensemble_gen.training.loss import _internal_spread

    torch.manual_seed(3)
    mask = torch.ones(5, 24)
    x = torch.randn(5, 24, 3) * 4.0
    base = _internal_spread(x, mask)

    q = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.det(q) < 0:
        q[:, 0] *= -1
    moved = x @ q + torch.tensor([12.0, -3.0, 5.0])
    assert torch.allclose(_internal_spread(moved, mask), base, atol=1e-3), \
        "a rigid motion of every frame must not change the spread"
    assert torch.allclose(_internal_spread(x * 2.0, mask), base * 2.0, rtol=1e-3), \
        "spread is a length, so it scales linearly"
    same = x[:1].expand(5, -1, -1).contiguous()
    assert float(_internal_spread(same, mask)) < 1e-3, "identical frames have no spread"


def test_spread_weight_zero_leaves_the_loss_untouched():
    """The default config must reproduce the previous loss exactly.

    Guards the whole feature: ``weight: 0`` has to be a true no-op, including not
    switching the σ draw to one-per-micro-batch.
    """
    import inspect

    from mol_ensemble_gen.training.config import SpreadConfig
    from mol_ensemble_gen.training import loss as L

    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return 0.0, {}

    real = L.flow_matching_loss
    try:
        L.flow_matching_loss = fake
        L.diffusion_loss("flow", spread=SpreadConfig(), flow=None)
        assert "spread_weight" not in captured, "weight 0 must not pass a spread term"
        assert "shared_sigma" not in captured, "weight 0 must not change the sigma draw"
        captured.clear()
        L.diffusion_loss("flow", spread=SpreadConfig(weight=0.3, atoms=64), flow=None)
        assert captured["spread_weight"] == 0.3
        assert captured["spread_atoms"] == 64
        assert captured["shared_sigma"] is True
    finally:
        L.flow_matching_loss = real

    # the plumbing exists on both entry points
    for fn in (L.flow_matching_loss, L.edm_diffusion_loss):
        p = inspect.signature(fn).parameters
        assert {"spread_weight", "spread_atoms", "shared_sigma"} <= set(p)
        assert p["spread_weight"].default == 0.0
        assert p["shared_sigma"].default is False


def test_spread_term_penalises_collapse_and_vanishes_when_matched():
    """The term must be ~0 when spreads agree and grow when the model collapses."""
    import math
    import torch

    from mol_ensemble_gen.training.loss import _internal_spread

    torch.manual_seed(11)
    mask = torch.ones(6, 30)
    gt = torch.randn(6, 30, 3) * 3.0
    sp_gt = _internal_spread(gt, mask)

    matched = math.log(float(_internal_spread(gt.clone(), mask)) / float(sp_gt)) ** 2
    assert matched < 1e-8, "identical spread must cost nothing"

    collapsed = gt.mean(0, keepdim=True).expand_as(gt).contiguous()
    penalty = math.log(max(float(_internal_spread(collapsed, mask)), 1e-6) / float(sp_gt)) ** 2
    assert penalty > 1.0, "a collapsed ensemble must be penalised heavily"
