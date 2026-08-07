"""Tests for the flow-matching interface (``mol_ensemble_gen.model.flow``).

Everything here runs offline on CPU against a tiny random-init denoiser. The
properties checked are exact identities or invariants that hold for *any* weights,
so they pin the parameterization rather than the trained behaviour:

* the ``σ ↔ t`` change of variables, and the ``ln σ_d`` shift that aligns flow's
  noise band with EDM's (the bug that cost a pilot run — see DESIGN.md §11);
* the velocity ↔ x₀ algebra;
* ``t_conditioning="add"`` being **bitwise** the pretrained model at init;
* the ODE sampler being deterministic and staying finite.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from mol_ensemble_gen.model.flow import (
    SIGMA_DATA,
    TRAIN_NOISE_LOG_MEAN,
    sigma_to_t,
    t_to_sigma,
)

from test_model_denoiser import TINY, _tiny_inputs

TINY_ADD = dataclasses.replace(TINY, t_conditioning="add", t_fourier_dim=8)
TINY_REPLACE = dataclasses.replace(TINY, t_conditioning="replace", t_fourier_dim=8)


@pytest.mark.unit
@pytest.mark.parametrize("sigma", [1e-3, 0.3, 1.0, 4.82, 16.0, 256.0])
def test_sigma_t_roundtrip(sigma):
    assert t_to_sigma(sigma_to_t(sigma)) == pytest.approx(sigma, rel=1e-9)


@pytest.mark.unit
def test_sample_flow_time_matches_edm_band():
    """logit-normal ``t`` must reproduce EDM's σ band, median ``σ_d·e^{p_mean}``."""
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.flow import sample_flow_time

    g = torch.Generator().manual_seed(0)
    t, sigma = sample_flow_time(200_000, time_dist="logitnormal", generator=g)
    assert sigma.median().item() == pytest.approx(
        SIGMA_DATA * math.exp(TRAIN_NOISE_LOG_MEAN), rel=0.02
    )
    # σ and t must stay mutually consistent element-wise.
    torch.testing.assert_close(t_to_sigma(t), sigma, rtol=1e-5, atol=1e-6)
    assert (t > 0).all() and (t < 1).all()


@pytest.mark.unit
def test_uniform_time_is_deliberately_not_edm_matched():
    """``time_dist="uniform"`` is uniform in path time: median σ ≈ 1, not 4.82."""
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.flow import sample_flow_time

    g = torch.Generator().manual_seed(0)
    _, sigma = sample_flow_time(100_000, time_dist="uniform", generator=g)
    assert sigma.median().item() == pytest.approx(1.0, rel=0.05)


@pytest.mark.unit
def test_unknown_time_dist_raises():
    pytest.importorskip("torch")
    from mol_ensemble_gen.model.flow import sample_flow_time

    with pytest.raises(ValueError, match="time_dist"):
        sample_flow_time(4, time_dist="cosine")


@pytest.mark.unit
def test_velocity_x0_algebra_is_exact():
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.flow import velocity_from_x0, x0_from_velocity

    g = torch.Generator().manual_seed(1)
    x_t = torch.randn(3, 12, 3, generator=g)
    x0 = torch.randn(3, 12, 3, generator=g)
    t = torch.tensor([0.1, 0.5, 0.9])

    v = velocity_from_x0(x_t, x0, t)
    torch.testing.assert_close(x0_from_velocity(x_t, v, t), x0, rtol=1e-6, atol=1e-6)
    # v = (x_t - x0)/t, elementwise
    torch.testing.assert_close(v, (x_t - x0) / t[:, None, None], rtol=0, atol=0)


@pytest.mark.unit
def test_flow_denoiser_velocity_matches_its_own_x0():
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule
    from mol_ensemble_gen.model.flow import FlowDenoiser

    model = DiffusionModule(TINY).eval()
    cond = _tiny_inputs(torch)
    n_atoms = cond["tok_idx"].shape[1]
    x_t = torch.randn(2, n_atoms, 3)
    t = torch.tensor([0.3, 0.8])

    fd = FlowDenoiser(model)
    with torch.no_grad():
        x0 = fd.predict_x0(x_t, t, num_diffusion_samples=2, **cond)
        v = fd.velocity(x_t, t, num_diffusion_samples=2, **cond)
    torch.testing.assert_close(v, (x_t - x0) / t[:, None, None], rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_add_mode_is_bitwise_identical_to_pretrained_at_init():
    """The whole point of zero-init ``add``: step 0 must be the pretrained model.

    If this drifts, a finetune no longer starts from the released model's
    behaviour, and the temperature/flow signal is no longer learned from an
    identity baseline.
    """
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule

    off = DiffusionModule(TINY).eval()
    add = DiffusionModule(TINY_ADD).eval()
    add.load_state_dict(off.state_dict(), strict=False)

    cond = _tiny_inputs(torch)
    n_atoms = cond["tok_idx"].shape[1]
    x = torch.randn(2, n_atoms, 3)
    t = torch.tensor([0.4, 0.85])
    with torch.no_grad():
        a = off(x_noisy=x, t_hat=t_to_sigma(t), num_diffusion_samples=2, **cond)["x_denoised"]
        b = add(
            x_noisy=x, t_hat=t_to_sigma(t), num_diffusion_samples=2, flow_t=t, **cond
        )["x_denoised"]
    assert torch.equal(a, b)


@pytest.mark.unit
def test_replace_mode_actually_changes_the_conditioning():
    """``replace`` drops the log-σ features, so it must *not* match ``off``."""
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule

    off = DiffusionModule(TINY).eval()
    rep = DiffusionModule(TINY_REPLACE).eval()
    rep.load_state_dict(off.state_dict(), strict=False)

    cond = _tiny_inputs(torch)
    x = torch.randn(1, cond["tok_idx"].shape[1], 3)
    t = torch.tensor([0.5])
    with torch.no_grad():
        a = off(x_noisy=x, t_hat=t_to_sigma(t), num_diffusion_samples=1, **cond)["x_denoised"]
        c = rep(
            x_noisy=x, t_hat=t_to_sigma(t), num_diffusion_samples=1, flow_t=t, **cond
        )["x_denoised"]
    assert not torch.allclose(a, c)


@pytest.mark.unit
def test_t_head_adds_only_expected_tensors():
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule

    off = set(DiffusionModule(TINY).state_dict())
    add = set(DiffusionModule(TINY_ADD).state_dict())
    assert add - off == {
        "conditioning.t_fourier.w",
        "conditioning.t_fourier.b",
        "conditioning.t_norm.weight",
        "conditioning.t_norm.bias",
        "conditioning.t_proj.weight",
    }
    assert not off - add


@pytest.mark.unit
def test_load_denoiser_permits_only_the_t_head_to_be_missing():
    """A real mismatch must still fail, even with the t-head exemption in play."""
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule, load_denoiser

    base = DiffusionModule(TINY).state_dict()
    # t-head missing: allowed.
    load_denoiser(base, config=TINY_ADD)

    # Something else missing: must raise.
    broken = {k: v for k, v in base.items() if k != "token_norm.weight"}
    with pytest.raises(RuntimeError, match="mismatch"):
        load_denoiser(broken, config=TINY_ADD)

    # An unexpected key: must raise.
    extra = dict(base)
    extra["not_a_real_param"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="mismatch"):
        load_denoiser(extra, config=TINY)


@pytest.mark.unit
@pytest.mark.parametrize("sampler", ["euler", "heun"])
def test_flow_ode_sample_is_deterministic_and_finite(sampler):
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule
    from mol_ensemble_gen.model.flow import flow_ode_sample

    model = DiffusionModule(TINY).eval()
    cond = _tiny_inputs(torch)
    n_atoms = cond["tok_idx"].shape[1]
    kw = dict(
        n_atoms=n_atoms,
        batch=1,
        steps=3,
        sampler=sampler,
        device="cpu",
        atom_mask=cond["ref_mask"][0],
    )

    a = flow_ode_sample(model, cond, generator=torch.Generator().manual_seed(5), **kw)
    b = flow_ode_sample(model, cond, generator=torch.Generator().manual_seed(5), **kw)
    assert a.shape == (1, n_atoms, 3)
    assert torch.isfinite(a).all()
    # Same seed ⇒ same trajectory: this is the probability-flow ODE, no churn.
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.unit
def test_flow_ode_sample_rejects_unknown_sampler():
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule
    from mol_ensemble_gen.model.flow import flow_ode_sample

    model = DiffusionModule(TINY).eval()
    cond = _tiny_inputs(torch)
    with pytest.raises(ValueError, match="sampler"):
        flow_ode_sample(
            model, cond, n_atoms=cond["tok_idx"].shape[1], steps=1,
            sampler="rk4", device="cpu",
        )


# ---------------------------------------------------------------------------
# config wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flow_is_the_default_scheme_with_flow_tuned_optimizer():
    """Defaults must match the default scheme, including the retuned clip/lr.

    Measured on the corrected pilot: flow's median grad norm was 8.9 against a
    clip of 1.0, so *every* step clipped and the LR schedule was fictional.
    """
    from mol_ensemble_gen.training.config import TrainConfig

    cfg = TrainConfig()
    assert cfg.optim.scheme == "flow"
    # Calibrated to ≈p85 of the measured grad-norm distribution; see OptimConfig
    # for the clip 1.0 -> 20.0 -> 35.0 history and why lr does not move with it.
    assert cfg.optim.grad_clip == 35.0
    assert cfg.optim.lr == pytest.approx(1e-5)
    assert cfg.model.backend == "ours"
    assert cfg.flow.t_conditioning == "add"


@pytest.mark.unit
def test_reference_backend_rejects_native_t_conditioning():
    """The reference denoiser has no ``flow_t`` input — fail loudly, not silently."""
    from mol_ensemble_gen.training.config import TrainConfig, _build, _validate

    cfg = _build(TrainConfig, {"model": {"backend": "reference"}})
    with pytest.raises(ValueError, match="backend: ours"):
        _validate(cfg)

    # Explicitly turning it off is fine.
    ok = _build(
        TrainConfig,
        {"model": {"backend": "reference"}, "flow": {"t_conditioning": "off"}},
    )
    _validate(ok)


@pytest.mark.unit
def test_yaml_bare_off_is_accepted_for_t_conditioning(tmp_path):
    """``t_conditioning: off`` unquoted is boolean ``False`` in YAML 1.1.

    Rather than making users remember quotes around a token we chose, normalize it.
    ``True`` stays an error because there is no unambiguous mode to map it to.
    """
    from mol_ensemble_gen.training.config import load_train_config

    p = tmp_path / "off.yaml"
    p.write_text("flow:\n  t_conditioning: off\nmodel:\n  backend: reference\n")
    assert load_train_config(p).flow.t_conditioning == "off"

    p2 = tmp_path / "on.yaml"
    p2.write_text("flow:\n  t_conditioning: on\n")
    with pytest.raises(ValueError, match="t_conditioning"):
        load_train_config(p2)


@pytest.mark.unit
def test_bad_backend_rejected():
    from mol_ensemble_gen.training.config import TrainConfig, _build, _validate

    with pytest.raises(ValueError, match="backend"):
        _validate(_build(TrainConfig, {"model": {"backend": "jax"}}))
