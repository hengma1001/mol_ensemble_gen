"""End-to-end GPU integration check for the finetuning loss path.

Opt-in (needs the ``genAI`` env + a GPU + weights download). Captures the real
denoiser conditioning via the featurization monkeypatch, runs one EDM loss step
through the actual ``diffusion_module``, and asserts a finite loss with gradients
flowing to both the diffusion module and the temperature embedder — the pieces
the trainer optimizes. Run with: ``pytest -m gpu``.
"""

from __future__ import annotations

import pytest

# A short, fully-standard sequence keeps the trunk forward cheap.
SEQUENCE = "GSAGKLETVW"


@pytest.mark.integration
@pytest.mark.gpu
def test_edm_loss_and_gradients_on_real_denoiser():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU available")

    from esm.models.esmfold2 import ESMFold2InputBuilder
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from mol_ensemble_gen.training.conditioning import build_temperature_embedder
    from mol_ensemble_gen.training.config import TemperatureConfig
    from mol_ensemble_gen.training.featurize import (
        _capture_conditioning,
        _protein_spi,
    )
    from mol_ensemble_gen.training.loss import edm_diffusion_loss

    device = "cuda"
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(device).eval()
    builder = ESMFold2InputBuilder()

    captured = _capture_conditioning(model, builder, _protein_spi(SEQUENCE), num_loops=4)

    # Keep the conditioning tensors on-device in fp32 (the loss runs the coord
    # math in fp32; the denoiser call is wrapped in autocast below).
    conditioning = {}
    for key, val in captured.items():
        conditioning[key] = val.to(device) if torch.is_tensor(val) else val

    ref_mask = conditioning["ref_mask"].bool().reshape(-1)          # (n_atoms,)
    ref_pos = conditioning["ref_pos"].to(torch.float32).reshape(1, -1, 3)
    n_atoms = ref_pos.shape[1]
    assert ref_mask.shape[0] == n_atoms

    # Use the idealized reference conformer as a stand-in ground truth, batched.
    b = 2
    gt_coords = ref_pos.expand(b, -1, -1).contiguous().to(device)

    head = model.structure_head
    diffusion_module = head.diffusion_module
    temp_embedder = build_temperature_embedder(TemperatureConfig()).to(device)

    diffusion_module.requires_grad_(True)
    temp_embedder.requires_grad_(True)

    gen = torch.Generator(device=device).manual_seed(0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, metrics = edm_diffusion_loss(
            diffusion_module,
            head,
            temp_embedder,
            conditioning,
            gt_coords,
            ref_mask,
            temperature=379.0,
            generator=gen,
        )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert metrics["mse"] >= 0.0

    loss.backward()

    # Gradients reach the diffusion module...
    dm_grad = [p.grad for p in diffusion_module.parameters() if p.grad is not None]
    assert dm_grad, "no gradient reached diffusion_module"
    assert all(torch.isfinite(g).all() for g in dm_grad)
    assert sum(float(g.abs().sum()) for g in dm_grad) > 0.0

    # ...and the temperature embedder's (zero-init) output layer.
    out_layer = temp_embedder.mlp[-1]
    assert out_layer.weight.grad is not None, "no gradient reached temp embedder"
    assert torch.isfinite(out_layer.weight.grad).all()


def _capture_flow_fixture(device):
    """Shared setup: real model + conditioning + batched GT for flow tests."""
    import torch

    from esm.models.esmfold2 import ESMFold2InputBuilder
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from mol_ensemble_gen.training.conditioning import build_temperature_embedder
    from mol_ensemble_gen.training.config import TemperatureConfig
    from mol_ensemble_gen.training.featurize import _capture_conditioning, _protein_spi

    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(device).eval()
    builder = ESMFold2InputBuilder()
    captured = _capture_conditioning(model, builder, _protein_spi(SEQUENCE), num_loops=4)
    conditioning = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in captured.items()
    }
    temp_embedder = build_temperature_embedder(TemperatureConfig()).to(device)
    return model, conditioning, temp_embedder


@pytest.mark.integration
@pytest.mark.gpu
def test_flow_loss_and_gradients_on_real_denoiser():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU available")

    from mol_ensemble_gen.training.config import FlowConfig
    from mol_ensemble_gen.training.loss import diffusion_loss, flow_matching_loss

    device = "cuda"
    model, conditioning, temp_embedder = _capture_flow_fixture(device)

    ref_mask = conditioning["ref_mask"].bool().reshape(-1)
    ref_pos = conditioning["ref_pos"].to(torch.float32).reshape(1, -1, 3)
    b = 2
    gt_coords = ref_pos.expand(b, -1, -1).contiguous().to(device)

    head = model.structure_head
    diffusion_module = head.diffusion_module
    diffusion_module.requires_grad_(True)
    temp_embedder.requires_grad_(True)

    gen = torch.Generator(device=device).manual_seed(0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, metrics = flow_matching_loss(
            diffusion_module, head, temp_embedder, conditioning,
            gt_coords, ref_mask, temperature=379.0, generator=gen,
        )

    assert torch.isfinite(loss) and loss.item() >= 0.0
    assert metrics["mse"] >= 0.0 and 0.0 < metrics["t_mean"] < 1.0

    loss.backward()
    dm_grad = [p.grad for p in diffusion_module.parameters() if p.grad is not None]
    assert dm_grad, "no gradient reached diffusion_module under flow scheme"
    assert all(torch.isfinite(g).all() for g in dm_grad)
    assert sum(float(g.abs().sum()) for g in dm_grad) > 0.0

    # Dispatcher parity: diffusion_loss("flow", ..., flow=cfg) hits the same path.
    diffusion_module.zero_grad(set_to_none=True)
    gen2 = torch.Generator(device=device).manual_seed(0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss2, _ = diffusion_loss(
            "flow", diffusion_module, head, temp_embedder, conditioning,
            gt_coords, ref_mask, temperature=379.0, generator=gen2, flow=FlowConfig(),
        )
    assert torch.isfinite(loss2)


@pytest.mark.integration
@pytest.mark.gpu
def test_flow_ode_sample_produces_finite_coords():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU available")

    from mol_ensemble_gen.training.sample import flow_ode_sample

    device = "cuda"
    model, conditioning, _ = _capture_flow_fixture(device)
    head = model.structure_head
    n_atoms = conditioning["tok_idx"].shape[1]

    # Override the captured broadcast factor to sample a 2-frame batch.
    conditioning = {**conditioning, "num_diffusion_samples": 2}

    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = flow_ode_sample(
            head.diffusion_module, head, steps=5, sampler="euler", sigma_max=256.0,
            generator=gen, **conditioning,
        )

    x = out["sample_atom_coords"]
    assert x.shape == (2, n_atoms, 3)
    assert torch.isfinite(x).all()

    # Heun path also runs and stays finite (few steps for speed).
    conditioning_h = {**conditioning, "num_diffusion_samples": 1}
    gen2 = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_h = flow_ode_sample(
            head.diffusion_module, head, steps=3, sampler="heun", sigma_max=256.0,
            generator=gen2, **conditioning_h,
        )
    assert torch.isfinite(out_h["sample_atom_coords"]).all()
