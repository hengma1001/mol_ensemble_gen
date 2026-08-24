"""Tests for our reimplementation of ESMFold2's diffusion denoiser.

Three tiers:

* **Structural** (offline, no weights): the module's ``state_dict`` must match the
  pretrained denoiser's exactly — 345 tensors, 131,501,446 parameters. This is
  what makes ``load_state_dict(strict=True)`` on real weights possible, and it
  catches any accidental rename or reshape immediately.
* **Behavioural** (offline, tiny random-init config): a forward pass runs on CPU
  and satisfies the EDM preconditioning identity, which holds for *any* weights.
* **Parity** (``@pytest.mark.gpu``): against the reference module on identical
  weights and real cached conditioning. The tolerance is set by the reference's
  own run-to-run nondeterminism (bf16 attention reductions), measured in the same
  test rather than hard-coded — the reference does not reproduce itself bitwise.
"""

from __future__ import annotations

import pytest

from mol_ensemble_gen.model.denoiser import (
    PRETRAINED_NUM_PARAMS,
    PRETRAINED_NUM_TENSORS,
    DenoiserConfig,
    state_dict_signature,
)

CACHE_DIR = "cache/featurized"
PILOT_CKPT = "runs/finetune_pilot/checkpoint.pt"
PARITY_DOMAIN = "1ha8A00"

#: A tiny but shape-valid config for CPU tests. ``c_atom``/``atom_num_heads`` are
#: kept at their real values because 3D RoPE needs ``head_dim >= 2*(3*spatial+uid)``
#: channels to rotate; shrinking the atom head dim would make the rotary slice
#: wider than the head itself.
TINY = DenoiserConfig(
    c_atom=128,
    atom_num_heads=4,
    atom_num_blocks=1,
    c_token=64,
    token_num_heads=4,
    token_num_blocks=1,
    c_z=16,
    c_s_inputs=8,
    fourier_dim=8,
)


def _tiny_inputs(torch, n_atoms=24, n_tokens=6, batch=2, cfg=TINY):
    """Synthetic conditioning with the real tensor ranks/shapes."""
    from mol_ensemble_gen.model.denoiser import CHAR_VOCAB_SIZE, MAX_ATOMIC_NUMBER, MAX_CHARS

    g = torch.Generator().manual_seed(7)
    rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    # Atoms are laid out contiguously per token, as the model requires.
    tok_idx = torch.arange(n_atoms) * n_tokens // n_atoms
    return {
        "ref_pos": rnd(1, n_atoms, 3),
        "ref_charge": rnd(1, n_atoms),
        "ref_mask": torch.ones(1, n_atoms),
        "ref_element": rnd(1, n_atoms, MAX_ATOMIC_NUMBER),
        "ref_atom_name_chars": rnd(1, n_atoms, MAX_CHARS, CHAR_VOCAB_SIZE),
        "ref_space_uid": tok_idx.clone().unsqueeze(0).float(),
        "tok_idx": tok_idx.unsqueeze(0),
        "s_inputs": rnd(1, n_tokens, cfg.c_s_inputs),
        "s_trunk": None,
        "z_trunk": rnd(1, n_tokens, n_tokens, cfg.c_z),
        "relative_position_encoding": rnd(1, n_tokens, n_tokens, cfg.c_z),
        "asym_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "residue_index": torch.arange(n_tokens).unsqueeze(0),
        "entity_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "token_index": torch.arange(n_tokens).unsqueeze(0),
        "sym_id": torch.zeros(1, n_tokens, dtype=torch.long),
        "token_attention_mask": torch.ones(1, n_tokens, dtype=torch.bool),
    }


@pytest.mark.unit
def test_state_dict_matches_pretrained_denoiser_exactly():
    """Weight compatibility is the whole point: same names, same shapes, same count."""
    pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule

    state = DiffusionModule().state_dict()
    assert state_dict_signature(state) == (PRETRAINED_NUM_TENSORS, PRETRAINED_NUM_PARAMS)


@pytest.mark.unit
def test_expected_top_level_groups_and_block_counts():
    """Guards the structure the pretrained checkpoint's key prefixes imply."""
    pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule

    keys = list(DiffusionModule().state_dict())
    groups: dict[str, int] = {}
    for k in keys:
        groups[k.split(".")[0]] = groups.get(k.split(".")[0], 0) + 1
    assert groups == {
        "conditioning": 31,
        "atom_encoder": 23,
        "atom_decoder": 22,
        "s_to_token": 1,
        "token_transformer": 264,
        "s_step_norm": 2,
        "token_norm": 2,
    }
    # 12 interleaved attention/transition blocks in the token transformer.
    assert sum(1 for k in keys if k.startswith("token_transformer.attn_blocks.11.")) > 0
    assert not any(k.startswith("token_transformer.attn_blocks.12.") for k in keys)


@pytest.mark.unit
def test_forward_runs_on_cpu_and_shapes_are_right():
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule

    model = DiffusionModule(TINY).eval()
    kw = _tiny_inputs(torch)
    n_atoms = kw["tok_idx"].shape[1]
    b = 2
    x = torch.randn(b, n_atoms, 3)
    with torch.no_grad():
        out = model(
            x_noisy=x,
            t_hat=torch.full((b,), 4.0),
            num_diffusion_samples=b,
            return_token_repr=True,
            **kw,
        )
    assert out["x_denoised"].shape == (b, n_atoms, 3)
    assert out["token_repr"].shape == (b, kw["s_inputs"].shape[1], TINY.c_token)
    assert out["atom_intermediates"] is None
    assert torch.isfinite(out["x_denoised"]).all()


@pytest.mark.unit
def test_edm_preconditioning_returns_input_as_noise_vanishes():
    """As t→0 the skip coefficient σ²/(σ²+t²)→1, so D(x;t)→x for any weights.

    Checks the EDM wiring rather than the network, so it is weight-independent —
    a sign error in either preconditioning coefficient breaks it.
    """
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule

    model = DiffusionModule(TINY).eval()
    kw = _tiny_inputs(torch)
    x = torch.randn(1, kw["tok_idx"].shape[1], 3)
    with torch.no_grad():
        out = model(x_noisy=x, t_hat=torch.full((1,), 1e-8), num_diffusion_samples=1, **kw)
    torch.testing.assert_close(out["x_denoised"], x, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_geometry_ops_align_recovers_a_known_rotation():
    """``_weighted_rigid_align`` must undo a rigid motion.

    Tolerance is fp32-scale even though the inputs are float64: the covariance is
    cast with ``H.float()`` before the SVD (matching the reference), so the
    recovered rotation carries fp32 error regardless of input precision.
    """
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import GeometryOps

    g = torch.Generator().manual_seed(3)
    x_gt = torch.randn(1, 40, 3, generator=g).double()
    rot = GeometryOps._random_rotations(1, torch.float64, torch.device("cpu"))
    moved = x_gt @ rot[0].T + torch.tensor([[[5.0, -2.0, 1.5]]], dtype=torch.float64)
    w = torch.ones(1, 40, dtype=torch.float64)
    back = GeometryOps._weighted_rigid_align(moved, x_gt, w, w)
    torch.testing.assert_close(back, x_gt, rtol=1e-4, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("scheme", ["edm", "flow"])
def test_training_loss_runs_offline_against_our_modules(scheme):
    """The real training loss, on CPU, with no ESMFold2 code loaded.

    This is what the reimplementation buys: ``GeometryOps`` substitutes for the
    reference ``structure_head`` (the loss only ever uses its two helpers), so
    both objectives — including the backward pass and the gradient into the
    temperature head — are covered offline instead of only under
    ``@pytest.mark.gpu``.
    """
    torch = pytest.importorskip("torch")
    from mol_ensemble_gen.model.denoiser import DiffusionModule, GeometryOps
    from mol_ensemble_gen.training.conditioning import build_temperature_embedder
    from mol_ensemble_gen.training.config import FlowConfig, TemperatureConfig
    from mol_ensemble_gen.training.loss import diffusion_loss

    model = DiffusionModule(TINY)
    temp_embedder = build_temperature_embedder(
        TemperatureConfig(embed_dim=TINY.c_s_inputs, hidden_dim=16, num_fourier=4)
    )
    cond = _tiny_inputs(torch)
    n_atoms = cond["tok_idx"].shape[1]
    gt = torch.randn(4, n_atoms, 3) * 10.0
    atom_mask = torch.ones(n_atoms, dtype=torch.bool)
    atom_mask[-3:] = False  # unmatched slots must be excluded, not zero-filled

    loss, metrics = diffusion_loss(
        scheme,
        model,
        GeometryOps(),
        temp_embedder,
        cond,
        gt,
        atom_mask,
        348.0,
        flow=FlowConfig(),
    )
    loss.backward()

    assert torch.isfinite(loss) and loss.item() > 0
    assert metrics["mse"] > 0 and metrics["sigma_mean"] > 0
    if scheme == "flow":
        assert 0.0 < metrics["t_mean"] < 1.0
    # Gradient must reach both trained parts.
    assert model.atom_decoder.output_linear.weight.grad.abs().max() > 0
    assert temp_embedder.mlp[-1].weight.grad is not None


@pytest.mark.unit
def test_denoiser_needs_no_esm_or_transformers():
    """Importing and running our denoiser must not pull in the heavy stack."""
    import subprocess
    import sys

    code = (
        "import sys, torch;"
        "from mol_ensemble_gen.model.denoiser import DiffusionModule, DenoiserConfig;"
        "m = DiffusionModule(DenoiserConfig(c_token=64, token_num_heads=4,"
        " token_num_blocks=1, atom_num_blocks=1, c_z=16, c_s_inputs=8, fourier_dim=8));"
        "bad = sorted(k for k in sys.modules if k.split('.')[0] in ('esm', 'transformers'));"
        "print('LOADED:' + ','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "LOADED:\n" in out or out.strip().endswith("LOADED:"), out


# ---------------------------------------------------------------------------
# parity against the reference implementation (needs GPU + weights + cache)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.gpu
def test_parity_with_reference_within_its_own_nondeterminism():
    """Our denoiser must agree with the reference to within the reference's own noise.

    The reference does not reproduce itself bitwise — its bf16 attention
    reductions vary run to run — so an absolute tolerance would be arbitrary.
    We measure the reference's self-difference on identical inputs and require
    ours to be in the same ballpark.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from pathlib import Path

    if not Path(PILOT_CKPT).exists():
        pytest.skip(f"no checkpoint at {PILOT_CKPT}")
    if not Path(CACHE_DIR, PARITY_DOMAIN, "conditioning.pt").exists():
        pytest.skip(f"no featurization cache for {PARITY_DOMAIN}")

    from transformers.models.esmfold2.modeling_esmfold2_common import (
        DiffusionModule as Reference,
    )

    from mol_ensemble_gen.model.denoiser import DiffusionModule
    from mol_ensemble_gen.training.featurize import load_conditioning

    dev = "cuda"
    torch.manual_seed(0)
    state = torch.load(PILOT_CKPT, map_location="cpu", weights_only=False)["diffusion_module"]
    ref = Reference().to(dev).eval()
    ref.load_state_dict(state)
    mine = DiffusionModule().to(dev).eval()
    mine.load_state_dict(state)

    raw = load_conditioning(CACHE_DIR, PARITY_DOMAIN, device=dev)
    cond = {k: (None if v is None else (v.float() if v.is_floating_point() else v)) for k, v in raw.items()}
    keys = (
        "ref_pos",
        "ref_charge",
        "ref_mask",
        "ref_element",
        "ref_atom_name_chars",
        "ref_space_uid",
        "tok_idx",
        "s_inputs",
        "s_trunk",
        "z_trunk",
        "relative_position_encoding",
        "asym_id",
        "residue_index",
        "entity_id",
        "token_index",
        "sym_id",
        "token_attention_mask",
    )
    kw = {k: cond.get(k) for k in keys}
    n_atoms = cond["tok_idx"].shape[1]

    def run(model, x, t):
        with torch.no_grad():
            return model(
                x_noisy=x,
                t_hat=t,
                num_diffusion_samples=x.shape[0],
                return_token_repr=False,
                return_atom_repr=False,
                inference_cache=None,
                **kw,
            )["x_denoised"]

    for batch, sigma in ((1, 4.82), (2, 60.0)):
        x = torch.randn(batch, n_atoms, 3, device=dev) * sigma
        t = torch.full((batch,), sigma, device=dev)
        r1, r2 = run(ref, x, t), run(ref, x, t)
        m1 = run(mine, x, t)
        ref_self = (r1 - r2).abs().max().item()
        cross = (r1 - m1).abs().max().item()
        # Allow a small multiple of the reference's own jitter, plus a floor so a
        # coincidentally-deterministic run cannot make this vacuous.
        assert cross <= max(4.0 * ref_self, 1e-3), f"sigma={sigma}: |ref-mine|={cross:.3e} vs |ref-ref|={ref_self:.3e}"
        assert torch.isfinite(m1).all()


@pytest.mark.integration
@pytest.mark.gpu
def test_load_denoiser_accepts_a_training_checkpoint():
    pytest.importorskip("torch")
    from pathlib import Path

    if not Path(PILOT_CKPT).exists():
        pytest.skip(f"no checkpoint at {PILOT_CKPT}")
    from mol_ensemble_gen.model.denoiser import load_denoiser

    model = load_denoiser(PILOT_CKPT, device="cpu")
    assert state_dict_signature(model.state_dict()) == (
        PRETRAINED_NUM_TENSORS,
        PRETRAINED_NUM_PARAMS,
    )


def test_augmentation_is_reproducible_with_a_generator():
    """The rotation/translation draw must honour ``generator``.

    Regression test for a measurement bug, not a modelling one: the augmentation
    used to come off the global RNG even when the caller supplied a seeded
    generator, so two validation passes over the *same weights and same frames*
    disagreed by 2-4% — the same magnitude as the training effects the validation
    number was being used to compare. Anything that reads a loss twice and
    subtracts needs this to hold.
    """
    import torch

    from mol_ensemble_gen.model.denoiser import GeometryOps, augment_with_generator

    geom = GeometryOps()
    x = torch.randn(2, 9, 3, dtype=torch.float32)
    mask = torch.ones(2, 9, dtype=torch.float32)

    def once(seed):
        g = torch.Generator().manual_seed(seed)
        out, _ = augment_with_generator(geom, x.clone(), mask, g)
        return out

    torch.manual_seed(0)
    a = once(1234)
    torch.manual_seed(999)  # perturb the global stream between calls
    b = once(1234)
    assert torch.equal(a, b), "same generator seed must give the same augmentation"

    assert not torch.equal(a, once(4321)), "a different seed must actually change it"


def test_augmentation_tolerates_heads_without_generator_support():
    """The reference ``DiffusionStructureHead`` takes no ``generator`` kwarg.

    Its signature is upstream's to change, so support is probed rather than
    assumed; a head lacking it must still work (unseeded), not raise TypeError.
    """
    import torch

    from mol_ensemble_gen.model.denoiser import GeometryOps, augment_with_generator

    class LegacyHead:
        """Mimics the reference signature: no ``generator`` parameter."""

        def __init__(self):
            self._inner = GeometryOps()
            self.calls = 0

        def _center_random_augmentation(self, x, atom_mask, second_coords=None):
            self.calls += 1
            return self._inner._center_random_augmentation(x, atom_mask, second_coords)

    head = LegacyHead()
    x = torch.randn(1, 5, 3, dtype=torch.float32)
    mask = torch.ones(1, 5, dtype=torch.float32)
    out, _ = augment_with_generator(head, x, mask, torch.Generator().manual_seed(7))
    assert out.shape == x.shape
    assert head.calls == 1
    # probe result is cached, so a second call does not re-inspect the signature
    augment_with_generator(head, x, mask, torch.Generator().manual_seed(7))
    assert head._accepts_augmentation_generator is False
