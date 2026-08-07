"""Denoising losses for finetuning the ESMFold2 diffusion module.

Two schemes share one implementation because the pretrained ``DiffusionModule``
is an EDM x₀-predictor ``D(x;σ)`` (all preconditioning built in): both draw a
per-frame noise level ``σ``, form ``x_noisy = x0 + σ·ε``, denoise, align, and take
a weighted coordinate MSE. They differ *only* in how ``σ`` is sampled and how each
frame is weighted:

- **edm** (Karras): ``σ = σ_d·exp(p_mean + p_std·N(0,1))``; weight
  ``λ(σ) = (σ² + σ_d²)/(σ·σ_d)²``.
- **flow** (rectified flow): the linear path ``x_t = (1−t)x0 + t·ε`` reparameterizes
  to the same EDM input via ``σ = t/(1−t)`` (so ``x_t/(1−t) = x0 + σ·ε``). Because
  ``x0`` is at raw coordinate scale, that ``σ`` is a *raw* noise level — the same
  units the denoiser consumes — so matching the EDM branch's coverage means drawing
  ``ln σ = ln σ_d + p``, ``p ~ N(p_mean,p_std)``, i.e. logit-normal
  ``t = sigmoid(ln σ_d + p)``. **The ``ln σ_d`` shift is load-bearing**: without it
  the denoiser trains at ``1/σ_d`` (1/16) of the intended noise level, disjoint from
  where the ODE sampler starts. Velocity matching reduces to a ``1/t²``-weighted
  coordinate MSE (``‖v_θ − v*‖² = ‖x̂₀ − x0‖²/t²``); ``"data"`` weighting uses 1.

One micro-batch is ``B`` frames of a single (domain, temperature): they share all
conditioning, so batch-1 cached tensors are broadcast to ``B`` via the denoiser's
``num_diffusion_samples`` (``repeat_interleave``) — no duplication of ``z_trunk``.
The whole coordinate math runs fp32; only matched model atoms contribute
(``atom_mask``), never padding.
"""

from __future__ import annotations

import math

from .config import SIGMA_DATA, TRAIN_NOISE_LOG_MEAN, TRAIN_NOISE_LOG_STD

# Conditioning tensor names forwarded verbatim to the denoiser (s_inputs is
# handled separately because temperature is injected into it).
_PASS_THROUGH = (
    "ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
    "ref_space_uid", "tok_idx", "s_trunk", "z_trunk", "relative_position_encoding",
    "asym_id", "residue_index", "entity_id", "token_index", "sym_id",
    "token_attention_mask",
)


def inject_temperature(conditioning: dict, temp_embedder, temperature: float, dtype):
    """Return a temperature-shifted ``s_inputs`` (padding tokens left unchanged).

    ``s_inputs`` is (1, L, C); the embedder yields a (1, C) bias added to every
    real token. At init the embedder outputs zero, so this is a no-op and the
    denoiser reproduces the pretrained model.
    """
    s_inputs = conditioning["s_inputs"].to(dtype)
    bias = temp_embedder(temperature).to(dtype)          # (1, C)
    tam = conditioning.get("token_attention_mask")
    if tam is not None:
        keep = tam.to(dtype)[..., None]                  # (1, L, 1)
        return s_inputs + bias[:, None, :] * keep
    return s_inputs + bias[:, None, :]


def _denoise_and_weighted_mse(
    diffusion_module,
    head,
    temp_embedder,
    conditioning: dict,
    gt_coords,           # (B, n_atoms, 3) float, model atom axis
    atom_mask,           # (n_atoms,) bool — matched model atoms
    temperature: float,
    sigma,               # (B,) per-frame noise level σ
    weight,              # (B,) per-frame loss weight
    generator=None,
    flow_t=None,         # (B,) flow path time, for native-t conditioning
):
    """Shared core: noise → denoise → fp32 Kabsch align → weighted coordinate MSE.

    Returns ``(loss, per_frame_mse)`` where ``per_frame_mse`` is the unweighted
    per-frame masked MSE; both schemes only differ in ``sigma`` and ``weight``.
    """
    import torch

    device = gt_coords.device
    b = gt_coords.shape[0]
    x0 = gt_coords.to(torch.float32)                             # (B, N, 3)
    mask = atom_mask.to(torch.float32).to(device)[None, :].expand(b, -1)  # (B, N)

    # Center + random rotation/translation augmentation of the ground truth.
    x0, _ = head._center_random_augmentation(x0, mask, second_coords=None)

    sigma = sigma.to(torch.float32)
    eps = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    x_noisy = x0 + sigma[:, None, None] * eps

    s_inputs = inject_temperature(conditioning, temp_embedder, temperature, x_noisy.dtype)
    # Forward every conditioning arg verbatim, keeping None values (e.g. s_trunk,
    # which is None for ESMFold2 but a required positional of the denoiser).
    kwargs = {k: conditioning.get(k) for k in _PASS_THROUGH}

    # Only our denoiser accepts the native flow-time input; the reference one
    # would raise on the unexpected kwarg.
    if flow_t is not None and getattr(diffusion_module, "supports_flow_time", False):
        kwargs["flow_t"] = flow_t

    out = diffusion_module(
        x_noisy=x_noisy,
        t_hat=sigma,
        s_inputs=s_inputs,
        num_diffusion_samples=b,      # broadcast batch-1 conditioning to B frames
        return_token_repr=False,
        return_atom_repr=False,
        inference_cache=None,
        **kwargs,
    )
    x_denoised = out["x_denoised"].to(torch.float32)               # (B, N, 3)

    # Alignment (Kabsch SVD/det) and the coordinate loss must run in fp32 with
    # autocast off: under bf16 autocast the align's matmuls downcast and
    # torch.linalg.det has no bfloat16 kernel. The caller (trainer) wraps this
    # whole function in autocast, so disable it explicitly here.
    with torch.autocast(device_type=device.type, enabled=False):
        # Align GT onto the prediction (fp32 Kabsch) before the coordinate loss.
        x0_aligned = head._weighted_rigid_align(x0, x_denoised, mask, mask)
        sq_err = ((x_denoised - x0_aligned) ** 2).sum(-1)             # (B, N)
        per_frame = (sq_err * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        loss = (weight.to(torch.float32) * per_frame).mean()
    return loss, per_frame


def edm_diffusion_loss(
    diffusion_module,
    head,
    temp_embedder,
    conditioning: dict,
    gt_coords,           # (B, n_atoms, 3) float, model atom axis
    atom_mask,           # (n_atoms,) bool — matched model atoms
    temperature: float,
    *,
    sigma_data: float = SIGMA_DATA,
    p_mean: float = TRAIN_NOISE_LOG_MEAN,
    p_std: float = TRAIN_NOISE_LOG_STD,
    generator=None,
):
    """EDM/Karras denoising loss for one (domain, temperature) micro-batch."""
    import torch

    device = gt_coords.device
    b = gt_coords.shape[0]
    log_sigma = p_mean + p_std * torch.randn(b, device=device, generator=generator)
    sigma = (sigma_data * torch.exp(log_sigma)).to(torch.float32)   # (B,)
    lam = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2    # (B,)

    loss, per_frame = _denoise_and_weighted_mse(
        diffusion_module, head, temp_embedder, conditioning,
        gt_coords, atom_mask, temperature, sigma, lam, generator=generator,
    )
    return loss, {"sigma_mean": float(sigma.mean()), "mse": float(per_frame.mean())}


def flow_matching_loss(
    diffusion_module,
    head,
    temp_embedder,
    conditioning: dict,
    gt_coords,
    atom_mask,
    temperature: float,
    *,
    sigma_data: float = SIGMA_DATA,
    p_mean: float = TRAIN_NOISE_LOG_MEAN,
    p_std: float = TRAIN_NOISE_LOG_STD,
    time_dist: str = "logitnormal",
    weighting: str = "velocity",
    t_min: float = 1e-3,
    generator=None,
):
    """Rectified-flow (flow-matching) loss, reusing the EDM denoiser via σ=t/(1−t).

    ``t`` is the flow time (0=data, 1=noise) and ``σ=t/(1−t)`` the equivalent EDM
    noise level, so ``x_noisy = x0 + σ·ε`` is built exactly as in the EDM branch.
    Velocity matching weights the aligned coordinate MSE by ``1/t²``.

    With ``time_dist="logitnormal"`` the time is drawn so that ``σ`` matches the EDM
    branch exactly: ``t = sigmoid(ln σ_d + p_mean + p_std·N(0,1))``. ``sigma_data``
    therefore *is* used here — it sets the centre of the noise band, and omitting it
    silently shifts training down by a factor of ``σ_d``.
    """
    import torch

    from ..model.flow import sample_flow_time

    device = gt_coords.device
    b = gt_coords.shape[0]
    # Single source of truth for the t draw (including the load-bearing ln σ_d
    # shift) — shared with the sampler so training and sampling cannot diverge.
    t, sigma = sample_flow_time(
        b,
        device=device,
        time_dist=time_dist,
        p_mean=p_mean,
        p_std=p_std,
        sigma_data=sigma_data,
        t_min=t_min,
        generator=generator,
    )

    if weighting == "data":
        weight = torch.ones_like(t)
    else:  # velocity matching: ‖v_θ − v*‖² = ‖x̂₀ − x0‖² / t²
        weight = 1.0 / (t * t)

    loss, per_frame = _denoise_and_weighted_mse(
        diffusion_module, head, temp_embedder, conditioning,
        gt_coords, atom_mask, temperature, sigma, weight,
        generator=generator, flow_t=t,
    )
    return loss, {
        "sigma_mean": float(sigma.mean()),
        "t_mean": float(t.mean()),
        "mse": float(per_frame.mean()),
    }


def diffusion_loss(scheme: str, *args, flow=None, **kwargs):
    """Dispatch to the EDM or flow-matching loss by ``scheme`` ("edm" | "flow").

    ``flow`` is a :class:`~.config.FlowConfig` supplying the flow-matching
    hyper-parameters; ignored for the EDM scheme.
    """
    if scheme == "edm":
        return edm_diffusion_loss(*args, **kwargs)
    if scheme == "flow":
        fkw = {}
        if flow is not None:
            fkw = dict(
                p_mean=flow.p_mean, p_std=flow.p_std,
                time_dist=flow.time_dist, weighting=flow.weighting,
                t_min=flow.t_min,
            )
        fkw.update(kwargs)
        return flow_matching_loss(*args, **fkw)
    raise ValueError(f"unknown scheme {scheme!r} (expected 'edm' or 'flow')")
