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

from ..model.denoiser import augment_with_generator as _augment
from .config import SIGMA_DATA, TRAIN_NOISE_LOG_MEAN, TRAIN_NOISE_LOG_STD

# Conditioning tensor names forwarded verbatim to the denoiser (s_inputs is
# handled separately because temperature is injected into it).
_PASS_THROUGH = (
    "ref_pos",
    "ref_charge",
    "ref_mask",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
    "tok_idx",
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


def inject_temperature(conditioning: dict, temp_embedder, temperature: float, dtype):
    """Return a temperature-shifted ``s_inputs`` (padding tokens left unchanged).

    ``s_inputs`` is (1, L, C); the embedder yields a (1, C) bias added to every
    real token. At init the embedder outputs zero, so this is a no-op and the
    denoiser reproduces the pretrained model.
    """
    s_inputs = conditioning["s_inputs"].to(dtype)
    bias = temp_embedder(temperature).to(dtype)  # (1, C)
    gain = temp_embedder.scale(temperature).to(dtype)  # (1, C); exactly 1.0 when film is off
    tam = conditioning.get("token_attention_mask")
    if tam is not None:
        keep = tam.to(dtype)[..., None]  # (1, L, 1)
        # Padding tokens keep gain 1 and bias 0, so masking applies to the
        # *deviation* from identity rather than to the gain itself.
        g = 1.0 + (gain[:, None, :] - 1.0) * keep
        return s_inputs * g + bias[:, None, :] * keep
    return s_inputs * gain[:, None, :] + bias[:, None, :]


def _internal_spread(x, mask, n_probe: int = 192):
    """Mean pairwise structural spread of a batch of frames, in Angstrom.

    Measured on **inter-atomic distances** rather than coordinates, so it is
    invariant to rotation and translation by construction — no Kabsch alignment,
    and therefore no SVD in the loss graph. For frames ``i,j`` the distance is
    ``rms(D_i - D_j)`` over a fixed evenly-spaced subset of present atoms, which
    keeps the cost at ``O(n_probe^2)`` regardless of chain length.

    ``x`` is ``(B, N, 3)``, ``mask`` is ``(B, N)``. Returns a scalar tensor.
    """
    import torch

    present = mask[0].nonzero(as_tuple=False).reshape(-1)
    if present.numel() < 3:
        return x.new_zeros(())
    if present.numel() > n_probe:  # evenly spaced, deterministic
        sel = torch.linspace(0, present.numel() - 1, n_probe, device=x.device).long()
        present = present[sel]
    p = x[:, present, :]  # (B, K, 3)
    d = torch.cdist(p, p)  # (B, K, K)
    k = d.shape[-1]
    iu = torch.triu_indices(k, k, offset=1, device=x.device)
    dv = d[:, iu[0], iu[1]]  # (B, K(K-1)/2)
    b = dv.shape[0]
    if b < 2:
        return x.new_zeros(())
    ii, jj = torch.triu_indices(b, b, offset=1, device=x.device)
    return ((dv[ii] - dv[jj]) ** 2).mean(dim=-1).clamp_min(1e-8).sqrt().mean()


def _denoise_and_weighted_mse(
    diffusion_module,
    head,
    temp_embedder,
    conditioning: dict,
    gt_coords,  # (B, n_atoms, 3) float, model atom axis
    atom_mask,  # (n_atoms,) bool — matched model atoms
    temperature: float,
    sigma,  # (B,) per-frame noise level σ
    weight,  # (B,) per-frame loss weight
    generator=None,
    flow_t=None,  # (B,) flow path time, for native-t conditioning
    spread_weight: float = 0.0,
    spread_atoms: int = 192,
    spread_sigma_max: float = 8.0,
):
    """Shared core: noise → denoise → fp32 Kabsch align → weighted coordinate MSE.

    Returns ``(loss, per_frame_mse, extra)``; ``per_frame_mse`` is the unweighted
    per-frame masked MSE and ``extra`` carries the spread diagnostics. Both schemes
    differ *only* in ``sigma`` and ``weight``.

    ``spread_weight > 0`` adds a **spread-matching** term. The coordinate MSE is
    minimised by predicting the conditional *mean*, so it actively rewards
    collapsing the ensemble, and nothing in it refers to temperature at all. The
    measured consequence: the diversity of x̂₀ across noise draws is
    temperature-flat (450K/320K ratio ≈ 1.0 at every σ) where MD needs 4.65x. This
    term compares the spread of the predicted frames against the spread of the
    ground-truth frames in the same micro-batch — which, because a micro-batch is
    B frames of one (domain, temperature), *is* the MD spread at that temperature.
    It is therefore temperature-dependent for free, with no new conditioning.

    Penalising ``log(pred/gt)`` squared makes it scale-free and matches the
    evaluation metric (mean ``|log(model/MD)|``), so training and scoring finally
    optimise the same quantity. Note the term deliberately biases the denoiser away
    from the exact conditional mean: with a perfect velocity field the ODE would
    already transport noise to the right marginal, so this trades theoretical
    exactness for a fix to a measured 2x under-dispersion.
    """
    import torch

    device = gt_coords.device
    b = gt_coords.shape[0]
    x0 = gt_coords.to(torch.float32)  # (B, N, 3)
    mask = atom_mask.to(torch.float32).to(device)[None, :].expand(b, -1)  # (B, N)

    # Center + random rotation/translation augmentation of the ground truth.
    x0, _ = _augment(head, x0, mask, generator)

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
        num_diffusion_samples=b,  # broadcast batch-1 conditioning to B frames
        return_token_repr=False,
        return_atom_repr=False,
        inference_cache=None,
        **kwargs,
    )
    x_denoised = out["x_denoised"].to(torch.float32)  # (B, N, 3)

    # Alignment (Kabsch SVD/det) and the coordinate loss must run in fp32 with
    # autocast off: under bf16 autocast the align's matmuls downcast and
    # torch.linalg.det has no bfloat16 kernel. The caller (trainer) wraps this
    # whole function in autocast, so disable it explicitly here.
    with torch.autocast(device_type=device.type, enabled=False):
        # Align GT onto the prediction (fp32 Kabsch) before the coordinate loss.
        x0_aligned = head._weighted_rigid_align(x0, x_denoised, mask, mask)
        sq_err = ((x_denoised - x0_aligned) ** 2).sum(-1)  # (B, N)
        per_frame = (sq_err * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        loss = (weight.to(torch.float32) * per_frame).mean()

        extra = {}
        # Gate on σ: the "match the MD spread" target is only valid where the
        # posterior is data-dominated (see SpreadConfig.sigma_max).
        in_band = bool(float(sigma.mean()) <= spread_sigma_max)
        if spread_weight > 0.0 and b > 1 and in_band:
            sp_pred = _internal_spread(x_denoised, mask, n_probe=spread_atoms)
            # The target is data, not a prediction — detach so no gradient flows
            # into it (it has no parameters, but keep the graph honest).
            sp_gt = _internal_spread(x0, mask, n_probe=spread_atoms).detach()
            if float(sp_gt) > 1e-6:
                ratio = torch.log(sp_pred.clamp_min(1e-6) / sp_gt)
                loss = loss + spread_weight * ratio.pow(2)
                extra = {
                    "spread_pred": float(sp_pred),
                    "spread_gt": float(sp_gt),
                    "spread_log_ratio": float(ratio),
                    "spread_applied": 1.0,
                }
        elif spread_weight > 0.0:
            extra = {"spread_applied": 0.0}
    return loss, per_frame, extra


def edm_diffusion_loss(
    diffusion_module,
    head,
    temp_embedder,
    conditioning: dict,
    gt_coords,  # (B, n_atoms, 3) float, model atom axis
    atom_mask,  # (n_atoms,) bool — matched model atoms
    temperature: float,
    *,
    sigma_data: float = SIGMA_DATA,
    p_mean: float = TRAIN_NOISE_LOG_MEAN,
    p_std: float = TRAIN_NOISE_LOG_STD,
    generator=None,
    spread_weight: float = 0.0,
    spread_atoms: int = 192,
    spread_sigma_max: float = 8.0,
    shared_sigma: bool = False,
):
    """EDM/Karras denoising loss for one (domain, temperature) micro-batch."""
    import torch

    device = gt_coords.device
    b = gt_coords.shape[0]
    n_draw = 1 if shared_sigma else b
    log_sigma = p_mean + p_std * torch.randn(n_draw, device=device, generator=generator)
    if shared_sigma:
        log_sigma = log_sigma.expand(b)
    sigma = (sigma_data * torch.exp(log_sigma)).to(torch.float32)  # (B,)
    lam = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2  # (B,)

    loss, per_frame, extra = _denoise_and_weighted_mse(
        diffusion_module,
        head,
        temp_embedder,
        conditioning,
        gt_coords,
        atom_mask,
        temperature,
        sigma,
        lam,
        generator=generator,
        spread_weight=spread_weight,
        spread_atoms=spread_atoms,
        spread_sigma_max=spread_sigma_max,
    )
    return loss, {"sigma_mean": float(sigma.mean()), "mse": float(per_frame.mean()), **extra}


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
    spread_weight: float = 0.0,
    spread_atoms: int = 192,
    spread_sigma_max: float = 8.0,
    shared_sigma: bool = False,
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
    # One σ for the whole micro-batch when the spread term is active: with a
    # per-frame σ the frames are denoised from wildly different noise levels, so
    # their mutual spread measures the σ draw rather than the ensemble.
    t, sigma = sample_flow_time(
        1 if shared_sigma else b,
        device=device,
        time_dist=time_dist,
        p_mean=p_mean,
        p_std=p_std,
        sigma_data=sigma_data,
        t_min=t_min,
        generator=generator,
    )
    if shared_sigma:
        t, sigma = t.expand(b), sigma.expand(b)

    if weighting == "data":
        weight = torch.ones_like(t)
    else:  # velocity matching: ‖v_θ − v*‖² = ‖x̂₀ − x0‖² / t²
        weight = 1.0 / (t * t)

    loss, per_frame, extra = _denoise_and_weighted_mse(
        diffusion_module,
        head,
        temp_embedder,
        conditioning,
        gt_coords,
        atom_mask,
        temperature,
        sigma,
        weight,
        generator=generator,
        flow_t=t,
        spread_weight=spread_weight,
        spread_atoms=spread_atoms,
        spread_sigma_max=spread_sigma_max,
    )
    return loss, {
        "sigma_mean": float(sigma.mean()),
        "t_mean": float(t.mean()),
        "mse": float(per_frame.mean()),
        **extra,
    }


def diffusion_loss(scheme: str, *args, flow=None, spread=None, **kwargs):
    """Dispatch to the EDM or flow-matching loss by ``scheme`` ("edm" | "flow").

    ``flow`` is a :class:`~.config.FlowConfig` supplying the flow-matching
    hyper-parameters; ignored for the EDM scheme. ``spread`` is a
    :class:`~.config.SpreadConfig`; ``shared_sigma`` is forwarded **only** when the
    weight is positive, so a default config reproduces the previous behaviour
    exactly rather than silently changing the σ draw.
    """
    if spread is not None and spread.weight > 0.0:
        kwargs.setdefault("spread_weight", spread.weight)
        kwargs.setdefault("spread_atoms", spread.atoms)
        kwargs.setdefault("spread_sigma_max", spread.sigma_max)
        kwargs.setdefault("shared_sigma", spread.shared_sigma)
    if scheme == "edm":
        return edm_diffusion_loss(*args, **kwargs)
    if scheme == "flow":
        fkw = {}
        if flow is not None:
            fkw = dict(
                p_mean=flow.p_mean,
                p_std=flow.p_std,
                time_dist=flow.time_dist,
                weighting=flow.weighting,
                t_min=flow.t_min,
            )
        fkw.update(kwargs)
        return flow_matching_loss(*args, **fkw)
    raise ValueError(f"unknown scheme {scheme!r} (expected 'edm' or 'flow')")
