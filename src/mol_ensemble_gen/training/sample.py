"""Temperature-conditioned sampling from a finetuned ESMFold2 checkpoint.

Loads the full model (the trunk is needed to condition arbitrary input
sequences), swaps in the finetuned ``diffusion_module`` weights, and injects the
learned temperature bias by wrapping ``structure_head.sample`` to add
``temp_embedder(T)`` to its ``s_inputs`` — the same injection point used in
training. The wrapped model is then driven through the existing
:class:`~mol_ensemble_gen.ensemble.ESMFold2Ensemble`, so all CIF/manifest/metadata
output conventions are reused unchanged; one temperature → one ``T<K>/`` subdir.

When the checkpoint was trained with ``scheme=="flow"`` the wrapper additionally
replaces the diffusion integration with :func:`flow_ode_sample` — a deterministic
rectified-flow ODE integrator that reuses the same pretrained EDM denoiser via
``σ = t/(1−t)`` (see DESIGN.md §11). Everything upstream (SPI build, trunk
forward, conditioning) and downstream (CIF decode/manifest) is identical, so both
schemes share one output path; only the sampling inner loop differs.
"""

from __future__ import annotations

from pathlib import Path

# Conditioning kwargs forwarded verbatim from ``head.sample`` to the denoiser
# inside the flow ODE loop (``x_noisy``/``t_hat``/``num_diffusion_samples`` are
# supplied per-step, so they are excluded here).
_DENOISER_KEYS = (
    "ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
    "ref_space_uid", "tok_idx", "s_inputs", "s_trunk", "z_trunk",
    "relative_position_encoding", "asym_id", "residue_index", "entity_id",
    "token_index", "sym_id", "token_attention_mask",
)


def flow_ode_sample(
    head,
    *,
    steps: int = 50,
    sampler: str = "euler",
    sigma_max: float = 256.0,
    t_min: float = 1e-3,
    generator=None,
    **conditioning,
) -> dict:
    """Deterministic rectified-flow ODE sampler over the pretrained EDM denoiser.

    A drop-in replacement for ``structure_head.sample`` that accepts the exact
    same conditioning kwargs and returns the same ``{"sample_atom_coords",
    "diff_token_repr"}`` dict, so the surrounding :class:`ESMFold2Ensemble`
    plumbing is reused unchanged.

    Integrates the probability-flow ODE on a **uniform-in-t** grid from noise
    (``t_max = σ_max/(1+σ_max)``) to data (``t_min``), mapping each time to the
    EDM noise level ``σ = t/(1−t)`` the denoiser expects. Each step calls the
    denoiser for ``x̂₀ = D(x;σ)``, aligns ``x`` onto it (fp32 Kabsch), and takes
    the score step ``x += (σ'−σ)·(x−x̂₀)/σ``. With ``sampler="heun"`` a 2nd-order
    correction re-evaluates the derivative at the endpoint. Deterministic: no
    stochastic churn (``γ=0``), unlike the Karras SDE in ``head.sample``.
    """
    import torch

    s_inputs = conditioning["s_inputs"]
    tok_idx = conditioning["tok_idx"]
    ref_mask = conditioning["ref_mask"]
    device = s_inputs.device
    n_atoms = tok_idx.shape[1]
    num_diffusion_samples = int(conditioning.get("num_diffusion_samples", 1) or 1)
    target_batch = s_inputs.shape[0] * num_diffusion_samples

    # Uniform-in-t grid from t_max (noise) down to t_min (data); σ = t/(1−t).
    t_max = sigma_max / (1.0 + sigma_max)
    ts = torch.linspace(t_max, t_min, steps + 1, device=device, dtype=torch.float32)
    sigmas = ts / (1.0 - ts)                       # σ_k, monotonically decreasing

    kwargs = {k: conditioning.get(k) for k in _DENOISER_KEYS}
    atom_mask = ref_mask.repeat_interleave(num_diffusion_samples, 0).float()

    def _denoise(x, sigma_val):
        out = head.diffusion_module(
            x_noisy=x,
            t_hat=torch.full((target_batch,), sigma_val, device=device, dtype=torch.float32),
            num_diffusion_samples=num_diffusion_samples,
            return_token_repr=True,
            return_atom_repr=False,
            inference_cache=None,
            **kwargs,
        )
        return out["x_denoised"], out["token_repr"]

    x = float(sigmas[0]) * torch.randn(
        target_batch, n_atoms, 3, device=device, dtype=torch.float32, generator=generator
    )
    token_repr = None

    for i in range(steps):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])

        x, _ = head._center_random_augmentation(x, atom_mask, second_coords=None)
        x_denoised, token_repr = _denoise(x, sigma)

        # Align the current coords onto the prediction before the score step
        # (fp32 Kabsch; det/SVD have no bf16 kernel — matches head.sample).
        with torch.autocast(device_type=device.type, enabled=False):
            x = self_align(head, x, x_denoised, atom_mask)
        x = x.to(x_denoised.dtype)

        d = (x - x_denoised) / sigma                 # score direction
        x_euler = x + (sigma_next - sigma) * d

        if sampler == "heun" and sigma_next > 0.0:
            xd_next, _ = _denoise(x_euler, sigma_next)
            with torch.autocast(device_type=device.type, enabled=False):
                x_euler_a = self_align(head, x_euler, xd_next, atom_mask)
            x_euler_a = x_euler_a.to(xd_next.dtype)
            d_next = (x_euler_a - xd_next) / sigma_next
            x = x + (sigma_next - sigma) * 0.5 * (d + d_next)
        else:
            x = x_euler

    return {"sample_atom_coords": x, "diff_token_repr": token_repr}


def self_align(head, x, x_denoised, atom_mask):
    """Kabsch-align ``x`` onto ``x_denoised`` (fp32), returning the moved ``x``."""
    return head._weighted_rigid_align(
        x.float(), x_denoised.float(), atom_mask, atom_mask
    )


def build_temperature_conditioned_model(checkpoint: str | Path, device: str = "cuda"):
    """Load a finetuned model and return ``(model, temp_holder)``.

    Set ``temp_holder["T"]`` to a temperature (K) before folding; the patched
    sampler adds the learned bias for that temperature. ``None`` disables the
    injection (recovering the finetuned-but-unconditioned behavior).
    """
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from .conditioning import build_temperature_embedder
    from .config import TrainConfig, _build

    state = torch.load(checkpoint, map_location=device)
    cfg: TrainConfig = _build(TrainConfig, state.get("config", {}))

    model = ESMFold2Model.from_pretrained(cfg.model.model_name).to(device).eval()

    # A checkpoint trained with native flow-time conditioning carries extra t-head
    # tensors that the reference denoiser has no slots for. Loading it non-strictly
    # would *silently drop the learned conditioning* and sample from a model that
    # is missing part of what was trained — so substitute our denoiser instead,
    # and load strictly into it.
    trained_keys = set(state["diffusion_module"])
    has_t_head = any(k.startswith("conditioning.t_") for k in trained_keys)
    if has_t_head or cfg.model.backend == "ours":
        from ..model.denoiser import DenoiserConfig, DiffusionModule

        t_cond = cfg.flow.t_conditioning if has_t_head else "off"
        module = DiffusionModule(DenoiserConfig(t_conditioning=t_cond)).to(device).eval()
        module.load_state_dict(state["diffusion_module"])  # strict: shapes must match
        model.structure_head.diffusion_module = module
        print(
            f"[sample] using mol_ensemble_gen denoiser "
            f"(t_conditioning={t_cond}, {len(trained_keys)} tensors)",
            flush=True,
        )
    else:
        model.structure_head.diffusion_module.load_state_dict(state["diffusion_module"])

    temp_embedder = build_temperature_embedder(cfg.temperature).to(device).eval()
    temp_embedder.load_state_dict(state["temp_embedder"])

    head = model.structure_head
    original = head.sample
    scheme = cfg.optim.scheme
    flow = cfg.flow
    holder: dict = {"T": None}

    def _patched(*args, **kwargs):
        temp = holder["T"]
        if temp is not None and "s_inputs" in kwargs:
            s = kwargs["s_inputs"]
            bias = temp_embedder(temp).to(s.dtype)          # (1, C)
            tam = kwargs.get("token_attention_mask")
            if tam is not None:
                kwargs["s_inputs"] = s + bias[:, None, :] * tam.to(s.dtype)[..., None]
            else:
                kwargs["s_inputs"] = s + bias[:, None, :]
        if scheme == "flow":
            # Reuse the CLI/ensemble sampling knobs where they map onto the ODE.
            steps = int(kwargs.get("num_sampling_steps") or flow.num_sampling_steps)
            smax = kwargs.get("max_inference_sigma")
            smax = float(smax) if smax is not None else flow.sigma_max
            return flow_ode_sample(
                head, steps=steps, sampler=flow.sampler, sigma_max=smax,
                t_min=flow.t_min, **kwargs,
            )
        return original(*args, **kwargs)

    head.sample = _patched
    return model, holder


def sample_at_temperature(
    checkpoint: str | Path,
    input_path: str | Path,
    temperature: float,
    out_dir: str | Path,
    *,
    members: int = 50,
    base_seed: int = 0,
    device: str = "cuda",
    sampling: dict | None = None,
    model_holder: tuple | None = None,
) -> list:
    """Sample an ensemble at one temperature; write to ``out_dir/T<K>/``.

    Pass ``model_holder`` (from :func:`build_temperature_conditioned_model`) to
    reuse a loaded model across several temperatures without reloading weights.
    """
    from ..ensemble import EnsembleSpec, ESMFold2Ensemble, SamplingParams

    if model_holder is None:
        model_holder = build_temperature_conditioned_model(checkpoint, device=device)
    model, holder = model_holder

    spec = EnsembleSpec(members=members, base_seed=base_seed, sampling=SamplingParams(**(sampling or {})))
    ens = ESMFold2Ensemble(spec, device=device, model=model)
    spi, input_id = ens.build_spi(input_path)

    holder["T"] = float(temperature)
    temp_dir = Path(out_dir) / f"T{int(round(temperature))}"
    members_out = ens.generate(spi, f"{input_id}_T{int(round(temperature))}", temp_dir)
    print(f"[sample] T={temperature:.0f}K: {len(members_out)} structures -> {temp_dir}", flush=True)
    return members_out


def sample_temperatures(
    checkpoint: str | Path,
    input_path: str | Path,
    temperatures: list[float],
    out_dir: str | Path,
    *,
    members: int = 50,
    base_seed: int = 0,
    device: str = "cuda",
    sampling: dict | None = None,
) -> dict[float, list]:
    """Sample ensembles at several temperatures, reloading the model only once."""
    holder = build_temperature_conditioned_model(checkpoint, device=device)
    return {
        t: sample_at_temperature(
            checkpoint, input_path, t, out_dir,
            members=members, base_seed=base_seed, device=device,
            sampling=sampling, model_holder=holder,
        )
        for t in temperatures
    }
