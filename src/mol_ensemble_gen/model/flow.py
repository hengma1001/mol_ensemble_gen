"""Rectified flow matching as a first-class interface on our denoiser.

The pretrained network is an EDM x₀-predictor ``D(x; σ)``. Flow matching lives on
the linear path

    x_t = (1 − t)·x₀ + t·ε ,   ε ~ N(0, I) ,   t: 0 = data → 1 = noise

and wants a *velocity* ``v(x_t, t) = dx_t/dt = ε − x₀``. The two are the same model
under a change of variables, which is what lets flow reuse the pretrained weights:

    x_t / (1 − t) = x₀ + σ·ε      with   σ = t / (1 − t)

so ``x̂₀ = D(x_t/(1−t); σ)`` and, substituting,

    v̂(x_t, t) = (x_t − x̂₀) / t

(the ``1/t`` is why velocity matching is a ``1/t²``-weighted x₀ loss — see
:mod:`mol_ensemble_gen.training.loss`).

This module makes that native rather than implicit:

* :func:`sample_flow_time` — the **single source of truth** for drawing ``t``,
  including the ``ln σ_d`` shift that aligns flow's noise band with the EDM
  branch's. Getting this wrong once already cost a pilot run (see DESIGN.md §11).
* :class:`FlowDenoiser` — ``velocity()`` / ``predict_x0()`` on top of any
  :class:`~mol_ensemble_gen.model.denoiser.DiffusionModule`, doing the ``(1−t)``
  rescaling that the σ-space caller would otherwise have to remember.
* :func:`flow_ode_sample` — the **one** probability-flow ODE integrator in the
  package (``training.sample`` re-exports this exact object). It steps the **EDM**
  variable on a Karras ρ-schedule; see the function docstring for why stepping
  ``x_t`` directly diverges, and why the σ spacing matters.

Note the scaling subtlety, which is easy to get wrong: the denoiser must be fed
``x_t/(1−t)``, not ``x_t``. An integrator that works in EDM space (where the state
*is* ``x₀ + σε``) skips that division legitimately; one that works in flow space
must not — and must also not rigid-align a noise-scale state onto an Ångström-scale
prediction, which is what broke the first attempt.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .denoiser import GeometryOps  # noqa: F401  (re-exported for callers)
from .denoiser import augment_with_generator as _augment

#: EDM/Karras constants of the pretrained model, duplicated here so this module
#: does not import the training package (which depends on it).
SIGMA_DATA = 16.0
TRAIN_NOISE_LOG_MEAN = -1.2
TRAIN_NOISE_LOG_STD = 1.5


# ---------------------------------------------------------------------------
# time <-> noise level
# ---------------------------------------------------------------------------


def t_to_sigma(t: Tensor | float) -> Tensor | float:
    """``σ = t/(1−t)`` — flow time to the EDM noise level the denoiser consumes."""
    return t / (1.0 - t)


def sigma_to_t(sigma: Tensor | float) -> Tensor | float:
    """``t = σ/(1+σ)`` — the inverse of :func:`t_to_sigma`."""
    return sigma / (1.0 + sigma)


def sample_flow_time(
    batch: int,
    *,
    device: torch.device | str = "cpu",
    time_dist: str = "logitnormal",
    p_mean: float = TRAIN_NOISE_LOG_MEAN,
    p_std: float = TRAIN_NOISE_LOG_STD,
    sigma_data: float = SIGMA_DATA,
    t_min: float = 1e-3,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Draw ``(t, σ)`` for a batch of frames. Returns both, already clamped.

    ``time_dist="logitnormal"`` reproduces the EDM branch's noise band exactly:
    EDM draws ``σ = σ_d·exp(p)`` with ``p ~ N(p_mean, p_std)``, so the matching
    flow time is ``t = σ/(1+σ) = sigmoid(ln σ_d + p)``. **The ``ln σ_d`` shift is
    load-bearing** — dropping it trains at ``1/σ_d`` of the intended noise level.

    ``time_dist="uniform"`` draws ``t ~ U(0,1)``, which is uniform in *path time*
    and deliberately not EDM-matched (median σ = 1).
    """
    if time_dist == "uniform":
        t = torch.rand(batch, device=device, generator=generator)
    elif time_dist == "logitnormal":
        log_sigma = math.log(sigma_data) + p_mean + p_std * torch.randn(batch, device=device, generator=generator)
        t = torch.sigmoid(log_sigma)
    else:
        raise ValueError(f"unknown time_dist {time_dist!r} (expected 'logitnormal' or 'uniform')")

    t = t.clamp(min=t_min, max=1.0 - t_min).to(torch.float32)
    return t, t_to_sigma(t)


def velocity_from_x0(x_t: Tensor, x0_hat: Tensor, t: Tensor) -> Tensor:
    """``v̂ = (x_t − x̂₀)/t`` — the flow velocity implied by an x₀ prediction."""
    return (x_t - x0_hat) / t.reshape(-1, *([1] * (x_t.dim() - 1)))


def x0_from_velocity(x_t: Tensor, v: Tensor, t: Tensor) -> Tensor:
    """``x̂₀ = x_t − t·v̂`` — the inverse of :func:`velocity_from_x0`."""
    return x_t - t.reshape(-1, *([1] * (x_t.dim() - 1))) * v


# ---------------------------------------------------------------------------
# flow-native wrapper
# ---------------------------------------------------------------------------


class FlowDenoiser:
    """Velocity-prediction view of an EDM x₀-predicting :class:`DiffusionModule`.

    Not an ``nn.Module`` — it holds no parameters of its own and deliberately does
    not appear in any ``state_dict``. It exists so callers can think in ``(x_t, t)``
    and get ``v̂`` back, with the ``(1−t)`` input rescaling handled in one place.
    """

    def __init__(self, module, *, sigma_data: float | None = None) -> None:
        self.module = module
        self.sigma_data = float(sigma_data if sigma_data is not None else module.sigma_data)
        # The reference DiffusionModule has no native flow-time input; passing it
        # would be a TypeError, so probe once rather than per call.
        self.supports_flow_time = bool(getattr(module, "supports_flow_time", False))

    def predict_x0(self, x_t: Tensor, t: Tensor, **conditioning) -> Tensor:
        """``x̂₀ = D(x_t/(1−t); t/(1−t))`` — note the ``(1−t)`` rescaling."""
        t = t.to(torch.float32).reshape(-1)
        shape = (-1, *([1] * (x_t.dim() - 1)))
        sigma = t_to_sigma(t)
        x_edm = x_t / (1.0 - t).reshape(*shape)
        if self.supports_flow_time:
            conditioning = {**conditioning, "flow_t": t}
        out = self.module(
            x_noisy=x_edm,
            t_hat=sigma,
            sigma_data=self.sigma_data,
            **conditioning,
        )
        return out["x_denoised"]

    def velocity(self, x_t: Tensor, t: Tensor, **conditioning) -> Tensor:
        """Flow velocity ``v̂(x_t, t)`` at the given path time."""
        t = t.to(torch.float32).reshape(-1)
        return velocity_from_x0(x_t, self.predict_x0(x_t, t, **conditioning), t)


# ---------------------------------------------------------------------------
# deterministic probability-flow ODE sampler
# ---------------------------------------------------------------------------

#: Conditioning kwargs forwarded verbatim to the denoiser inside the ODE loop.
#: ``x_noisy``/``t_hat``/``num_diffusion_samples`` are supplied per step.
_DENOISER_KEYS = (
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


@torch.no_grad()
def flow_ode_sample(
    denoiser,
    geometry,
    *,
    steps: int = 50,
    sampler: str = "euler",
    sigma_max: float = 256.0,
    t_min: float = 1e-3,
    schedule: str = "karras",
    rho: float = 7.0,
    generator=None,
    **conditioning,
) -> dict:
    """Deterministic probability-flow ODE sampler over the pretrained EDM denoiser.

    A drop-in replacement for ``structure_head.sample``: it takes the same
    conditioning kwargs and returns the same ``{"sample_atom_coords",
    "diff_token_repr"}`` dict, so the surrounding ensemble plumbing is unchanged.

    The integration variable is the **EDM coordinate** ``x = x₀ + σ·ε`` with
    ``σ = t/(1−t)``, stepping
    ``x += (σ' − σ)·(x − x̂₀)/σ``. Both parameterizations describe the same
    continuous ODE, but stepping in EDM space is the one that is numerically safe:

    > Integrating in *flow* space (state ``x_t = (1−t)x₀ + tε``) puts the state at
    > noise scale (‖x‖ ≈ 1) while ``x̂₀`` is at Ångström scale (‖x̂₀‖ ≈ 25). The
    > per-step rigid align then translates the state onto ``x̂₀``'s centroid — a
    > shift several times larger than the state itself — which feeds back and
    > diverges by roughly 10× per step. Measured, not hypothetical. In EDM space
    > the state and ``x̂₀`` share a scale, so the align is meaningful.

    ``sampler="heun"`` adds a second-order correction at the cost of one extra
    denoiser call per step. There is no stochastic churn (γ=0), unlike the Karras
    SDE in ``head.sample``.

    Parameters
    ----------
    denoiser:
        The diffusion module — anything with the ``DiffusionModule`` forward.
    geometry:
        Supplies ``_center_random_augmentation`` and ``_weighted_rigid_align``;
        either :class:`~mol_ensemble_gen.model.denoiser.GeometryOps` or the
        reference ``structure_head``.
    """
    s_inputs = conditioning["s_inputs"]
    tok_idx = conditioning["tok_idx"]
    ref_mask = conditioning["ref_mask"]
    device = s_inputs.device
    n_atoms = tok_idx.shape[1]
    num_diffusion_samples = int(conditioning.get("num_diffusion_samples", 1) or 1)
    target_batch = s_inputs.shape[0] * num_diffusion_samples

    if sampler not in ("euler", "heun"):
        raise ValueError(f"unknown sampler {sampler!r} (expected 'euler' or 'heun')")

    # The σ grid. Both options span the same endpoints; they differ in how the
    # steps are distributed, and that distribution matters a great deal.
    #
    # "uniform_t" walks t linearly, which crushes the whole high-σ range into the
    # first step: with sigma_max=256 and 50 steps it goes 256 → 41 → 22 → 15,
    # so exactly one step lands above the training p95 (σ≈57). Since the early
    # high-σ steps are what select the global mode, that decision is made by a
    # single enormous Euler step evaluated where the model was trained on 0.4% of
    # its draws.
    #
    # "karras" (default) is the EDM ρ-schedule, roughly geometric in σ, which
    # spends a proper fraction of the steps across the high-σ decade.
    sigma_min = t_min / (1.0 - t_min)
    if schedule == "karras":
        i = torch.arange(steps + 1, device=device, dtype=torch.float32) / max(1, steps)
        sigmas = (sigma_max ** (1.0 / rho) + i * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    elif schedule == "uniform_t":
        t_max = sigma_max / (1.0 + sigma_max)
        ts = torch.linspace(t_max, t_min, steps + 1, device=device, dtype=torch.float32)
        sigmas = ts / (1.0 - ts)
    else:
        raise ValueError(f"unknown schedule {schedule!r} (expected 'karras' or 'uniform_t')")
    ts = sigmas / (1.0 + sigmas)  # flow times, for native-t conditioning

    kwargs = {k: conditioning.get(k) for k in _DENOISER_KEYS}
    supports_flow_t = bool(getattr(denoiser, "supports_flow_time", False))
    atom_mask = ref_mask.repeat_interleave(num_diffusion_samples, 0).float()

    def _denoise(x, sigma_val, t_val):
        extra = {}
        if supports_flow_t:
            extra["flow_t"] = torch.full((target_batch,), t_val, device=device, dtype=torch.float32)
        out = denoiser(
            x_noisy=x,
            t_hat=torch.full((target_batch,), sigma_val, device=device, dtype=torch.float32),
            num_diffusion_samples=num_diffusion_samples,
            return_token_repr=True,
            return_atom_repr=False,
            inference_cache=None,
            **kwargs,
            **extra,
        )
        return out["x_denoised"], out["token_repr"]

    def _align(x, target):
        # SVD/det have no bf16 kernel, so force fp32 with autocast off.
        with torch.autocast(device_type=device.type, enabled=False):
            return geometry._weighted_rigid_align(x.float(), target.float(), atom_mask, atom_mask)

    x = float(sigmas[0]) * torch.randn(
        target_batch, n_atoms, 3, device=device, dtype=torch.float32, generator=generator
    )
    token_repr = None

    for i in range(steps):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        t_now, t_next = float(ts[i]), float(ts[i + 1])

        # Seeded like the initial noise above: without the generator here the
        # per-step augmentation comes off the global RNG, so ``generator`` does not
        # actually pin the trajectory and a member is not reproducible from its
        # seed. The sampled *distribution* is unaffected either way.
        x, _ = _augment(geometry, x, atom_mask, generator)
        x_denoised, token_repr = _denoise(x, sigma, t_now)
        x = _align(x, x_denoised).to(x_denoised.dtype)

        d = (x - x_denoised) / sigma  # score direction
        x_euler = x + (sigma_next - sigma) * d

        if sampler == "heun" and sigma_next > 0.0:
            xd_next, _ = _denoise(x_euler, sigma_next, t_next)
            x_euler_a = _align(x_euler, xd_next).to(xd_next.dtype)
            d_next = (x_euler_a - xd_next) / sigma_next
            x = x + (sigma_next - sigma) * 0.5 * (d + d_next)
        else:
            x = x_euler

    return {"sample_atom_coords": x, "diff_token_repr": token_repr}
