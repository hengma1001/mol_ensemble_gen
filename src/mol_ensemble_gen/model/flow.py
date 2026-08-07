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
* :func:`flow_ode_sample` — a deterministic probability-flow ODE integrator that
  steps **in flow space** (``x_t``, uniform in ``t``), not in EDM σ-space.

Note the scaling subtlety, which is easy to get wrong: the denoiser must be fed
``x_t/(1−t)``, not ``x_t``. An integrator that works in EDM space (where the state
*is* ``x₀ + σε``) skips that division legitimately; one that works in flow space
must not.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .denoiser import GeometryOps

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
        log_sigma = (
            math.log(sigma_data)
            + p_mean
            + p_std * torch.randn(batch, device=device, generator=generator)
        )
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
# deterministic probability-flow ODE sampler (in flow space)
# ---------------------------------------------------------------------------


@torch.no_grad()
def flow_ode_sample(
    module,
    conditioning: dict,
    *,
    n_atoms: int,
    batch: int = 1,
    steps: int = 50,
    sampler: str = "euler",
    sigma_max: float = 256.0,
    t_min: float = 1e-3,
    atom_mask: Tensor | None = None,
    align_each_step: bool = True,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cuda",
    geometry: GeometryOps | None = None,
) -> Tensor:
    """Integrate ``dx/dt = v̂(x, t)`` from noise to data. Deterministic.

    Steps on a **uniform-in-t** grid from ``t_max = σ_max/(1+σ_max)`` down to
    ``t_min``, in flow space: the state is ``x_t``, initialized at ``t_max`` from
    ``t_max·ε`` (at ``t = 1`` the path is exactly ``ε``). ``sampler="heun"`` adds a
    second-order correction, costing one extra denoiser call per step.

    There is no stochastic churn — this is the probability-flow ODE, not the Karras
    SDE that ``structure_head.sample`` runs. Returns ``x₀`` of shape
    ``(batch, n_atoms, 3)``.
    """
    geometry = geometry or GeometryOps()
    flow = FlowDenoiser(module)

    t_max = sigma_max / (1.0 + sigma_max)
    ts = torch.linspace(t_max, t_min, steps + 1, device=device, dtype=torch.float32)

    if atom_mask is None:
        atom_mask = torch.ones(batch, n_atoms, device=device, dtype=torch.float32)
    elif atom_mask.dim() == 1:
        atom_mask = atom_mask.to(device).float().unsqueeze(0).expand(batch, -1)
    else:
        atom_mask = atom_mask.to(device).float()

    # At t = t_max the path is x = t_max·ε (the (1-t)·x0 term has all but vanished).
    x = float(t_max) * torch.randn(
        batch, n_atoms, 3, device=device, dtype=torch.float32, generator=generator
    )

    def _velocity(state: Tensor, t_scalar: float) -> Tensor:
        t = torch.full((batch,), t_scalar, device=device, dtype=torch.float32)
        return flow.velocity(state, t, num_diffusion_samples=batch, **conditioning)

    for i in range(steps):
        t_now = float(ts[i])
        t_next = float(ts[i + 1])
        dt = t_next - t_now  # negative: integrating toward data

        v = _velocity(x, t_now)
        if align_each_step:
            # Remove accumulated rigid drift against the current x0 estimate. The
            # Kabsch SVD has no bf16 kernel, so force fp32 with autocast off.
            x0_hat = x0_from_velocity(
                x, v, torch.full((batch,), t_now, device=device, dtype=torch.float32)
            )
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                x = geometry._weighted_rigid_align(
                    x.float(), x0_hat.float(), atom_mask, atom_mask
                )
            v = _velocity(x, t_now)

        if sampler == "heun" and i < steps - 1:
            x_euler = x + dt * v
            v_next = _velocity(x_euler, t_next)
            x = x + dt * 0.5 * (v + v_next)
        elif sampler in ("euler", "heun"):
            x = x + dt * v
        else:
            raise ValueError(f"unknown sampler {sampler!r} (expected 'euler' or 'heun')")

    # The final state sits at t_min, not 0; one last x0 read-out lands on the data
    # manifold instead of leaving a t_min·ε residual.
    v = _velocity(x, float(ts[-1]))
    return x0_from_velocity(
        x, v, torch.full((batch,), float(ts[-1]), device=device, dtype=torch.float32)
    )
