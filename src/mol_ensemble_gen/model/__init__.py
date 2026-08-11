"""Our own implementation of the ESMFold2 pieces this package trains and samples.

:mod:`~mol_ensemble_gen.model.denoiser` reimplements ESMFold2's diffusion
denoiser (``structure_head.diffusion_module``) and the two geometry helpers the
loss and samplers use, weight-compatibly with the pretrained checkpoint. Because
trunk conditioning is cached offline, this is the whole surface needed to train
and to sample from a cached domain — with no ``transformers``/``esm`` import.

Imports are lazy so that ``import mol_ensemble_gen`` stays torch-free.
"""

from __future__ import annotations

_DENOISER_EXPORTS = (
    "DenoiserConfig",
    "DiffusionModule",
    "GeometryOps",
    "load_denoiser",
    "state_dict_signature",
    "PRETRAINED_NUM_PARAMS",
    "PRETRAINED_NUM_TENSORS",
)

_FLOW_EXPORTS = (
    "FlowDenoiser",
    "flow_ode_sample",
    "sample_flow_time",
    "t_to_sigma",
    "sigma_to_t",
    "velocity_from_x0",
    "x0_from_velocity",
)

__all__ = [*_DENOISER_EXPORTS, *_FLOW_EXPORTS]


def __getattr__(name: str):
    if name in _DENOISER_EXPORTS:
        from . import denoiser

        return getattr(denoiser, name)
    if name in _FLOW_EXPORTS:
        from . import flow

        return getattr(flow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
