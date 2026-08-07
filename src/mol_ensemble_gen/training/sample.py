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

# The ODE sampler lives in mol_ensemble_gen.model.flow — one implementation,
# re-exported here because this module is where callers expect to find it.
from ..model.flow import flow_ode_sample  # noqa: E402,F401


def plan_denoiser(trained_keys, backend: str, t_conditioning: str) -> tuple[bool, str]:
    """Decide which denoiser a checkpoint must be loaded into.

    Returns ``(use_our_denoiser, t_conditioning_to_build)``.

    Pure, so the decision is unit-testable without loading a 1.5 GB model — this is
    the logic that silently broke ``sample-md`` for every native-``t`` checkpoint.
    A checkpoint carrying ``conditioning.t_*`` tensors has no slots for them in the
    reference denoiser, and loading it non-strictly there would drop the learned
    conditioning without a word, so those checkpoints *must* use ours.
    """
    has_t_head = any(str(k).startswith("conditioning.t_") for k in trained_keys)
    if has_t_head:
        return True, t_conditioning if t_conditioning != "off" else "add"
    return backend == "ours", "off"


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
    use_ours, t_cond = plan_denoiser(trained_keys, cfg.model.backend, cfg.flow.t_conditioning)
    if use_ours:
        from ..model.denoiser import DenoiserConfig, DiffusionModule

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
            # head supplies the geometry helpers; head.diffusion_module the network.
            return flow_ode_sample(
                head.diffusion_module, head, steps=steps, sampler=flow.sampler,
                sigma_max=smax, t_min=flow.t_min, **kwargs,
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
