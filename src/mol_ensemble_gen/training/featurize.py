"""Offline per-domain featurization: cache the frozen trunk's conditioning.

Every tensor the denoiser consumes is a pure function of sequence and the
idealized reference conformer — independent of the diffusion coordinate and of
temperature. So we run the real ESMFold2 forward **once per domain** and capture
exactly the kwargs it passes to ``structure_head.sample`` (byte-identical to what
the denoiser sees at inference), then reuse them for every frame and temperature.

Capture is done by monkeypatching ``structure_head.sample`` to record its bound
arguments and abort before any diffusion step — so the cost is a single trunk
forward, and there is zero risk of drifting from the model's own conditioning by
re-implementing the trunk.

Alongside the conditioning we store the mdCATH→model :class:`~.atom_map.AtomMap`,
remapped into the model's *padded* atom axis (using the captured ``ref_mask`` to
locate real-atom positions), so the streaming dataset scatters straight into the
denoiser's coordinate layout with no layout assumptions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .atom_map import AtomMap, build_atom_map
from .mdcath import read_topology

# Float conditioning tensors stored as fp16 to save disk; ints/bools kept native.
_FLOAT_KEYS = {"ref_pos", "s_inputs", "z_trunk", "relative_position_encoding"}

# The exact kwargs structure_head.sample forwards to the denoiser (all
# sequence-derived, x/t/T-independent). s_trunk is None for ESMFold2.
_CONDITIONING_KEYS = (
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


class _CaptureDone(Exception):
    """Raised inside the patched sample() to abort once conditioning is captured."""


def domain_cache_dir(cache_dir: str | Path, domain: str) -> Path:
    return Path(cache_dir) / domain


def is_featurized(cache_dir: str | Path, domain: str) -> bool:
    d = domain_cache_dir(cache_dir, domain)
    return (d / "conditioning.pt").exists() and (d / "atom_map.npz").exists()


def _protein_spi(sequence: str):
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    return StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])


def _capture_conditioning(model, builder, spi, num_loops: int) -> dict:
    """Run one forward and capture the kwargs passed to ``structure_head.sample``."""
    import inspect

    head = model.structure_head
    original = head.sample
    captured: dict = {}

    def _spy(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        captured.update(bound.arguments)
        raise _CaptureDone()

    head.sample = _spy
    try:
        builder.fold(model, spi, seed=0, num_loops=num_loops, num_sampling_steps=1, num_diffusion_samples=1)
    except _CaptureDone:
        pass
    except Exception:
        if not captured:
            raise
    finally:
        head.sample = original
    if not captured:
        raise RuntimeError("failed to capture structure_head.sample conditioning")
    return captured


def _to_cache_tensors(captured: dict) -> dict:
    """Extract + downcast the conditioning tensors for on-disk storage."""
    import torch

    out: dict = {}
    for key in _CONDITIONING_KEYS:
        val = captured.get(key)
        if val is None:
            out[key] = None
            continue
        t = val.detach().to("cpu")
        if key in _FLOAT_KEYS:
            t = t.to(torch.float16)
        out[key] = t
    return out


def _remap_to_padded_axis(amap: AtomMap, ref_mask_row: np.ndarray, heavy_indices: np.ndarray) -> AtomMap:
    """Lift a real-atom-slot AtomMap into the model's padded atom axis.

    ``ref_mask_row`` is the captured per-atom mask (n_atoms,) marking real atoms.
    Its True positions, in order, are the model slots for our contiguous layout;
    we index through them so no "real atoms come first" assumption is needed.
    """
    real_positions = np.where(ref_mask_row)[0]
    if real_positions.shape[0] != amap.num_slots:
        raise ValueError(
            f"model real-atom count {real_positions.shape[0]} != mapped layout "
            f"{amap.num_slots}; sequence/topology mismatch"
        )
    n_atoms = int(ref_mask_row.shape[0])
    full_slot = real_positions[amap.slot_index].astype(np.int64)
    full_present = np.zeros(n_atoms, dtype=bool)
    full_present[real_positions[amap.present_mask]] = True
    return AtomMap(
        num_slots=n_atoms,
        gather_md_row=amap.gather_md_row,
        slot_index=full_slot,
        present_mask=full_present,
        n_md_heavy=amap.n_md_heavy,
        heavy_indices=heavy_indices,
    )


def _save_atom_map(cache: Path, amap: AtomMap) -> None:
    tmp = cache / "atom_map_tmp.npz"
    with open(tmp, "wb") as fh:  # file handle avoids np.savez's auto-".npz" suffix
        np.savez(
            fh,
            num_slots=np.int64(amap.num_slots),
            gather_md_row=amap.gather_md_row,
            slot_index=amap.slot_index,
            present_mask=amap.present_mask,
            n_md_heavy=np.int64(amap.n_md_heavy),
            heavy_indices=amap.heavy_indices,
        )
    tmp.replace(cache / "atom_map.npz")


def load_atom_map(cache_dir: str | Path, domain: str) -> AtomMap:
    """Load a cached :class:`~.atom_map.AtomMap` (padded model-atom axis)."""
    path = domain_cache_dir(cache_dir, domain) / "atom_map.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path)
    return AtomMap(
        num_slots=int(d["num_slots"]),
        gather_md_row=d["gather_md_row"],
        slot_index=d["slot_index"],
        present_mask=d["present_mask"],
        n_md_heavy=int(d["n_md_heavy"]),
        heavy_indices=d["heavy_indices"],
    )


def load_conditioning(cache_dir: str | Path, domain: str, device: str = "cpu") -> dict:
    """Load a domain's cached denoiser conditioning tensors onto ``device``."""
    import torch

    path = domain_cache_dir(cache_dir, domain) / "conditioning.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location=device)
    return raw


def load_meta(cache_dir: str | Path, domain: str) -> dict:
    return json.loads((domain_cache_dir(cache_dir, domain) / "meta.json").read_text())


def featurize_domain(
    domain: str,
    *,
    mdcath_dir: str | Path,
    cache_dir: str | Path,
    model,
    builder,
    num_loops: int = 20,
    min_matched_fraction: float = 0.98,
    max_len: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Featurize one domain: cache conditioning + atom map. Returns a status dict.

    Idempotent — skips a domain that already has a cache unless ``overwrite``.
    Domains whose atom map matches < ``min_matched_fraction`` of model atoms are
    recorded as ``dropped`` and not cached (their gradients would be unreliable).
    """
    import torch

    cache = domain_cache_dir(cache_dir, domain)
    if is_featurized(cache_dir, domain) and not overwrite:
        return {"domain": domain, "status": "cached"}

    topo = read_topology(mdcath_dir, domain)
    length = len(topo.sequence)
    if max_len is not None and length > max_len:
        return {"domain": domain, "status": "too_long", "length": length}

    amap = build_atom_map(topo.sequence, topo.md_records)
    if amap.matched_fraction < min_matched_fraction:
        return {"domain": domain, "status": "dropped", "matched_fraction": amap.matched_fraction, "length": length}

    spi = _protein_spi(topo.sequence)
    with torch.no_grad():
        captured = _capture_conditioning(model, builder, spi, num_loops)

    ref_mask = captured["ref_mask"]
    ref_mask_row = ref_mask.detach().to("cpu").numpy().reshape(ref_mask.shape[-1]).astype(bool)
    full_map = _remap_to_padded_axis(amap, ref_mask_row, topo.heavy_indices)

    cache.mkdir(parents=True, exist_ok=True)
    tensors = _to_cache_tensors(captured)
    tmp = cache / "conditioning.pt.tmp"
    torch.save(tensors, tmp)
    tmp.replace(cache / "conditioning.pt")
    _save_atom_map(cache, full_map)
    meta = {
        "domain": domain,
        "sequence": topo.sequence,
        "length": length,
        "n_atoms": int(ref_mask_row.shape[0]),
        "n_real_atoms": int(amap.num_slots),
        "matched_fraction": amap.matched_fraction,
        "n_md_heavy": amap.n_md_heavy,
    }
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"domain": domain, "status": "featurized", **{k: meta[k] for k in ("length", "matched_fraction")}}


def featurize_all(cfg, *, overwrite: bool = False, verbose: bool = True) -> list[dict]:
    """Featurize every training domain in ``cfg`` on a single GPU.

    Loads the model once (the expensive resource) and streams domains through it,
    mirroring the load-once/stream-many pattern in ``ensemble.py``.
    """
    from esm.models.esmfold2 import ESMFold2InputBuilder
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from .config import resolve_domains

    device = "cuda"
    model = ESMFold2Model.from_pretrained(cfg.model.model_name).to(device).eval()
    builder = ESMFold2InputBuilder()
    domains = resolve_domains(cfg.data) + list(cfg.data.val_domains)

    results: list[dict] = []
    for i, domain in enumerate(domains):
        try:
            res = featurize_domain(
                domain,
                mdcath_dir=cfg.data.mdcath_dir,
                cache_dir=cfg.data.cache_dir,
                model=model,
                builder=builder,
                num_loops=cfg.model.num_loops,
                min_matched_fraction=cfg.data.min_matched_fraction,
                max_len=cfg.data.max_len,
                overwrite=overwrite,
            )
        except Exception as exc:  # keep going; one bad domain shouldn't kill the sweep
            res = {"domain": domain, "status": "error", "error": repr(exc)}
        results.append(res)
        if verbose:
            print(
                f"[featurize {i + 1}/{len(domains)}] {domain}: {res['status']}"
                + (f" ({res.get('matched_fraction'):.3f})" if "matched_fraction" in res else ""),
                flush=True,
            )
    return results
