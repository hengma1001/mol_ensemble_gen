"""Tests for the sampling path — the code that turns a checkpoint into structures.

This path had no coverage until it broke: ``sample-md`` loaded every native-``t``
checkpoint's 350 tensors into the reference denoiser's 345 slots with
``strict=True`` and died, and the obvious "fix" (``strict=False``) would have been
worse — silently discarding the learned temperature head and sampling from a model
missing part of what was trained.

The decision is now :func:`plan_denoiser`, a pure function, so the logic is checked
on CPU in milliseconds. The end-to-end fold stays behind ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import pytest

from mol_ensemble_gen.training.sample import plan_denoiser

# A pretrained/EDM checkpoint: 345 tensors, no t-head.
EDM_KEYS = {"conditioning.s_proj.weight", "token_transformer.attn_blocks.0.q_proj.weight"}
# A native-t checkpoint additionally carries conditioning.t_*.
FLOW_KEYS = EDM_KEYS | {
    "conditioning.t_fourier.w",
    "conditioning.t_norm.weight",
    "conditioning.t_proj.weight",
}


@pytest.mark.unit
def test_t_head_checkpoint_always_uses_our_denoiser():
    """The regression: a t-head checkpoint must never go to the reference module.

    Even when the stored config says ``backend: reference`` — an inconsistent
    checkpoint must not silently lose its conditioning.
    """
    for backend in ("ours", "reference"):
        use_ours, t_cond = plan_denoiser(FLOW_KEYS, backend, "add")
        assert use_ours, f"t-head checkpoint routed away from our denoiser ({backend=})"
        assert t_cond == "add"


@pytest.mark.unit
def test_t_head_checkpoint_never_builds_an_off_head():
    """Building with 'off' would leave the t-head tensors unmatched → load error."""
    use_ours, t_cond = plan_denoiser(FLOW_KEYS, "ours", "off")
    assert use_ours and t_cond != "off"


@pytest.mark.unit
def test_replace_mode_is_preserved():
    _, t_cond = plan_denoiser(FLOW_KEYS, "ours", "replace")
    assert t_cond == "replace"


@pytest.mark.unit
@pytest.mark.parametrize("backend,expect_ours", [("ours", True), ("reference", False)])
def test_plain_checkpoint_follows_the_configured_backend(backend, expect_ours):
    use_ours, t_cond = plan_denoiser(EDM_KEYS, backend, "add")
    assert use_ours is expect_ours
    # No t-head in the weights ⇒ must not build one, whatever the config says.
    assert t_cond == "off"


# ---------------------------------------------------------------------------
# end to end (needs GPU + weights + a real checkpoint)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    "ckpt", ["runs/finetune_pilot_flow_v4/checkpoint.pt", "runs/finetune_pilot_edm_v3/checkpoint.pt"]
)
def test_sample_at_temperature_writes_usable_structures(ckpt, tmp_path):
    """Fold one structure per checkpoint and check it is a plausible protein.

    Covers what unit tests cannot: the real strict load, the temperature-bias
    monkeypatch, the sampler, and the CIF/metadata write — the whole path a
    ``sample-md`` invocation exercises.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from pathlib import Path

    if not Path(ckpt).exists():
        pytest.skip(f"no checkpoint at {ckpt}")

    import numpy as np

    from mol_ensemble_gen.cif import ca_coords, parse_cif_atoms
    from mol_ensemble_gen.training.mdcath import read_topology
    from mol_ensemble_gen.training.sample import sample_at_temperature

    mdcath = "/nfs/lambda_stor_01/homes/heng.ma/dataset/mdcath"
    if not Path(mdcath, "mdcath_dataset_1ha8A00.h5").exists():
        pytest.skip("mdCATH domain not available")
    seq = read_topology(mdcath, "1ha8A00").sequence
    fasta = tmp_path / "d.fasta"
    fasta.write_text(f">A\n{seq}\n")

    members = sample_at_temperature(ckpt, fasta, 320.0, tmp_path / "out", members=1, base_seed=0, device="cuda")
    assert len(members) == 1

    atoms = parse_cif_atoms(members[0].cif_path)
    _, ca = ca_coords(atoms)
    assert len(ca) == len(seq), "one Cα per residue"
    assert np.isfinite(ca).all()

    # A folded domain, not a diverged cloud or a collapsed point.
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
    assert 5.0 < rg < 40.0, f"radius of gyration {rg:.1f} Å is not protein-like"

    # Consecutive Cα atoms sit ~3.8 Å apart in any real chain.
    d = np.linalg.norm(np.diff(ca, axis=0), axis=-1)
    assert 2.5 < float(np.median(d)) < 4.5, f"median Cα–Cα {float(np.median(d)):.2f} Å"
