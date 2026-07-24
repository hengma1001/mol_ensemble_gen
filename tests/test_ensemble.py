"""Offline unit tests for the ensemble sampler (no GPU / no model load)."""

from __future__ import annotations

import json

import pytest

from python_package.ensemble import (
    EnsembleMember,
    EnsembleSpec,
    ESMFold2Ensemble,
    SamplingParams,
    derive_seed,
)


class _FakeArr:
    def __init__(self, value: float):
        self._value = value

    def mean(self) -> float:
        return self._value


class _FakeComplex:
    def __init__(self, tag: str):
        self._tag = tag

    def to_mmcif(self) -> str:
        return f"data_{self._tag}\n"


class _FakeResult:
    def __init__(self, tag: str, iptm: float | None):
        self.complex = _FakeComplex(tag)
        self.plddt = _FakeArr(0.9)
        self.ptm = 0.8
        self.iptm = iptm


def _ensemble(spec: EnsembleSpec, results_per_fold) -> ESMFold2Ensemble:
    """Build an ESMFold2Ensemble without importing esm/transformers or loading a model."""
    obj = ESMFold2Ensemble.__new__(ESMFold2Ensemble)
    obj.spec = spec
    obj.device = "cpu"
    obj.model = object()

    class _Builder:
        def fold(self, model, spi, *, seed, **kwargs):  # noqa: ARG002
            return results_per_fold(seed)

    obj._builder = _Builder()
    return obj


@pytest.mark.unit
def test_derive_seed_is_deterministic_and_unique():
    a = derive_seed(42, "P1", 0)
    assert a == derive_seed(42, "P1", 0)            # reproducible
    assert a != derive_seed(42, "P1", 1)            # varies by member
    assert a != derive_seed(42, "P2", 0)            # varies by input
    assert 0 <= a < 2**31                           # int32-safe positive


@pytest.mark.unit
def test_fold_kwargs_drops_none():
    p = SamplingParams(num_loops=5, lm_dropout=0.3)  # noise_scale etc. left None
    assert p.fold_kwargs() == {"num_loops": 5, "num_sampling_steps": 200, "num_diffusion_samples": 1, "lm_dropout": 0.3}


@pytest.mark.unit
def test_size_counts_diffusion_samples():
    spec = EnsembleSpec(members=4, sampling=SamplingParams(num_diffusion_samples=3))
    assert spec.size == 12


@pytest.mark.unit
def test_generate_handles_single_result(tmp_path):
    spec = EnsembleSpec(members=3, base_seed=1)
    ens = _ensemble(spec, lambda seed: _FakeResult(f"s{seed}", iptm=None))  # single obj, not a list

    members = ens.generate(spi=None, input_id="mono", out_dir=tmp_path)

    assert len(members) == 3
    assert all(m.iptm is None for m in members)                 # monomer -> None, not a crash
    cifs = sorted(p.name for p in tmp_path.glob("*.cif"))
    assert cifs == ["mono_m0000_s00.cif", "mono_m0001_s00.cif", "mono_m0002_s00.cif"]  # unique names


@pytest.mark.unit
def test_generate_handles_list_results_and_metadata(tmp_path):
    spec = EnsembleSpec(members=2, sampling=SamplingParams(num_diffusion_samples=2))
    ens = _ensemble(spec, lambda seed: [_FakeResult(f"{seed}a", 0.7), _FakeResult(f"{seed}b", 0.6)])

    members = ens.generate(spi=None, input_id="cplx", out_dir=tmp_path)

    assert len(members) == 4                                    # 2 seeds x 2 samples
    assert len(list(tmp_path.glob("*.cif"))) == 4
    assert {m.sample_idx for m in members} == {0, 1}

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["spec"]["size"] == 4
    assert len(manifest["members"]) == 4
    assert (tmp_path / "metadata.csv").exists()
    header = (tmp_path / "metadata.csv").read_text().splitlines()[0]
    assert "param_seed" in header and "iptm" in header
