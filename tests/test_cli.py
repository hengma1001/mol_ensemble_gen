"""Offline unit tests for the CLI config layer (no GPU / no model load)."""

from __future__ import annotations

import pytest

from python_package.cli import (
    RunConfig,
    _apply_overrides,
    build_parser,
    build_spec,
    load_config,
    parse_gpus,
)

_CONFIG = """
input: examples/example.fasta
out_dir: runs/t
ensemble:
  members: 5
  base_seed: 7
sampling:
  num_loops: 3
  num_diffusion_samples: 2
  lm_dropout: 0.3
"""


@pytest.mark.unit
def test_load_config_and_build_spec(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_CONFIG)

    cfg = load_config(cfg_path)
    assert cfg.input == "examples/example.fasta"
    assert cfg.model_name == "biohub/ESMFold2"    # default applied

    spec = build_spec(cfg)
    assert spec.members == 5
    assert spec.base_seed == 7
    assert spec.size == 10                          # 5 seeds x 2 diffusion samples
    assert spec.sampling.num_loops == 3
    assert spec.sampling.lm_dropout == 0.3


@pytest.mark.unit
def test_load_config_rejects_unknown_top_level_key(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("input: x.fasta\nbogus: 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(cfg_path)


@pytest.mark.unit
def test_build_spec_rejects_unknown_sampling_knob():
    cfg = RunConfig(input="x.fasta", sampling={"not_a_knob": 1})
    with pytest.raises(ValueError, match="unknown sampling knobs"):
        build_spec(cfg)


@pytest.mark.unit
def test_load_config_requires_input(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("out_dir: runs/x\n")
    with pytest.raises(ValueError, match="must define 'input'"):
        load_config(cfg_path)


@pytest.mark.unit
def test_parser_accepts_overrides():
    args = build_parser().parse_args(["run", "c.yaml", "--members", "50", "--out-dir", "runs/e2"])
    assert args.command == "run"
    assert args.members == 50
    assert args.out_dir == "runs/e2"


@pytest.mark.unit
def test_parse_gpus():
    assert parse_gpus("0,1,2,3,4,5,6,7") == [0, 1, 2, 3, 4, 5, 6, 7]
    assert parse_gpus("0") == [0]
    assert parse_gpus("0, 2 ,4") == [0, 2, 4]


@pytest.mark.unit
def test_gpus_override_sets_local_backend():
    args = build_parser().parse_args(["run", "c.yaml", "--gpus", "0,1,2,3"])
    cfg = _apply_overrides(RunConfig(input="x.fasta"), args)
    assert cfg.execution["backend"] == "local"
    assert cfg.execution["gpus"] == [0, 1, 2, 3]
