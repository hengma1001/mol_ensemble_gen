"""Offline unit tests for the finetuning YAML config loader (stdlib + pyyaml)."""

from __future__ import annotations

import textwrap

import pytest

from mol_ensemble_gen.training.config import (
    DataConfig,
    TrainConfig,
    config_to_dict,
    load_train_config,
    resolve_domains,
)


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text))
    return p


@pytest.mark.unit
def test_defaults_when_empty(tmp_path):
    cfg = load_train_config(_write(tmp_path, ""))
    assert isinstance(cfg, TrainConfig)
    assert cfg.out_dir == "runs/finetune"
    assert cfg.data.temperatures == [320, 348, 379, 413, 450]
    assert cfg.temperature.embed_dim == 451
    assert cfg.resume is True


@pytest.mark.unit
def test_nested_overrides_and_roundtrip(tmp_path):
    cfg = load_train_config(
        _write(
            tmp_path,
            """
            out_dir: runs/x
            seed: 7
            data:
              mdcath_dir: /data/mdcath
              domains: [aA00, bB01]
              skip_frames: 5
            temperature:
              hidden_dim: 128
            optim:
              lr: 2.0e-4
              max_steps: 100
            amp_dtype: float16
            slurm:
              nodes: 4
              partition: gpu
            """,
        )
    )
    assert cfg.out_dir == "runs/x" and cfg.seed == 7
    assert cfg.data.mdcath_dir == "/data/mdcath"
    assert cfg.data.domains == ["aA00", "bB01"]
    assert cfg.data.skip_frames == 5
    assert cfg.temperature.hidden_dim == 128
    assert cfg.optim.lr == pytest.approx(2e-4) and cfg.optim.max_steps == 100
    assert cfg.amp_dtype == "float16"
    assert cfg.slurm == {"nodes": 4, "partition": "gpu"}
    # asdict round-trip preserves the nested structure
    d = config_to_dict(cfg)
    assert d["data"]["domains"] == ["aA00", "bB01"]
    assert d["optim"]["max_steps"] == 100


@pytest.mark.unit
def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown TrainConfig keys"):
        load_train_config(_write(tmp_path, "nope: 1\n"))


@pytest.mark.unit
def test_unknown_nested_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown DataConfig keys"):
        load_train_config(_write(tmp_path, "data:\n  bogus: 1\n"))


@pytest.mark.unit
def test_non_mapping_section_rejected(tmp_path):
    with pytest.raises(ValueError, match="expected a mapping for DataConfig"):
        load_train_config(_write(tmp_path, "data: [1, 2, 3]\n"))


@pytest.mark.unit
def test_wandb_defaults_off():
    cfg = TrainConfig()
    assert cfg.wandb.enabled is False
    assert cfg.wandb.project == "mol-ensemble-gen"
    assert cfg.wandb.mode == "online"
    assert cfg.wandb.tags == []


@pytest.mark.unit
def test_wandb_block_loads(tmp_path):
    cfg = load_train_config(
        _write(
            tmp_path,
            """
            wandb:
              enabled: true
              project: my-proj
              entity: my-team
              run_name: run-1
              tags: [a, b]
              mode: offline
            """,
        )
    )
    assert cfg.wandb.enabled is True
    assert cfg.wandb.project == "my-proj"
    assert cfg.wandb.entity == "my-team"
    assert cfg.wandb.run_name == "run-1"
    assert cfg.wandb.tags == ["a", "b"]
    assert cfg.wandb.mode == "offline"
    assert config_to_dict(cfg)["wandb"]["tags"] == ["a", "b"]


@pytest.mark.unit
def test_wandb_unknown_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown WandbConfig keys"):
        load_train_config(_write(tmp_path, "wandb:\n  bogus: 1\n"))


@pytest.mark.unit
def test_wandb_bad_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="wandb.mode"):
        load_train_config(_write(tmp_path, "wandb:\n  mode: sometimes\n"))


@pytest.mark.unit
def test_init_wandb_disabled_returns_none():
    from mol_ensemble_gen.training.config import config_to_dict as _c2d
    from mol_ensemble_gen.training.trainer import _init_wandb

    cfg = TrainConfig()
    assert _init_wandb(cfg, _c2d(cfg), resume_step=0) is None


@pytest.mark.unit
def test_init_wandb_enabled_without_wandb_raises(monkeypatch):
    import builtins

    from mol_ensemble_gen.training.config import WandbConfig, config_to_dict as _c2d
    from mol_ensemble_gen.training.trainer import _init_wandb

    # Simulate wandb not being installed regardless of the environment.
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    cfg = TrainConfig(wandb=WandbConfig(enabled=True))
    with pytest.raises(ImportError, match="wandb.enabled is true"):
        _init_wandb(cfg, _c2d(cfg), resume_step=0)


@pytest.mark.unit
def test_resolve_domains_dedup_and_val_exclusion():
    data = DataConfig(domains=["a", "b", "a", "c"], val_domains=["b"])
    assert resolve_domains(data) == ["a", "c"]


@pytest.mark.unit
def test_resolve_domains_reads_file(tmp_path):
    f = tmp_path / "doms.txt"
    f.write_text("# header\nd1\n\nd2\nd1\n")
    data = DataConfig(domains=["d0"], domains_file=str(f), val_domains=["d2"])
    assert resolve_domains(data) == ["d0", "d1"]
