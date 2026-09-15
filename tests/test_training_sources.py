"""Tests for the data-source dispatch (``cfg.data.source``) and its split resolvers.

Pure config-level tests: no torch, no mdtraj, no h5py. What they pin is that
selecting a source routes the stream, the split and the topology reader together --
a run that streamed BioEmu frames while resolving mdCATH domains would train on
whatever the cache happened to hold.
"""

from __future__ import annotations

import pytest

from mol_ensemble_gen.training.config import (
    DataConfig,
    resolve_bioemu_domains,
    resolve_bioemu_val_domains,
    resolve_domains,
)
from mol_ensemble_gen.training.sources import resolve_train_val_domains


@pytest.mark.unit
def test_default_source_is_mdcath():
    assert DataConfig().source == "mdcath"


@pytest.mark.unit
def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown data.source"):
        resolve_train_val_domains(DataConfig(source="nope"))


@pytest.mark.unit
def test_bioemu_val_domains_are_carved_out_of_train():
    data = DataConfig(
        source="bioemu",
        bioemu_domains=["cath2_a", "cath2_b", "cath2_c", "cath2_b"],
        bioemu_val_domains=["cath2_c"],
    )
    train, val = resolve_train_val_domains(data)
    assert train == ["cath2_a", "cath2_b"]  # deduped, val removed
    assert val == ["cath2_c"]


@pytest.mark.unit
def test_source_selects_which_split_is_resolved():
    """Both splits populated: the source decides which one the trainer sees."""
    common = dict(
        domains=["mdA", "mdB"],
        val_domains=["mdV"],
        bioemu_domains=["cath2_x"],
        bioemu_val_domains=["cath2_v"],
    )
    assert resolve_train_val_domains(DataConfig(source="mdcath", **common)) == (["mdA", "mdB"], ["mdV"])
    assert resolve_train_val_domains(DataConfig(source="bioemu", **common)) == (["cath2_x"], ["cath2_v"])


@pytest.mark.unit
def test_id_file_ignores_comments_blanks_and_trailing_tab_fields(tmp_path):
    """prepare_msr_cath2.py writes a tab-separated tag column into the val file."""
    path = tmp_path / "val.txt"
    path.write_text("# a comment\n\ncath2_a\theld\ncath2_b\trandom\n   \ncath2_c\n")
    data = DataConfig(source="bioemu", bioemu_val_domains_file=str(path))
    assert resolve_bioemu_val_domains(data) == ["cath2_a", "cath2_b", "cath2_c"]


@pytest.mark.unit
def test_bioemu_resolvers_do_not_read_the_mdcath_fields(tmp_path):
    data = DataConfig(source="bioemu", domains=["mdA"], bioemu_domains=["cath2_x"])
    assert resolve_bioemu_domains(data) == ["cath2_x"]
    assert resolve_domains(data) == ["mdA"]


@pytest.mark.unit
def test_make_dataset_requires_bioemu_dir():
    from mol_ensemble_gen.training.bioemu import make_dataset

    class Cfg:
        data = DataConfig(source="bioemu", bioemu_domains=["cath2_x"])

    with pytest.raises(ValueError, match="bioemu_dir"):
        make_dataset(Cfg())


@pytest.mark.unit
def test_real_msr_cath2_config_resolves_a_disjoint_split():
    """The shipped config: 993 train / 50 val, and none of val_496/test_60 trains."""
    from pathlib import Path

    from mol_ensemble_gen.training.config import load_train_config

    cfg_path = Path("examples/finetune_msr_cath2_4gpu.yaml")
    if not (cfg_path.is_file() and Path("splits/msr_cath2_train.txt").is_file()):
        pytest.skip("MSR_cath2 splits not built")
    cfg = load_train_config(cfg_path)
    train, val = resolve_train_val_domains(cfg.data)
    assert set(train) & set(val) == set()
    held = set()
    for name in ("val_496.txt", "test_60.txt"):
        held |= {ln.strip() for ln in Path("splits", name).read_text().splitlines() if ln.strip()}
    leaked = [s for s in train if s.split("_", 1)[-1] in held]
    assert leaked == [], f"training on held-out mdCATH domains: {leaked}"
