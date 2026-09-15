"""Dispatch the training stream to a data source (``cfg.data.source``).

The trainer only ever touches ``batch.domain``, ``batch.gt_coords``,
``batch.atom_mask`` and ``batch.temperature``, and looks the featurization cache up
by ``batch.domain``. So a source is fully described by two things: something that
yields :class:`~.mdcath.FrameBatch`, and a domain-id namespace that keys the cache.
Keeping that dispatch here rather than in the trainer means adding a source never
touches the training loop.

BioEmu system ids carry their own prefix (``cath2_1a1wA00``), so mdCATH and BioEmu
caches cannot collide even for the 175 CATH domains present in both — which is
correct, since the two have different topologies, force fields and temperatures.
"""

from __future__ import annotations

#: Registered sources. ``featurize`` names the topology reader in :mod:`.featurize`.
SOURCES = ("mdcath", "bioemu")


def _check(source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"unknown data.source {source!r}; expected one of {list(SOURCES)}")
    return source


def resolve_train_val_domains(data) -> tuple[list[str], list[str]]:
    """``(train, val)`` domain ids for the configured source."""
    source = _check(getattr(data, "source", "mdcath"))
    if source == "bioemu":
        from .config import resolve_bioemu_domains, resolve_bioemu_val_domains

        return resolve_bioemu_domains(data), resolve_bioemu_val_domains(data)
    from .config import resolve_domains, resolve_val_domains

    return resolve_domains(data), resolve_val_domains(data)


def make_dataset(cfg, **kwargs):
    """Build the frame-streaming dataset for ``cfg.data.source``."""
    if _check(getattr(cfg.data, "source", "mdcath")) == "bioemu":
        from .bioemu import make_dataset as build

        return build(cfg, **kwargs)
    from .mdcath import make_dataset as build

    return build(cfg, **kwargs)


def read_topology(data, domain: str):
    """Parse one domain's topology using the configured source's reader."""
    if _check(getattr(data, "source", "mdcath")) == "bioemu":
        from .bioemu import read_topology as read

        return read(data.bioemu_dir, domain)
    from .mdcath import read_topology as read

    return read(data.mdcath_dir, domain)
