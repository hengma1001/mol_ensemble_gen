"""Read BioEmu-format MD systems (``topology.pdb`` + ``trajs/*.xtc``) as references.

The BioEmu CATH release (Lewis et al., *Science* 2025; Zenodo 10.5281/zenodo.15629740)
ships one directory per system::

    $ROOT/$SYSTEM/dataset.json                      # metadata (force field, T, ...)
    $ROOT/$SYSTEM/topology.pdb                      # topology for the xtc files
    $ROOT/$SYSTEM/trajs/run###_protein.cmprsd.xtc   # coordinates
    $ROOT/$SYSTEM/trajs/run###_protein.json         # per-run metadata [optional]

This module is the BioEmu counterpart of :mod:`.mdcath`: it exposes the same
``read_topology`` / ``read_reference_ca`` pair so :mod:`.eval` can score a sampled
ensemble against either source. Two differences from mdCATH matter and are handled
here rather than by the caller:

* **Units.** xtc stores nanometres; mdCATH's HDF5 stores Ångström, and every other
  coordinate in this package is Ångström. Frames are scaled by 10 on read.
* **Temperature.** BioEmu's CATH MD is a *single* thermodynamic state at or near
  300 K, where mdCATH gives five per domain. There is no temperature axis to index,
  so ``temperature`` here is checked against ``dataset.json`` rather than used as a
  key, and :func:`system_temperature` is what a caller should use to decide which
  temperature to condition the sampler on.

The ONE_cath1 subset (50 systems, >100 µs cumulative each) is the intended use: its
sampling is converged far past mdCATH's 5x500 ns, so distribution-overlap metrics
computed against it mean something that the same metrics against mdCATH do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .atom_map import PROTEIN_1TO3, canonical_md_atom_name
from .mdcath import Topology, parse_topology

#: BioEmu's CATH MD was run at (or near) 300 K. Used when ``dataset.json`` names no
#: temperature; note this is *below* mdCATH's 320-450 K range, so a model finetuned
#: on mdCATH is extrapolating when asked for it.
DEFAULT_TEMPERATURE: float = 300.0

#: mdtraj reports nanometres; this package works in Ångström throughout.
NM_TO_ANGSTROM: float = 10.0

#: ``dataset.json`` key names seen for the thermostat setpoint, most specific first.
_TEMPERATURE_KEYS = ("temperature", "temperature_K", "temp", "T", "thermostat_temperature")


def system_dir(root: str | Path, system: str) -> Path:
    """Path to one system's directory."""
    return Path(root) / system


def has_trajectories(system_path: Path) -> bool:
    """True when a system directory actually holds coordinate data.

    The MSR_cath2 release ships three systems (``cath2_3bdlA01``, ``cath2_3bpqD00``,
    ``cath2_3dh3A01``) with a ``topology.pdb`` and a ``dataset.json`` but **no**
    ``trajs/`` at all — verified against the zip's own central directory, so it is
    the release and not a partial extraction. Treating a topology as proof of a
    usable system put all three into a training split and killed a 4-GPU run at
    step ~1,300.
    """
    traj_dir = system_path / "trajs"
    return traj_dir.is_dir() and any(traj_dir.glob("*.xtc"))


def available_systems(root: str | Path) -> list[str]:
    """Every *usable* system directory under ``root``, sorted.

    A directory counts as a system when it holds a ``topology.pdb`` **and** at least
    one trajectory; see :func:`has_trajectories`. The Zenodo zip may unpack with an
    extra top-level directory (``$ROOT/ONE_cath1/$SYSTEM``), so that one level of
    nesting is descended into when ``root`` itself holds no systems.
    """
    base = Path(root)
    systems = sorted(
        p.name for p in base.iterdir() if p.is_dir() and (p / "topology.pdb").is_file() and has_trajectories(p)
    )
    if systems:
        return systems
    nested = [p for p in sorted(base.iterdir()) if p.is_dir()]
    if len(nested) == 1:
        return available_systems(nested[0])
    return []


def resolve_root(root: str | Path) -> Path:
    """The directory that actually contains the system folders.

    Mirrors :func:`available_systems`'s single-level descent so callers can hold one
    path rather than re-deriving it.
    """
    base = Path(root)
    if any(p.is_dir() and (p / "topology.pdb").is_file() for p in base.iterdir()):
        return base
    nested = [p for p in sorted(base.iterdir()) if p.is_dir()]
    if len(nested) == 1:
        return resolve_root(nested[0])
    return base


def read_metadata(root: str | Path, system: str) -> dict:
    """Parse a system's ``dataset.json`` (empty dict when absent)."""
    path = system_dir(root, system) / "dataset.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _find_temperature(meta: dict) -> float | None:
    """Pull a Kelvin temperature out of a metadata dict, however it is spelled.

    Searches the named keys at the top level first, then any nested dict, then any
    key whose name merely *contains* "temp" -- the release's own card does not
    document the schema, and per-run json files may differ from ``dataset.json``.
    """
    for key in _TEMPERATURE_KEYS:
        if key in meta:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                pass
    for value in meta.values():
        if isinstance(value, dict):
            found = _find_temperature(value)
            if found is not None:
                return found
    for key, value in meta.items():
        if "temp" in key.lower():
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def system_temperature(root: str | Path, system: str) -> float:
    """Simulation temperature (K), from ``dataset.json`` or :data:`DEFAULT_TEMPERATURE`."""
    found = _find_temperature(read_metadata(root, system))
    return DEFAULT_TEMPERATURE if found is None else float(found)


def _canonicalize(topo: Topology) -> Topology:
    """Rewrite Amber/GROMACS atom names to ESMFold2 slot names, dropping the rest.

    Applied here and **not** on the mdCATH path: mdCATH's 5,398 cached atom maps
    were built without it, and changing the shared parser would silently invalidate
    them. See :func:`.atom_map.canonical_md_atom_name` for what is renamed and why.

    ``md_records`` and ``heavy_indices`` are filtered together — they are parallel
    arrays over the same atoms, and dropping from one alone would shift every
    subsequent coordinate by a slot.
    """
    records: list[tuple[int, str]] = []
    indices: list[int] = []
    for (res_idx, atom_name), full_idx in zip(topo.md_records, topo.heavy_indices):
        res3 = PROTEIN_1TO3.get(topo.sequence[res_idx], "UNK")
        canon = canonical_md_atom_name(res3, atom_name)
        if canon is None:
            continue
        records.append((res_idx, canon))
        indices.append(int(full_idx))
    return Topology(
        sequence=topo.sequence,
        md_records=records,
        heavy_indices=np.asarray(indices, dtype=np.int64),
        n_atoms=topo.n_atoms,
    )


def read_topology(root: str | Path, system: str) -> Topology:
    """Parse ``topology.pdb`` with the same parser mdCATH's embedded blob uses.

    Gives the one-letter sequence (what :mod:`.sample` needs as a FASTA input, so
    that the sampled residue count matches the reference) alongside the heavy-atom
    records, with force-field atom names folded to model slot names.
    """
    path = system_dir(root, system) / "topology.pdb"
    if not path.is_file():
        raise FileNotFoundError(f"no topology.pdb for system {system!r} under {root}")
    return _canonicalize(parse_topology(path.read_text()))


def trajectory_paths(root: str | Path, system: str, replicas: list[int] | None = None) -> list[Path]:
    """Sorted ``trajs/*.xtc`` for a system, optionally restricted to run numbers.

    ``replicas`` selects by position in the sorted list (0-based), matching how
    :func:`.mdcath.read_reference_ca` treats its own ``replicas`` argument as a
    selector rather than a filename.
    """
    traj_dir = system_dir(root, system) / "trajs"
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"no trajs/ for system {system!r} under {root}")
    paths = sorted(traj_dir.glob("*.xtc"))
    if not paths:
        raise FileNotFoundError(f"no .xtc files in {traj_dir}")
    if replicas is not None:
        paths = [paths[i] for i in replicas if 0 <= i < len(paths)]
    return paths


def ca_indices(topology_path: str | Path, n_res: int | None = None) -> np.ndarray:
    """Cα atom indices in mdtraj's atom order, one per residue.

    Selected through mdtraj rather than by re-parsing the PDB text: the indices are
    used as ``atom_indices`` for the xtc reader, so they must be in *mdtraj's* frame,
    and a mismatch there would silently mis-scatter coordinates rather than error.
    """
    import mdtraj as md

    top = md.load_topology(str(topology_path))
    idx = top.select("protein and name CA")
    if idx.size == 0 or (n_res is not None and idx.size != n_res):
        # ``protein`` drops residues mdtraj does not recognise; fall back to a plain
        # name match when it disagrees with the residue count the PDB parser found.
        alt = top.select("name CA")
        if alt.size and (n_res is None or alt.size == n_res or idx.size == 0):
            idx = alt
    if idx.size == 0:
        raise ValueError(f"no Cα atoms found in {topology_path}")
    return np.asarray(idx, dtype=np.int64)


def _load_ca_frames(traj_path: Path, topology_path: Path, ca: np.ndarray, stride: int) -> np.ndarray:
    """Strided Cα frames ``(F, n_res, 3)`` in Å from one xtc.

    Read in chunks: ONE_cath1 trajectories run to >100 µs, so a whole-file load would
    be tens of GB even before the Cα selection is applied. ``chunk`` is rounded up to
    a multiple of ``stride`` because mdtraj's chunked reader applies the stride within
    each chunk, and an unaligned chunk boundary shifts which frames are kept.
    """
    import mdtraj as md

    stride = max(1, int(stride))
    chunk = stride * max(1, 1000 // stride)
    frames = [
        traj.xyz.astype(np.float32)
        for traj in md.iterload(str(traj_path), top=str(topology_path), chunk=chunk, stride=stride, atom_indices=ca)
    ]
    if not frames:
        return np.zeros((0, ca.size, 3), dtype=np.float32)
    return np.concatenate(frames, axis=0) * NM_TO_ANGSTROM


def _subsample(coords: np.ndarray, max_frames: int) -> np.ndarray:
    """Thin ``coords`` to at most ``max_frames``, evenly across the whole stack.

    Even spacing rather than a head slice: the point of ONE_cath1 is that its
    trajectories are long enough to have visited every basin, and truncating to the
    first N frames throws exactly that away.
    """
    if coords.shape[0] <= max_frames:
        return coords
    keep = np.linspace(0, coords.shape[0] - 1, max_frames).round().astype(np.int64)
    return coords[np.unique(keep)]


def read_reference_ca(
    root: str | Path,
    system: str,
    temperature: float | int | None = None,
    *,
    replicas: list[int] | None = None,
    skip: int = 10,
    max_frames: int | None = None,
) -> np.ndarray:
    """Stack a system's Cα coordinates ``(F, n_res, 3)`` in Ångström.

    Signature-compatible with :func:`.mdcath.read_reference_ca` so :mod:`.eval` can
    take either. ``temperature`` is *validated*, not indexed: BioEmu simulated one
    state per system, so asking for a temperature the system was not run at is a
    caller error worth raising rather than silently scoring against the wrong state.
    """
    root = resolve_root(root)
    if temperature is not None:
        actual = system_temperature(root, system)
        if abs(float(temperature) - actual) > 1.0:
            raise KeyError(
                f"{system} was simulated at {actual:g} K; no frames at {float(temperature):g} K "
                "(BioEmu's CATH set is single-temperature)"
            )

    topology_path = system_dir(root, system) / "topology.pdb"
    topo = read_topology(root, system)
    ca = ca_indices(topology_path, n_res=len(topo.sequence))

    stacks = [
        frames
        for path in trajectory_paths(root, system, replicas)
        if (frames := _load_ca_frames(path, topology_path, ca, skip)).shape[0]
    ]
    if not stacks:
        raise ValueError(f"no coordinate frames for {system}")
    out = np.concatenate(stacks, axis=0)
    return out if max_frames is None else _subsample(out, max_frames)


def write_fasta(root: str | Path, systems: list[str], out_dir: str | Path) -> list[Path]:
    """Write one ``<system>.fasta`` per system, for :mod:`.sample` to fold.

    The sequence comes from ``topology.pdb``, so the sampled ensemble has exactly the
    residue count :func:`read_reference_ca` returns -- :func:`.eval.evaluate_temperature`
    refuses to score a mismatch.
    """
    root = resolve_root(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for system in systems:
        seq = read_topology(root, system).sequence
        path = out / f"{system}.fasta"
        path.write_text(f">{system}\n{seq}\n")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Training stream
# ---------------------------------------------------------------------------
#
# Unlike mdCATH's HDF5, xtc has no usable random access: seeking to an arbitrary
# frame means decompressing everything before it. So a domain's frames are read
# once, whole, and chunked in memory — which also lets chunks be shuffled across
# the domain's ~39 trajectories rather than only within one, keeping consecutive
# gradients from sharing a single trajectory's slow degrees of freedom.
#
# The cost is bounded and small: heavy atoms only, ~500 per domain at these
# lengths, ~7.8k frames after striding, i.e. ~47 MB held per worker.

#: Warn past this much resident coordinate data for one domain (MB). Not a hard
#: cap — a long domain should still train, just noisily.
WARN_DOMAIN_MB: float = 512.0


def _load_domain_heavy(root: Path, system: str, heavy: np.ndarray, skip: int) -> np.ndarray:
    """Every trajectory's heavy-atom frames for one system, ``(F, n_heavy, 3)`` in Å."""
    topology_path = system_dir(root, system) / "topology.pdb"
    stacks = [
        frames
        for path in trajectory_paths(root, system)
        if (frames := _load_ca_frames(path, topology_path, heavy, skip)).shape[0]
    ]
    if not stacks:
        return np.zeros((0, heavy.size, 3), dtype=np.float32)
    out = np.concatenate(stacks, axis=0)
    mb = out.nbytes / 1e6
    if mb > WARN_DOMAIN_MB:
        print(f"[bioemu] {system}: {mb:.0f} MB resident ({out.shape[0]} frames)", flush=True)
    return out


def make_dataset(
    cfg,
    rank: int = 0,
    world_size: int = 1,
    domains: list[str] | None = None,
    shuffle: bool | None = None,
    units_per_domain: int | None = None,
    seed_offset: int = 0,
):
    """Build a BioEmu frame-streaming dataset from a :class:`~.config.TrainConfig`.

    Signature-compatible with :func:`.mdcath.make_dataset` so the trainer can pick
    either from ``cfg.data.source``. ``cfg.data.bioemu_dir`` is the dataset root and
    ``resolve_bioemu_domains`` the split; temperature comes from each system's own
    ``dataset.json`` unless ``cfg.data.bioemu_temperature`` overrides it.
    """
    from .config import resolve_bioemu_domains

    data = cfg.data
    if not data.bioemu_dir:
        raise ValueError("data.bioemu_dir must be set for source: bioemu")
    picked = resolve_bioemu_domains(data) if domains is None else list(domains)
    return _dataset_cls()(
        root=resolve_root(data.bioemu_dir),
        cache_dir=data.cache_dir,
        domains=picked,
        temperature=data.bioemu_temperature,
        skip_frames=data.skip_frames,
        frames_per_step=data.frames_per_step,
        rank=rank,
        world_size=world_size,
        shuffle=getattr(data, "shuffle", True) if shuffle is None else shuffle,
        units_per_domain=units_per_domain,
        shuffle_seed=getattr(data, "shuffle_seed", 20260827) + int(seed_offset),
    )


def _make_iterable_dataset():
    import torch

    from .mdcath import FrameBatch, _chunks, _shard, _worker_rank

    class _BioEmuDataset(torch.utils.data.IterableDataset):
        """Stream ``(system, 300 K, B frames)`` micro-batches, sharded by rank/worker.

        Mirrors :class:`.mdcath.MDCathDataset`, including the requirement that each
        domain have a featurization cache (domains without one are skipped with a
        warning). The one structural difference is that there is no temperature axis
        to interleave: BioEmu ran a single state per system, so a domain's units
        differ only by trajectory and frame offset.
        """

        def __init__(
            self,
            *,
            root,
            cache_dir,
            domains,
            temperature,
            skip_frames,
            frames_per_step,
            rank,
            world_size,
            shuffle=True,
            shuffle_seed=20260827,
            units_per_domain=None,
        ):
            super().__init__()
            self.root = Path(root)
            self.cache_dir = Path(cache_dir)
            self.domains = list(domains)
            self.temperature = temperature
            self.skip_frames = max(1, int(skip_frames))
            self.frames_per_step = max(1, int(frames_per_step))
            self.rank = rank
            self.world_size = world_size
            self.shuffle = bool(shuffle)
            self.shuffle_seed = int(shuffle_seed)
            self.units_per_domain = units_per_domain
            self._epoch = 0

        def _rng(self, flat_rank):
            import random

            return random.Random((self.shuffle_seed, self._epoch, flat_rank).__hash__())

        def __iter__(self):
            from .featurize import load_atom_map

            flat_rank, flat_world = _worker_rank(self.rank, self.world_size)
            my_domains = _shard(self.domains, flat_rank, flat_world)
            rng = self._rng(flat_rank)
            if self.shuffle:
                my_domains = list(my_domains)
                rng.shuffle(my_domains)
            self._epoch += 1
            for system in my_domains:
                try:
                    amap = load_atom_map(self.cache_dir, system)
                except FileNotFoundError:
                    print(f"[bioemu] no cache for {system}; skipping", flush=True)
                    continue
                if not (self.root / system / "topology.pdb").is_file():
                    print(f"[bioemu] missing {system}/topology.pdb; skipping", flush=True)
                    continue
                # Skip rather than raise: a split built before has_trajectories()
                # existed, or a partially-synced mount, must not take down every
                # DDP rank over one domain.
                if not has_trajectories(self.root / system):
                    print(f"[bioemu] no trajectories for {system}; skipping", flush=True)
                    continue
                yield from self._stream_domain(system, amap, rng)

        def _stream_domain(self, system, amap, rng=None):
            temp = float(self.temperature) if self.temperature is not None else system_temperature(self.root, system)
            coords = _load_domain_heavy(self.root, system, amap.heavy_indices, self.skip_frames)
            if coords.shape[0] == 0:
                print(f"[bioemu] no frames for {system}; skipping", flush=True)
                return
            chunks = list(_chunks(list(range(coords.shape[0])), self.frames_per_step))
            if self.shuffle and rng is not None:
                rng.shuffle(chunks)
            if self.units_per_domain:
                chunks = chunks[: self.units_per_domain]
            for chunk in chunks:
                gt = amap.scatter_batch(coords[chunk])  # (B, num_slots, 3)
                yield FrameBatch(
                    domain=system,
                    temperature=temp,
                    gt_coords=gt,
                    atom_mask=amap.present_mask.copy(),
                )

    return _BioEmuDataset


def _dataset_cls():
    """Build (once) and return the ``BioEmuDataset`` class (see :mod:`.mdcath`)."""
    cls = globals().get("BioEmuDataset")
    if cls is None:
        cls = globals()["BioEmuDataset"] = _make_iterable_dataset()
    return cls


def __getattr__(name):
    if name == "BioEmuDataset":
        return _dataset_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
