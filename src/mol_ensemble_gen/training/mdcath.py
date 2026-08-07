"""Read mdCATH per-domain HDF5 files and stream trajectory frames for training.

Each ``mdcath_dataset_<domain>.h5`` holds one CATH domain::

    f[domain]                      group; attrs: numProteinAtoms, numResidues,
                                     numNoHAtoms, numFrames, ...
    f[domain]/<pdb-string dataset> full-atom topology as a PDB text blob
    f[domain]/z                    atomic numbers (numProteinAtoms,)
    f[domain]/{temp}/{repl}/coords (frames, numProteinAtoms, 3) float32, Å
    f[domain]/{temp}/{repl}/forces ...

The PDB blob's ATOM order equals the ``coords`` atom order, so parsing it gives,
per atom, ``(res_idx, atom_name, element)`` — everything the atom map needs. The
:class:`MDCathDataset` streams strided frames, selects heavy atoms, and scatters
them into model slots using a cached :class:`~.atom_map.AtomMap`.

Coordinates are already in Ångström (ESMFold2's unit), and the EDM loss centers
each frame, so no unit or translation handling is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..pdb import _THREE_TO_ONE
from .atom_map import PROTEIN_1TO3, canonical_resname, is_heavy_atom
from .config import TEMPERATURES

# One-letter code per canonical residue, for building the model sequence.
_ONE_LETTER = {**{v: k for k, v in PROTEIN_1TO3.items()}}  # 3->1 for the 20 standard


def domain_path(mdcath_dir: str | Path, domain: str) -> Path:
    """Path to a domain's HDF5 file (``mdcath_dataset_<domain>.h5``)."""
    return Path(mdcath_dir) / f"mdcath_dataset_{domain}.h5"


@dataclass
class Topology:
    """Per-domain topology derived from the embedded PDB blob."""

    sequence: str                  # one-letter, length numResidues
    md_records: list[tuple[int, str]]   # (res_idx, atom_name) per heavy atom, coord order
    heavy_indices: np.ndarray      # (n_heavy,) indices into the full atom axis
    n_atoms: int                   # full atom count (== numProteinAtoms)


def _parse_pdb_atoms(pdb_text: str) -> list[tuple[str, str, str, str, str, str]]:
    """Parse ATOM/HETATM records → ``(atom_name, res_name, chain, res_seq, icode, element)``.

    Reads the first model only and honours the primary altloc, preserving file
    order (which matches the coordinate array).
    """
    atoms: list[tuple[str, str, str, str, str, str]] = []
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record == "ENDMDL":
            break
        if record not in ("ATOM", "HETATM"):
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain_id = line[21].strip() or "A"
        res_seq = line[22:26].strip()
        i_code = line[26]
        element = line[76:78].strip() if len(line) >= 78 else ""
        atoms.append((atom_name, res_name, chain_id, res_seq, i_code, element))
    return atoms


def parse_topology(pdb_text: str) -> Topology:
    """Build a :class:`Topology` from a full-atom PDB text blob."""
    atoms = _parse_pdb_atoms(pdb_text)
    if not atoms:
        raise ValueError("no ATOM records in mdCATH topology blob")

    sequence: list[str] = []
    md_records: list[tuple[int, str]] = []
    heavy_indices: list[int] = []
    res_idx = -1
    prev_key: tuple[str, str, str] | None = None

    for i, (atom_name, res_name, chain, res_seq, icode, element) in enumerate(atoms):
        key = (chain, res_seq, icode)
        if key != prev_key:
            res_idx += 1
            prev_key = key
            res3 = canonical_resname(res_name)
            sequence.append(_ONE_LETTER.get(res3, _THREE_TO_ONE.get(res3, "X")))
        if is_heavy_atom(atom_name, element or None):
            md_records.append((res_idx, atom_name))
            heavy_indices.append(i)

    return Topology(
        sequence="".join(sequence),
        md_records=md_records,
        heavy_indices=np.asarray(heavy_indices, dtype=np.int64),
        n_atoms=len(atoms),
    )


def _find_pdb_blob(group, expected_atoms: int | None) -> str:
    """Return the topology PDB text from a domain group.

    mdCATH's blob dataset name varies across releases, so pick the string dataset
    whose parsed atom count matches ``numProteinAtoms`` (falling back to the first
    that parses). Raises if none is found.
    """
    candidates: list[str] = []
    for name in group:
        item = group[name]
        # a scalar/1-elem string dataset (bytes or vlen-str)
        if getattr(item, "shape", None) is not None and item.dtype.kind in ("S", "O", "U"):
            candidates.append(name)
    # Prefer conventional names first.
    ordered = sorted(candidates, key=lambda n: (n not in ("pdbProteinAtoms", "pdb"), n))
    best: str | None = None
    for name in ordered:
        raw = group[name][()]
        text = raw.decode() if isinstance(raw, (bytes, bytearray)) else (
            raw[0].decode() if isinstance(raw, np.ndarray) and raw.dtype.kind == "S" else str(raw)
        )
        if "ATOM" not in text and "HETATM" not in text:
            continue
        if expected_atoms is None:
            return text
        n = len(_parse_pdb_atoms(text))
        if n == expected_atoms:
            return text
        best = text  # parses but count mismatched; keep as fallback
    if best is not None:
        return best
    raise ValueError("could not locate a PDB topology dataset in mdCATH domain group")


def read_topology(mdcath_dir: str | Path, domain: str) -> Topology:
    """Open a domain file and parse its topology (heavy-atom map inputs)."""
    import h5py

    path = domain_path(mdcath_dir, domain)
    with h5py.File(path, "r") as f:
        group = f[domain]
        expected = int(group.attrs["numProteinAtoms"]) if "numProteinAtoms" in group.attrs else None
        topo = parse_topology(_find_pdb_blob(group, expected))
    return topo


def ca_atom_indices(pdb_text: str) -> np.ndarray:
    """Full-atom indices of the Cα atoms (one per residue, residue order)."""
    idx = [i for i, a in enumerate(_parse_pdb_atoms(pdb_text)) if a[0].strip().upper() == "CA"]
    return np.asarray(idx, dtype=np.int64)


def read_reference_ca(
    mdcath_dir: str | Path,
    domain: str,
    temperature: int,
    *,
    replicas: list[int] | None = None,
    skip: int = 10,
    max_frames: int | None = None,
) -> np.ndarray:
    """Stack mdCATH Cα coordinates ``(F, n_res, 3)`` at one temperature.

    Concatenates the requested replicas with a frame stride; the ground-truth
    ensemble to compare a sampled ensemble against.
    """
    import h5py

    path = domain_path(mdcath_dir, domain)
    frames: list[np.ndarray] = []
    with h5py.File(path, "r") as f:
        group = f[domain]
        expected = int(group.attrs["numProteinAtoms"]) if "numProteinAtoms" in group.attrs else None
        ca = ca_atom_indices(_find_pdb_blob(group, expected))
        tkey = str(temperature)
        if tkey not in group:
            raise KeyError(f"temperature {temperature} absent for {domain}")
        tgroup = group[tkey]
        reps = replicas if replicas is not None else available_replicas(tgroup)
        for repl in reps:
            rkey = str(repl)
            if rkey not in tgroup or "coords" not in tgroup[rkey]:
                continue
            dset = tgroup[rkey]["coords"]
            sel = list(range(0, dset.shape[0], max(1, skip)))
            frames.append(dset[sel, :, :][:, ca, :].astype(np.float32))
    if not frames:
        raise ValueError(f"no coordinate frames for {domain} at {temperature} K")
    out = np.concatenate(frames, axis=0)
    if max_frames is not None and out.shape[0] > max_frames:
        stride = out.shape[0] // max_frames
        out = out[::stride][:max_frames]
    return out


def available_replicas(group_temp) -> list[int]:
    """Replica indices present under a temperature group, sorted."""
    reps = []
    for name in group_temp:
        try:
            reps.append(int(name))
        except ValueError:
            continue
    return sorted(reps)


def _chunks(seq: list[int], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@dataclass
class FrameBatch:
    """A micro-step's worth of frames from one (domain, temperature)."""

    domain: str
    temperature: float             # Kelvin
    gt_coords: np.ndarray          # (B, num_slots, 3) float32, Å
    atom_mask: np.ndarray          # (num_slots,) bool — matched model slots


def _shard(items: list, rank: int, world_size: int) -> list:
    """Contiguous-strided shard for (rank of world_size); balanced by count."""
    return items[rank::world_size] if world_size > 1 else items


def _worker_rank(rank: int, world_size: int) -> tuple[int, int]:
    """Combine DDP rank with DataLoader worker id into a single flat shard index."""
    try:
        import torch

        info = torch.utils.data.get_worker_info()
    except Exception:  # torch missing or not in a worker
        info = None
    if info is None:
        return rank, world_size
    return rank * info.num_workers + info.id, world_size * info.num_workers


def make_dataset(cfg, rank: int = 0, world_size: int = 1):
    """Build an :class:`MDCathDataset` from a :class:`~.config.TrainConfig`."""
    from .config import resolve_domains

    data = cfg.data
    return _dataset_cls()(
        mdcath_dir=data.mdcath_dir,
        cache_dir=data.cache_dir,
        domains=resolve_domains(data),
        temperatures=data.temperatures or list(TEMPERATURES),
        replicas=data.replicas,
        skip_frames=data.skip_frames,
        frames_per_step=data.frames_per_step,
        rank=rank,
        world_size=world_size,
    )


def _make_iterable_dataset():
    import torch

    class _MDCathDataset(torch.utils.data.IterableDataset):
        """Stream ``(domain, temperature, B frames)`` micro-batches, sharded by rank/worker.

        Requires each domain to have a featurization cache (see
        :mod:`.featurize`) providing its :class:`~.atom_map.AtomMap`; domains
        without a cache are skipped with a warning (they were dropped for a low
        matched fraction or not yet featurized).
        """

        def __init__(self, *, mdcath_dir, cache_dir, domains, temperatures,
                     replicas, skip_frames, frames_per_step, rank, world_size):
            super().__init__()
            self.mdcath_dir = str(mdcath_dir)
            self.cache_dir = Path(cache_dir)
            self.domains = list(domains)
            self.temperatures = [int(t) for t in temperatures]
            self.replicas = None if replicas is None else [int(r) for r in replicas]
            self.skip_frames = max(1, int(skip_frames))
            self.frames_per_step = max(1, int(frames_per_step))
            self.rank = rank
            self.world_size = world_size

        def __iter__(self):
            import h5py

            from .featurize import load_atom_map

            flat_rank, flat_world = _worker_rank(self.rank, self.world_size)
            my_domains = _shard(self.domains, flat_rank, flat_world)
            for domain in my_domains:
                try:
                    amap = load_atom_map(self.cache_dir, domain)
                except FileNotFoundError:
                    print(f"[mdcath] no cache for {domain}; skipping", flush=True)
                    continue
                path = domain_path(self.mdcath_dir, domain)
                if not path.exists():
                    print(f"[mdcath] missing {path}; skipping", flush=True)
                    continue
                with h5py.File(path, "r") as f:
                    group = f[domain]
                    heavy = amap.heavy_indices
                    yield from self._stream_domain(domain, group, amap, heavy)

        def _stream_domain(self, domain, group, amap, heavy):
            for temp in self.temperatures:
                tkey = str(temp)
                if tkey not in group:
                    continue
                tgroup = group[tkey]
                reps = self.replicas if self.replicas is not None else available_replicas(tgroup)
                for repl in reps:
                    rkey = str(repl)
                    if rkey not in tgroup or "coords" not in tgroup[rkey]:
                        continue
                    dset = tgroup[rkey]["coords"]
                    n_frames = dset.shape[0]
                    frame_ids = list(range(0, n_frames, self.skip_frames))
                    for chunk in _chunks(frame_ids, self.frames_per_step):
                        coords = dset[chunk, :, :].astype(np.float32)      # (B, n_atoms, 3)
                        gt = amap.scatter_batch(coords[:, heavy, :])       # (B, num_slots, 3)
                        yield FrameBatch(
                            domain=domain,
                            temperature=float(temp),
                            gt_coords=gt,
                            atom_mask=amap.present_mask.copy(),
                        )

    return _MDCathDataset


def _dataset_cls():
    """Build (once) and return the ``MDCathDataset`` class.

    Must be called rather than referencing the name directly from inside this
    module: the lazy name is served by ``__getattr__``, which Python consults
    only for *attribute* access on the module (``mdcath.MDCathDataset``), never
    for global-name lookup in a function defined here.
    """
    cls = globals().get("MDCathDataset")
    if cls is None:
        cls = globals()["MDCathDataset"] = _make_iterable_dataset()
    return cls


def __getattr__(name):
    if name == "MDCathDataset":
        return _dataset_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
