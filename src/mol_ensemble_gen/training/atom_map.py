"""Map mdCATH heavy atoms onto ESMFold2's per-residue atom slots.

This is the correctness keystone: a silent misalignment here corrupts every
training gradient. ESMFold2 lays atoms out **residue-by-residue in
``PROTEIN_HEAVY_ATOMS[resname]`` order** (``esm/.../prepare_input.py``
``tokenize_protein``), contiguously, real atoms first. So the model "slot" of an
atom is fully determined by ``(residue_index, atom_name)``.

We build that slot layout from a sequence, then match each mdCATH heavy atom to a
slot by ``(res_idx, atom_name)``. Unmatched model slots are masked out of the
loss (never zero-filled). The heavy-atom table is injectable so the mapping logic
can be unit-tested without importing ``esm``; the default pulls the real table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# One-letter → three-letter for the 20 standard amino acids. Anything else maps
# to UNK (backbone-only in PROTEIN_HEAVY_ATOMS).
PROTEIN_1TO3: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# Non-standard / MD force-field residue names → their canonical parent. mdCATH is
# classical FF, so protonation-state variants of HIS and modified residues appear.
_RESNAME_CANON: dict[str, str] = {
    "MSE": "MET",                                     # selenomethionine
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",         # CHARMM HIS states
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",         # AMBER HIS states
    "CYX": "CYS", "CYM": "CYS",                       # disulfide / deprotonated CYS
    "LYN": "LYS", "ASH": "ASP", "GLH": "GLU",         # alt protonation
    "ARN": "ARG",
}

# Atom names to ignore when reading an MD topology (terminal oxygen; any H).
_SKIP_ATOMS = {"OXT", "OT1", "OT2", "OT"}


def canonical_resname(name: str) -> str:
    """Fold a force-field/modified residue name to a canonical 3-letter code."""
    name = name.strip().upper()
    return _RESNAME_CANON.get(name, name)


def is_heavy_atom(atom_name: str, element: str | None = None) -> bool:
    """True for a non-hydrogen atom that participates in the model layout."""
    atom_name = atom_name.strip().upper()
    if atom_name in _SKIP_ATOMS:
        return False
    if element is not None:
        return element.strip().upper() not in {"H", "D"}
    # Fall back to the PDB convention: a leading digit then H, or a leading H.
    stripped = atom_name.lstrip("0123456789")
    return not stripped.startswith("H")


def load_heavy_atom_table() -> dict[str, list[str]]:
    """Return ``PROTEIN_HEAVY_ATOMS`` (res3 → ordered heavy-atom names) from esm."""
    from esm.models.esmfold2.constants import PROTEIN_HEAVY_ATOMS

    return {k: list(v) for k, v in PROTEIN_HEAVY_ATOMS.items()}


def build_model_atom_layout(
    sequence: str, *, heavy_atoms: dict[str, list[str]] | None = None
) -> list[tuple[int, str]]:
    """Ordered ``(res_idx, atom_name)`` for every real model atom slot.

    Matches ``tokenize_protein``: residues in sequence order, atoms in
    ``PROTEIN_HEAVY_ATOMS[resname]`` order, contiguous. The length equals the
    model's real (pre-padding) atom count.
    """
    heavy = heavy_atoms if heavy_atoms is not None else load_heavy_atom_table()
    fallback = heavy.get("UNK", ["N", "CA", "C", "O"])
    layout: list[tuple[int, str]] = []
    for i, letter in enumerate(sequence):
        res3 = PROTEIN_1TO3.get(letter.upper(), "UNK")
        atoms = heavy.get(res3, fallback)
        for atom_name in atoms:
            layout.append((i, atom_name))
    return layout


@dataclass
class AtomMap:
    """A frozen mapping from mdCATH heavy-atom rows to model atom slots.

    Apply per frame with :meth:`scatter`: ``gt[slot_index] = heavy[gather_md_row]``.
    """

    num_slots: int                 # real model atom count (== len(layout))
    gather_md_row: np.ndarray      # (K,) mdCATH heavy-atom rows that matched a slot
    slot_index: np.ndarray         # (K,) model slot each matched row fills
    present_mask: np.ndarray       # (num_slots,) bool: slot got an MD coordinate
    n_md_heavy: int                # heavy-atom count read from the MD topology
    # Indices of heavy atoms into the full trajectory atom axis. Populated at
    # featurization time (from the topology) so the streaming dataset can select
    # heavy coords with a single fancy-index before scattering.
    heavy_indices: np.ndarray | None = None

    @property
    def matched_fraction(self) -> float:
        """Fraction of model slots that received an MD coordinate."""
        return float(self.present_mask.mean()) if self.num_slots else 0.0

    def scatter(self, heavy_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scatter one frame's heavy-atom coords ``(n_md_heavy, 3)`` into model slots.

        Returns ``(gt_coords (num_slots, 3), present_mask (num_slots,))``; unmatched
        slots are left at zero and masked False (callers must not put them in the loss).
        """
        if heavy_coords.shape[0] != self.n_md_heavy:
            raise ValueError(
                f"frame has {heavy_coords.shape[0]} heavy atoms, map expects {self.n_md_heavy}"
            )
        gt = np.zeros((self.num_slots, 3), dtype=np.float32)
        gt[self.slot_index] = heavy_coords[self.gather_md_row]
        return gt, self.present_mask.copy()

    def scatter_batch(self, heavy_coords: np.ndarray) -> np.ndarray:
        """Scatter a batch of frames ``(B, n_md_heavy, 3)`` → ``(B, num_slots, 3)``.

        Unmatched slots are left at zero; use :attr:`present_mask` for the loss mask.
        """
        if heavy_coords.ndim != 3 or heavy_coords.shape[1] != self.n_md_heavy:
            raise ValueError(
                f"expected (B, {self.n_md_heavy}, 3), got {tuple(heavy_coords.shape)}"
            )
        b = heavy_coords.shape[0]
        gt = np.zeros((b, self.num_slots, 3), dtype=np.float32)
        gt[:, self.slot_index] = heavy_coords[:, self.gather_md_row]
        return gt


def build_atom_map(
    sequence: str,
    md_records: list[tuple[int, str]],
    *,
    heavy_atoms: dict[str, list[str]] | None = None,
) -> AtomMap:
    """Build the mdCATH→model :class:`AtomMap` for one domain.

    ``md_records`` is ``(res_idx, atom_name)`` per mdCATH heavy atom, in the order
    their coordinates appear in the trajectory (see :func:`.mdcath.read_topology`).
    Each is matched to the model slot with the same ``(res_idx, atom_name)``; a slot
    is filled at most once (first match wins).
    """
    layout = build_model_atom_layout(sequence, heavy_atoms=heavy_atoms)
    num_slots = len(layout)
    slot_of_key: dict[tuple[int, str], int] = {}
    for slot, key in enumerate(layout):
        slot_of_key.setdefault((key[0], key[1].strip().upper()), slot)

    gather_rows: list[int] = []
    slots: list[int] = []
    present = np.zeros(num_slots, dtype=bool)
    for md_row, (res_idx, atom_name) in enumerate(md_records):
        slot = slot_of_key.get((res_idx, atom_name.strip().upper()))
        if slot is None or present[slot]:
            continue
        gather_rows.append(md_row)
        slots.append(slot)
        present[slot] = True

    return AtomMap(
        num_slots=num_slots,
        gather_md_row=np.asarray(gather_rows, dtype=np.int64),
        slot_index=np.asarray(slots, dtype=np.int64),
        present_mask=present,
        n_md_heavy=len(md_records),
    )
