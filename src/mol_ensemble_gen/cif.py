"""Minimal, header-driven mmCIF ``atom_site`` reader.

Just enough to pull coordinates and B-factors out of the CIF files ESMFold2
writes. Header-driven (column order is read from the ``loop_`` block, not
hard-coded), stdlib + numpy only, so ensemble analysis needs no structure
library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AtomArray:
    """Flat per-atom arrays for the ``ATOM`` records of one structure."""

    chain_id: np.ndarray   # (N,) str
    res_seq: np.ndarray    # (N,) int   (label_seq_id)
    res_name: np.ndarray   # (N,) str
    atom_name: np.ndarray  # (N,) str
    coords: np.ndarray     # (N, 3) float
    b_factor: np.ndarray   # (N,) float  (pLDDT for ESMFold2 output)

    def __len__(self) -> int:
        return len(self.coords)


def parse_cif_atoms(path: str | Path) -> AtomArray:
    """Parse the ``_atom_site`` loop of an mmCIF file into an :class:`AtomArray`."""
    lines = Path(path).read_text().splitlines()
    fields: list[str] | None = None
    rows: list[list[str]] = []

    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        # Collect this loop's column headers.
        i += 1
        headers: list[str] = []
        while i < n and lines[i].lstrip().startswith("_"):
            headers.append(lines[i].strip().split()[0])
            i += 1
        if not any(h.startswith("_atom_site.") for h in headers):
            continue  # some other loop; keep scanning
        fields = [h.split(".", 1)[1] for h in headers]
        # Read data rows until the loop terminates.
        while i < n:
            s = lines[i].strip()
            if s in ("", "#", "loop_") or s.startswith("_"):
                break
            rows.append(s.split())
            i += 1
        break

    if fields is None:
        raise ValueError(f"no _atom_site loop found in {path}")

    idx = {f: k for k, f in enumerate(fields)}
    required = ("group_PDB", "label_atom_id", "label_comp_id", "label_asym_id",
                "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z")
    missing = [f for f in required if f not in idx]
    if missing:
        raise ValueError(f"{path}: _atom_site loop missing columns {missing}")

    atom_rows = [r for r in rows if r[idx["group_PDB"]] == "ATOM"]
    if not atom_rows:
        raise ValueError(f"{path}: no ATOM records")

    def column(name):
        k = idx[name]
        return [r[k] for r in atom_rows]

    b_key = "B_iso_or_equiv"
    return AtomArray(
        chain_id=np.array(column("label_asym_id")),
        res_seq=np.array([int(v) for v in column("label_seq_id")]),
        res_name=np.array(column("label_comp_id")),
        atom_name=np.array(column("label_atom_id")),
        coords=np.array([[float(r[idx["Cartn_x"]]), float(r[idx["Cartn_y"]]), float(r[idx["Cartn_z"]])]
                         for r in atom_rows]),
        b_factor=np.array([float(r[idx[b_key]]) for r in atom_rows]) if b_key in idx
        else np.zeros(len(atom_rows)),
    )


def ca_coords(atoms: AtomArray, chain: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(res_seq, coords)`` for Cα atoms, ordered by (chain, residue).

    ``chain=None`` concatenates all chains (useful for complex-wide RMSD).
    """
    mask = atoms.atom_name == "CA"
    if chain is not None:
        mask &= atoms.chain_id == chain
    if not mask.any():
        raise ValueError(f"no Cα atoms{'' if chain is None else f' for chain {chain}'}")
    order = np.lexsort((atoms.res_seq[mask], atoms.chain_id[mask]))
    return atoms.res_seq[mask][order], atoms.coords[mask][order]
