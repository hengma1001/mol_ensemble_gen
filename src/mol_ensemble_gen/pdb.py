"""Minimal, dependency-free PDB parsing.

ESMFold2 folds *from sequence*, so to seed an ensemble from an experimental
structure we only need the per-chain sequence. This reads it straight from the
``ATOM`` records (one residue per Cα), which keeps the step offline-testable and
free of a heavy structure-library import.
"""

from __future__ import annotations

from pathlib import Path

# Standard amino acids + a few common non-standard residues mapped to parents.
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "HYP": "P",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
    "CSO": "C",
    "HIE": "H",
    "HID": "H",
    "HIP": "H",
}


def parse_pdb_sequences(pdb_path: str | Path, *, unknown: str = "X") -> dict[str, str]:
    """Extract per-chain protein sequences from a PDB file.

    Reads the first model only (NMR/multi-model files), takes one residue per Cα
    atom, honours the primary altloc, and maps three-letter residue names to
    one-letter codes (unknown residues become ``unknown``).

    Returns
    -------
    dict[str, str]
        Ordered mapping of ``chain_id -> sequence`` (chains in first-seen order).
    """
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, int, str]] = set()  # (chain, resSeq, iCode) already recorded

    with open(pdb_path) as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "ENDMDL":
                break  # only the first model
            if record != "ATOM":
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            altloc = line[16]
            if altloc not in (" ", "A"):
                continue  # keep only the primary alternate location

            res_name = line[17:20].strip()
            chain_id = line[21].strip() or "A"
            res_seq = int(line[22:26])
            i_code = line[26]

            key = (chain_id, res_seq, i_code)
            if key in seen:
                continue
            seen.add(key)

            chains.setdefault(chain_id, []).append(_THREE_TO_ONE.get(res_name, unknown))

    if not chains:
        raise ValueError(f"no protein Cα atoms found in {pdb_path}")

    return {cid: "".join(residues) for cid, residues in chains.items()}
