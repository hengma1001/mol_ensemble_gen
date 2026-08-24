"""Configuration-ensemble generation with ESMFold2.

ESMFold2 is a diffusion-based, all-atom, multi-entity structure predictor. A
single ``fold`` call is stochastic, so an *ensemble* is produced by folding the
same input under many independent seeds (and, optionally, several diffusion
samples per seed) while sweeping the sampling knobs that control diversity.

Every member is attributed to the exact ``(seed, sampling params)`` that produced
it, so runs are reproducible and members are traceable.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # heavy deps imported lazily at call time (see _load_model / _builder)
    from esm.models.esmfold2 import StructurePredictionInput
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

# Fields StructurePredictionInput actually accepts; Boltz files may carry extras
# (constraints/properties/templates) that must be dropped or mapped, not passed.
_SPI_FIELDS = {"sequences", "pocket", "distogram_conditioning", "covalent_bonds"}


def derive_seed(base_seed: int, input_id: str, member_idx: int) -> int:
    """Deterministically derive a per-member seed.

    Stable across runs given the same ``(base_seed, input_id, member_idx)``, so an
    ensemble is reproducible and shardable across workers without coordination.
    """
    digest = hashlib.blake2b(f"{base_seed}:{input_id}:{member_idx}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)  # positive, int32-safe


@dataclass
class SamplingParams:
    """ESMFold2 ``fold`` knobs. ``None`` means "use the library default".

    Only non-``None`` values are passed to ``fold`` so unset knobs keep the
    model's defaults. These are the dials that turn a deterministic call into a
    diverse ensemble.
    """

    num_loops: int = 20
    num_sampling_steps: int = 200
    num_diffusion_samples: int = 1
    lm_dropout: float | None = None  # MC-dropout on the LM features
    lm_mask_pct: float | None = None  # sequence masking
    noise_scale: float | None = None  # diffusion temperature
    step_scale: float | None = None
    max_inference_sigma: float | None = None
    msa_max_depth: int | None = None
    msa_column_mask_rate: float | None = None

    def fold_kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EnsembleSpec:
    """How large an ensemble to draw and with what sampling."""

    members: int = 10  # number of independent seeds
    base_seed: int = 0
    sampling: SamplingParams = field(default_factory=SamplingParams)

    @property
    def size(self) -> int:
        """Total structures produced: one per (seed, diffusion sample)."""
        return self.members * self.sampling.num_diffusion_samples


@dataclass
class EnsembleMember:
    """One structure in the ensemble, with full provenance."""

    member_idx: int
    sample_idx: int
    seed: int
    cif_path: str
    plddt: float
    ptm: float
    iptm: float | None
    params: dict[str, Any]

    def as_row(self) -> dict[str, Any]:
        row = {k: v for k, v in asdict(self).items() if k != "params"}
        row.update({f"param_{k}": v for k, v in self.params.items()})
        return row


class ESMFold2Ensemble:
    """Generate configuration ensembles from a single ESMFold2 model."""

    def __init__(
        self,
        spec: EnsembleSpec,
        *,
        device: str = "cuda",
        model_name: str = "biohub/ESMFold2",
        model: ESMFold2Model | None = None,
    ) -> None:
        from esm.models.esmfold2 import ESMFold2InputBuilder
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        self.spec = spec
        self.device = device
        # Load once; the model is the expensive resource. Callers on a multi-GPU
        # executor pass a preloaded model so each worker loads a single time.
        self.model = model if model is not None else ESMFold2Model.from_pretrained(model_name).to(device).eval()
        self._builder = ESMFold2InputBuilder()

    # -- folding -----------------------------------------------------------

    def _fold(self, spi: StructurePredictionInput, seed: int) -> list:
        """Fold once and always return a list (``num_diffusion_samples`` may be >1)."""
        result = self._builder.fold(self.model, spi, seed=seed, **self.spec.sampling.fold_kwargs())
        return result if isinstance(result, list) else [result]

    @staticmethod
    def _confidence(result: Any) -> tuple[float, float, float | None]:
        plddt = float(result.plddt.mean())
        ptm = float(result.ptm)
        iptm = None if result.iptm is None else float(result.iptm)  # None for monomers
        return plddt, ptm, iptm

    # -- generation --------------------------------------------------------

    def fold_member(
        self, spi: StructurePredictionInput, input_id: str, member_idx: int, out_dir: str | Path
    ) -> list[EnsembleMember]:
        """Fold a single ensemble member (one seed → ``num_diffusion_samples`` CIFs).

        This is the unit of work shared by the single-GPU loop and the
        multi-GPU executor. Writes the member's ``.cif`` files but no manifest.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        kwargs = self.spec.sampling.fold_kwargs()
        seed = derive_seed(self.spec.base_seed, input_id, member_idx)

        records: list[EnsembleMember] = []
        for s, result in enumerate(self._fold(spi, seed)):
            cif_path = out_dir / f"{input_id}_m{member_idx:04d}_s{s:02d}.cif"
            cif_path.write_text(result.complex.to_mmcif())
            plddt, ptm, iptm = self._confidence(result)
            records.append(
                EnsembleMember(
                    member_idx=member_idx,
                    sample_idx=s,
                    seed=seed,
                    cif_path=str(cif_path),
                    plddt=plddt,
                    ptm=ptm,
                    iptm=iptm,
                    params={"seed": seed, **kwargs},
                )
            )
        return records

    def generate(self, spi: StructurePredictionInput, input_id: str, out_dir: str | Path) -> list[EnsembleMember]:
        """Draw the full ensemble for one input on this device and write it out.

        Writes one ``.cif`` per (member, diffusion sample) with a unique name,
        plus ``manifest.json`` (spec + provenance) and ``metadata.csv``.
        """
        out_dir = Path(out_dir)
        members: list[EnsembleMember] = []
        for m in range(self.spec.members):
            members.extend(self.fold_member(spi, input_id, m, out_dir))
        write_manifest(out_dir, input_id, self.spec, members)
        return members

    def build_spi(self, input_path: str | Path) -> tuple[StructurePredictionInput, str]:
        """Build a StructurePredictionInput from a .fasta / .pdb / .yaml file.

        Returns ``(spi, input_id)`` where ``input_id`` is the file stem. Used by
        both the CLI and the multi-GPU executor to reconstruct the input per
        worker without pickling model objects across processes.
        """
        path = Path(input_path)
        suffix = path.suffix.lower()
        if suffix in {".fasta", ".fa", ".faa"}:
            from esm.models.esmfold2 import StructurePredictionInput

            from .utils import fasta_to_scheme

            return StructurePredictionInput(**fasta_to_scheme(path)), path.stem
        if suffix in {".pdb", ".ent"}:
            from .pdb import parse_pdb_sequences

            return self._protein_spi(parse_pdb_sequences(path)), path.stem
        if suffix in {".yaml", ".yml"}:
            import yaml

            from .utils import update_boltz_schema

            with open(path) as f:
                schema = update_boltz_schema(yaml.safe_load(f))
            from esm.models.esmfold2 import StructurePredictionInput

            dropped = sorted(set(schema) - _SPI_FIELDS)
            if dropped:
                print(f"[ensemble] ignoring unsupported input keys: {dropped}")
            return StructurePredictionInput(**{k: v for k, v in schema.items() if k in _SPI_FIELDS}), path.stem
        raise ValueError(f"unsupported input '{input_path}': expected .fasta, .pdb, or .yaml")

    def generate_from_fasta(self, input_fasta: str | Path, out_dir: str | Path) -> list[EnsembleMember]:
        from esm.models.esmfold2 import StructurePredictionInput

        from .utils import fasta_to_scheme

        spi = StructurePredictionInput(**fasta_to_scheme(Path(input_fasta)))
        return self.generate(spi, Path(input_fasta).stem, out_dir)

    def generate_from_pdb(self, pdb_path: str | Path, out_dir: str | Path) -> list[EnsembleMember]:
        """Seed an ensemble from an experimental structure (uses its sequence only)."""
        from .pdb import parse_pdb_sequences

        sequences = parse_pdb_sequences(pdb_path)
        spi = self._protein_spi(sequences)
        return self.generate(spi, Path(pdb_path).stem, out_dir)

    @staticmethod
    def _protein_spi(sequences: dict[str, str]):
        """Build a protein-only StructurePredictionInput from {chain_id: sequence}."""
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

        return StructurePredictionInput(
            sequences=[ProteinInput(id=cid, sequence=seq) for cid, seq in sequences.items()]
        )

    def generate_from_yaml(self, input_yaml: str | Path, out_dir: str | Path) -> list[EnsembleMember]:
        import yaml
        from esm.models.esmfold2 import StructurePredictionInput

        from .utils import update_boltz_schema

        with open(input_yaml) as f:
            schema = update_boltz_schema(yaml.safe_load(f))
        # StructurePredictionInput only accepts a fixed field set; drop anything
        # else (constraints/properties/templates) rather than crash on it.
        dropped = sorted(set(schema) - _SPI_FIELDS)
        if dropped:
            print(f"[ensemble] ignoring unsupported input keys: {dropped}")
        spi = StructurePredictionInput(**{k: v for k, v in schema.items() if k in _SPI_FIELDS})
        return self.generate(spi, Path(input_yaml).stem, out_dir)


def write_manifest(out_dir: str | Path, input_id: str, spec: EnsembleSpec, members: list[EnsembleMember]) -> None:
    """Write ``manifest.json`` (spec + provenance) and ``metadata.csv``.

    Members are sorted by (member_idx, sample_idx) so output ordering is stable
    regardless of which worker produced them.
    """
    out_dir = Path(out_dir)
    members = sorted(members, key=lambda m: (m.member_idx, m.sample_idx))
    manifest = {
        "input_id": input_id,
        "spec": {
            "members": spec.members,
            "base_seed": spec.base_seed,
            "size": spec.size,
            "sampling": asdict(spec.sampling),
        },
        "members": [asdict(m) for m in members],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows = [m.as_row() for m in members]
    if rows:
        with open(out_dir / "metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
