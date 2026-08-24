from dataclasses import dataclass
from pathlib import Path

import yaml
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

from .utils import fasta_to_scheme, update_boltz_schema


@dataclass
class ESMFold2config:
    num_loops: int = 3
    num_sampling_steps: int = 50
    num_diffusion_samples: int = 1
    seed: int = 0


class ESMFold2Model_API(object):
    def __init__(self, config: ESMFold2config):
        # super().__init__(config)
        self.config = config

        self.model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()

    def predict_structure(self, input_spi: StructurePredictionInput):

        result = ESMFold2InputBuilder().fold(self.model, input_spi, **self.config.__dict__)

        return result

    def predict_structure_from_yml_file(self, input_yaml: Path):

        with open(input_yaml, "r") as f:
            schema = yaml.safe_load(f)

        schema = update_boltz_schema(schema)
        input_spi = StructurePredictionInput(**schema)

        return self.predict_structure(input_spi)

    def predict_structure_from_fasta(self, input_fasta: Path):

        schema = fasta_to_scheme(input_fasta)
        input_spi = StructurePredictionInput(**schema)

        return self.predict_structure(input_spi)

    def write_structure_to_cif(self, structure, output_cif: Path):
        with open(output_cif, "w") as f:
            f.write(structure.complex.to_mmcif())

    def parse_output(self, structure, output_dir: Path) -> dict:
        if output_dir is not None:
            self.write_structure_to_cif(structure, output_dir / "predicted_structure.cif")

        return {
            "structure": str(output_dir / "predicted_structure.cif"),
            "plddt": structure.plddt.mean(),
            "ptm": structure.ptm,
            "iptm": structure.iptm,
        }
