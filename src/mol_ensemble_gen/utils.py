from pathlib import Path

from esm.utils.structure import input_builder


def update_boltz_schema(  # noqa: C901, PLR0915, PLR0912
    schema: dict,
) -> dict:
    """Parse a Boltz input yaml / json for esmfold2.

    The input file should be a dictionary with the following format:

    version: 1
    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
            msa: esmfold2.MSA
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
            msa: path/to/msa2.a3m
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
    # constraints:
    #     - bond:
    #         atom1: [A, 1, CA]
    #         atom2: [A, 2, N]
    #     - pocket:
    #         binder: E
    #         contacts: [[B, 1], [B, 2]]
    #         max_distance: 6
    #     - contact:
    #         token1: [A, 1]
    #         token2: [B, 1]
    #         max_distance: 6
    # templates:
    #     - cif: path/to/template.cif
    # properties:
    #     - affinity:
    #         binder: E

    Parameters
    ----------
    schema : dict
        The input schema.

    Returns
    -------
    scheme: dict
        The updated input schema, with the "sequences" field converted to a list of ProteinInput, DNAInput, RNAInput or LigandInput objects. The "version" field is removed if exists. The "constraints" and "properties" fields are not processed and returned as is, since they are not used in ESMFold2. The "templates" field is also not processed.

    """

    # First group items that have the same type, sequence and modifications
    schema.pop("version", None)  # Version is not needed for parsing, pop it out if exists
    schema["sequences"] = convert_dict_to_input(schema["sequences"])
    return schema


def convert_dict_to_input(sequence_dict: list) -> list:
    sequences = []
    for item in sequence_dict:
        # Get entity type
        entity_type = next(iter(item.keys())).lower()
        if entity_type not in {"protein", "dna", "rna", "ligand"}:
            msg = f"Invalid entity type: {entity_type}"
            raise ValueError(msg)

        if entity_type == "protein":
            sequence = item["protein"]
            sequences.append(input_builder.ProteinInput(**sequence))
        elif entity_type == "dna":
            sequence = item["dna"]
            sequences.append(input_builder.DNAInput(**sequence))
        elif entity_type == "rna":
            sequence = item["rna"]
            sequences.append(input_builder.RNAInput(**sequence))
        elif entity_type == "ligand":
            sequence = item["ligand"]
            sequences.append(input_builder.LigandInput(**sequence))

    return sequences


def fasta_to_scheme(input_fasta: Path) -> dict:
    """Convert a fasta file to a Boltz input schema.
    only supports protein sequences for now.

    The fasta file should have the following format:

    >A
    MADQLTEEQIAEFKEAFSLF
    >B
    AKLSILPWGHC

    Parameters
    ----------
    input_fasta : str
        The path to the input fasta file.

    Returns
    -------
    scheme: dict
        The input schema, with the "sequences" field converted to a list of ProteinInput objects. The "constraints", "properties" and "templates" fields are not included since they are tricky to define in fasta format.

    """

    sequences = []
    with open(input_fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                id = line[1:].strip()
                sequence = next(f).strip()
                sequences.append(input_builder.ProteinInput(id=id, sequence=sequence))

    return {"sequences": sequences}
