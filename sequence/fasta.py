"""FASTA file handling: tryptic digestion via PyOpenMS and decoy generation."""

import random

import pyopenms as pms


def tryptic_digest_pyopenms(
    file_path,
    min_len=5,
    max_len=50,
    missed_cleavages=2,
    decoy_method="reverse",
    decoy_prefix="rev_",
    seq_types=["original", "decoy"],
):
    """
    Perform in-silico tryptic digestion with decoy generation using PyOpenMS.

    Uses Trypsin/P enzyme (cleaves at K/R, not before P). For each protein,
    generates both original and decoy peptides. Decoys are created by reversing
    or scrambling the protein sequence before digestion.

    Args:
        file_path: Path to the FASTA file.
        min_len: Minimum peptide length to keep (default: 5).
        max_len: Maximum peptide length to keep (default: 50).
        missed_cleavages: Number of allowed missed cleavages (default: 2).
        decoy_method: "reverse" or "scramble" (default: "reverse").
        decoy_prefix: Prefix for decoy protein names (default: "rev_").
        seq_types: List of sequence types to generate (default: ["original", "decoy"]).

    Returns:
        List of tuples: (protein_name, start, end, "{protein}|{start}|{end}", peptide_sequence).
        Peptides containing "X" are filtered out.
    """
    # Read the FASTA file
    fasta = pms.FASTAFile()
    entries = []
    fasta.load(file_path, entries)

    # Set up the enzyme digestion
    digestor = pms.ProteaseDigestion()
    digestor.setEnzyme("Trypsin/P")
    digestor.setMissedCleavages(missed_cleavages)

    peptides = []
    for entry in entries:
        # Process both original and decoy sequences
        for seq_type in seq_types:
            if seq_type == "original":
                protein_sequence = str(entry.sequence)
            else:
                if decoy_method == "reverse":
                    protein_sequence = str(entry.sequence)[::-1]
                elif decoy_method == "scramble":
                    seq_list = list(str(entry.sequence))
                    random.shuffle(seq_list)
                    protein_sequence = "".join(seq_list)
                else:
                    raise ValueError(
                        "Invalid decoy method. Choose 'reverse' or 'scramble'."
                    )

            protein_name = entry.identifier.split()[
                0
            ]  # Adjust based on your FASTA format

            # Perform the tryptic digest
            result = []
            digestor.digest(pms.AASequence.fromString(protein_sequence), result)

            for peptide in result:
                peptide_sequence = str(peptide.toString())
                len_pep_seq = len(peptide_sequence)
                start = protein_sequence.find(peptide_sequence)
                end = start + len_pep_seq
                if "X" in peptide_sequence:
                    continue
                if len_pep_seq >= min_len and len_pep_seq <= max_len:
                    if seq_type == "original":
                        peptides.append(
                            (
                                protein_name,
                                start,
                                end,
                                f"{protein_name}|{start}|{end}",
                                peptide_sequence,
                            )
                        )
                    else:
                        peptides.append(
                            (
                                f"{decoy_prefix}{protein_name}",
                                start,
                                end,
                                f"{decoy_prefix}{protein_name}|{start}|{end}",
                                peptide_sequence,
                            )
                        )

    return peptides


def write_to_fasta(df, output_file="vectorized_output.fasta"):
    """
    Write a DataFrame to FASTA format.

    Args:
        df: Pandas DataFrame with "id" and "peptide" columns.
        output_file: Output FASTA file path (default: "vectorized_output.fasta").
    """
    # Combine 'id' and 'peptide' with a newline character
    fasta_series = ">" + df["id"] + "\n" + df["peptide"]

    # Join all rows with a newline character
    fasta_content = "\n".join(fasta_series)

    # Write the content to a file
    with open(output_file, "w") as fasta_file:
        fasta_file.write(fasta_content)
