from Levenshtein import distance as levenshtein_distance
from collections import Counter
from prettytable import PrettyTable
from utils import compute_gc_content_vectorized, compute_cai, compute_mfe_energy, codon_gc_counts, decode_sequence
import torch

def compute_identity(seq1, seq2):
    """Percentage of identical characters in the same positions."""
    length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return (matches / length) * 100

def compare_to_natural(generated_sequences, natural_sequence):
    results = []
    for i, seq in enumerate(generated_sequences):
        lev_dist = levenshtein_distance(seq, natural_sequence)
        identity = compute_identity(seq, natural_sequence)
        results.append({
            "Sequence": f"Generated {i+1}",
            "Levenshtein": lev_dist,
            "Identity %": round(identity, 2),
            "Length difference": abs(len(seq) - len(natural_sequence))
        })
    return results


def analyze_sequence_properties(seqs_tensor, natural_tensor, labels=None, out_path="sequence_comparison_table.txt"):

    table = PrettyTable()
    table.field_names = ["Seq", "GC %", "MFE", "CAI", "Levenshtein", "Identity %"]

    decoded_nat = decode_sequence(natural_tensor)

    for i, s in enumerate(seqs_tensor):
        gc = compute_gc_content_vectorized(s, codon_gc_counts).item()
        mfe = compute_mfe_energy(s).item()
        cai = compute_cai(s).item()

        decoded_s = decode_sequence(s)
        lev = levenshtein_distance(decoded_s, decoded_nat)
        identity = compute_identity(decoded_s, decoded_nat)

        label = labels[i] if labels and i < len(labels) else f"Gen {i+1}"

        table.add_row([label, f"{gc:.2f}", f"{mfe:.2f}", f"{cai:.2f}", lev, f"{identity:.2f}"])

    # Add natural sequence comparison
    gc = compute_gc_content_vectorized(natural_tensor, codon_gc_counts).item()
    mfe = compute_mfe_energy(natural_tensor).item()
    cai = compute_cai(natural_tensor).item()

    table.add_row(["Natural", f"{gc:.2f}", f"{mfe:.2f}", f"{cai:.2f}", 0, "100.00"])

    # Save to file
    with open(out_path, "w") as f:
        f.write(table.get_string())
