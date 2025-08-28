from MFE_calculator import RNAFolder
from CAI_calculator import CAICalculator
import torch
import yaml
import numpy as np
from types import SimpleNamespace
from typing import List, Tuple, Optional, Dict, Any, Sequence
import math

# --- Biological Constants ---

# Stop codons
STOP_CODONS: List[str] = ["UAA", "UAG", "UGA"]

# Codon table mapping amino acids (or stop *) to codons, example of protein sequence : ACDEFGHIKLMNPQ
CODON_TABLE: Dict[str, List[str]] = {
    "A": ["GCU", "GCC", "GCA", "GCG"],
    "C": ["UGU", "UGC"],
    "D": ["GAU", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["UUU", "UUC"],
    "G": ["GGU", "GGC", "GGA", "GGG"],
    "H": ["CAU", "CAC"],
    "I": ["AUU", "AUC", "AUA"],
    "K": ["AAA", "AAG"],
    "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "M": ["AUG"],
    "N": ["AAU", "AAC"],
    "P": ["CCU", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
    "T": ["ACU", "ACC", "ACA", "ACG"],
    "V": ["GUU", "GUC", "GUA", "GUG"],
    "W": ["UGG"],
    "Y": ["UAU", "UAC"],
    "*": ["UAA", "UAG", "UGA"],  # Stop codons
}


# Amino acid list
AA_LIST: List[str] = list(CODON_TABLE.keys())
ALL_CODONS: List[str] = sorted(
    list(set(c for codons in CODON_TABLE.values() for c in codons))
)
N_CODONS: int = len(ALL_CODONS)
CODON_TO_IDX: Dict[str, int] = {codon: idx for idx, codon in enumerate(ALL_CODONS)}
IDX_TO_CODON: Dict[int, str] = {idx: codon for codon, idx in CODON_TO_IDX.items()}


codon_gc_counts = torch.tensor(
    [codon.count("G") + codon.count("C") for codon in ALL_CODONS], dtype=torch.float
)


def dna_to_mrna(dna: str) -> str:
    """Convert a DNA sequence to an mRNA sequence by replacing T with U."""
    dna = dna.upper().replace(" ", "")
    mrna = dna.replace("T", "U")
    return mrna


def codon_idx_to_aa(codon_idx: int) -> str:
    """
    Given a codon index (0 <= codon_idx < N_CODONS),
    return the corresponding amino acid (including '*' for stop codons).
    """
    codon = IDX_TO_CODON[codon_idx]
    for aa, codons in CODON_TABLE.items():
        if codon in codons:
            return aa
    raise ValueError(f"Codon {codon} not found in CODON_TABLE.")


def extract_sequence_from_fasta(fasta_path: str) -> str:

    with open(fasta_path, "r") as f:
        lines = f.readlines()
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    return sequence


# --Helper: Tokenize codon string to LongTensor
def tokenize_sequence_to_tensor(seq):
    codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
    indices = [CODON_TO_IDX[c] for c in codons if c in CODON_TO_IDX]
    return torch.tensor(indices, dtype=torch.long)


def decode_sequence(tensor_seq):
    return "".join([IDX_TO_CODON[int(i)] for i in tensor_seq])


# --- Utility Functions ---
def get_synonymous_indices(amino_acid: str) -> List[int]:
    """
    Return the list of global codon indices that encode the given amino acid.
    Handles standard amino acids and '*'.
    """
    codons = CODON_TABLE.get(amino_acid, [])
    return [CODON_TO_IDX[c] for c in codons]


def mRNA_string_to_tensor(rna: str):

    rna_index = []
    for i in range(0, len(rna) - 3, 3):
        index = CODON_TO_IDX[rna[i : i + 3]]
        rna_index.append(index)

    rna_tensor = torch.tensor(rna_index)
    return rna_tensor


def to_mRNA_string(rna_tensor: torch.Tensor):

    rna_string = ""
    for i in range(0, len(rna_tensor)):
        cd = IDX_TO_CODON[int(rna_tensor[i].item())]
        rna_string += cd

    return rna_string


def load_config(path: str):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)


def compute_gc_content_vectorized(
    indices: torch.Tensor, codon_gc_counts: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized GC content calculation using precomputed codon GC counts
    """
    device = indices.device
    gc_counts = codon_gc_counts[indices].sum(dim=0)
    total_nucleotides = indices.shape[0] * 3
    gc_content = gc_counts / total_nucleotides * 100

    return gc_content.to(device)


def compute_mfe_energy(
    indices: torch.Tensor, energies=None, loop_min=4
) -> torch.Tensor:
    """
    Compute the minimum free energy (MFE) of an RNA sequence using Zucker Algorithm.
    """
    device = indices.device
    mfe_energies = []
    rna_str = to_mRNA_string(indices)
    try:
        sol = RNAFolder(energies=energies, loop_min=loop_min)
        s = sol.solve(rna_str)
        energy = s.energy()
    except Exception as e:
        print(f"Energy computation failed for: {rna_str}, error: {e}")
        energy = float("inf")

    mfe_energies.append(energy)
    return torch.tensor(mfe_energies, dtype=torch.float32).to(device)


def compute_cai(indices: torch.Tensor, energies=None, loop_min=4) -> torch.Tensor:

    device = indices.device
    cai_scores = []
    rna_str = to_mRNA_string(indices)
    try:
        calc = CAICalculator(rna_str)
        score = calc.compute_cai()
    except Exception as e:
        print(f"CAI computation failed for: {rna_str}, error: {e}")
        score = float("inf")
    cai_scores.append(score)
    return torch.tensor(cai_scores, dtype=torch.float32).to(device)


def compute_reward_components(state, codon_gc_counts):
    gc_content = compute_gc_content_vectorized(state, codon_gc_counts).item()
    mfe_energy = compute_mfe_energy(state).item()
    cai_score = compute_cai(state).item()
    return gc_content, mfe_energy, cai_score

def compute_reward(
    state,
    codon_gc_counts,
    weights: Sequence[float],
    *,
    gc_target: float = 0.50,    # target GC content (fraction)
    gc_width: float = 0.10,     # Gaussian width for GC reward
    mfe_min: float = -500.0,    # lower bound for MFE scaling (most negative)
    mfe_max: float = 0.0,       # upper bound for MFE scaling
    cai_min: float = 0.0,       # min CAI for scaling
    cai_max: float = 1.0,       # max CAI for scaling
) -> Tuple[float, Tuple[float, float, float]]:

    gc_val, mfe_val, cai_val = compute_reward_components(state, codon_gc_counts)

    # --- normalize---
    # 1) GC reward: gaussian around gc_target
    if gc_width == 0:
        gc_reward = 1.0 if math.isclose(gc_val, gc_target) else 0.0
    else:
        gc_reward = math.exp(-0.5 * ((gc_val - gc_target) / gc_width) ** 2)

    # 2) MFE reward: scaled to [0,1] with clipping then inverted (lower MFE -> higher reward)
    mfe_span = mfe_max - mfe_min
    if mfe_span == 0:
        mfe_scaled = 0.0
    else:
        mfe_scaled = (mfe_val - mfe_min) / mfe_span
    mfe_scaled = float(np.clip(mfe_scaled, 0.0, 1.0))
    mfe_reward = 1.0 - mfe_scaled

    # 3) CAI reward: scaled to [0,1] with clipping (higher CAI -> higher reward)
    cai_span = cai_max - cai_min
    if cai_span == 0:
        cai_reward = 0.0
    else:
        cai_reward = float(np.clip((cai_val - cai_min) / cai_span, 0.0, 1.0))

    # --- weights and final reward ---
    if len(weights) != 3:
        raise ValueError("weights must be length-3 for [gc, mfe, cai]")

    w = [float(wi) for wi in weights]
    comp_rewards = [gc_reward, mfe_reward, cai_reward]
    reward = float(w[0] * comp_rewards[0] + w[1] * comp_rewards[1] + w[2] * comp_rewards[2])

    if not math.isfinite(reward):
        reward = float(np.clip(reward, -1e9, 1e9))

    return reward, (gc_val, mfe_val, cai_val)





















# def compute_reward(
#     state,
#     codon_gc_counts,
#     weights: Sequence[float],
#     *,
#     clip: Tuple[float, float] = (-100.0, 100.0),
#     normalize: Optional[str] = None,
#     stats: Optional[Dict[str, Any]] = None,
# ) -> Tuple[float, Tuple[float, float, float]]:
#     """
#     Compute weighted reward from components for a single state.

#     Args:
#       state: sequence representation (tensor / list / numpy) accepted by your compute_* helpers.
#       codon_gc_counts: used by compute_gc_content_vectorized.
#       weights: sequence-like of length 3 for [gc, -mfe, cai] respectively.
#       clip: (min, max) clipping applied to each component before weighting.
#       normalize: optional normalization mode: None | "zscore" | "minmax".
#         If specified, `stats` must be provided with required fields:
#           - for "zscore": stats = {"means": (m_gc, m_mfe_inv, m_cai), "stds": (s_gc, s_mfe_inv, s_cai)}
#           - for "minmax": stats = {"mins": (min_gc, min_mfe_inv, min_cai), "maxs": (...)}
#       stats: optional dict used by normalization.

#     Returns:
#       (reward, (gc, mfe, cai)) where reward is a plain Python float and
#       components are the raw (gc, mfe, cai) values (mfe is raw negative energy).
#     """
#     gc, mfe, cai = compute_reward_components(state, codon_gc_counts)

#     comp = [float(gc), float(-mfe), float(cai)]

#     min_clip, max_clip = clip
#     comp = [max(min(c, max_clip), min_clip) for c in comp]

#     if normalize is not None:
#         if stats is None:
#             raise ValueError("stats must be provided when normalize is set")
#         if normalize == "zscore":
#             means = stats.get("means")
#             stds = stats.get("stds")
#             if means is None or stds is None:
#                 raise ValueError("stats must contain 'means' and 'stds' for zscore normalization")
#             comp = [(c - m) / (s if s != 0.0 else 1.0) for c, m, s in zip(comp, means, stds)]
#         elif normalize == "minmax":
#             mins = stats.get("mins")
#             maxs = stats.get("maxs")
#             if mins is None or maxs is None:
#                 raise ValueError("stats must contain 'mins' and 'maxs' for minmax normalization")
#             comp = [
#                 (c - mn) / (mx - mn) if (mx - mn) != 0.0 else 0.0
#                 for c, mn, mx in zip(comp, mins, maxs)
#             ]
#         else:
#             raise ValueError(f"Unknown normalize mode: {normalize}")

#     if len(weights) != 3:
#         raise ValueError("weights must be length-3 for [gc, -mfe, cai]")

#     try:
#         w = [float(x) for x in weights]
#     except Exception:
#         w = [float(torch.as_tensor(weights[i]).item()) if 'torch' in globals() else float(weights[i]) for i in range(3)]

#     reward = sum(w_i * c_i for w_i, c_i in zip(w, comp))

#     if not (isinstance(reward, (float, int)) and math.isfinite(float(reward))):
#         reward = float(max(min(reward, max_clip), min_clip))

#     return float(reward), (gc, mfe, cai)