from MFE_calculator import RNAFolder
from CAI_calculator import CAICalculator
import torch
import yaml
import numpy as np
from types import SimpleNamespace
from typing import List, Dict
import random
import os
from datetime import datetime
from torchgfn.src.gfn.modules import ScalarEstimator
from torchgfn.src.gfn.utils.modules import MLP



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# --Helper: Tokenize codon string to LongTensor Indices
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
    gc_content = (gc_counts / total_nucleotides) * 100

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
        energy = 0  # Default MFE value

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
        score = 0
        
    cai_scores.append(score)
    return torch.tensor(cai_scores, dtype=torch.float32).to(device)


def compute_reward_components(state, codon_gc_counts):
    gc_content = compute_gc_content_vectorized(state, codon_gc_counts).item()
    mfe_energy = compute_mfe_energy(state).item()
    cai_score = compute_cai(state).item()
    return gc_content, mfe_energy, cai_score


def create_output_directory(args, config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    experiment_type = "conditional" if hasattr(args, 'conditional') and args.conditional else "unconditional"
    protein_size = getattr(config, 'type', 'unknown')
    run_name = args.run_name if args.run_name else config.run_name
    output_dir = f"outputs/{experiment_type}/{protein_size}/{run_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, timestamp


def create_experiment_summary(args, config, output_dir, timestamp, experiment_type, protein_size, device):
    """Create experiment summary file with all relevant parameters and generated files."""
    summary_file = f"{output_dir}/experiment_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Experiment Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Experiment Type: {experiment_type}\n")
        f.write(f"Protein Size: {protein_size}\n")
        f.write(f"Run Name: {args.run_name}\n")
        f.write(f"Architecture: {getattr(config, 'arch', 'MLP')}\n")
        f.write(f"Protein Sequence Length: {len(config.protein_seq)}\n")
        f.write(f"Training Iterations: {args.n_iterations}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Hidden Dimension: {args.hidden_dim}\n")
        f.write(f"Number of Hidden Layers: {args.n_hidden}\n")
        f.write(f"SubTB Lambda: {args.subTB_lambda}\n")
        f.write(f"Epsilon: {args.epsilon}\n")
        f.write(f"WandB Project: {config.wandb_project}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nProtein Sequence: {config.protein_seq}\n")
        f.write(f"Natural mRNA Sequence: {config.natural_mRNA_seq}\n")
        f.write(f"\nGenerated Files:\n")
        f.write(f"- experiment_summary.txt (this file)\n")
        f.write(f"- trained_gflownet_{args.run_name}_{timestamp}.pth (model weights)\n")
        f.write(f"- generated_sequences_{timestamp}.txt (generated sequences)\n")
        f.write(f"- metric_distributions_{timestamp}.png (metric histograms)\n")
        f.write(f"- pareto_scatter_{timestamp}.png (Pareto front)\n")
        f.write(f"- cai_vs_mfe_{timestamp}.png (CAI vs MFE plot)\n")
        f.write(f"- gc_vs_mfe_{timestamp}.png (GC vs MFE plot)\n")
        f.write(f"- comprehensive_comparison_{timestamp}.txt (comparison table)\n")
        f.write(f"- metrics_summary_{timestamp}.txt (detailed metrics)\n")
        f.write(f"- enhanced_diversity_analysis_{timestamp}.png (diversity plots)\n")


def set_up_logF_estimator(
    args, preprocessor, pf_module
):
    """Returns a LogStateFlowEstimator."""

    module = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=1,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden,
            trunk=(
                pf_module.trunk
                if args.tied and isinstance(pf_module.trunk, torch.nn.Module)
                else None
            ),
        )

    return ScalarEstimator(module=module, preprocessor=preprocessor)



# ----------------------------- Protein Sequence Encoding ----------------------------
def encode_protein_sequence(protein_seq: str, device: torch.device) -> torch.Tensor:

    # Amino acid to index mapping
    aa_to_idx = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '*': 20
    }

    one_hot_seq = torch.zeros(21, device=device)  # 20 amino acids + stop codon

    for aa in protein_seq:
        if aa in aa_to_idx:
            idx = aa_to_idx[aa]
            one_hot_seq[idx] = 1.0


    return one_hot_seq

