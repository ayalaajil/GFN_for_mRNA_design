from env import to_mRNA_string
from MFE_calculator import RNAFolder
from CAI_calculator import CAICalculator
import torch
import yaml
from types import SimpleNamespace

def load_config(path: str):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)


def compute_gc_content_vectorized(indices: torch.LongTensor, codon_gc_counts: torch.Tensor) -> torch.FloatTensor:
    """
    Vectorized GC content calculation using precomputed codon GC counts
    """
    gc_counts = codon_gc_counts[indices].sum(dim=0)
    total_nucleotides = indices.shape[0] * 3
    gc_content = gc_counts / total_nucleotides * 100

    return gc_content

def compute_mfe_energy(indices: torch.LongTensor, energies=None, loop_min=4) -> torch.FloatTensor:
    """
    Compute the minimum free energy (MFE) of an RNA sequence using Zucker Algorithm.
    """
    mfe_energies = []
    rna_str = to_mRNA_string(indices)
    try:
            sol = RNAFolder(energies=energies, loop_min=loop_min)
            s = sol.solve(rna_str)
            energy = s.energy()
    except Exception as e:
            print(f"Energy computation failed for: {rna_str}, error: {e}")
            energy = float('inf')

    mfe_energies.append(energy)
    return torch.tensor(mfe_energies, dtype=torch.float32)


def compute_cai(indices: torch.LongTensor, energies=None, loop_min=4) -> torch.FloatTensor:
    cai_scores = []
    rna_str = to_mRNA_string(indices)
    try:
            calc = CAICalculator(rna_str)
            score=calc.compute_cai()
    except Exception as e:
            print(f"CAI computation failed for: {rna_str}, error: {e}")
            score = float('inf')
    cai_scores.append(score)
    return torch.tensor(cai_scores, dtype=torch.float32)

def compute_reward_components(state, codon_gc_counts):
    gc_content = compute_gc_content_vectorized(state, codon_gc_counts).item()
    mfe_energy = compute_mfe_energy(state).item()
    cai_score = compute_cai(state).item()
    return gc_content, mfe_energy, cai_score

def compute_reward(state, codon_gc_counts, weights):
    gc, mfe, cai = compute_reward_components(state, codon_gc_counts)
    reward = sum(w * r for w, r in zip(weights, [gc, -mfe, cai]))    # weighted sum
    return reward, (gc, mfe, cai)


