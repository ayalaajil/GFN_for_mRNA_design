import torch
import string
from typing import List, Dict, Tuple
from gfn.states import DiscreteStates
import numpy as np
import gfn
import random
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from MFE_calculator import RNAFolder
from CAI_calculator import CAICalculator

# --- Biological Constants ---

# Stop codons
STOP_CODONS: List[str] = ["UAA", "UAG", "UGA"]

# Codon table mapping amino acids (or stop *) to codons, example of protein sequence : ACDEFGHIKLMNPQ
CODON_TABLE : Dict[str, List[str]] = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'C': ['UGU', 'UGC'],
    'D': ['GAU', 'GAC'],
    'E': ['GAA', 'GAG'],
    'F': ['UUU', 'UUC'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],
    'H': ['CAU', 'CAC'],
    'I': ['AUU', 'AUC', 'AUA'],
    'K': ['AAA', 'AAG'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'M': ['AUG'],
    'N': ['AAU', 'AAC'],
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    'W': ['UGG'],
    'Y': ['UAU', 'UAC'],
    '*': ['UAA', 'UAG', 'UGA'],  # Stop codons
}

# Amino acid list 
AA_LIST: List[str] = list(CODON_TABLE.keys())
ALL_CODONS: List[str] = sorted(list(set(c for codons in CODON_TABLE.values() for c in codons)))
N_CODONS: int = len(ALL_CODONS)
CODON_TO_IDX: Dict[str, int] = {codon: idx for idx, codon in enumerate(ALL_CODONS)}
IDX_TO_CODON: Dict[int, str] = {idx: codon for codon, idx in CODON_TO_IDX.items()}

# --- Utility Functions ---
def get_synonymous_indices(amino_acid: str) -> List[int]:
    """
    Return the list of global codon indices that encode the given amino acid.
    Handles standard amino acids and '*'.
    """
    codons = CODON_TABLE.get(amino_acid, [])
    return [CODON_TO_IDX[c] for c in codons]

def compute_gc_content_vectorized(indices: torch.LongTensor, codon_gc_counts: torch.Tensor) -> torch.FloatTensor:
    """
    Vectorized GC content calculation using precomputed codon GC counts
    """
    gc_counts = codon_gc_counts[indices].sum(dim=1)
    total_nucleotides = indices.shape[1] * 3
    return gc_counts / total_nucleotides * 100

def mRNA_string_to_tensor(rna: string):

    rna_index = []
    for i in range(0,len(rna)-3,3):
        index = CODON_TO_IDX[rna[i:i+3]]
        rna_index.append(index)
    
    rna_tensor = torch.tensor(rna_index)
    return rna_tensor

def to_mRNA_string(rna_tensor : torch.Tensor):

    rna_string = ''
    for i in range(0,len(rna_tensor)):
        cd = IDX_TO_CODON[rna_tensor[i].item()]
        rna_string += cd

    return rna_string 

def compute_mfe_energy(indices: torch.LongTensor, energies=None, loop_min=4) -> torch.FloatTensor:
    """
    Compute the minimum free energy (MFE) of an RNA sequence using Zucker Algorithm.
    Input: indices (B, L) tensor of codon indices
    Output: Tensor of shape (B,) with MFE values
    """
    batch_size = indices.shape[0]
    mfe_energies = []

    for i in range(batch_size):

        rna_str = to_mRNA_string(indices[i])
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

    batch_size = indices.shape[0]
    cai_scores = []

    for i in range(batch_size):

        rna_str = to_mRNA_string(indices[i])
        try:

            calc = CAICalculator(rna_str)
            score=calc.compute_cai()

        except Exception as e:
            print(f"CAI computation failed for: {rna_str}, error: {e}")
            score = float('inf')

        cai_scores.append(score)

    return torch.tensor(cai_scores, dtype=torch.float32)

def mfe_score(mfe, min_mfe=-100, max_mfe=-10):
    """Map MFE into [0,1] range; low energy = high score"""
    
    clipped = max(min(mfe, max_mfe), min_mfe)
    return (max_mfe - clipped) / (max_mfe - min_mfe)


def gc_score(gc_percent, target=57.5, tol=5.0):
    """Return a GC score ∈ [0,1], highest when close to target GC%"""
    return max(0.0, 1.0 - abs(gc_percent - target) / tol)

# --- mRNA Design Environment ---
class CodonDesignEnv(DiscreteEnv):
    """
    Environment for designing mRNA codon sequences for a given protein.
    Action space is the global codon set (size N_CODONS) plus an exit action.
    Dynamic masks restrict actions at each step:
    - At step t < seq_length: only synonymous codons for protein_seq[t] are allowed.
    - At step t == seq_length: only the exit action is allowed.

    Rewards are the GC-content of the full generated sequence.
    """

    def __init__(
        self,
        protein_seq: str,
        sf=None,
        device: torch.device = torch.device('cpu'),):

        self.protein_seq = protein_seq
        self.seq_length = len(protein_seq)

        # Total possible actions = N_CODONS + 1 (for exit action)
        self.n_actions = N_CODONS + 1
        self.exit_action_index = N_CODONS # Index for the exit action

        self.syn_indices = [get_synonymous_indices(aa) for aa in protein_seq]

        # Precompute GC counts for all codons
        self.codon_gc_counts = torch.tensor([
            codon.count('G') + codon.count('C') for codon in ALL_CODONS
        ], device=device, dtype=torch.float)

        s0 = torch.full((self.seq_length,), fill_value=-1, dtype=torch.long, device=device)
        sf = torch.full((self.seq_length,), fill_value=0, dtype=torch.long, device=device)

        self.weights = torch.tensor([0.3, 0.3, 0.4])

        super().__init__(
            n_actions=self.n_actions,
            s0=s0,
            state_shape = (self.seq_length,),    
            action_shape=(1,), # Each action is a single index
            sf=sf,
        )

        self.idx_to_codon = IDX_TO_CODON


    def set_weights(self, w : list):
        """
        Store the current preference weights (ω) for conditional reward.
        """
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, dtype=torch.float32)

        self.weights = w

    def step(self,
            states: DiscreteStates, 
            actions: Actions,
        ) ->  DiscreteStates:   
            
        states_tensor = states.tensor
        batch_size = states_tensor.shape[0]
        current_length = (states_tensor!= -1).sum(dim=1)
    
        max_length = states_tensor.shape[1]
        new_states = states_tensor.clone()

        if isinstance(actions, Actions):
            valid_actions = actions.tensor.squeeze(-1)
        else:
            valid_actions = actions.squeeze(-1)

        for i in range(batch_size):
            if current_length[i].item() < max_length and valid_actions[i].item() != self.exit_action_index:
                new_states[i, current_length[i].item()] = valid_actions[i].item()
        
        # return self.States(new_states)
        return new_states

    def backward_step(
        self,
        states: DiscreteStates,
        actions: Actions,
    ) -> torch.Tensor:
        
        states_tensor = states.tensor
        batch_size, seq_len = states_tensor.shape
        current_length = (states_tensor != -1).sum(dim=1)
        new_states = states_tensor.clone()
        
        for i in range(batch_size):
            if current_length[i] > 0:
                new_states[i, current_length[i]-1] = -1 
                       
        # return self.States(new_states)
        return new_states


    def update_masks(self, states: DiscreteStates) -> None:

        states_tensor = states.tensor
        batch_size = states_tensor.shape[0]
        current_length = (states_tensor != -1).sum(dim=1)
        forward_masks = torch.zeros((batch_size, self.n_actions), 
                                   dtype=torch.bool, device=self.device)
        backward_masks = torch.zeros((batch_size, self.n_actions - 1), 
                                    dtype=torch.bool, device=self.device)
  
        for i in range(batch_size):

            cl = current_length[i].item()
            # Forward masks
            if cl < self.seq_length:
                # Allow synonymous codons
                syns = self.syn_indices[cl]
                forward_masks[i, syns] = True
            elif cl == self.seq_length:
                # Allow only exit action
                forward_masks[i, self.exit_action_index] = True
            # Backward masks
            if cl > 0:
                last_codon = states_tensor[i, cl - 1].item()
                if last_codon >= 0:
                    backward_masks[i, last_codon] = True

        states.forward_masks = forward_masks
        states.backward_masks = backward_masks

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:

        states_tensor = final_states.tensor
        valid_mask = states_tensor != -1
        valid_indices = torch.where(valid_mask, states_tensor, 0)

        gc_percent = compute_gc_content_vectorized(valid_indices, self.codon_gc_counts)
        mfe_energy = compute_mfe_energy(valid_indices)
        cai_score = compute_cai(valid_indices)

        reward_components = torch.stack([gc_percent, -mfe_energy, cai_score], dim=-1)  # shape: (batch, 3)
        reward = (reward_components * self.weights).sum(dim=-1)  # shape: (batch,)
        
        return reward

    def is_terminal(self, states: DiscreteStates) -> torch.BoolTensor:
        states_tensor = states
        current_length = (states_tensor != -1).sum(dim=1)
        return current_length >= self.seq_length
    
    @staticmethod
    def make_sink_states_tensor(shape, device=None):
        return torch.zeros(shape, dtype=torch.long, device=device)
    