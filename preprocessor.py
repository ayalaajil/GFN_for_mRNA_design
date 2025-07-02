import torch
from env import CodonDesignEnv, N_CODONS
import torch.nn as nn
from gfn.states import DiscreteStates, GraphStates, States
from gfn.preprocessors import Preprocessor


class CodonSequencePreprocessor(Preprocessor):
    """Preprocessor for codon sequence states"""
    def __init__(self, seq_length: int, embedding_dim: int, device: torch.device):
        super().__init__(output_dim=seq_length * embedding_dim)
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(N_CODONS + 1, embedding_dim, padding_idx=N_CODONS)    # maybe learnable
        self.device = device
        
    def preprocess(self, states: States) -> torch.Tensor:

        states_tensor = states.tensor.long().clone()
        states_tensor[states_tensor == -1] = N_CODONS
        embedded = self.embedding(states_tensor)

        out = embedded.view(states_tensor.shape[0], -1)
        return out