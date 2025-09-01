import torch
from env import N_CODONS
import torch.nn as nn
from torchgfn.src.gfn.states import States
from torchgfn.src.gfn.preprocessors import Preprocessor


class CodonSequencePreprocessor(Preprocessor):
    """Preprocessor for codon sequence states"""

    def __init__(self, seq_length: int, embedding_dim: int, device: torch.device):
        super().__init__(output_dim=seq_length * embedding_dim)
        self.device = device
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            N_CODONS + 1, embedding_dim, padding_idx=N_CODONS
        ).to(device)

    def preprocess(self, states: States) -> torch.Tensor:

        states_tensor = states.tensor.to(device=self.device, dtype=torch.long)
        states_tensor[states_tensor == -1] = N_CODONS

        embedded = self.embedding(states_tensor)
        out = embedded.view(states_tensor.shape[0], -1)

        return out

    # def preprocess(self, states: States) -> torch.Tensor:

    #     states_tensor = states.tensor.long().clone().to(self.device)
    #     states_tensor[states_tensor == -1] = N_CODONS
    #     embedded = self.embedding(states_tensor)
    #     out = embedded.view(states_tensor.shape[0], -1)
    #     out = out.to(self.embedding.weight.device)
    #     return out
