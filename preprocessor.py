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




class CodonSequencePreprocessor2(Preprocessor):
    """
    Preprocessor that supports variable-length states by padding/truncating to
    a fixed `max_seq_length`. Returns:
      - embeddings: Tensor of shape (batch, max_seq_length * embedding_dim)
      - mask:        Bool tensor of shape (batch, max_seq_length) where True=valid position
    """

    def __init__(
        self,
        max_seq_length: int,
        embedding_dim: int,
        device: torch.device,
        use_positional: bool = True,
    ):
        # output_dim = max_seq_length * embedding_dim (flattened)
        super().__init__(output_dim=max_seq_length * embedding_dim)
        self.device = device
        self.max_seq_length = int(max_seq_length)
        self.embedding_dim = int(embedding_dim)

        # token embedding: N_CODONS real codons + 1 padding token at index N_CODONS
        self.embedding = nn.Embedding(N_CODONS + 1, embedding_dim, padding_idx=N_CODONS).to(device)

        # optional learned positional embeddings so model knows positions
        self.use_positional = use_positional
        if self.use_positional:
            self.pos_embedding = nn.Embedding(self.max_seq_length, embedding_dim).to(device)
        else:
            self.pos_embedding = None

    def _pad_or_truncate(self, states_tensor: torch.LongTensor) -> torch.LongTensor:
        """
        states_tensor: (B, L) where L = env.seq_length (may be <= max_seq_length)
        Returns a tensor of shape (B, max_seq_length) with padding idx N_CODONS.
        """
        b, L = states_tensor.shape
        if L > self.max_seq_length:
            # Prefer explicit failure so you know your max length is too small
            raise ValueError(f"state length L={L} > max_seq_length={self.max_seq_length}")

        if L == self.max_seq_length:
            return states_tensor

        # create padded tensor filled with padding index
        padded = torch.full(
            (b, self.max_seq_length),
            fill_value=N_CODONS,
            dtype=torch.long,
            device=states_tensor.device,
        )
        padded[:, :L] = states_tensor
        return padded

    def preprocess(self, states: States):
        """
        Input:
          states.tensor: (B, L) with entries in {0..N_CODONS-1} and -1 for empty
        Returns:
          out_flat: Tensor (B, max_seq_length * embedding_dim)  -- flattened embeddings
          mask:     Bool Tensor (B, max_seq_length) -- True where there is a valid codon (not padding)
        """

        # move to device and cast
        states_tensor = states.tensor.to(device=self.device, dtype=torch.long)

        # Replace -1 (empty slots) with padding idx so embedding padding_idx works
        states_tensor = states_tensor.clone()
        states_tensor[states_tensor == -1] = N_CODONS

        # pad or truncate to max_seq_length
        states_tensor = self._pad_or_truncate(states_tensor)  # (B, max_seq_length)

        # mask: valid positions are those != padding idx
        mask = (states_tensor != N_CODONS)

        # embed tokens: (B, max_seq_length, embedding_dim)
        embedded = self.embedding(states_tensor)

        # add positional embeddings if requested
        if self.use_positional:
            # pos ids [0..max_seq_length-1]: (max_seq_length,)
            pos_ids = torch.arange(self.max_seq_length, device=self.device).unsqueeze(0)  # (1, max_seq_length)
            pos_emb = self.pos_embedding(pos_ids)  # (1, max_seq_length, embedding_dim)
            embedded = embedded + pos_emb  # broadcast over batch

        # flatten: (B, max_seq_length * embedding_dim)
        out_flat = embedded.view(embedded.shape[0], -1)

        return out_flat
