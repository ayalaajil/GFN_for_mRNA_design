import torch
from utils import (
    N_CODONS,
    CODON_TABLE,
    ALL_CODONS,
    IDX_TO_CODON,
    compute_gc_content_vectorized,
    compute_mfe_energy,
    compute_cai,
    get_synonymous_indices,
    codon_idx_to_aa,
)
from torchgfn.src.gfn.actions import Actions
from torchgfn.src.gfn.states import DiscreteStates
from torchgfn.src.gfn.env import DiscreteEnv
from typing import Union


class MutationDesignEnv(DiscreteEnv):
    """
    Environment that starts from a given mRNA (codon indices) and
    lets the agent iteratively mutate it by synonymous codon replacements.
    """

    def __init__(
        self,
        natural_seq_indices: torch.LongTensor,  # shape (L,)
        device: torch.device,
        weights: Union[list, torch.Tensor],
    ):

        self._device = device
        self.natural = natural_seq_indices.to(device)  # tensor of length L
        self.seq_length = self.natural.size(0)

        # Precompute, for each position, the list of synonymous codon indices (including the “current” one, though we’ll mask that out)

        self.syn_indices_per_pos = []

        for codon_idx in self.natural.tolist():
            aa_syns = get_synonymous_indices(
                codon_idx_to_aa(codon_idx)
            )  # synonyms for that amino acid
            self.syn_indices_per_pos.append(aa_syns)

        # Total possible actions = N_CODONS + 1 (for exit action)
        self.n_actions = N_CODONS + 1
        self.exit_action_index = N_CODONS

        # initial and sink states
        s0 = self.natural.clone().unsqueeze(0)  # shape (1, L), batch dim=1
        sf = torch.zeros_like(s0)  # not really used

        self.weights = torch.tensor([0.3, 0.3, 0.4]).to(device=self._device)

        super().__init__(
            n_actions=self.n_actions,
            s0=s0,
            state_shape=(self.seq_length,),
            action_shape=(1,),
            sf=sf,
        )

        self.idx_to_codon = IDX_TO_CODON
        self.States: type[DiscreteStates] = self.States

    # def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:

    #     st = states.tensor.to(self.device).clone()
    #     for i, a in enumerate(actions.tensor.view(-1)):
    #         pos, codon = self._decode_action(int(a.item()))
    #         if pos is None:
    #             # exit action: we could mark terminal, but DiscreteEnv uses is_terminal()
    #             continue
    #         # apply the mutation
    #         st[i, pos] = codon
    #     return self.States(st)
