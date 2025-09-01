import torch
import math
from utils import *
from gfn.actions import Actions
from gfn.states import DiscreteStates
from gfn.env import DiscreteEnv
from typing import Union, List

def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# --- mRNA Design Environment ---
class CodonDesignEnv(DiscreteEnv):
    """
    Environment for designing mRNA codon sequences for a given protein.
    Action space is the global codon set (size N_CODONS) plus an exit action.
    Dynamic masks restrict actions at each step:
    - At step t < seq_length: only synonymous codons for protein_seq[t] are allowed.
    - At step t == seq_length: only the exit action is allowed.

    """

    def __init__(
        self,
        protein_seq: str,
        device: torch.device,
        sf=None,
    ):

        self._device = device
        self.protein_seq = protein_seq
        self.seq_length = len(protein_seq)

        # Total possible actions = N_CODONS + 1 (for exit action)
        self.n_actions = N_CODONS + 1
        self.exit_action_index = N_CODONS  # Index for the exit action

        self.syn_indices = [get_synonymous_indices(aa) for aa in protein_seq]

        # Precompute GC counts for all codons
        self.codon_gc_counts = torch.tensor(
            [codon.count("G") + codon.count("C") for codon in ALL_CODONS],
            device=self._device,
            dtype=torch.float,
        )

        s0 = torch.full(
            (self.seq_length,), fill_value=-1, dtype=torch.long, device=self._device
        )
        sf = torch.full(
            (self.seq_length,), fill_value=0, dtype=torch.long, device=self._device
        )

        self.weights = torch.tensor([0.3, 0.3, 0.4]).to(device=self._device)

        super().__init__(
            n_actions=self.n_actions,
            s0=s0,
            state_shape=(self.seq_length,),
            action_shape=(1,),  # Each action is a single index
            sf=sf,
        )

        self.idx_to_codon = IDX_TO_CODON
        self.States: type[DiscreteStates] = self.States

    def set_weights(self, w: Union[list, torch.Tensor]):
        """
        Store the current preference weights (w) for conditional reward.
        """
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, dtype=torch.float32)

        if w is not None:
            self.weights = w

    def step(
        self,
        states,
        actions: Actions,
    ) -> DiscreteStates:

        states_tensor = states.tensor.to(self._device)
        batch_size = states_tensor.shape[0]
        current_length = (states_tensor != -1).sum(dim=1)

        max_length = states_tensor.shape[1]
        new_states = states_tensor.clone()
        valid_actions = actions.tensor.squeeze(-1)
        for i in range(batch_size):

            if (
                current_length[i].item() < max_length
                and valid_actions[i].item() != self.exit_action_index
            ):
                new_states[i, int(current_length[i].item())] = valid_actions[i].item()
        return self.States(new_states)

    def backward_step(
        self,
        states,
        actions: Actions,
    ) -> DiscreteStates:

        states_tensor = states.tensor
        batch_size, seq_len = states_tensor.shape
        current_length = (states_tensor != -1).sum(dim=1)
        new_states = states_tensor.clone()

        for i in range(batch_size):
            if current_length[i] > 0:
                new_states[i, current_length[i] - 1] = -1

        return self.States(new_states)

    def update_masks(self, states: DiscreteStates) -> None:

        states_tensor = states.tensor
        batch_size = states_tensor.shape[0]
        current_length = (states_tensor != -1).sum(dim=1)

        forward_masks = torch.zeros(
            (batch_size, self.n_actions), dtype=torch.bool, device=self._device
        )
        backward_masks = torch.zeros(
            (batch_size, self.n_actions - 1), dtype=torch.bool, device=self._device
        )

        for i in range(batch_size):

            # print(current_length[i])

            cl = current_length[i].item()

            # print(cl)

            if cl < self.seq_length:

                # Allow synonymous codons
                syns = self.syn_indices[int(cl)]
                forward_masks[i, syns] = True

            elif cl == self.seq_length:
                # Allow only exit action
                forward_masks[i, self.exit_action_index] = True

            # Backward masks
            if cl > 0:
                last_codon = states_tensor[i, int(cl) - 1].item()
                if last_codon >= 0:
                    backward_masks[i, int(last_codon)] = True

        # print(states_tensor.shape)

        states.forward_masks = forward_masks
        states.backward_masks = backward_masks

    def reward(
        self,
        states,
        gc_target: float = 0.50,    # target GC content (fraction)
        gc_width: float = 0.10,     # Gaussian width for GC reward
        mfe_min: float = -500.0,    # lower bound for MFE scaling (most negative)
        mfe_max: float = 0.0,       # upper bound for MFE scaling
        cai_min: float = 0.0,       # min CAI for scaling
        cai_max: float = 1.0,       # max CAI for scaling
    ) -> torch.Tensor:

        states_tensor = states.tensor
        device = states_tensor.device
        batch_size = states_tensor.shape[0]

        if isinstance(self.weights, torch.Tensor):
            try:
                weights_seq = self.weights.detach().cpu().tolist()
            except Exception:
                weights_seq = [float(x) for x in self.weights]
        else:
            weights_seq = list(self.weights)

        rewards: list[float] = []

        for i in range(batch_size):
            seq_indices = states_tensor[i]

            # compute_reward returns (reward_float, (gc, mfe, cai))
            r, _ = compute_reward(
                seq_indices,
                self.codon_gc_counts,
                weights=weights_seq,
                gc_target=gc_target,
                gc_width=gc_width,
                mfe_min=mfe_min,
                mfe_max=mfe_max,
                cai_min=cai_min,
                cai_max=cai_max,
            )
            rewards.append(float(r))

        if len(rewards) == 0:
            return torch.zeros((0,), device=device, dtype=torch.float32)

        combined = torch.tensor(rewards, device=device, dtype=torch.float32)

        combined = torch.where(torch.isfinite(combined), combined, torch.zeros_like(combined))
        combined = torch.clamp(combined, -1e6, 1e6)

        return combined


    def is_terminal(self, states: DiscreteStates) -> torch.Tensor:
        states_tensor = states.tensor
        current_length = (states_tensor != -1).sum(dim=1)
        return (current_length >= self.seq_length).bool()

    @staticmethod
    def make_sink_states_tensor(shape, device=None):
        return torch.zeros(shape, dtype=torch.long, device=device)



    # def reward(self, states) -> torch.Tensor:

    #     states_tensor = states.tensor
    #     batch_size = states_tensor.shape[0]

    #     gc_percents = []
    #     mfe_energies = []
    #     cai_scores = []

    #     # Process each sequence individually
    #     for i in range(batch_size):

    #         seq_indices = states_tensor[i]

    #         # Compute GC content
    #         gc_percent = compute_gc_content_vectorized(
    #             seq_indices, codon_gc_counts=self.codon_gc_counts
    #         )
    #         gc_percents.append(gc_percent)

    #         # Compute MFE
    #         mfe_energy = compute_mfe_energy(seq_indices)
    #         mfe_energies.append(mfe_energy)

    #         # Compute CAI
    #         cai_score = compute_cai(seq_indices)
    #         cai_scores.append(cai_score)

    #     device = states_tensor.device
    #     gc_percent = torch.tensor(gc_percents, device=device, dtype=torch.float32)
    #     mfe_energy = torch.tensor(mfe_energies, device=device, dtype=torch.float32)
    #     cai_score = torch.tensor(cai_scores, device=device, dtype=torch.float32)

    #     # Calculate weighted reward
    #     reward_components = torch.stack([gc_percent, -mfe_energy, cai_score], dim=-1)
    #     reward = (reward_components * self.weights.to(device)).sum(dim=-1)

    #     return reward

