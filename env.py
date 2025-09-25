import torch
from utils import *
from gfn.actions import Actions
from gfn.states import DiscreteStates
from gfn.env import DiscreteEnv
from typing import Union
from reward import compute_simple_reward

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
        max_seq_length: int = None
    ):

        self._device = device

        self.protein_seq = protein_seq
        self.seq_length = len(protein_seq)

        # Total possible actions = N_CODONS + 1 (for exit action)
        self.n_actions = N_CODONS + 1
        self.exit_action_index = N_CODONS

        self.syn_indices = [get_synonymous_indices(aa) for aa in protein_seq]

        # Precompute GC counts for all codons
        self.codon_gc_counts = torch.tensor(
            [codon.count("G") + codon.count("C") for codon in ALL_CODONS],
            device=self._device,
            dtype=torch.float,
        )


        # Use fixed-size state tensors of length max_seq_length
        s0 = torch.full((self.seq_length,), fill_value=-1, dtype=torch.long, device=self._device)
        sf = torch.full((self.seq_length,), fill_value=0, dtype=torch.long, device=self._device)

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

        w_t = torch.tensor(w, dtype=torch.float32, device=self._device) if not isinstance(w, torch.Tensor) else w.to(self._device)
        self.weights = w_t

    def step(
        self,
        states,
        actions: Actions,
    ) -> DiscreteStates:

        states_tensor = states.tensor.to(self._device)
        batch_size = states_tensor.shape[0]
        current_length = (states_tensor != -1).sum(dim=1)

        new_states = states_tensor.clone()
        valid_actions = actions.tensor.squeeze(-1)
        for i in range(batch_size):

            if (
                current_length[i].item() < self.seq_length
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
        batch_size = states_tensor.shape[0]
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

            cl = current_length[i].item()

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

        states.forward_masks = forward_masks
        states.backward_masks = backward_masks

    def reward(self, states) -> torch.Tensor:

        if isinstance(states, torch.Tensor):
            states_tensor = states
        else:
            states_tensor = states.tensor

        device = states_tensor.device
        batch_size = states_tensor.shape[0]

        w = self.weights

        rewards: list[float] = []

        for i in range(batch_size):

            w_i = w[i] if w.dim() == 2 else w
            seq_indices = states_tensor[i]

            current_length = (seq_indices != -1).sum().item()

            if current_length != self.seq_length:
                rewards.append(0)
            else:

                valid_codons = seq_indices[seq_indices != -1]
                r, _ = compute_simple_reward(
                    valid_codons,
                    self.codon_gc_counts,
                    weights=w_i,
                )
                rewards.append(float(r))

        if len(rewards) == 0:
            return torch.zeros((0,), device=device, dtype=torch.float32)

        return torch.tensor(rewards, device=device, dtype=torch.float32)

    def is_terminal(self, states: DiscreteStates) -> torch.Tensor:
        states_tensor = states.tensor
        return ((states_tensor != -1).sum(dim=1) == self.seq_length).bool()

    @staticmethod
    def make_sink_states_tensor(shape, device=None):
        return torch.zeros(shape, dtype=torch.long, device=device)


