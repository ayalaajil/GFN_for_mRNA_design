import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
from tensordict import TensorDict
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.nn import DirGNNConv, GCNConv, GINConv

from gfn.actions import GraphActionType

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal

#from torchgfn.src.gfn.utils.ENN_module import input_dim


class GaussianIndexer:

    def __init__(self, index_dim: int):
        self.index_dim = index_dim

    def __call__(self, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            np.random.seed(seed)
        z = np.random.randn(self.index_dim)  # N(0, I)
        return torch.tensor(z, dtype=torch.float32)




class MLP_ENN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        index_dim: int = 6,
        hidden_dim: int = 256,
        epinet_hidden_dim: int = 128,
        epinet_layers: int = 2,
        trunk: Optional[nn.Module] = None,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh", "elu"]] = "relu",
        add_layer_norm: bool = False,
        prior_scale: float = 1,
        stop_gradient: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.index_dim = index_dim
        self.stop_gradient = stop_gradient
        self.prior_scale = prior_scale
        self._fixed_index = None  # for fixed head index

        # Activation function
        if activation_fn == "elu":
            activation = nn.ELU
        elif activation_fn == "relu":
            activation = nn.ReLU
        elif activation_fn == "tanh":
            activation = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")
        self.activation = activation()

        # Trunk
        if trunk is not None:
            self.trunk = trunk
        else:
            base_layers = [nn.Linear(input_dim, hidden_dim)]
            if add_layer_norm:
                base_layers.append(nn.LayerNorm(hidden_dim))
            base_layers.append(self.activation)

            for _ in range(n_hidden_layers - 1):
                base_layers.append(nn.Linear(hidden_dim, hidden_dim))
                if add_layer_norm:
                    base_layers.append(nn.LayerNorm(hidden_dim))
                base_layers.append(self.activation)

            self.trunk = nn.Sequential(*base_layers)

        self.base_head = nn.Linear(hidden_dim, output_dim)

        def make_epinet():
            layers = [nn.Linear(hidden_dim, epinet_hidden_dim), self.activation]
            for _ in range(epinet_layers - 1):
                layers.append(nn.Linear(epinet_hidden_dim, epinet_hidden_dim))
                layers.append(self.activation)
            layers.append(nn.Linear(epinet_hidden_dim, output_dim * index_dim))
            return nn.Sequential(*layers)

        self.epinet_train = make_epinet()
        self.epinet_prior = make_epinet()

        for param in self.epinet_prior.parameters():
            param.requires_grad = False

    def set_index(self, index: Optional[int] = None):
        """
        Sets a fixed index for the epinet prior to use during forward passes.
        If `index` is None, a random index will be used.
        """
        if index is not None and not (0 <= index < self.index_dim):
            raise ValueError(f"Index must be in [0, {self.index_dim - 1}]")
        self._fixed_index = index

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        if preprocessed_states.dtype != torch.float:
            preprocessed_states = preprocessed_states.float()

        features = self.trunk(preprocessed_states)
        if self.stop_gradient:
            features = features.detach()

        base_out = self.base_head(features)

        # Sample z from standard Gaussian
        z = torch.randn(self.index_dim, dtype=torch.float32, device=preprocessed_states.device)

        def project(epinet_net, features: torch.Tensor, detach: bool = False, head_idx: Optional[int] = None):
            if detach:
                features = features.detach()
            epinet_out = epinet_net(features)
            epinet_out = epinet_out.view(-1, self.output_dim, self.index_dim)  # shape: (B, C, K)
            return torch.einsum('bcz,z->bc', epinet_out, z)

            #head_idx = self._fixed_index
            #if head_idx is None:
                #head_idx = torch.randint(0, self.index_dim, (1,), device=preprocessed_states.device).item()

        # Epinet train is projected using sampled z
        epi_train = project(self.epinet_train, features, detach=False)

        # Epinet prior uses only a selected head index (either fixed or random)

        head_idx = torch.randint(0, self.index_dim, (1,), device=preprocessed_states.device).item()
        #head_idx  = self._fixed_index

        epi_prior = project(self.epinet_prior, features, detach=True, head_idx=head_idx)

        final_train = base_out + epi_train
        final_prior = self.prior_scale * epi_prior

        return final_train + final_prior
        



