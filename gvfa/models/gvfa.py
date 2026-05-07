"""
Main GVFA Model

Implements Algorithm 1 from the paper:
Graph Vector Function Architecture for zero-shot graph representations.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

from .combine import phi1, phi2, phi3, phi4, PHI_FUNCTIONS
from .aggregate import aggregate_neighbors
from .normalize import sign_normalize, clip_normalize, l2_normalize


class GVFA(nn.Module):
    """
    Graph Vector Function Architecture model.

    Implements the GVFA algorithm for computing graph/node representations
    using hyperdimensional computing operations.

    Args:
        num_layers: Number of GVFA layers (default: 3)
        phi: Combine function variant ('phi1', 'phi2', 'phi3', 'phi4')
        normalize: Normalization function ('sign', 'clip', 'l2', 'none')
        kappa: Clipping threshold for clip normalization (default: 1.0)
    """

    def __init__(
        self,
        num_layers: int = 3,
        phi: Literal['phi1', 'phi2', 'phi3', 'phi4'] = 'phi3',
        normalize: Literal['sign', 'clip', 'l2', 'none'] = 'sign',
        kappa: float = 1.0
    ):
        super().__init__()
        self.num_layers = num_layers
        self.phi_fn = PHI_FUNCTIONS[phi]
        self.normalize = normalize
        self.kappa = kappa

    def _normalize(self, h: torch.Tensor) -> torch.Tensor:
        """Apply normalization function."""
        if self.normalize == 'sign':
            return sign_normalize(h)
        elif self.normalize == 'clip':
            return clip_normalize(h, self.kappa)
        elif self.normalize == 'l2':
            return l2_normalize(h)
        else:
            return h

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        return_all_levels: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of GVFA.

        Args:
            h: Initial node features [N, D] (should be projected/binarized)
            edge_index: Graph edges [2, E]
            return_all_levels: If True, concatenate all level representations.
                              If False, return only the final level.

        Returns:
            Node representations [N, D * num_layers] if return_all_levels,
            otherwise [N, D]
        """
        num_nodes = h.shape[0]
        all_levels = [h]

        # num_layers - 1 iterations
        for _ in range(self.num_layers - 1):
            # F(i) = Σ_{j∈N_i} H_j
            f = aggregate_neighbors(h, edge_index, num_nodes)
            # Φ(H_i, F(i))
            h = self.phi_fn(h, f)
            # Λ normalization
            h = self._normalize(h)
            all_levels.append(h)

        if return_all_levels:
            return torch.cat(all_levels, dim=1)
        else:
            return h

    def forward_graph(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        pooling: Literal['sum', 'mean'] = 'sum'
    ) -> torch.Tensor:
        """
        Forward pass for graph-level representations.

        Args:
            h: Initial node features [N, D]
            edge_index: Graph edges [2, E]
            batch: Batch assignment for each node [N] (for batched graphs)
            pooling: Graph pooling type ('sum' or 'mean')

        Returns:
            Graph representations [num_graphs, D * num_layers]
        """
        # Get node representations
        node_repr = self.forward(h, edge_index, return_all_levels=True)

        # Pool to graph level
        if batch is None:
            # Single graph
            if pooling == 'sum':
                return node_repr.sum(dim=0, keepdim=True)
            else:
                return node_repr.mean(dim=0, keepdim=True)
        else:
            # Batched graphs
            num_graphs = batch.max().item() + 1
            D = node_repr.shape[1]
            graph_repr = torch.zeros(num_graphs, D, dtype=node_repr.dtype, device=node_repr.device)

            if pooling == 'sum':
                graph_repr.index_add_(0, batch, node_repr)
            else:
                graph_repr.index_add_(0, batch, node_repr)
                counts = torch.bincount(batch).float().unsqueeze(1)
                graph_repr = graph_repr / counts

            return graph_repr
