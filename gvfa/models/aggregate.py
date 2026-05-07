"""
Aggregate Function: F

F(i) = Σ_{j∈N_i} H_j - Superposition of neighbor hypervectors (Equation 3)
"""

import torch


def aggregate_neighbors(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int
) -> torch.Tensor:
    """
    Aggregate neighbor hypervectors: F(i) = Σ_{j∈N_i} H_j

    Computes the sum of neighbor hypervectors for each node.

    Args:
        h: Node hypervectors [N, D]
        edge_index: Edge indices [2, E] where edge_index[0] are source nodes
                   and edge_index[1] are target nodes
        num_nodes: Total number of nodes

    Returns:
        Aggregated neighbor representations [N, D]
    """
    row, col = edge_index
    out = torch.zeros(num_nodes, h.shape[1], dtype=h.dtype, device=h.device)
    out.index_add_(0, row, h[col])
    return out
