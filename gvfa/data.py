"""
Data utilities for GVFA.

Exposes the helpers re-exported from ``gvfa/__init__.py``:

- :func:`random_projection` — project raw features into a D-dim hypervector
  space with an optional sign normalization (the standard GVFA initialization).
- :class:`S2VGraph` — a lightweight container for a single graph instance, used
  by graph-level experiments (e.g. molecular regression) to keep node features,
  the edge list, and an optional label together.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def random_projection(
    features: torch.Tensor,
    D: int,
    seed: int,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Project features to ``D`` dimensions with a random Gaussian matrix.

    The projection matrix ``W`` is sampled as ``N(0, 1/D)`` and applied as
    ``h = features @ W``. When ``normalize=True``, the result is sign-binarized
    to ``{-1, +1}``, which is the standard GVFA initialization for downstream
    layers that expect bipolar hypervectors.

    Args:
        features: Input features ``[N, input_dim]``.
        D: Target hypervector dimension.
        seed: Random seed controlling the projection matrix.
        normalize: If True, apply ``sign`` normalization after the projection.

    Returns:
        Projected features ``[N, D]``.
    """
    torch.manual_seed(seed)
    input_dim = features.shape[1]
    W = torch.randn(input_dim, D) / np.sqrt(D)
    h = features @ W
    if normalize:
        h = torch.sign(h)
    return h


@dataclass
class S2VGraph:
    """
    Lightweight container for a single graph instance.

    Mirrors the minimal interface graph-level experiments need: per-node
    features, the edge list, and an optional graph-level label. Heavier
    PyG-style attributes (batch vectors, edge attributes, 3D coords) are kept
    out of this base container — domain-specific subpackages (e.g.
    ``gvfa.chem``) layer them on top.

    Attributes:
        node_features: ``[N, F]`` raw node features.
        edge_index: ``[2, E]`` edge list. Undirected graphs should include
            both directions explicitly.
        label: Optional graph-level target.
        num_nodes: Convenience accessor; inferred from ``node_features`` when
            not provided.
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    label: Optional[torch.Tensor] = None
    num_nodes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_nodes is None:
            self.num_nodes = int(self.node_features.shape[0])
