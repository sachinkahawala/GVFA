"""
Edge-aware GVFA encoder.

The base :class:`gvfa.models.GVFA` ignores edge attributes — its aggregation
is just ``F(i) = Σ_{j ∈ N_i} H_j`` (a sum of neighbor hypervectors). Many
molecular tasks benefit from explicitly conditioning each message on the
*bond* between two atoms (single vs. aromatic, conjugated, in-ring, 3D
length). This module adds that capability while reusing every existing
GVFA primitive.

Two pieces:

- :class:`EdgeBinder` — pure binding op between a neighbor hypervector and
  an edge hypervector. Two kinds:
    - ``"circular"`` — FFT circular convolution from
      :func:`gvfa.operations.bind` (the canonical VSA binding).
    - ``"hadamard"`` — element-wise multiplication (the variant used by
      ``Molecular_Solubility/GVFA_with_edge``).

- :class:`EdgeAwareGVFA` — subclass of :class:`gvfa.models.GVFA`. When given
  per-edge hypervectors ``edge_h``, it replaces the aggregation step with
  ``F(i) = Σ_{j → i} EdgeBinder(H_j, e_{j→i})`` and reuses the parent's
  combine + normalize + concat logic. With ``edge_h=None`` it delegates to
  the base class verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import torch

from ..models import GVFA
from ..operations import bind


EdgeBindKind = Literal["circular", "hadamard"]


@dataclass
class EdgeBinder:
    """Bind neighbor and edge hypervectors into a per-edge message.

    Args:
        kind: ``"circular"`` (FFT) or ``"hadamard"`` (element-wise multiply).
    """

    kind: EdgeBindKind = "hadamard"

    def __call__(self, neighbor_h: torch.Tensor, edge_h: torch.Tensor) -> torch.Tensor:
        if self.kind == "hadamard":
            return neighbor_h * edge_h
        if self.kind == "circular":
            return bind(neighbor_h, edge_h)
        raise ValueError(f"Unknown edge bind kind: {self.kind!r}")


def aggregate_with_edges(
    node_h: torch.Tensor,
    edge_h: torch.Tensor,
    edge_index: torch.Tensor,
    edge_binder: EdgeBinder,
    num_nodes: int,
) -> torch.Tensor:
    """Aggregate edge-bound neighbor messages.

    For each edge ``(j → i)`` (with ``j = edge_index[1, e]`` as the source
    and ``i = edge_index[0, e]`` as the target — matching
    :func:`gvfa.models.aggregate.aggregate_neighbors`), compute
    ``msg = EdgeBinder(H_j, e)`` and accumulate at node ``i``.

    Returns ``[N, D]`` aggregated messages.
    """
    row, col = edge_index
    messages = edge_binder(node_h[col], edge_h)
    out = torch.zeros(num_nodes, node_h.shape[1], dtype=node_h.dtype, device=node_h.device)
    out.index_add_(0, row, messages)
    return out


class EdgeAwareGVFA(GVFA):
    """GVFA encoder with optional per-edge binding in the aggregation step.

    Args (in addition to :class:`gvfa.models.GVFA`):
        edge_binder: an :class:`EdgeBinder` instance. If ``None``, the
            encoder behaves identically to the base ``GVFA``.
    """

    def __init__(
        self,
        edge_binder: Optional[EdgeBinder] = None,
        **gvfa_kwargs,
    ) -> None:
        super().__init__(**gvfa_kwargs)
        self.edge_binder = edge_binder

    # ------------------------------------------------------------------
    # Forward variants. We keep the parent's signature compatibility:
    #   parent.forward(h, edge_index, return_all_levels=True)
    # but add an extra ``edge_h`` keyword. When ``edge_h`` is None or the
    # edge binder is missing, we delegate to ``super().forward`` so the
    # behavior is exactly the base GVFA — no surprises in edge-free runs.
    # ------------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: Optional[torch.Tensor] = None,
        return_all_levels: bool = True,
    ) -> torch.Tensor:
        if self.edge_binder is None or edge_h is None:
            return super().forward(h, edge_index, return_all_levels=return_all_levels)
        levels = self.forward_levels(h, edge_index, edge_h)
        return torch.cat(levels, dim=1) if return_all_levels else levels[-1]

    def forward_levels(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Run the encoder and return the *list* of per-level node embeddings.

        This is the entry point augmentations (:class:`Reservoir`,
        :class:`SigmaPi`) consume — they need each level individually.

        With ``edge_h`` provided and an :class:`EdgeBinder` configured, the
        aggregation step uses :func:`aggregate_with_edges`. Otherwise it
        falls back to the base :func:`gvfa.models.aggregate.aggregate_neighbors`.
        """
        from ..models.aggregate import aggregate_neighbors

        num_nodes = int(h.shape[0])
        levels: List[torch.Tensor] = [h]
        for _ in range(self.num_layers - 1):
            if self.edge_binder is not None and edge_h is not None:
                f = aggregate_with_edges(h, edge_h, edge_index, self.edge_binder, num_nodes)
            else:
                f = aggregate_neighbors(h, edge_index, num_nodes)
            h = self.phi_fn(h, f)
            h = self._normalize(h)
            levels.append(h)
        return levels

    def forward_graph(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_h: Optional[torch.Tensor] = None,
        pooling: Literal["sum", "mean"] = "sum",
    ) -> torch.Tensor:
        """Graph-level forward.

        Mirrors :meth:`gvfa.models.GVFA.forward_graph` but threads ``edge_h``
        into the node-level forward.
        """
        node_repr = self.forward(
            h, edge_index, edge_h=edge_h, return_all_levels=True
        )
        if batch is None:
            return (
                node_repr.sum(dim=0, keepdim=True)
                if pooling == "sum"
                else node_repr.mean(dim=0, keepdim=True)
            )
        num_graphs = int(batch.max().item()) + 1
        D = int(node_repr.shape[1])
        graph_repr = torch.zeros(
            num_graphs, D, dtype=node_repr.dtype, device=node_repr.device
        )
        graph_repr.index_add_(0, batch, node_repr)
        if pooling == "mean":
            counts = torch.bincount(batch).float().clamp(min=1).unsqueeze(1)
            graph_repr = graph_repr / counts
        return graph_repr
