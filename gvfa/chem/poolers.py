"""
Graph-level pooling readouts.

The default ``forward_graph`` paths in :class:`gvfa.models.GVFA` and
:class:`gvfa.chem.EdgeAwareGVFA` use sum/mean pooling. Some recipes — notably
the Best3 ``multi_stat_pool`` from the legacy
``Molecular_Solubility/GVFA_with_edge`` code — concatenate three statistics
per graph instead. This module exposes that as a swappable pooler so the
same pipeline can run either readout via a single config flag.

Currently bundled:

- :class:`MultiStatPool` — concat of ``[mean(F_v) | max(F_v) | mean(bind(F_v, F_v))]``
  per graph. Reuses :func:`gvfa.operations.bind` for the squared-mean term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..operations import bind as fft_bind


PoolBindKind = Literal["circular", "hadamard"]


def _pool_bind(x: torch.Tensor, y: torch.Tensor, kind: PoolBindKind) -> torch.Tensor:
    if kind == "hadamard":
        return x * y
    if kind == "circular":
        return fft_bind(x, y)
    raise ValueError(f"Unknown multi-stat bind kind: {kind!r}")


@dataclass
class MultiStatPool:
    """Concat of ``[mean(F_v) | max(F_v) | mean(bind(F_v, F_v))]`` per graph.

    Operates on a single graph's per-node representation ``F_v: [N, D]`` and
    returns ``[1, 3·D]``. Mirrors the Best3 ``multi_stat_pool`` semantics —
    both ``g_mean`` and ``g_mean_sq`` are *true* per-atom averages (divide by
    ``N``); ``g_max`` is the per-feature max across nodes.

    Args:
        bind_kind: ``"circular"`` (FFT — canonical VSA bind) or ``"hadamard"``
            (element-wise multiply). Best3 uses circular.
    """

    bind_kind: PoolBindKind = "circular"

    def __call__(self, F_v: torch.Tensor) -> torch.Tensor:
        if F_v.ndim != 2:
            raise ValueError(f"MultiStatPool expects [N, D], got {tuple(F_v.shape)}")
        if F_v.shape[0] == 0:
            # Empty graph guard: zeros for all three stats.
            D = int(F_v.shape[1])
            return torch.zeros(1, 3 * D, dtype=F_v.dtype, device=F_v.device)

        g_mean = F_v.mean(dim=0, keepdim=True)
        g_max = F_v.max(dim=0, keepdim=True).values
        F_sq = _pool_bind(F_v, F_v, self.bind_kind)
        g_mean_sq = F_sq.mean(dim=0, keepdim=True)
        return torch.cat([g_mean, g_max, g_mean_sq], dim=-1)
