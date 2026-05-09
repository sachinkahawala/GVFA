"""
Per-graph post-pool size-aware transforms.

Two knobs, both off by default:

- **``scale``** — divide each graph's pooled embedding by ``N^p`` where
  ``p ∈ {0, 0.5, 1.0, 1.5}`` (``"none" | "sqrt_n" | "n" | "n_pow_1_5"``).
  Useful when the pooler is a sum (where ``p ≈ 0.5`` matches degree-aware
  GNN normalisation), or when a higher exponent is desired as part of a
  specific recipe (the legacy Best3 stack uses ``sqrt_n`` *on top of*
  ``MultiStatPool``'s internal divide-by-``N`` for an effective ``1/N^{1.5}``).

- **``append_size``** — append a single column of size information to each
  graph embedding. Two encodings:
    - ``"raw"`` — ``num_nodes`` (Best3's choice).
    - ``"log1p_over_log10"`` — ``log1p(N) / log1p(10)``, ranges roughly
      ``[0, ~2]`` and is comparable in magnitude to bipolar HV dims (Best2's
      choice). Better behaved under L2-regularised heads without an upstream
      ``StandardScaler``.

The class is intentionally configuration-driven and keeps both forms so that
the size-aware ablation (does flipping ``scale`` between ``none`` and
``sqrt_n`` move metrics?) is a single config edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


SizeScale = Literal["none", "sqrt_n", "n", "n_pow_1_5"]
SizeAppendKind = Literal["raw", "log1p_over_log10"]

_SCALE_EXPONENT = {"none": 0.0, "sqrt_n": 0.5, "n": 1.0, "n_pow_1_5": 1.5}


@dataclass
class SizeAwarePost:
    scale: SizeScale = "none"
    append_size: bool = False
    append_size_kind: SizeAppendKind = "raw"

    def __call__(
        self, graph_emb: torch.Tensor, num_nodes: torch.Tensor
    ) -> torch.Tensor:
        """Apply size-aware scaling and (optionally) append a size column.

        Args:
            graph_emb: ``[num_graphs, F]`` per-graph embeddings.
            num_nodes: ``[num_graphs]`` per-graph atom counts.

        Returns:
            ``[num_graphs, F]`` (or ``[num_graphs, F + 1]`` if
            ``append_size`` is True).
        """
        if self.scale not in _SCALE_EXPONENT:
            raise ValueError(f"Unknown size scale: {self.scale!r}")

        out = graph_emb
        n = num_nodes.to(out.dtype).clamp(min=1.0)

        p = _SCALE_EXPONENT[self.scale]
        if p > 0.0:
            divisor = n.pow(p).view(-1, 1).clamp(min=1e-6)
            out = out / divisor

        if self.append_size:
            if self.append_size_kind == "raw":
                col = n.view(-1, 1)
            elif self.append_size_kind == "log1p_over_log10":
                ten = torch.tensor(10.0, dtype=out.dtype, device=out.device)
                col = (torch.log1p(n) / torch.log1p(ten)).view(-1, 1)
            else:
                raise ValueError(f"Unknown append_size_kind: {self.append_size_kind!r}")
            out = torch.cat([out, col], dim=1)

        return out
