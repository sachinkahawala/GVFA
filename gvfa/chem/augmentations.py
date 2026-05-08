"""
Optional encoder augmentations: Reservoir tap-buffer + Sigma-Pi expansion.

Both operate on the *list of per-level node embeddings* produced by
:meth:`gvfa.chem.edge_aware.EdgeAwareGVFA.forward_levels` (or the
equivalent unrolled output from a base :class:`gvfa.models.GVFA` encoder),
not on the concatenation. This is what makes them swappable in the
pipeline: they replace the default "concatenate all levels" head with a
different way of combining levels into a fixed-size representation.

The math follows the legacy ``Molecular_Solubility/GVFA_with_edge`` code:

- :class:`Reservoir`:  ``F⁽¹⁾_v = Σ_k λ^k · ρ^k(h_v^{(k)})``, then L2-normalize.
  ``λ`` is ``hop_decay`` (typical 0.6–0.95). ``ρ^k`` is a cyclic shift
  by ``k`` along the feature axis.

- :class:`SigmaPi`: a recursive polynomial expansion. With ``∗_t`` defined by
  ``∗_0(F) = F`` and ``∗_t(F) = bind(π(∗_{t-1}(F)), F)``,
  the output is ``F = Σ_{t ∈ orders} ∗_t(F⁽¹⁾)`` (then L2-normalized).
  ``π`` is a cyclic shift by ``max(1, D // 3)``. ``bind`` uses the
  configured kind (circular FFT or Hadamard) — typically Hadamard so the
  expansion stays in real space.

Either can be used alone or composed (Reservoir → SigmaPi).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Sequence

import torch
import torch.nn.functional as F

from ..operations import bind as fft_bind


SigmaPiBindKind = Literal["circular", "hadamard"]


def _sigma_pi_bind(x: torch.Tensor, y: torch.Tensor, kind: SigmaPiBindKind) -> torch.Tensor:
    if kind == "hadamard":
        return x * y
    if kind == "circular":
        return fft_bind(x, y)
    raise ValueError(f"Unknown sigma-pi bind kind: {kind!r}")


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


# ---------------------------------------------------------------------------
# Reservoir tap buffer
# ---------------------------------------------------------------------------


@dataclass
class Reservoir:
    """RC-style tap buffer: weighted, permutation-shifted superposition of layers.

    Args:
        hop_decay: ``λ`` in ``[0, 1]``. Typical 0.6–0.95. Larger values weight
            far-hop layers more heavily.
        normalize: if True, L2-normalize the result along the feature axis.
    """

    hop_decay: float = 0.85
    normalize: bool = True

    def __call__(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        """Combine per-layer node embeddings into a single ``[N, D]`` representation."""
        if not levels:
            raise ValueError("Reservoir requires at least one level")
        out = torch.zeros_like(levels[0])
        for k, h_k in enumerate(levels):
            weight = self.hop_decay ** k
            permuted = torch.roll(h_k, shifts=k, dims=-1)
            out = out + weight * permuted
        if self.normalize:
            out = _l2_normalize(out)
        return out


# ---------------------------------------------------------------------------
# Sigma-Pi polynomial expansion
# ---------------------------------------------------------------------------


@dataclass
class SigmaPi:
    """Sigma-Pi recursive polynomial expansion of a hypervector field.

    Args:
        orders: the orders ``t`` to *include* in the sum. Order 0 is the
            identity, order 1 is ``π(F) ⊙ F``, order 2 is ``π(π(F) ⊙ F) ⊙ F``,
            and so on. The recursion is always built up to ``max(orders)``;
            orders not in the list are skipped in the final sum but still
            participate in the recursive chain.
        bind_kind: ``"hadamard"`` (default, real-space, matches legacy) or
            ``"circular"`` (FFT-based, the canonical VSA binding).
        normalize: L2-normalize the recursive intermediates and the result.
    """

    orders: Sequence[int] = (0, 1)
    bind_kind: SigmaPiBindKind = "hadamard"
    normalize: bool = True

    def __call__(self, F1: torch.Tensor) -> torch.Tensor:
        """Expand ``F1`` ``[N, D]`` and return the requested-order sum ``[N, D]``."""
        D = int(F1.shape[-1])
        shift = max(1, D // 3)
        wanted = set(int(o) for o in self.orders)
        max_order = max(wanted) if wanted else 0

        result = torch.zeros_like(F1)
        ast_prev = F1
        for t in range(max_order + 1):
            if t == 0:
                ast_t = F1
            else:
                bound = _sigma_pi_bind(
                    torch.roll(ast_prev, shifts=shift, dims=-1), F1, self.bind_kind
                )
                ast_t = _l2_normalize(bound) if self.normalize else bound
            ast_prev = ast_t
            if t in wanted:
                result = result + ast_t

        if self.normalize:
            result = _l2_normalize(result)
        return result


# ---------------------------------------------------------------------------
# Convenience composition
# ---------------------------------------------------------------------------


@dataclass
class ReservoirSigmaPi:
    """Compose :class:`Reservoir` followed by :class:`SigmaPi`.

    The pipeline can also instantiate the two stages separately and chain
    them itself; this dataclass is just a convenience for the common case
    used by the legacy ``GVFA_with_edge`` reservoir variant.
    """

    reservoir: Reservoir = field(default_factory=Reservoir)
    sigma_pi: SigmaPi = field(default_factory=SigmaPi)

    def __call__(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        F1 = self.reservoir(levels)
        return self.sigma_pi(F1)
