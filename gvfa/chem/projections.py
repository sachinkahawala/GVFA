"""
Projections: raw feature matrices ``[N, F]`` → hypervectors ``[N, D]``.

The pipeline separates two concerns that were entangled in the legacy
``VSA_conversion.py``:

- :class:`BoundedScaler` — per-column scaling that maps each raw feature
  column to roughly ``[-1, +1]`` using BINARY / MIN-MAX / TANH rules.
  Stateful: ``fit`` captures column minima/ranges from training data so
  ``transform`` can apply the same statistics to test data without leakage.

- :class:`RandomProjection` — applies a fixed ``[F, D]`` random matrix.
  Two variants (configured by ``kind``):
    - ``"gaussian"``: ``W ~ N(0, 1 / F)`` (Johnson-Lindenstrauss style).
    - ``"orthogonal"``: ``W`` taken from a QR decomposition of a random
      Gaussian matrix; preserves norms better than Gaussian for ``D ≤ F``.
  Optional ``sign_normalize`` binarizes the output to ``{-1, +1}`` after
  projection (the standard GVFA initialization).

A pipeline typically does ``BoundedScaler.fit_transform`` (train) →
``RandomProjection.fit_transform`` (train), then ``transform`` on test.
The same two stages are applied independently to atom features and bond
features when an edge-aware variant is configured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# BoundedScaler
# ---------------------------------------------------------------------------


@dataclass
class BoundedScaler:
    """Map each feature column to ~``[-1, +1]`` using one of three rules.

    Args:
        binary_cols: column indices treated as ``{0, 1}`` flags. Mapped via
            ``2x - 1``.
        minmax_cols: column indices min-max scaled to ``[-1, +1]`` using
            training statistics.
        tanh_cols: ``{column_index: divisor}``. Each column is mapped via
            ``tanh(x / divisor)``. Use a small divisor for narrow inputs
            (e.g. Gasteiger charges) and a larger one for broader ranges.
    """

    binary_cols: Sequence[int] = ()
    minmax_cols: Sequence[int] = ()
    tanh_cols: Dict[int, float] = field(default_factory=dict)

    # Populated by .fit():
    col_min: Optional[torch.Tensor] = None
    col_range: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "BoundedScaler":
        """Capture min/range over the min-max columns from training data."""
        if len(self.minmax_cols) == 0:
            self.col_min = torch.zeros(0)
            self.col_range = torch.zeros(0)
            return self
        cols = list(self.minmax_cols)
        sub = X[:, cols]
        col_min = sub.min(dim=0).values
        col_range = (sub.max(dim=0).values - col_min).clamp(min=1e-6)
        self.col_min = col_min
        self.col_range = col_range
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.col_min is None or self.col_range is None:
            raise RuntimeError("BoundedScaler.transform called before fit().")
        out = X.clone().float()
        if self.binary_cols:
            bc = list(self.binary_cols)
            out[:, bc] = out[:, bc] * 2.0 - 1.0
        if self.minmax_cols:
            mm = list(self.minmax_cols)
            out[:, mm] = ((out[:, mm] - self.col_min) / self.col_range) * 2.0 - 1.0
        for col, scale in self.tanh_cols.items():
            out[:, col] = torch.tanh(out[:, col] / float(scale))
        return out

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# RandomProjection
# ---------------------------------------------------------------------------


ProjectionKind = Literal["gaussian", "orthogonal"]


def _build_projection_matrix(
    in_dim: int, out_dim: int, kind: ProjectionKind, seed: int
) -> torch.Tensor:
    """Sample the projection matrix.

    Gaussian: ``N(0, 1/in_dim)``. Orthogonal: QR of a random Gaussian; for
    ``out_dim > in_dim`` we transpose-and-truncate to keep the rows
    orthonormal in the larger space.
    """
    g = torch.Generator().manual_seed(int(seed))
    if kind == "gaussian":
        W = torch.randn(in_dim, out_dim, generator=g)
        return W / math.sqrt(in_dim)
    if kind == "orthogonal":
        if out_dim <= in_dim:
            Q, _ = torch.linalg.qr(torch.randn(in_dim, out_dim, generator=g))
            return Q[:, :out_dim]
        Q, _ = torch.linalg.qr(torch.randn(out_dim, in_dim, generator=g))
        return Q[:, :in_dim].T
    raise ValueError(f"Unknown projection kind: {kind!r}")


@dataclass
class RandomProjection:
    """Fixed random projection ``[N, F] @ W → [N, D]``.

    Args:
        D: target hypervector dimension.
        kind: ``"gaussian"`` or ``"orthogonal"``.
        seed: seed for the projection matrix.
        sign_normalize: if True, apply ``torch.sign`` after the projection
            (GVFA's standard bipolar initialization).
    """

    D: int
    kind: ProjectionKind = "gaussian"
    seed: int = 0
    sign_normalize: bool = False

    W: Optional[torch.Tensor] = None  # populated by fit()

    def fit(self, X: torch.Tensor) -> "RandomProjection":
        """Sample W from the input dimension. Idempotent when called with the same
        ``X.shape[1]`` and ``seed``."""
        in_dim = int(X.shape[1])
        self.W = _build_projection_matrix(in_dim, self.D, self.kind, self.seed)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            raise RuntimeError("RandomProjection.transform called before fit().")
        h = X @ self.W
        if self.sign_normalize:
            h = torch.sign(h)
        return h

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Reasonable default scaler configurations
# ---------------------------------------------------------------------------

# Column layouts matching the EXTENDED_ATOM_FEATURES layout (18 columns).
# These mirror the BINARY_COLS / MINMAX_COLS / TANH_COLS constants from the
# legacy GVFA_with_edge/src/VSA_conversion.py.
EXTENDED_ATOM_BINARY_COLS: tuple[int, ...] = (3, 4, 5, 6, 8, 9, 10, 11, 16)
EXTENDED_ATOM_MINMAX_COLS: tuple[int, ...] = (0, 1, 2, 12, 15, 17)
EXTENDED_ATOM_TANH_COLS: Dict[int, float] = {7: 1.0, 13: 0.3, 14: 0.5}


def default_extended_atom_scaler() -> BoundedScaler:
    """Bounded scaler matching the legacy 18-col atom feature layout."""
    return BoundedScaler(
        binary_cols=list(EXTENDED_ATOM_BINARY_COLS),
        minmax_cols=list(EXTENDED_ATOM_MINMAX_COLS),
        tanh_cols=dict(EXTENDED_ATOM_TANH_COLS),
    )


# Bond features (default 4 cols: bond_type, is_conjugated, in_ring, bond_length_3d).
# bond_type ∈ {1..4} → min-max; flags → binary; length → tanh.
DEFAULT_BOND_BINARY_COLS: tuple[int, ...] = (1, 2)
DEFAULT_BOND_MINMAX_COLS: tuple[int, ...] = (0,)
DEFAULT_BOND_TANH_COLS: Dict[int, float] = {3: 1.5}  # ~typical C-C bond length


def default_bond_scaler() -> BoundedScaler:
    """Bounded scaler for the default 4-col bond feature layout."""
    return BoundedScaler(
        binary_cols=list(DEFAULT_BOND_BINARY_COLS),
        minmax_cols=list(DEFAULT_BOND_MINMAX_COLS),
        tanh_cols=dict(DEFAULT_BOND_TANH_COLS),
    )
