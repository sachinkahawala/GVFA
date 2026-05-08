"""Projection determinism + no-leakage tests."""

import torch

from gvfa.chem import (
    BoundedScaler,
    RandomProjection,
    default_extended_atom_scaler,
)


def test_gaussian_projection_is_deterministic_with_seed():
    X = torch.randn(20, 8)
    p1 = RandomProjection(D=64, kind="gaussian", seed=7).fit(X)
    p2 = RandomProjection(D=64, kind="gaussian", seed=7).fit(X)
    assert torch.allclose(p1.W, p2.W)
    assert torch.allclose(p1.transform(X), p2.transform(X))


def test_orthogonal_projection_columns_are_orthonormal_when_d_le_f():
    # D <= F: rows of W should be orthonormal columns
    X = torch.randn(10, 32)
    p = RandomProjection(D=8, kind="orthogonal", seed=42).fit(X)
    # W is [F, D] = [32, 8]; columns should be orthonormal
    gram = p.W.T @ p.W
    assert torch.allclose(gram, torch.eye(8), atol=1e-5)


def test_sign_normalize_yields_bipolar_output():
    X = torch.randn(20, 16)
    p = RandomProjection(D=64, kind="gaussian", seed=0, sign_normalize=True).fit(X)
    h = p.transform(X)
    assert torch.unique(h).tolist() == [-1.0, 1.0]


def test_bounded_scaler_no_leakage():
    """Test data must be transformed with train-time min/range, not its own."""
    train = torch.tensor([[0.0, 1.0], [10.0, 0.0], [5.0, 1.0]])  # col0 ∈ [0, 10], col1 ∈ {0, 1}
    test = torch.tensor([[20.0, 1.0]])  # OOD on col0 — should NOT redefine min/range
    scaler = BoundedScaler(binary_cols=[1], minmax_cols=[0])
    scaler.fit(train)
    train_min_before = scaler.col_min.clone()
    train_range_before = scaler.col_range.clone()
    _ = scaler.transform(test)
    # transform() must not mutate fit-time stats
    assert torch.allclose(scaler.col_min, train_min_before)
    assert torch.allclose(scaler.col_range, train_range_before)
    # And the OOD test value must extrapolate based on train stats — not be clamped
    out = scaler.transform(test)
    # col0 = 20: (20 - 0) / 10 = 2.0, then * 2 - 1 = 3.0 (extrapolation)
    assert out[0, 0].item() > 1.0


def test_extended_atom_scaler_brings_values_into_unit_range():
    """The 18-col extended layout should map to roughly [-1, +1] after fit_transform."""
    # Synthesize plausible values: integers for binary/minmax cols, charges for tanh.
    n = 50
    X = torch.zeros(n, 18)
    X[:, 0] = torch.randint(1, 30, (n,)).float()      # atomic number
    X[:, 1] = torch.randint(1, 5, (n,)).float()       # degree
    X[:, 2] = torch.randint(1, 9, (n,)).float()       # valence
    X[:, 3:6] = torch.randint(0, 2, (n, 3)).float()   # hybridization (binary)
    X[:, 6] = torch.randint(0, 2, (n,)).float()       # aromaticity (binary)
    X[:, 7] = torch.randn(n)                           # formal charge (tanh, scale=1.0)
    X[:, 8:10] = torch.randint(0, 2, (n, 2)).float()  # hbond_flags (binary)
    X[:, 10:12] = torch.randint(0, 2, (n, 2)).float() # chirality (binary)
    X[:, 12] = torch.randint(0, 5, (n,)).float()      # num_attached_h (minmax)
    X[:, 13] = torch.randn(n) * 0.3                    # gasteiger charge (tanh, scale=0.3)
    X[:, 14] = torch.randn(n) * 0.5                    # crippen logP (tanh, scale=0.5)
    X[:, 15] = torch.randint(0, 30, (n,)).float()      # tpsa (minmax)
    X[:, 16] = torch.randint(0, 2, (n,)).float()       # is_in_aromatic_ring (binary)
    X[:, 17] = torch.randint(0, 7, (n,)).float()       # smallest_ring_size (minmax)
    scaler = default_extended_atom_scaler()
    Y = scaler.fit_transform(X)
    # Tanh outputs hover within [-1, +1]; binary maps exactly; minmax maps exactly to [-1, +1].
    assert Y.min().item() >= -1.001
    assert Y.max().item() <= 1.001
