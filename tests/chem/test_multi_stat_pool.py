"""MultiStatPool + SizeAwarePost contract tests."""

import math

import torch

from gvfa.chem import MultiStatPool, SizeAwarePost


def test_multi_stat_pool_output_shape():
    """[N, D] -> [1, 3*D] regardless of N."""
    for n in [1, 5, 17]:
        F_v = torch.randn(n, 64)
        out = MultiStatPool()(F_v)
        assert out.shape == (1, 192), out.shape


def test_multi_stat_pool_permutation_invariant():
    F_v = torch.randn(11, 32)
    pool = MultiStatPool(bind_kind="circular")
    out = pool(F_v)
    perm = torch.randperm(11)
    out_shuffled = pool(F_v[perm])
    assert torch.allclose(out, out_shuffled, atol=1e-5)


def test_multi_stat_pool_empty_graph_returns_zeros():
    out = MultiStatPool()(torch.zeros(0, 16))
    assert out.shape == (1, 48)
    assert (out == 0).all()


def test_multi_stat_pool_hadamard_vs_circular_differ():
    F_v = torch.randn(8, 64)
    out_h = MultiStatPool(bind_kind="hadamard")(F_v)
    out_c = MultiStatPool(bind_kind="circular")(F_v)
    # First two thirds (mean, max) match; last third (mean of bind) differs.
    D = 64
    assert torch.allclose(out_h[:, : 2 * D], out_c[:, : 2 * D])
    assert not torch.allclose(out_h[:, 2 * D :], out_c[:, 2 * D :])


def test_size_aware_post_none_is_identity():
    emb = torch.randn(4, 16)
    sizes = torch.tensor([3, 7, 12, 20])
    out = SizeAwarePost(scale="none", append_size=False)(emb, sizes)
    assert torch.allclose(out, emb)


def test_size_aware_post_sqrt_n_with_raw_append():
    emb = torch.ones(3, 4)
    sizes = torch.tensor([4, 9, 25])
    out = SizeAwarePost(scale="sqrt_n", append_size=True, append_size_kind="raw")(emb, sizes)
    # Each row scaled by 1/√N, then `N` appended.
    expected = torch.tensor([
        [0.5, 0.5, 0.5, 0.5, 4.0],     # 1/√4 = 0.5
        [1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3, 9.0],
        [0.2, 0.2, 0.2, 0.2, 25.0],    # 1/√25 = 0.2
    ])
    assert torch.allclose(out, expected, atol=1e-5)


def test_size_aware_post_n_pow_1_5_matches_n_times_sqrt_n():
    """1/N^1.5 should equal (1/N) * (1/√N) — sanity check on the scale exponent."""
    emb = torch.ones(2, 8)
    sizes = torch.tensor([4, 16])
    out_15 = SizeAwarePost(scale="n_pow_1_5")(emb, sizes)
    out_then = SizeAwarePost(scale="n")(SizeAwarePost(scale="sqrt_n")(emb, sizes), sizes)
    assert torch.allclose(out_15, out_then, atol=1e-6)


def test_size_aware_post_log1p_over_log10_append_value():
    emb = torch.zeros(1, 1)
    sizes = torch.tensor([10])
    out = SizeAwarePost(
        scale="none", append_size=True, append_size_kind="log1p_over_log10"
    )(emb, sizes)
    # log1p(10) / log1p(10) = 1.0
    assert torch.allclose(out[:, -1], torch.tensor([1.0]), atol=1e-6)


def test_size_aware_post_unknown_scale_raises():
    import pytest
    with pytest.raises(ValueError):
        SizeAwarePost(scale="cubic_n")(torch.zeros(1, 1), torch.tensor([1]))
