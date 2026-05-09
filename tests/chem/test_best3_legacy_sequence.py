"""Best3 operation-sequence checks against the legacy GVFA_with_edge code."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from gvfa.chem import EdgeAwareGVFA, EdgeBinder, MultiStatPool, RandomProjection
from gvfa.chem import Reservoir, SigmaPi
from gvfa.operations import bind as modular_bind


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = REPO_ROOT.parent / "Molecular_Solubility" / "GVFA_with_edge"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from models.graphcnnVSA_Binding_FULL import GraphCNN  # noqa: E402
from src.VSA_conversion import _random_projection_matrix  # noqa: E402


def _legacy_model(D: int, *, edge_feat_dim: int = 0) -> GraphCNN:
    return GraphCNN(
        D,
        4,
        0,
        "sum",
        "sum",
        torch.device("cpu"),
        10,
        edge_feat_dim=edge_feat_dim,
        use_reservoir=True,
        hop_decay=0.85,
        sigma_pi_orders=[0, 1],
        rng_seed=0,
    )


def test_modular_fft_bind_matches_legacy_graphcnn_bind():
    torch.manual_seed(0)
    D = 64
    x = torch.randn(7, D)
    y = torch.randn(7, D)
    legacy = _legacy_model(D)

    assert torch.allclose(legacy.bind(x, y), modular_bind(x, y))


def test_modular_reservoir_matches_legacy_tap_buffer():
    torch.manual_seed(1)
    D = 64
    levels = [torch.randn(5, D), torch.sign(torch.randn(5, D)), torch.sign(torch.randn(5, D))]
    legacy = _legacy_model(D)

    old = legacy.tap_buffer(levels)
    new = Reservoir(hop_decay=0.85, normalize=True)(levels)

    assert torch.allclose(old, new)


def test_modular_circular_sigma_pi_matches_legacy_expansion():
    torch.manual_seed(2)
    D = 64
    F1 = torch.randn(6, D)
    legacy = _legacy_model(D)

    old = legacy.sigma_pi_expansion(F1)
    new = SigmaPi(orders=[0, 1], bind_kind="circular", normalize=True)(F1)

    assert torch.allclose(old, new)


def test_modular_multi_stat_pool_matches_legacy_pool():
    torch.manual_seed(3)
    D = 64
    F_v = torch.randn(9, D)
    legacy = _legacy_model(D)
    idx = torch.tensor([[0] * F_v.shape[0], list(range(F_v.shape[0]))])
    elem = torch.ones(F_v.shape[0])
    graph_pool = torch.sparse_coo_tensor(idx, elem, torch.Size([1, F_v.shape[0]]))

    old = legacy.multi_stat_pool(F_v, graph_pool, [0, F_v.shape[0]])
    new = MultiStatPool(bind_kind="circular")(F_v)

    assert torch.allclose(old, new)


def test_modular_edge_phi1_matches_legacy_equation10_delta0():
    torch.manual_seed(4)
    D = 64
    h = torch.randn(4, D)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    undirected_edge_h = torch.randn(3, D)
    edge_h = torch.stack(
        [
            undirected_edge_h[0],
            undirected_edge_h[0],
            undirected_edge_h[1],
            undirected_edge_h[1],
            undirected_edge_h[2],
            undirected_edge_h[2],
        ]
    )
    legacy = _legacy_model(D)

    old = legacy.next_layer_eps(
        h,
        0,
        Adj_block=None,
        delta=0,
        equation=10,
        edge_index=edge_index,
        edge_H=edge_h,
        num_nodes=h.shape[0],
    )
    encoder = EdgeAwareGVFA(
        edge_binder=EdgeBinder("circular"),
        num_layers=2,
        phi="phi1",
        normalize="sign",
    )
    new = encoder.forward_levels(h, edge_index, edge_h=edge_h)[1]

    assert torch.allclose(old, new)


def test_modular_raw_node_projection_matches_legacy_projection_matrix():
    torch.manual_seed(5)
    F_in = 18
    D = 64
    seed = 7
    X = torch.randn(11, F_in)

    W_old = _random_projection_matrix(F_in, D, orthogonal=True, seed=seed)
    projection = RandomProjection(
        D=D,
        kind="orthogonal",
        seed=seed,
        sign_normalize=False,
    ).fit(X)

    assert torch.allclose(W_old, projection.W)
    assert torch.allclose(X @ W_old, projection.transform(X))
