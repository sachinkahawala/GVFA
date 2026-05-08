"""EdgeAwareGVFA + augmentation contract tests."""

import torch

from gvfa.chem import (
    AtomFeaturizer,
    BondFeaturizer,
    EdgeAwareGVFA,
    EdgeBinder,
    EXTENDED_ATOM_FEATURES,
    RandomProjection,
    Reservoir,
    SigmaPi,
    default_bond_scaler,
    default_extended_atom_scaler,
    smiles_to_mol_graph,
)
from gvfa.models import GVFA


def _build_graph_with_hvs(D: int = 64, seed: int = 0):
    af = AtomFeaturizer(EXTENDED_ATOM_FEATURES)
    bf = BondFeaturizer()
    g = smiles_to_mol_graph("c1ccccc1O", target=-1.0, atom_featurizer=af, bond_featurizer=bf)
    assert g is not None
    ns = default_extended_atom_scaler().fit(g.x_atom)
    np_ = RandomProjection(D=D, kind="orthogonal", seed=seed, sign_normalize=True).fit(ns.transform(g.x_atom))
    bs = default_bond_scaler().fit(g.x_bond)
    bp = RandomProjection(D=D, kind="orthogonal", seed=seed + 1, sign_normalize=True).fit(bs.transform(g.x_bond))
    g.h_atom = np_.transform(ns.transform(g.x_atom))
    g.h_bond = bp.transform(bs.transform(g.x_bond))
    return g


def test_edge_aware_without_binder_matches_base_gvfa():
    g = _build_graph_with_hvs()
    base = GVFA(num_layers=3, phi="phi3", normalize="sign")
    ea = EdgeAwareGVFA(num_layers=3, phi="phi3", normalize="sign", edge_binder=None)
    out_base = base.forward(g.h_atom, g.edge_index, return_all_levels=True)
    out_ea = ea.forward(g.h_atom, g.edge_index, return_all_levels=True)
    assert torch.allclose(out_base, out_ea)


def test_hadamard_and_circular_binders_produce_different_outputs():
    g = _build_graph_with_hvs()
    ea_h = EdgeAwareGVFA(num_layers=3, phi="phi3", normalize="sign", edge_binder=EdgeBinder("hadamard"))
    ea_c = EdgeAwareGVFA(num_layers=3, phi="phi3", normalize="sign", edge_binder=EdgeBinder("circular"))
    out_h = ea_h.forward(g.h_atom, g.edge_index, edge_h=g.h_bond)
    out_c = ea_c.forward(g.h_atom, g.edge_index, edge_h=g.h_bond)
    assert out_h.shape == out_c.shape
    assert not torch.allclose(out_h, out_c)


def test_edge_aware_with_binder_differs_from_edge_free():
    g = _build_graph_with_hvs()
    ea = EdgeAwareGVFA(num_layers=3, phi="phi3", normalize="sign", edge_binder=EdgeBinder("hadamard"))
    out_with = ea.forward(g.h_atom, g.edge_index, edge_h=g.h_bond)
    out_without = ea.forward(g.h_atom, g.edge_index, edge_h=None)
    assert not torch.allclose(out_with, out_without)


def test_forward_levels_returns_one_tensor_per_layer():
    g = _build_graph_with_hvs()
    ea = EdgeAwareGVFA(num_layers=4, phi="phi3", normalize="sign", edge_binder=EdgeBinder("hadamard"))
    levels = ea.forward_levels(g.h_atom, g.edge_index, edge_h=g.h_bond)
    assert len(levels) == 4
    for l in levels:
        assert l.shape == (g.num_nodes, 64)


def test_reservoir_preserves_node_dim():
    g = _build_graph_with_hvs()
    ea = EdgeAwareGVFA(num_layers=3, phi="phi3", normalize="sign", edge_binder=EdgeBinder("hadamard"))
    levels = ea.forward_levels(g.h_atom, g.edge_index, edge_h=g.h_bond)
    F1 = Reservoir(hop_decay=0.85)(levels)
    assert F1.shape == (g.num_nodes, 64)


def test_sigma_pi_orders_change_output():
    g = _build_graph_with_hvs()
    ea = EdgeAwareGVFA(num_layers=3, phi="phi3", normalize="sign", edge_binder=EdgeBinder("hadamard"))
    levels = ea.forward_levels(g.h_atom, g.edge_index, edge_h=g.h_bond)
    F1 = Reservoir()(levels)
    out_012 = SigmaPi(orders=[0, 1, 2])(F1)
    out_01 = SigmaPi(orders=[0, 1])(F1)
    assert not torch.allclose(out_012, out_01)
