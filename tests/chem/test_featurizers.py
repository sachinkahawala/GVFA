"""Featurizer shape and basic correctness tests."""

import torch

from gvfa.chem import (
    AtomFeaturizer,
    BondFeaturizer,
    DescriptorFeaturizer,
    DEFAULT_ATOM_FEATURES,
    EXTENDED_ATOM_FEATURES,
    DEFAULT_BOND_FEATURES,
    EXTENDED_BOND_FEATURES,
    smiles_to_mol_graph,
)


def test_atom_featurizer_default_dim_is_8():
    af = AtomFeaturizer(DEFAULT_ATOM_FEATURES)
    assert af.output_dim == 8


def test_atom_featurizer_extended_dim_is_18():
    af = AtomFeaturizer(EXTENDED_ATOM_FEATURES)
    assert af.output_dim == 18


def test_bond_featurizer_default_dim_is_4():
    bf = BondFeaturizer(DEFAULT_BOND_FEATURES)
    assert bf.output_dim == 4


def test_bond_featurizer_extended_dim_is_5():
    bf = BondFeaturizer(EXTENDED_BOND_FEATURES)
    assert bf.output_dim == 5


def test_smiles_to_mol_graph_shapes():
    af = AtomFeaturizer(EXTENDED_ATOM_FEATURES)
    bf = BondFeaturizer()
    g = smiles_to_mol_graph("c1ccccc1O", target=-1.5, atom_featurizer=af, bond_featurizer=bf)
    assert g is not None
    # phenol: 7 atoms, 7 bonds (6 ring + 1 OH) → 14 directed edges
    assert g.x_atom.shape == (7, 18)
    assert g.edge_index.shape == (2, 14)
    assert g.x_bond.shape == (14, 4)
    assert torch.allclose(g.target, torch.tensor([-1.5]))


def test_smiles_to_mol_graph_handles_invalid():
    af = AtomFeaturizer()
    g = smiles_to_mol_graph("not-a-smiles", target=0.0, atom_featurizer=af)
    assert g is None


def test_descriptor_featurizer_dim_is_96():
    from rdkit import Chem
    df = DescriptorFeaturizer()
    df.fit([Chem.MolFromSmiles("CCO")])
    v = df.transform(Chem.MolFromSmiles("CCO"))
    assert v.shape == (96,)
    # No NaN/Inf — we strip those in transform()
    assert torch.isfinite(v).all()


def test_atom_featurizer_unknown_feature_raises():
    import pytest
    with pytest.raises(ValueError):
        AtomFeaturizer(["not_a_feature"])
