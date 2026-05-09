"""
gvfa.chem — molecular-domain primitives for graph-level chemistry tasks.

This subpackage layers chemistry-specific stages on top of the stateless
encoder in :mod:`gvfa.models`. The pieces are deliberately small and
independently swappable so a downstream pipeline can compose ablations:

- :mod:`gvfa.chem.featurizers` — SMILES → atom/bond features → :class:`MolGraph`.
- :mod:`gvfa.chem.projections` — raw features → hypervectors via random
  projections (Gaussian / Orthogonal) with optional bounded per-column scaling.
- :mod:`gvfa.chem.edge_aware` — edge-aware extension of the base GVFA encoder
  (added in Phase 2).
- :mod:`gvfa.chem.augmentations` — Sigma-Pi expansion and reservoir tap buffer
  (added in Phase 2).
- :mod:`gvfa.chem.splits` — random / scaffold / k-fold splitters.
- :mod:`gvfa.chem.datasets` — CSV and MoleculeNet dataset loaders.
"""

from .featurizers import (
    AtomFeaturizer,
    BondFeaturizer,
    DescriptorFeaturizer,
    MolGraph,
    DEFAULT_ATOM_FEATURES,
    EXTENDED_ATOM_FEATURES,
    DEFAULT_BOND_FEATURES,
    EXTENDED_BOND_FEATURES,
    DEFAULT_DESCRIPTOR_NAMES,
    smiles_to_mol_graph,
)
from .projections import (
    BoundedScaler,
    RandomProjection,
    default_extended_atom_scaler,
    default_bond_scaler,
)
from .splits import random_split, scaffold_split, kfold_split
from .datasets import SolubilityDataset, MoleculeNetDataset, index_subset
from .edge_aware import EdgeBinder, EdgeAwareGVFA, aggregate_with_edges
from .augmentations import Reservoir, SigmaPi, ReservoirSigmaPi
from .poolers import MultiStatPool
from .size_aware import SizeAwarePost

__all__ = [
    # featurizers
    "AtomFeaturizer",
    "BondFeaturizer",
    "DescriptorFeaturizer",
    "MolGraph",
    "DEFAULT_ATOM_FEATURES",
    "EXTENDED_ATOM_FEATURES",
    "DEFAULT_BOND_FEATURES",
    "EXTENDED_BOND_FEATURES",
    "DEFAULT_DESCRIPTOR_NAMES",
    "smiles_to_mol_graph",
    # projections
    "BoundedScaler",
    "RandomProjection",
    "default_extended_atom_scaler",
    "default_bond_scaler",
    # splits
    "random_split",
    "scaffold_split",
    "kfold_split",
    # datasets
    "SolubilityDataset",
    "MoleculeNetDataset",
    "index_subset",
    # edge-aware encoder
    "EdgeBinder",
    "EdgeAwareGVFA",
    "aggregate_with_edges",
    # augmentations
    "Reservoir",
    "SigmaPi",
    "ReservoirSigmaPi",
    # poolers + size-aware post
    "MultiStatPool",
    "SizeAwarePost",
]
