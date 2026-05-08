"""
Molecular featurizers.

Three featurizers are exposed:

- :class:`AtomFeaturizer` — per-atom feature vectors built from RDKit Mol
  objects. The feature *set* is configurable; the default (``DEFAULT_ATOM_FEATURES``)
  reproduces the 8-column layout used by the original ``Molecular_Solubility/GVFA``
  variant, while ``EXTENDED_ATOM_FEATURES`` reproduces the 18-column layout used
  by the ``GVFA_with_edge`` variant.
- :class:`BondFeaturizer` — per-bond feature vectors. The 3D bond-length feature
  triggers an ETKDGv3 conformer + MMFF optimization the first time it's needed.
- :class:`DescriptorFeaturizer` — molecule-level RDKit descriptors (~96 of them),
  used by the traditional baseline.

Each featurizer follows a sklearn-style ``fit`` / ``transform`` contract. The
fit step captures any training statistics needed to apply the same transform
to test data without leakage.

The :class:`MolGraph` dataclass is the unit that flows between pipeline
stages: it carries SMILES, raw atom/bond features, an edge list, an optional
target, and post-projection hypervector slots that downstream stages populate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

# RDKit is an optional dep (gvfa[chem]). We import lazily inside functions so
# that importing this module without the chem extra installed still gives a
# clean error message at the call site rather than at import time.

# ----------------------------------------------------------------------------
# MolGraph: the per-graph payload that flows through the pipeline.
# ----------------------------------------------------------------------------


@dataclass
class MolGraph:
    """Per-molecule payload carried through every pipeline stage.

    Raw fields (populated by the featurizer):
        smiles: original SMILES string.
        x_atom: ``[N, F_atom]`` raw atom features.
        edge_index: ``[2, E]`` directed edge list (both directions present).
        x_bond: ``[E, F_bond]`` raw bond features, aligned with ``edge_index``.
            ``None`` if no bond featurizer is configured.
        target: optional graph-level target (e.g. LogS).

    Post-projection fields (populated by the projection stage):
        h_atom: ``[N, D]`` projected atom hypervectors.
        h_bond: ``[E, D]`` projected bond hypervectors. ``None`` unless an
            edge featurizer + projection is configured.
    """

    smiles: str
    x_atom: torch.Tensor
    edge_index: torch.Tensor
    x_bond: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None

    h_atom: Optional[torch.Tensor] = None
    h_bond: Optional[torch.Tensor] = None

    @property
    def num_nodes(self) -> int:
        return int(self.x_atom.shape[0])


# ----------------------------------------------------------------------------
# Per-Z heuristic tables (ported verbatim from the legacy implementation).
# Kept as module-level constants because they're reference data, not config.
# ----------------------------------------------------------------------------

_NOBLE_GASES = {2, 10, 18, 36, 54, 86, 118}
_HALOGENS = {9, 17, 35, 53, 85, 117}
_CHALCOGENS = {8, 16, 34, 52, 84, 116}
_PNICTOGENS = {7, 15, 33, 51, 83, 115}
_GROUP14 = {6, 14, 32, 50, 82, 114}
_GROUP13 = {5, 13, 31, 49, 81, 113}
_ALKALI = {1, 3, 11, 19, 37, 55, 87}
_ALKALINE_EARTH = {4, 12, 20, 38, 56, 88}
_D_BLOCK = (
    set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))
)
_F_BLOCK = set(range(57, 72)) | set(range(89, 104))

_DONOR_LIKE = {7, 8, 16}
_ACCEPTOR_LIKE = {7, 8, 9, 16, 17, 35, 53}
_AROMATIC_LIKE = {6, 7, 8, 16, 34, 52}

# Valence electrons keyed by atomic number (Z=0 is reserved for unknown).
_VALENCE_DICT: dict[int, int] = {
    0: 0,
    # Period 1
    1: 1, 2: 2,
    # Period 2
    3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
    # Period 3
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
    # Period 4
    19: 1, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2,
    30: 2, 31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8,
    # Period 5
    37: 1, 38: 2, 39: 2, 40: 2, 41: 2, 42: 2, 43: 2, 44: 2, 45: 2, 46: 2, 47: 2,
    48: 2, 49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8,
    # Period 6
    55: 1, 56: 2,
    57: 2, 58: 2, 59: 2, 60: 2, 61: 2, 62: 2, 63: 2, 64: 2, 65: 2, 66: 2, 67: 2,
    68: 2, 69: 2, 70: 2, 71: 2,
    72: 2, 73: 2, 74: 2, 75: 2, 76: 2, 77: 2, 78: 2, 79: 2, 80: 2,
    81: 3, 82: 4, 83: 5, 84: 6, 85: 7, 86: 8,
    # Period 7
    87: 1, 88: 2,
    89: 2, 90: 2, 91: 2, 92: 2, 93: 2, 94: 2, 95: 2, 96: 2, 97: 2, 98: 2, 99: 2,
    100: 2, 101: 2, 102: 2, 103: 2,
    104: 2, 105: 2, 106: 2, 107: 2, 108: 2, 109: 2, 110: 2, 111: 2, 112: 2,
    113: 3, 114: 4, 115: 5, 116: 6, 117: 7, 118: 8,
}


def _hybridization_one_hot(z: int) -> List[int]:
    """Return [sp, sp2, sp3] one-hot for the legacy Z-based heuristic."""
    if z == 0:
        return [0, 0, 0]
    if z in _NOBLE_GASES:
        return [0, 0, 0]
    if z in _D_BLOCK or z in _F_BLOCK:
        return [0, 0, 1]
    if z in _ALKALI or z in _ALKALINE_EARTH:
        return [0, 0, 1]
    if z in _HALOGENS:
        return [0, 0, 1]
    if z in _CHALCOGENS:
        return [0, 1, 1]
    if z in _PNICTOGENS:
        return [0, 1, 1]
    if z in _GROUP14:
        return [1, 1, 1] if z == 6 else [0, 1, 1]
    if z in _GROUP13:
        return [0, 1, 1]
    if z == 1:
        return [0, 0, 1]
    return [0, 1, 1]


# ----------------------------------------------------------------------------
# AtomFeaturizer
# ----------------------------------------------------------------------------

# Each entry: name → (output dim, builder fn signature `(atom, mol, ring_info) -> List[float]`)
# Some features depend on additional precomputed quantities (Gasteiger charges,
# Crippen contribs, TPSA contribs); these are computed once per molecule in
# ``AtomFeaturizer._compute_mol_aux`` and threaded through.

DEFAULT_ATOM_FEATURES: tuple[str, ...] = (
    "atomic_number",
    "degree",
    "valence_electrons",
    "hybridization",
    "hbond_flags",
)
"""Layout used by the original ``Molecular_Solubility/GVFA`` variant (8 columns)."""

EXTENDED_ATOM_FEATURES: tuple[str, ...] = (
    "atomic_number",        # 1
    "degree",               # 1
    "valence_electrons",    # 1
    "hybridization",        # 3
    "aromaticity_z",        # 1   — Z-based heuristic
    "formal_charge",        # 1
    "hbond_flags",          # 2
    "chirality",            # 2
    "num_attached_h",       # 1
    "gasteiger_charge",     # 1
    "crippen_logp",         # 1
    "tpsa_contrib",         # 1
    "is_in_aromatic_ring",  # 1   — RDKit GetIsAromatic
    "smallest_ring_size",   # 1
)
"""Layout used by the ``GVFA_with_edge`` variant (18 columns)."""

_ATOM_FEATURE_DIMS: dict[str, int] = {
    "atomic_number": 1,
    "degree": 1,
    "valence_electrons": 1,
    "hybridization": 3,
    "aromaticity_z": 1,
    "formal_charge": 1,
    "hbond_flags": 2,
    "chirality": 2,
    "num_attached_h": 1,
    "gasteiger_charge": 1,
    "crippen_logp": 1,
    "tpsa_contrib": 1,
    "is_in_aromatic_ring": 1,
    "smallest_ring_size": 1,
}


class AtomFeaturizer:
    """Per-atom feature extractor.

    Args:
        feature_set: ordered tuple of feature names. Output column order
            matches the order of names in this tuple. Names must be keys of
            :data:`_ATOM_FEATURE_DIMS`.
    """

    def __init__(
        self,
        feature_set: Sequence[str] = DEFAULT_ATOM_FEATURES,
    ) -> None:
        unknown = [f for f in feature_set if f not in _ATOM_FEATURE_DIMS]
        if unknown:
            raise ValueError(
                f"Unknown atom feature(s): {unknown}. "
                f"Allowed: {list(_ATOM_FEATURE_DIMS)}"
            )
        self.feature_set: tuple[str, ...] = tuple(feature_set)

    @property
    def output_dim(self) -> int:
        return sum(_ATOM_FEATURE_DIMS[f] for f in self.feature_set)

    def fit(self, mols: Iterable) -> "AtomFeaturizer":
        """No-op (the featurizer is stateless); kept for API symmetry."""
        return self

    def transform(self, mol) -> torch.Tensor:
        """Compute the ``[N, F_atom]`` feature matrix for a single RDKit Mol."""
        from rdkit import Chem  # noqa: F401  (imported for side-effects elsewhere)

        aux = self._compute_mol_aux(mol)
        rows: List[List[float]] = []
        for atom in mol.GetAtoms():
            row: List[float] = []
            for name in self.feature_set:
                row.extend(self._atom_feature(name, atom, mol, aux))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32)

    def fit_transform(self, mols: Iterable) -> List[torch.Tensor]:
        return [self.transform(m) for m in mols]

    # -- internals -----------------------------------------------------------

    def _compute_mol_aux(self, mol) -> dict:
        """Precompute per-mol auxiliary tables used by multiple features."""
        from rdkit.Chem import AllChem, rdMolDescriptors

        n = mol.GetNumAtoms()
        aux: dict = {
            "gasteiger": [0.0] * n,
            "crippen_logp": [0.0] * n,
            "tpsa": [0.0] * n,
            "smallest_ring_size": [0] * n,
        }

        try:
            AllChem.ComputeGasteigerCharges(mol, throwOnParamFailure=False)
            for i in range(n):
                a = mol.GetAtomWithIdx(i)
                if a.HasProp("_GasteigerCharge"):
                    q = float(a.GetProp("_GasteigerCharge"))
                    if not (np.isnan(q) or np.isinf(q)):
                        aux["gasteiger"][i] = q
        except Exception:
            pass

        try:
            for i, (logp, _) in enumerate(rdMolDescriptors._CalcCrippenContribs(mol)):
                if i < n and not (np.isnan(logp) or np.isinf(logp)):
                    aux["crippen_logp"][i] = float(logp)
        except Exception:
            pass

        try:
            for i, t in enumerate(rdMolDescriptors._CalcTPSAContribs(mol)):
                if i < n and not (np.isnan(t) or np.isinf(t)):
                    aux["tpsa"][i] = float(t)
        except Exception:
            pass

        try:
            ring_info = mol.GetRingInfo()
            for ring in ring_info.AtomRings():
                size = len(ring)
                for aid in ring:
                    cur = aux["smallest_ring_size"][aid]
                    if cur == 0 or size < cur:
                        aux["smallest_ring_size"][aid] = size
        except Exception:
            pass

        return aux

    def _atom_feature(self, name: str, atom, mol, aux: dict) -> List[float]:
        z = atom.GetAtomicNum()
        idx = atom.GetIdx()

        if name == "atomic_number":
            return [float(z)]
        if name == "degree":
            return [float(atom.GetDegree())]
        if name == "valence_electrons":
            return [float(_VALENCE_DICT.get(z, 0))]
        if name == "hybridization":
            return [float(v) for v in _hybridization_one_hot(z)]
        if name == "aromaticity_z":
            return [1.0 if z in _AROMATIC_LIKE else 0.0]
        if name == "formal_charge":
            return [float(atom.GetFormalCharge())]
        if name == "hbond_flags":
            return [
                1.0 if z in _DONOR_LIKE else 0.0,
                1.0 if z in _ACCEPTOR_LIKE else 0.0,
            ]
        if name == "chirality":
            if atom.HasProp("_CIPCode"):
                cip = atom.GetProp("_CIPCode")
                if cip == "R":
                    return [1.0, 0.0]
                if cip == "S":
                    return [0.0, 1.0]
            return [0.0, 0.0]
        if name == "num_attached_h":
            return [float(atom.GetTotalNumHs())]
        if name == "gasteiger_charge":
            return [aux["gasteiger"][idx]]
        if name == "crippen_logp":
            return [aux["crippen_logp"][idx]]
        if name == "tpsa_contrib":
            return [aux["tpsa"][idx]]
        if name == "is_in_aromatic_ring":
            return [1.0 if atom.GetIsAromatic() else 0.0]
        if name == "smallest_ring_size":
            return [float(aux["smallest_ring_size"][idx])]
        raise ValueError(f"Unhandled atom feature: {name}")


# ----------------------------------------------------------------------------
# BondFeaturizer
# ----------------------------------------------------------------------------

DEFAULT_BOND_FEATURES: tuple[str, ...] = (
    "bond_type",
    "is_conjugated",
    "in_ring",
    "bond_length_3d",
)
EXTENDED_BOND_FEATURES: tuple[str, ...] = (
    "bond_type",
    "is_conjugated",
    "in_ring",
    "bond_length_3d",
    "stereo",
)

_BOND_FEATURE_DIMS: dict[str, int] = {
    "bond_type": 1,
    "is_conjugated": 1,
    "in_ring": 1,
    "bond_length_3d": 1,
    "stereo": 1,
}

_BOND_TYPE_ID: dict = {}  # populated lazily after rdkit import

_BOND_STEREO_ID: dict = {}  # populated lazily


def _ensure_bond_lookup_tables() -> None:
    global _BOND_TYPE_ID, _BOND_STEREO_ID
    if _BOND_TYPE_ID and _BOND_STEREO_ID:
        return
    from rdkit import Chem
    _BOND_TYPE_ID = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }
    _BOND_STEREO_ID = {
        Chem.rdchem.BondStereo.STEREONONE: 0.0,
        Chem.rdchem.BondStereo.STEREOCIS: 1.0,
        Chem.rdchem.BondStereo.STEREOTRANS: 2.0,
        Chem.rdchem.BondStereo.STEREOZ: 3.0,
        Chem.rdchem.BondStereo.STEREOE: 4.0,
    }


class BondFeaturizer:
    """Per-bond feature extractor.

    The ``bond_length_3d`` feature requires a 3D conformer; an ETKDGv3 + MMFF
    embed is run lazily the first time it's needed for a given molecule. If
    embedding fails, the length defaults to 0.0 (matching legacy fallback).

    Args:
        feature_set: ordered tuple of feature names from :data:`_BOND_FEATURE_DIMS`.
        embed_seed: random seed for ETKDGv3 conformer generation.
    """

    def __init__(
        self,
        feature_set: Sequence[str] = DEFAULT_BOND_FEATURES,
        embed_seed: int = 0xF00D,
    ) -> None:
        unknown = [f for f in feature_set if f not in _BOND_FEATURE_DIMS]
        if unknown:
            raise ValueError(
                f"Unknown bond feature(s): {unknown}. "
                f"Allowed: {list(_BOND_FEATURE_DIMS)}"
            )
        self.feature_set: tuple[str, ...] = tuple(feature_set)
        self.embed_seed = embed_seed
        self._needs_3d = "bond_length_3d" in feature_set

    @property
    def output_dim(self) -> int:
        return sum(_BOND_FEATURE_DIMS[f] for f in self.feature_set)

    def fit(self, mols: Iterable) -> "BondFeaturizer":
        return self

    def transform(self, mol, edge_index: torch.Tensor) -> torch.Tensor:
        """Build per-edge features aligned with ``edge_index`` ``[2, E]``."""
        _ensure_bond_lookup_tables()
        E = int(edge_index.shape[1])
        if E == 0:
            return torch.zeros((0, self.output_dim), dtype=torch.float32)

        pos = self._embed_3d(mol) if self._needs_3d else None
        rows: List[List[float]] = []
        for e in range(E):
            u = int(edge_index[0, e])
            v = int(edge_index[1, e])
            bond = mol.GetBondBetweenAtoms(u, v)
            row: List[float] = []
            for name in self.feature_set:
                row.extend(self._bond_feature(name, bond, u, v, pos))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32)

    # -- internals -----------------------------------------------------------

    def _embed_3d(self, mol) -> Optional[np.ndarray]:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        try:
            mol3d = Chem.RWMol(mol)
            try:
                Chem.SanitizeMol(mol3d)
            except Exception:
                mol3d.UpdatePropertyCache(strict=False)
            mol3d = Chem.AddHs(mol3d)
            params = AllChem.ETKDGv3()
            params.randomSeed = self.embed_seed
            if AllChem.EmbedMolecule(mol3d, params) != 0:
                return None
            AllChem.MMFFOptimizeMolecule(mol3d)
            conf = mol3d.GetConformer()
            n = mol3d.GetNumAtoms()
            pos = np.zeros((n, 3), dtype=np.float32)
            for i in range(n):
                p = conf.GetAtomPosition(i)
                pos[i] = (p.x, p.y, p.z)
            return pos
        except Exception:
            return None

    def _bond_feature(
        self, name: str, bond, u: int, v: int, pos: Optional[np.ndarray]
    ) -> List[float]:
        if bond is None:
            return [0.0] * _BOND_FEATURE_DIMS[name]
        if name == "bond_type":
            return [float(_BOND_TYPE_ID.get(bond.GetBondType(), 0))]
        if name == "is_conjugated":
            return [float(bond.GetIsConjugated())]
        if name == "in_ring":
            return [float(bond.IsInRing())]
        if name == "bond_length_3d":
            if pos is None or u >= len(pos) or v >= len(pos):
                return [0.0]
            return [float(np.linalg.norm(pos[u] - pos[v]))]
        if name == "stereo":
            return [float(_BOND_STEREO_ID.get(bond.GetStereo(), 0.0))]
        raise ValueError(f"Unhandled bond feature: {name}")


# ----------------------------------------------------------------------------
# DescriptorFeaturizer
# ----------------------------------------------------------------------------

DEFAULT_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v",
    "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3",
    "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8",
    "EState_VSA9",
    "FractionCSP3", "HallKierAlpha", "HeavyAtomCount",
    "Kappa1", "Kappa2", "Kappa3",
    "LabuteASA", "MolLogP", "MolMR", "MolWt",
    "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
    "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13",
    "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5",
    "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9",
    "RingCount",
    "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5",
    "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9",
    "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2",
    "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7",
    "SlogP_VSA8", "SlogP_VSA9",
    "TPSA",
    "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4",
    "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9",
)
"""96 RDKit descriptors used by the legacy ``Traditional_features`` baseline."""


class DescriptorFeaturizer:
    """Molecule-level RDKit descriptor extractor (for the traditional baseline)."""

    def __init__(self, names: Sequence[str] = DEFAULT_DESCRIPTOR_NAMES) -> None:
        self.names = tuple(names)
        self._calc = None

    @property
    def output_dim(self) -> int:
        return len(self.names)

    def _ensure_calc(self) -> None:
        if self._calc is not None:
            return
        from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
        self._calc = MolecularDescriptorCalculator(list(self.names))

    def fit(self, mols: Iterable) -> "DescriptorFeaturizer":
        self._ensure_calc()
        return self

    def transform(self, mol) -> torch.Tensor:
        self._ensure_calc()
        descrs = self._calc.CalcDescriptors(mol)
        # NaN/Inf protection — descriptors occasionally produce NaN on edge cases.
        clean = [0.0 if (np.isnan(v) or np.isinf(v)) else float(v) for v in descrs]
        return torch.tensor(clean, dtype=torch.float32)


# ----------------------------------------------------------------------------
# SMILES → MolGraph helper (the "MolFeaturizer" stage of the pipeline)
# ----------------------------------------------------------------------------


def smiles_to_mol_graph(
    smiles: str,
    target: Optional[float],
    atom_featurizer: AtomFeaturizer,
    bond_featurizer: Optional[BondFeaturizer] = None,
) -> Optional[MolGraph]:
    """Parse SMILES → :class:`MolGraph`. Returns ``None`` if RDKit can't parse the SMILES."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    src: List[int] = []
    dst: List[int] = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        # Both directions, mirroring ZINC convention used by graph_regression_zinc.py.
        src += [u, v]
        dst += [v, u]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    x_atom = atom_featurizer.transform(mol)
    x_bond = bond_featurizer.transform(mol, edge_index) if bond_featurizer is not None else None

    y = None if target is None else torch.tensor([float(target)], dtype=torch.float32)
    return MolGraph(
        smiles=smiles,
        x_atom=x_atom,
        edge_index=edge_index,
        x_bond=x_bond,
        target=y,
    )
