"""
Train/test splitters.

Three split strategies, each returning a list of ``(train_idx, test_idx)``
tuples (a single tuple for random / scaffold; ``k`` tuples for k-fold).

- :func:`random_split` — shuffled index split with a configurable train ratio.
- :func:`scaffold_split` — assigns molecules to the train or test set based
  on Bemis-Murcko scaffold so that scaffolds in the test set never appear
  in the train set. The standard MoleculeNet protocol for solubility.
- :func:`kfold_split` — k-fold over a shuffled index permutation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple

import numpy as np


def random_split(
    n: int, train_ratio: float = 0.9, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Shuffled index split.

    Args:
        n: total number of items.
        train_ratio: fraction assigned to train.
        seed: shuffle seed.

    Returns:
        A list with a single ``(train_idx, test_idx)`` tuple, both numpy
        arrays of indices into the original ordering.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(n * train_ratio))
    return [(perm[:cut], perm[cut:])]


def scaffold_split(
    smiles: Sequence[str], train_ratio: float = 0.8, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Bemis-Murcko scaffold split.

    Molecules are grouped by scaffold; scaffold groups are then assigned
    greedily to train (largest groups first) until ``train_ratio`` is filled,
    with the remainder going to test. This ensures no scaffold straddles
    the split — the standard scaffold-split protocol used by MoleculeNet.

    Falling back to a random split is intentional only when RDKit can't be
    imported (the function will raise instead).

    Args:
        smiles: list of SMILES strings, one per molecule.
        train_ratio: fraction (by molecule count) assigned to train.
        seed: tie-breaker seed for groups of equal size.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    n = len(smiles)
    scaffolds: defaultdict = defaultdict(list)
    for i, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                key = ""
            else:
                key = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
        except Exception:
            key = ""
        scaffolds[key].append(i)

    # Sort groups: largest first, with seeded tie-break for determinism.
    rng = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    sort_keys = [(-len(g), rng.random()) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: sort_keys[i])

    train_target = int(round(n * train_ratio))
    train_idx: List[int] = []
    test_idx: List[int] = []
    for k in order:
        group = groups[k]
        if len(train_idx) + len(group) <= train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    return [(np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64))]


def kfold_split(
    n: int, k: int = 5, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K-fold over a shuffled permutation.

    Args:
        n: total number of items.
        k: number of folds.
        seed: shuffle seed.

    Returns:
        A list of ``k`` ``(train_idx, test_idx)`` tuples.
    """
    if k < 2:
        raise ValueError(f"kfold requires k >= 2, got {k}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train_idx, test_idx))
    return out
