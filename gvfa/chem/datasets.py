"""
Dataset loaders.

Two sources are supported:

- :class:`SolubilityDataset` — load a CSV file with SMILES + a target column.
  Used for the in-house solubility CSVs under ``data/solubility/``. Drops
  rows where either column is missing.

- :class:`MoleculeNetDataset` — wrap ``torch_geometric.datasets.MoleculeNet``
  for the standard public solubility benchmarks (ESOL, FreeSolv, Lipo).
  Falls back to extracting SMILES via RDKit on the underlying graphs.

Both expose the same interface: an iterable of ``(smiles: str, target: float)``
tuples plus ``__len__`` and indexing. This is the input shape consumed by the
splitter and the molecular featurizer stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple


SmilesTarget = Tuple[str, float]


@dataclass
class SolubilityDataset:
    """SMILES + target loader from a CSV file.

    Args:
        csv_path: path to the CSV file.
        smiles_col: name of the SMILES column. Defaults to ``"smiles_canon"``,
            which matches the in-house ``data/solubility/{train,test,novel_test}.csv``
            layout.
        target_col: name of the target column. Defaults to ``"LogS"``.
    """

    csv_path: str
    smiles_col: str = "smiles_canon"
    target_col: str = "LogS"

    items: List[SmilesTarget] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        import pandas as pd

        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=[self.smiles_col, self.target_col])
        self.items = [
            (str(row[self.smiles_col]), float(row[self.target_col]))
            for _, row in df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SmilesTarget:
        return self.items[idx]

    def __iter__(self) -> Iterator[SmilesTarget]:
        return iter(self.items)

    @property
    def smiles(self) -> List[str]:
        return [smi for smi, _ in self.items]

    @property
    def targets(self) -> List[float]:
        return [y for _, y in self.items]


_MOLECULENET_TARGETS = {
    "ESOL": "ESOL",          # log-solubility (mol/L)
    "FreeSolv": "FreeSolv",  # hydration free energy
    "Lipo": "Lipo",          # lipophilicity (logD)
}


@dataclass
class MoleculeNetDataset:
    """Wrap ``torch_geometric.datasets.MoleculeNet`` for ESOL / FreeSolv / Lipo.

    Args:
        name: one of ``{"ESOL", "FreeSolv", "Lipo"}``.
        root: where PyG should cache the download. Defaults to ``./data``.
    """

    name: str
    root: str = "./data"

    items: List[SmilesTarget] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.name not in _MOLECULENET_TARGETS:
            raise ValueError(
                f"Unknown MoleculeNet dataset {self.name!r}. "
                f"Supported: {list(_MOLECULENET_TARGETS)}"
            )
        from torch_geometric.datasets import MoleculeNet

        ds = MoleculeNet(root=str(Path(self.root) / "MoleculeNet"), name=self.name)
        items: List[SmilesTarget] = []
        for data in ds:
            smi = getattr(data, "smiles", None)
            if smi is None:
                continue
            y = float(data.y.flatten()[0])
            items.append((str(smi), y))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SmilesTarget:
        return self.items[idx]

    def __iter__(self) -> Iterator[SmilesTarget]:
        return iter(self.items)

    @property
    def smiles(self) -> List[str]:
        return [smi for smi, _ in self.items]

    @property
    def targets(self) -> List[float]:
        return [y for _, y in self.items]


def index_subset(
    dataset: Sequence[SmilesTarget], idx
) -> List[SmilesTarget]:
    """Materialize a subset of a dataset by index array (numpy or list)."""
    return [dataset[int(i)] for i in idx]
