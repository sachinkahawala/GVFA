"""Compare legacy and modular solubility featurizers on the same molecules.

This is an audit helper, not a training script. It explains why exact metric
parity is not expected when the modular pipeline keeps chemically correct
SMILES-derived features: the legacy GVFA_with_edge builder reconstructs a
molecule from atomic numbers and graph connectivity, which makes every bond a
single bond and changes several downstream atom/bond features.

Usage:
    python -m gvfa.experiments.molecular_solubility.compare_legacy_featurizers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from gvfa.chem import AtomFeaturizer, BondFeaturizer, smiles_to_mol_graph
from gvfa.chem.featurizers import EXTENDED_ATOM_FEATURES, EXTENDED_BOND_FEATURES


def _default_legacy_root() -> Path:
    outer_root = Path(__file__).resolve().parents[4]
    return outer_root / "Molecular_Solubility" / "GVFA_with_edge"


def _sorted_unique(values: Iterable[float]) -> list[float]:
    return sorted({float(v) for v in values})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare legacy GVFA_with_edge and modular molecular features."
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of molecules to inspect.")
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=_default_legacy_root(),
        help="Path to Molecular_Solubility/GVFA_with_edge.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV to inspect. Defaults to <legacy-root>/final_data/solubility_1.csv.",
    )
    args = parser.parse_args(argv)

    legacy_root = args.legacy_root.resolve()
    csv_path = args.csv or legacy_root / "final_data" / "solubility_1.csv"
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))

    from src.load_data import ZINCLikeCSV  # noqa: WPS433
    from src.create_graphs import create_graph_list  # noqa: WPS433

    df = pd.read_csv(csv_path).dropna(subset=["SMILES", "logS"]).head(args.limit)
    legacy_dataset = ZINCLikeCSV(df, smiles_col="SMILES", target_col="logS")
    legacy_graphs = create_graph_list(legacy_dataset)

    atom_featurizer = AtomFeaturizer(EXTENDED_ATOM_FEATURES)
    bond_featurizer = BondFeaturizer(EXTENDED_BOND_FEATURES)

    print("Legacy/modular feature comparison")
    print(f"CSV: {csv_path}")
    print(
        "Expected differences: legacy degree is doubled by directed-edge counting; "
        "legacy reconstructed molecules use single bonds, so bond type, aromatic "
        "ring flags, ring sizes, and 3D lengths can differ."
    )
    print()

    for i, (_, row) in enumerate(df.iterrows()):
        smiles = str(row["SMILES"])
        y = float(row["logS"])
        legacy_graph = legacy_graphs[i]
        modular_graph = smiles_to_mol_graph(
            smiles,
            y,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
        )
        if modular_graph is None:
            print(f"[{i}] {smiles} -> modular parse failed")
            continue

        old_x = legacy_graph.node_features.float()
        new_x = modular_graph.x_atom.float()
        old_e = legacy_graph.edge_attr.float()
        new_e = modular_graph.x_bond.float() if modular_graph.x_bond is not None else torch.zeros(0, 5)

        node_diff = (
            float((old_x - new_x).abs().max())
            if old_x.shape == new_x.shape
            else float("nan")
        )
        edge_diff = (
            float((old_e - new_e).abs().max())
            if old_e.shape == new_e.shape
            else float("nan")
        )
        old_degree_sum = float(old_x[:, 1].sum())
        new_degree_sum = float(new_x[:, 1].sum())
        degree_ratio = old_degree_sum / new_degree_sum if new_degree_sum else float("nan")

        print(f"[{i}] {smiles}")
        print(f"  node_shape legacy={tuple(old_x.shape)} modular={tuple(new_x.shape)}")
        print(f"  edge_shape legacy={tuple(old_e.shape)} modular={tuple(new_e.shape)}")
        print(f"  max_abs_node_diff={node_diff:.6g}  max_abs_edge_diff={edge_diff:.6g}")
        print(f"  degree_sum legacy={old_degree_sum:.3f} modular={new_degree_sum:.3f} ratio={degree_ratio:.3f}")
        print(
            "  bond_type_unique "
            f"legacy={_sorted_unique(old_e[:, 0].tolist()) if old_e.numel() else []} "
            f"modular={_sorted_unique(new_e[:, 0].tolist()) if new_e.numel() else []}"
        )
        print(
            "  aromatic_ring_atom_count "
            f"legacy={int(old_x[:, 16].sum().item())} "
            f"modular={int(new_x[:, 16].sum().item())}"
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
