"""
Graph Regression on ZINC using GVFA

ZINC is a molecular dataset where the task is to regress constrained solubility.
Target = logP - SAS - cycles

Dataset:
- 10k train, 1k val, 1k test graphs (subset=True)
- 28 categorical atom types (one-hot encoded)
- Bond types available but not used by GVFA

Run:
    python -m gvfa.experiments.graph_regression_zinc
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import ZINC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from typing import Literal, List

from gvfa import GVFA
from gvfa.data import random_projection


def process_graphs(
    graphs: List,
    model: GVFA,
    D: int,
    seed: int,
    pooling: str
) -> tuple:
    """
    Process a list of graphs into embeddings and labels.

    Args:
        graphs: List of PyG Data objects
        model: GVFA model
        D: Hypervector dimension
        seed: Random seed for projection
        pooling: Graph pooling type ('sum' or 'mean')

    Returns:
        (embeddings [num_graphs, D*num_layers], labels [num_graphs])
    """
    embeddings = []
    labels = []

    for g in graphs:
        # One-hot encode atom types (28 categories)
        x = F.one_hot(g.x.squeeze().long(), num_classes=28).float()

        # Random projection + sign normalize
        h = random_projection(x, D, seed=seed, normalize=True)

        # GVFA graph embedding
        emb = model.forward_graph(h, g.edge_index, pooling=pooling)
        embeddings.append(emb.numpy())
        labels.append(g.y.item())

    return np.vstack(embeddings), np.array(labels)


def run_experiment(
    D: int = 5000,
    num_layers: int = 3,
    phi: Literal['phi1', 'phi2', 'phi3', 'phi4'] = 'phi3',
    normalize: Literal['sign', 'clip', 'l2', 'none'] = 'sign',
    pooling: Literal['sum', 'mean'] = 'sum',
    alpha: float = 1.0,
    seed: int = 42
) -> dict:
    """
    Run graph regression experiment on ZINC.

    Args:
        D: Hypervector dimension (default: 5000)
        num_layers: Number of GVFA layers (default: 3)
        phi: Phi function variant (default: 'phi3')
        normalize: Normalization function (default: 'sign')
        pooling: Graph pooling type (default: 'sum')
        alpha: Ridge regression alpha (default: 1.0)
        seed: Random seed (default: 42)

    Returns:
        Dictionary with train_mae and test_mae
    """
    # Load ZINC dataset (subset=True for 12k graphs)
    train_data = ZINC(root='./data', subset=True, split='train')
    val_data = ZINC(root='./data', subset=True, split='val')
    test_data = ZINC(root='./data', subset=True, split='test')

    # Combine train + val for training (no hyperparameter tuning needed)
    train_graphs = list(train_data) + list(val_data)
    test_graphs = list(test_data)

    print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")

    # GVFA model
    model = GVFA(num_layers=num_layers, phi=phi, normalize=normalize)

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Process graphs
    print("Processing training graphs...")
    X_train, y_train = process_graphs(train_graphs, model, D, seed, pooling)
    print("Processing test graphs...")
    X_test, y_test = process_graphs(test_graphs, model, D, seed, pooling)

    print(f"Embedding shape: {X_train.shape}")

    # StandardScaler + Ridge regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = Ridge(alpha=alpha)
    reg.fit(X_train_scaled, y_train)

    # Evaluate
    train_pred = reg.predict(X_train_scaled)
    test_pred = reg.predict(X_test_scaled)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'config': {
            'D': D,
            'num_layers': num_layers,
            'phi': phi,
            'normalize': normalize,
            'pooling': pooling,
            'alpha': alpha,
            'seed': seed
        }
    }


def main():
    """Run baseline experiment and print results."""
    print("=" * 60)
    print("GVFA Graph Regression on ZINC")
    print("=" * 60)
    print()

    # Baseline configuration
    print("Configuration:")
    print("  D = 5000")
    print("  num_layers = 3")
    print("  phi = phi3")
    print("  normalize = sign")
    print("  pooling = sum")
    print()

    results = run_experiment()

    print()
    print("-" * 40)
    print(f"Train MAE: {results['train_mae']:.4f}")
    print(f"Test MAE:  {results['test_mae']:.4f}")
    print("-" * 40)
    print()
    print("Reference: State-of-art GNNs achieve MAE ~0.07-0.08")
    print("=" * 60)


if __name__ == '__main__':
    main()
