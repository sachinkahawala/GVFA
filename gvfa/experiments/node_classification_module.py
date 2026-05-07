"""
Node Classification Experiment using GVFA Module

Demonstrates usage of the GVFA class for node classification.
Reproduces Table 2 from the paper.

Run:
    python -m gvfa.experiments.node_classification_module
"""

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score

from gvfa import GVFA
from gvfa.data import random_projection


def run_experiment(
    dataset_name: str,
    D: int = 5000,
    num_layers: int = 3,
    phi: str = 'phi3',
    normalize: str = 'sign',
    num_runs: int = 20
) -> tuple:
    """
    Run node classification using GVFA module.

    Args:
        dataset_name: 'Cora', 'citeseer', or 'Pubmed'
        D: Hypervector dimension
        num_layers: Number of GVFA layers
        phi: Combine function ('phi1', 'phi2', 'phi3', 'phi4')
        normalize: Normalization ('sign', 'clip', 'l2', 'none')
        num_runs: Number of runs

    Returns:
        (mean_accuracy, std_accuracy)
    """
    # Load dataset
    dataset = Planetoid(root='./data', name=dataset_name)
    data = dataset[0]

    features = data.x.float()
    labels = data.y.numpy()
    train_mask = (data.train_mask | data.val_mask).numpy()
    test_mask = data.test_mask.numpy()
    edge_index = to_undirected(data.edge_index)

    # Initialize GVFA model
    model = GVFA(
        num_layers=num_layers,
        phi=phi,
        normalize=normalize
    )

    accuracies = []
    for run in range(num_runs):
        # Project features with sign normalization
        h = random_projection(features, D, seed=run, normalize=True)

        # Run GVFA forward pass
        embeddings = model(h, edge_index, return_all_levels=True)
        X = embeddings.numpy()

        # StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        # RidgeClassifierCV
        alphas = np.logspace(-1, 4, 10)
        alphas = np.insert(alphas, 0, 10**-8)
        clf = RidgeClassifierCV(alphas=alphas)
        clf.fit(X_train, labels[train_mask])

        acc = accuracy_score(labels[test_mask], clf.predict(X_test)) * 100
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)


def main():
    """Run experiments on all datasets."""
    print("=" * 60)
    print("GVFA Node Classification (using GVFA module)")
    print("=" * 60)
    print()

    # Model configuration
    config = {
        'D': 5000,
        'num_layers': 3,
        'phi': 'phi3',
        'normalize': 'sign'
    }
    print(f"Config: D={config['D']}, layers={config['num_layers']}, "
          f"phi={config['phi']}, norm={config['normalize']}")
    print()

    datasets = ['Cora', 'citeseer', 'Pubmed']
    targets = {
        'Cora': '85.1 ± 0.5',
        'citeseer': '75.8 ± 0.4',
        'Pubmed': '83.3 ± 0.6'
    }

    print(f"{'Dataset':<12} {'Result':<15} {'Target':<15}")
    print("-" * 42)

    for dataset in datasets:
        mean, std = run_experiment(dataset, **config)
        result = f"{mean:.1f} ± {std:.1f}"
        print(f"{dataset:<12} {result:<15} {targets[dataset]:<15}")

    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
