"""
Node Classification Experiment for GVFA

Reproduces Table 2 from the paper using Φ3 (equation 11, delta=0) + sign normalization.

Target Results (Paper Table 2):
| Dataset   | Paper Result  |
|-----------|---------------|
| Cora      | 85.1 ± 0.5    |
| citeseer  | 75.8 ± 0.4    |
| Pubmed    | 83.3 ± 0.6    |

Run:
    python -m gvfa.experiments.node_classification
"""

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score


def random_projection(features: torch.Tensor, D: int, seed: int) -> torch.Tensor:
    """
    Project features to D dimensions and binarize with sign.

    Args:
        features: Input features [N, input_dim]
        D: Target dimension
        seed: Random seed

    Returns:
        Projected and binarized features [N, D]
    """
    torch.manual_seed(seed)
    input_dim = features.shape[1]
    W = torch.randn(input_dim, D) / np.sqrt(D)
    h = features @ W
    h = torch.sign(h)  # Binarize after projection
    return h


def gvfa_forward(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    num_layers: int = 3
) -> torch.Tensor:
    """
    Run GVFA with Φ3 (equation 11) and sign normalization.

    Implements Algorithm 1 from paper:
    1. Initialize H^(0) from projected features (already sign-normalized)
    2. For each layer k (num_layers - 1 iterations):
       - Aggregate: F(i) = Σ_{j∈N_i} H_j^(k)
       - Combine: Φ3 = ρ(H_i^(k) + F(i))
       - Normalize: sign(...)
    3. Concatenate all level representations

    Args:
        features: Projected node features [N, D] (already binarized)
        edge_index: Graph edges [2, E]
        num_layers: Number of layers (default: 3, giving 3 total levels)

    Returns:
        Concatenated representations [N, D * num_layers]
    """
    num_nodes = features.shape[0]
    D = features.shape[1]
    row, col = edge_index

    h = features
    all_levels = [h]

    # num_layers - 1 iterations (matching original GraphCNN behavior)
    for _ in range(num_layers - 1):
        # F(i) = Σ_{j∈N_i} H_j
        f = torch.zeros(num_nodes, D, dtype=h.dtype, device=h.device)
        f.index_add_(0, row, h[col])
        # Φ3: ρ(H_i ⊕ F(i)) - add then permute
        h = torch.roll(h + f, shifts=1, dims=-1)
        # Λ = sign normalization
        h = torch.sign(h)
        all_levels.append(h)

    return torch.cat(all_levels, dim=1)


def run_experiment(
    dataset_name: str,
    D: int = 5000,
    num_layers: int = 3,
    num_runs: int = 20
) -> tuple:
    """
    Run node classification experiment.

    Args:
        dataset_name: 'Cora', 'citeseer', or 'Pubmed'
        D: Hypervector dimension (default: 5000)
        num_layers: Number of GVFA layers (default: 3)
        num_runs: Number of experiment runs (default: 20)

    Returns:
        (mean_accuracy, std_accuracy)
    """
    # Load dataset
    dataset = Planetoid(root='./data', name=dataset_name)
    data = dataset[0]

    features = data.x.float()
    labels = data.y.numpy()
    # Combine train + validation masks (no hyperparameter tuning needed)
    train_mask = (data.train_mask | data.val_mask).numpy()
    test_mask = data.test_mask.numpy()

    # Ensure edges are undirected
    edge_index = to_undirected(data.edge_index)

    accuracies = []
    for run in range(num_runs):
        # Random projection with sign normalization
        h = random_projection(features, D, seed=run)

        # Run GVFA
        embeddings = gvfa_forward(h, edge_index, num_layers)
        X = embeddings.numpy()

        # StandardScaler normalization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        # RidgeClassifierCV with specific alpha range
        alphas = np.logspace(-1, 4, 10)
        alphas = np.insert(alphas, 0, 10**-8)
        clf = RidgeClassifierCV(alphas=alphas)
        clf.fit(X_train, labels[train_mask])

        # Evaluate
        pred = clf.predict(X_test)
        acc = accuracy_score(labels[test_mask], pred) * 100
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)


def main():
    """Run experiments on all datasets and print results."""
    print("=" * 50)
    print("GVFA Node Classification Experiment")
    print("Reproducing Table 2 from paper")
    print("=" * 50)
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
        mean, std = run_experiment(dataset)
        result = f"{mean:.1f} ± {std:.1f}"
        target = targets[dataset]
        print(f"{dataset:<12} {result:<15} {target:<15}")

    print()
    print("=" * 50)


if __name__ == '__main__':
    main()
