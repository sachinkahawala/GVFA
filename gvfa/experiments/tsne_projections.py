"""
TSNE Projection Visualization for GVFA Configurations

Visualizes node embeddings from 4 GVFA configurations (Φ1–Φ4) on Planetoid
datasets (Cora, Citeseer, Pubmed) using t-SNE. Produces a comparison layout:
  - Left panel: raw node features (no GVFA)
  - Right 2×2: embeddings from Φ1–Φ4

Run (all datasets):
    python -m gvfa.experiments.tsne_projections

Run (single dataset):
    python -m gvfa.experiments.tsne_projections --dataset Cora
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

from gvfa import GVFA
from gvfa.data import random_projection


PHI_LABELS = {
    'raw':  'Raw Node Features (no GVFA)',
    'phi1': r'$\Phi_1$: $H_i \oplus \rho(F_i)$',
    'phi2': r'$\Phi_2$: $H_i \oplus H_i \odot \rho(F_i)$',
    'phi3': r'$\Phi_3$: $\rho(H_i \oplus F_i)$',
    'phi4': r'$\Phi_4$: $\rho(H_i \oplus H_i \odot F_i)$',
}

# Okabe-Ito colorblind-safe palette (7 classes)
PALETTE = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermilion
    '#CC79A7',  # pink
]

# ── Per-dataset defaults ─────────────────────────────────────────────────────
DATASET_CONFIGS = {
    'Cora': {
        'name': 'Cora',
        'D': 5000,
        'num_layers': 3,
        'normalize': 'sign',
        'perplexity': 30,
        'class_names': [
            'Case-Based', 'Genetic Alg.', 'Neural Nets',
            'Prob. Methods', 'Reinforcement', 'Rule Learning', 'Theory',
        ],
    },
    'citeseer': {
        'name': 'citeseer',
        'D': 5000,
        'num_layers': 3,
        'normalize': 'sign',
        'perplexity': 30,
        'class_names': [
            'Agents', 'AI', 'DB',
            'IR', 'ML', 'HCI',
        ],
    },
    'Pubmed': {
        'name': 'Pubmed',
        'D': 5000,
        'num_layers': 3,
        'normalize': 'sign',
        'perplexity': 30,
        'class_names': [
            'Diabetes Type 1', 'Diabetes Type 2', 'Experimental',
        ],
    },
}


def compute_embeddings(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    phi: str,
    D: int = 5000,
    num_layers: int = 3,
    normalize: str = 'sign',
    seed: int = 42,
) -> np.ndarray:
    """
    Compute GVFA node embeddings for a given phi configuration.

    Args:
        features: Raw node features [N, F]
        edge_index: Graph edges [2, E]
        phi: Combine function ('phi1'–'phi4')
        D: Hypervector dimension
        num_layers: Number of GVFA layers
        normalize: Normalization type
        seed: Random seed for projection

    Returns:
        Node embeddings as numpy array [N, D * num_layers]
    """
    model = GVFA(num_layers=num_layers, phi=phi, normalize=normalize)
    h = random_projection(features, D, seed=seed, normalize=True)
    embeddings = model(h, edge_index, return_all_levels=True)
    return embeddings.detach().numpy()


def run_tsne(embeddings: np.ndarray, perplexity: float = 30, seed: int = 42) -> np.ndarray:
    """
    Run t-SNE on embeddings.

    Args:
        embeddings: High-dimensional embeddings [N, D]
        perplexity: t-SNE perplexity
        seed: Random state

    Returns:
        2D projections [N, 2]
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto')
    return tsne.fit_transform(embeddings)


def _style_ax(ax: plt.Axes, title: str, fontsize: int = 13) -> None:
    """Apply common axis styling."""
    ax.set_title(title, fontsize=fontsize, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scatter(ax: plt.Axes, proj: np.ndarray, labels: np.ndarray,
             colors: list, node_size: int) -> None:
    """Draw a scatter plot using the custom palette."""
    for cls_idx, color in enumerate(colors):
        mask = labels == cls_idx
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            color=color, s=node_size, alpha=0.80,
            edgecolors='none', rasterized=True,
        )


def plot_tsne_comparison(
    projections: dict,
    labels: np.ndarray,
    class_names: list = None,
    node_size: int = 18,
    save_path: str = None,
    title: str = 'Raw Features vs. GVFA t-SNE Projections',
) -> None:
    """
    Slide-friendly (16:9) comparison layout:
      Left column  — Raw node features (spans full height)
      Right 2×2   — Φ1–Φ4 GVFA configurations

    Args:
        projections: Ordered dict with keys 'raw', 'phi1', 'phi2', 'phi3', 'phi4',
                     each mapping to a 2D t-SNE projection [N, 2]
        labels: Node class labels [N]
        class_names: Optional list of class names
        node_size: Scatter point size (default 18)
        save_path: Optional path to save the figure
        title: Figure title
    """
    num_classes = len(np.unique(labels))
    colors = PALETTE[:num_classes]

    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    # ── Figure & GridSpec ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 8.5))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        title,
        fontsize=16, fontweight='bold', y=0.99,
    )

    # 2 rows × 3 cols; left col is wider (raw), right 2 cols hold phi panels
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1.25, 1, 1],
        hspace=0.28,
        wspace=0.08,
        left=0.03,
        right=0.97,
        top=0.91,
        bottom=0.12,
    )

    # ── Left: raw features (spans both rows) ────────────────────────────────
    ax_raw = fig.add_subplot(gs[:, 0])
    _scatter(ax_raw, projections['raw'], labels, colors, node_size)
    _style_ax(ax_raw, PHI_LABELS['raw'], fontsize=13)
    ax_raw.set_facecolor('#f8f8f8')

    # ── Right 2×2: phi1–phi4 ────────────────────────────────────────────────
    phi_grid = [('phi1', 0, 1), ('phi2', 0, 2), ('phi3', 1, 1), ('phi4', 1, 2)]
    for phi_name, row, col in phi_grid:
        ax = fig.add_subplot(gs[row, col])
        _scatter(ax, projections[phi_name], labels, colors, node_size)
        _style_ax(ax, PHI_LABELS[phi_name], fontsize=12)
        ax.set_facecolor('#f8f8f8')

    # ── Shared legend at bottom ──────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[i], markersize=9,
                   label=class_names[i])
        for i in range(num_classes)
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=num_classes,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.2,
    )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# Keep the original 2×2 grid function for backward compatibility
def plot_tsne_grid(
    projections: dict,
    labels: np.ndarray,
    class_names: list = None,
    node_size: int = 18,
    save_path: str = None,
) -> None:
    """
    Plot a 2×2 grid of t-SNE projections (phi1–phi4 only).

    Args:
        projections: Dict mapping phi name to 2D projections [N, 2]
        labels: Node class labels [N]
        class_names: Optional list of class names
        node_size: Scatter point size (default 18)
        save_path: Optional path to save the figure
    """
    num_classes = len(np.unique(labels))
    colors = PALETTE[:num_classes]

    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('GVFA t-SNE Projections on Cora', fontsize=18, fontweight='bold', y=0.98)

    phi_names = ['phi1', 'phi2', 'phi3', 'phi4']
    for idx, phi_name in enumerate(phi_names):
        ax = axes[idx // 2][idx % 2]
        _scatter(ax, projections[phi_name], labels, colors, node_size)
        _style_ax(ax, PHI_LABELS[phi_name], fontsize=14)
        ax.set_facecolor('#f8f8f8')

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[i], markersize=8,
                   label=class_names[i])
        for i in range(num_classes)
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=min(num_classes, 7),
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def run_single_dataset(
    dataset_name: str,
    D: int = None,
    num_layers: int = None,
    normalize: str = None,
    perplexity: float = None,
    seed: int = 42,
    node_size: int = 18,
    save_path: str = None,
):
    """
    Run the t-SNE comparison for a single dataset.

    Parameters default to those stored in DATASET_CONFIGS when not supplied.

    Args:
        dataset_name: 'Cora', 'citeseer', or 'Pubmed'
        D: Hypervector dimension (default from config)
        num_layers: Number of GVFA layers (default from config)
        normalize: Normalization type (default from config)
        perplexity: t-SNE perplexity (default from config)
        seed: Random seed
        node_size: Scatter point size
        save_path: Optional path to save the figure
    """
    cfg = DATASET_CONFIGS[dataset_name]
    D = D or cfg['D']
    num_layers = num_layers or cfg['num_layers']
    normalize = normalize or cfg['normalize']
    perplexity = perplexity or cfg['perplexity']
    class_names = cfg['class_names']

    print("=" * 60)
    print(f"GVFA t-SNE Projection — {dataset_name}")
    print("=" * 60)

    dataset = Planetoid(root='./data', name=cfg['name'])
    data = dataset[0]
    features = data.x.float()
    labels = data.y.numpy()
    edge_index = to_undirected(data.edge_index)

    print(f"Nodes: {features.shape[0]}, Features: {features.shape[1]}, Classes: {len(class_names)}")
    print(f"Config: D={D}, layers={num_layers}, normalize={normalize}")
    print()

    projections = {}

    # ── Raw features (as-is, no projection) ──────────────────────────────────
    print("Running t-SNE on raw features ...")
    raw_np = features.numpy()
    print(f"  Feature shape: {raw_np.shape}")
    projections['raw'] = run_tsne(raw_np, perplexity=perplexity, seed=seed)
    print("  Done.")

    # ── GVFA configurations ───────────────────────────────────────────────────
    for phi in ['phi1', 'phi2', 'phi3', 'phi4']:
        print(f"Computing embeddings for {PHI_LABELS[phi]} ...")
        emb = compute_embeddings(features, edge_index, phi=phi, D=D, num_layers=num_layers,
                                 normalize=normalize, seed=seed)
        print(f"  Embedding shape: {emb.shape}")
        projections[phi] = run_tsne(emb, perplexity=perplexity, seed=seed)
        print("  Done.")

    print()
    print("Plotting ...")
    title = f'Raw Features vs. GVFA t-SNE Projections — {dataset_name}'
    plot_tsne_comparison(projections, labels, class_names=class_names,
                         node_size=node_size, save_path=save_path, title=title)
    print("Complete.")


def main(
    datasets: list = None,
    seed: int = 42,
    node_size: int = 18,
    save_dir: str = None,
):
    """
    Run t-SNE comparison for one or more datasets.

    Args:
        datasets: List of dataset names (default: all three)
        seed: Random seed
        node_size: Scatter point size
        save_dir: If set, save figures as <save_dir>/tsne_<dataset>.png
    """
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())

    for ds_name in datasets:
        save_path = None
        if save_dir is not None:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = str(Path(save_dir) / f'tsne_{ds_name.lower()}.png')

        run_single_dataset(ds_name, seed=seed, node_size=node_size, save_path=save_path)
        print()


if __name__ == '__main__':
    import argparse

    valid_datasets = list(DATASET_CONFIGS.keys())

    parser = argparse.ArgumentParser(description='GVFA t-SNE Projections')
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
                        choices=valid_datasets,
                        help=f'Dataset(s) to visualize (default: all {valid_datasets})')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--node-size', type=int, default=18, help='Scatter point size')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save figures (e.g. figures/)')
    args = parser.parse_args()

    main(
        datasets=args.dataset,
        seed=args.seed,
        node_size=args.node_size,
        save_dir=args.save_dir,
    )
