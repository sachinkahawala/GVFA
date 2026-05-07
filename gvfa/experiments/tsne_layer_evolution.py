"""
Layer-by-Layer t-SNE Evolution for GVFA

Shows how node representations evolve across GVFA layers.
Produces a 1×K horizontal strip (one panel per layer) demonstrating
how neighborhood aggregation progressively structures the embedding space.

Uses Φ3 (best-performing) by default.

Run:
    python -m gvfa.experiments.tsne_layer_evolution
    python -m gvfa.experiments.tsne_layer_evolution --dataset Cora --phi phi3
    python -m gvfa.experiments.tsne_layer_evolution --dataset citeseer --save-dir figures/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

from gvfa.models.aggregate import aggregate_neighbors
from gvfa.models.combine import PHI_FUNCTIONS
from gvfa.models.normalize import sign_normalize, clip_normalize, l2_normalize
from gvfa.data import random_projection


# ── Palette (Okabe-Ito, up to 7 classes) ────────────────────────────────────
PALETTE = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442',
    '#0072B2', '#D55E00', '#CC79A7',
]

# ── Per-dataset defaults ────────────────────────────────────────────────────
DATASET_CONFIGS = {
    'Cora': {
        'name': 'Cora',
        'D': 5000,
        'num_layers': 5,
        'phi': 'phi3',
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
        'num_layers': 5,
        'phi': 'phi3',
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
        'num_layers': 5,
        'phi': 'phi3',
        'normalize': 'sign',
        'perplexity': 30,
        'class_names': [
            'Diabetes Type 1', 'Diabetes Type 2', 'Experimental',
        ],
    },
}

PHI_LABELS = {
    'phi1': r'$\Phi_1$',
    'phi2': r'$\Phi_2$',
    'phi3': r'$\Phi_3$',
    'phi4': r'$\Phi_4$',
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _apply_normalize(h: torch.Tensor, norm: str, kappa: float = 1.0) -> torch.Tensor:
    if norm == 'sign':
        return sign_normalize(h)
    elif norm == 'clip':
        return clip_normalize(h, kappa)
    elif norm == 'l2':
        return l2_normalize(h)
    return h


def compute_per_layer_embeddings(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    phi: str = 'phi3',
    D: int = 5000,
    num_layers: int = 3,
    normalize: str = 'sign',
    seed: int = 42,
) -> list:
    """
    Compute GVFA node embeddings at *each* layer separately.

    Returns:
        List of K numpy arrays, each [N, D], one per layer.
    """
    phi_fn = PHI_FUNCTIONS[phi]
    h = random_projection(features, D, seed=seed, normalize=True)
    num_nodes = h.shape[0]

    layers = [h.detach().numpy()]

    for _ in range(num_layers - 1):
        f = aggregate_neighbors(h, edge_index, num_nodes)
        h = phi_fn(h, f)
        h = _apply_normalize(h, normalize)
        layers.append(h.detach().numpy())

    return layers


def run_tsne(embeddings: np.ndarray, perplexity: float = 30, seed: int = 42) -> np.ndarray:
    tsne = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=seed, init='pca', learning_rate='auto',
    )
    return tsne.fit_transform(embeddings)


# ── Plotting ─────────────────────────────────────────────────────────────────

def _scatter(ax, proj, labels, colors, node_size):
    for cls_idx, color in enumerate(colors):
        mask = labels == cls_idx
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            color=color, s=node_size, alpha=0.80,
            edgecolors='none', rasterized=True,
        )


def plot_layer_evolution(
    layer_projs: list,
    labels: np.ndarray,
    class_names: list,
    phi: str = 'phi3',
    dataset_name: str = 'Cora',
    node_size: int = 16,
    save_path: str = None,
    panel_titles: list = None,
) -> None:
    """
    Plot a 1×K horizontal strip, one panel per entry in layer_projs.
    """
    K = len(layer_projs)
    num_classes = len(np.unique(labels))
    colors = PALETTE[:num_classes]

    panel_w = min(5.5, 28 / K)  # scale panels so total width stays ≤ ~28"
    fig, axes = plt.subplots(
        1, K,
        figsize=(panel_w * K, 5.5),
        constrained_layout=False,
    )
    fig.patch.set_facecolor('white')

    phi_label = PHI_LABELS.get(phi, phi)
    fig.suptitle(
        f'Layer-by-Layer t-SNE Evolution  —  {dataset_name} ({phi_label}, sign norm)',
        fontsize=14, fontweight='bold',
    )

    if panel_titles is None:
        panel_titles = [
            r'Layer 0: $H^{(0)}$' + '\n(projected features)',
        ] + [
            rf'Layer {k}: $H^{{({k})}}$' + f'\n({k} aggregation{"s" if k > 1 else ""})'
            for k in range(1, K)
        ]

    for k, (proj, title) in enumerate(zip(layer_projs, panel_titles)):
        ax = axes[k] if K > 1 else axes
        _scatter(ax, proj, labels, colors, node_size)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('#f8f8f8')

        # Arrow between panels
        if k < K - 1:
            fig.text(
                (k + 0.95) / K, 0.48, '\u2192',
                fontsize=24, ha='center', va='center',
                transform=fig.transFigure, color='#888888',
            )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=colors[i], markersize=9,
                    label=class_names[i])
        for i in range(num_classes)
    ]
    fig.legend(
        handles=handles, loc='lower center',
        ncol=min(num_classes, 7), fontsize=10,
        frameon=False, bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.85, wspace=0.08)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def run_single_dataset(
    dataset_name: str,
    phi: str = None,
    num_layers: int = None,
    seed: int = 42,
    node_size: int = 16,
    save_path: str = None,
):
    cfg = DATASET_CONFIGS[dataset_name]
    phi = phi or cfg['phi']
    D = cfg['D']
    num_layers = num_layers or cfg['num_layers']
    normalize = cfg['normalize']
    perplexity = cfg['perplexity']
    class_names = cfg['class_names']

    print("=" * 60)
    print(f"Layer-by-Layer t-SNE — {dataset_name} ({phi})")
    print("=" * 60)

    dataset = Planetoid(root='./data', name=cfg['name'])
    data = dataset[0]
    features = data.x.float()
    labels = data.y.numpy()
    edge_index = to_undirected(data.edge_index)

    print(f"Nodes: {features.shape[0]}, Classes: {len(class_names)}, Layers: {num_layers}")
    print()

    print("Computing per-layer embeddings ...")
    print("  Raw: running t-SNE on raw features ...")
    raw_proj = run_tsne(features.numpy(), perplexity=perplexity, seed=seed)
    print("  Done.")

    layers = compute_per_layer_embeddings(
        features, edge_index, phi=phi, D=D,
        num_layers=num_layers, normalize=normalize, seed=seed,
    )

    # Build titles and projections — raw first, then GVFA layers
    panel_titles = ['Raw Node Features\n(no GVFA)']
    layer_projs = [raw_proj]

    panel_titles += [
        r'Layer 0: $H^{(0)}$' + '\n(projected features)',
    ] + [
        rf'Layer {k}: $H^{{({k})}}$' + f'\n({k} aggregation{"s" if k > 1 else ""})'
        for k in range(1, num_layers)
    ]

    for k, emb in enumerate(layers):
        print(f"  Layer {k}: shape {emb.shape}, running t-SNE ...")
        layer_projs.append(run_tsne(emb, perplexity=perplexity, seed=seed))
    print("  Done.")

    print("Plotting ...")
    plot_layer_evolution(
        layer_projs, labels, class_names=class_names,
        phi=phi, dataset_name=dataset_name,
        node_size=node_size, save_path=save_path,
        panel_titles=panel_titles,
    )
    print("Complete.")


def main(
    datasets: list = None,
    phi: str = None,
    num_layers: int = None,
    seed: int = 42,
    node_size: int = 16,
    save_dir: str = None,
):
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())

    for ds_name in datasets:
        save_path = None
        if save_dir is not None:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = str(Path(save_dir) / f'tsne_layers_{ds_name.lower()}.png')

        run_single_dataset(ds_name, phi=phi, num_layers=num_layers, seed=seed,
                           node_size=node_size, save_path=save_path)
        print()


if __name__ == '__main__':
    import argparse

    valid_datasets = list(DATASET_CONFIGS.keys())

    parser = argparse.ArgumentParser(
        description='GVFA Layer-by-Layer t-SNE Evolution',
    )
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
                        choices=valid_datasets,
                        help=f'Dataset(s) (default: all {valid_datasets})')
    parser.add_argument('--phi', type=str, default=None,
                        choices=['phi1', 'phi2', 'phi3', 'phi4'],
                        help='Phi function (default: from dataset config, phi3)')
    parser.add_argument('--num-layers', '--layers', dest='num_layers', type=int, default=None,
                        help='Number of GVFA layers (default: 5 from config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--node-size', type=int, default=16, help='Scatter point size')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save figures')
    args = parser.parse_args()

    main(
        datasets=args.dataset,
        phi=args.phi,
        num_layers=args.num_layers,
        seed=args.seed,
        node_size=args.node_size,
        save_dir=args.save_dir,
    )