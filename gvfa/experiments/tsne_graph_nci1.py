"""
TSNE Projection Visualization for GVFA on NCI1 (Graph Classification)

Visualizes graph-level embeddings from 4 GVFA configurations (Φ1–Φ4)
on the NCI1 dataset using t-SNE. Layout: raw graph features on the left,
Φ1–Φ4 in a 2×2 grid on the right.

NCI1: 4110 chemical compound graphs, 2 classes (active / inactive),
      37 discrete atom types one-hot encoded.

Run:
    python -m gvfa.experiments.tsne_graph_nci1
    python -m gvfa.experiments.tsne_graph_nci1 --save-dir figures/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from typing import List

from gvfa import GVFA
from gvfa.data import random_projection, S2VGraph
from gvfa.data.loader import load_tu_dataset


# ── Labels ───────────────────────────────────────────────────────────────────
PHI_LABELS = {
    'raw':  'Raw Graph Features (no GVFA)',
    'phi1': r'$\Phi_1$: $H_i \oplus \rho(F_i)$',
    'phi2': r'$\Phi_2$: $H_i \oplus H_i \odot \rho(F_i)$',
    'phi3': r'$\Phi_3$: $\rho(H_i \oplus F_i)$',
    'phi4': r'$\Phi_4$: $\rho(H_i \oplus H_i \odot F_i)$',
}

# Okabe-Ito colorblind-safe palette
PALETTE = [
    '#0072B2',  # blue  (class 0 — inactive)
    '#D55E00',  # vermilion (class 1 — active)
]

# ── Default config ───────────────────────────────────────────────────────────
NCI1_CONFIG = {
    'D': 5000,
    'num_layers': 3,
    'normalize': 'sign',
    'pooling': 'sum',
    'perplexity': 30,
    'class_names': ['Inactive', 'Active'],
    'data_dir': 'Old/new_code/dataset',
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _graph_raw_features(graphs: List[S2VGraph]) -> np.ndarray:
    """
    Compute a simple raw graph-level feature for each graph by summing
    node features (bag-of-nodes), no GVFA.

    Returns:
        [num_graphs, feat_dim]
    """
    embeddings = []
    for g in graphs:
        embeddings.append(g.node_features.sum(dim=0).numpy())
    return np.vstack(embeddings)


def _graph_gvfa_embeddings(
    graphs: List[S2VGraph],
    phi: str,
    D: int,
    num_layers: int,
    normalize: str,
    pooling: str,
    seed: int,
) -> np.ndarray:
    """
    Compute GVFA graph-level embeddings for a list of graphs.

    Returns:
        [num_graphs, D * num_layers]
    """
    model = GVFA(num_layers=num_layers, phi=phi, normalize=normalize)
    embeddings = []

    for g in graphs:
        h = random_projection(g.node_features, D, seed=seed, normalize=True)
        emb = model.forward_graph(h, g.edge_mat, pooling=pooling)
        embeddings.append(emb.detach().numpy())

    return np.vstack(embeddings)


def run_tsne(embeddings: np.ndarray, perplexity: float = 30, seed: int = 42) -> np.ndarray:
    """Run t-SNE on embeddings → [N, 2]."""
    tsne = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=seed, init='pca', learning_rate='auto',
    )
    return tsne.fit_transform(embeddings)


# ── Plotting ─────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str, fontsize: int = 13) -> None:
    ax.set_title(title, fontsize=fontsize, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scatter(ax, proj, labels, colors, node_size):
    for cls_idx, color in enumerate(colors):
        mask = labels == cls_idx
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            color=color, s=node_size, alpha=0.70,
            edgecolors='none', rasterized=True,
        )


def plot_tsne_comparison(
    projections: dict,
    labels: np.ndarray,
    class_names: list,
    node_size: int = 14,
    save_path: str = None,
    title: str = 'Raw Features vs. GVFA t-SNE — NCI1',
) -> None:
    """
    Slide-friendly (16:9) comparison layout.
    Left: raw, Right 2×2: Φ1–Φ4.
    """
    num_classes = len(np.unique(labels))
    colors = PALETTE[:num_classes]

    fig = plt.figure(figsize=(20, 8.5))
    fig.patch.set_facecolor('white')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.25, 1, 1],
        hspace=0.28, wspace=0.08,
        left=0.03, right=0.97,
        top=0.91, bottom=0.12,
    )

    # left — raw
    ax_raw = fig.add_subplot(gs[:, 0])
    _scatter(ax_raw, projections['raw'], labels, colors, node_size)
    _style_ax(ax_raw, PHI_LABELS['raw'], fontsize=13)
    ax_raw.set_facecolor('#f8f8f8')

    # right 2×2 — phi1‒phi4
    phi_grid = [('phi1', 0, 1), ('phi2', 0, 2), ('phi3', 1, 1), ('phi4', 1, 2)]
    for phi_name, row, col in phi_grid:
        ax = fig.add_subplot(gs[row, col])
        _scatter(ax, projections[phi_name], labels, colors, node_size)
        _style_ax(ax, PHI_LABELS[phi_name], fontsize=12)
        ax.set_facecolor('#f8f8f8')

    # legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=colors[i], markersize=9,
                    label=class_names[i])
        for i in range(num_classes)
    ]
    fig.legend(
        handles=handles, loc='lower center',
        ncol=num_classes, fontsize=12, frameon=False,
        bbox_to_anchor=(0.5, 0.0), columnspacing=2.0,
    )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    seed: int = 42,
    node_size: int = 14,
    save_path: str = None,
):
    cfg = NCI1_CONFIG
    D = cfg['D']
    num_layers = cfg['num_layers']
    normalize = cfg['normalize']
    pooling = cfg['pooling']
    perplexity = cfg['perplexity']
    class_names = cfg['class_names']

    print("=" * 60)
    print("GVFA t-SNE Projection — NCI1 (Graph Classification)")
    print("=" * 60)

    graphs, num_classes = load_tu_dataset('NCI1', data_dir=cfg['data_dir'])
    labels = np.array([g.label for g in graphs])

    print(f"Graphs: {len(graphs)}, Classes: {num_classes}")
    print(f"Config: D={D}, layers={num_layers}, normalize={normalize}, pooling={pooling}")
    print()

    projections = {}

    # raw
    print("Computing raw graph features (sum of node features) ...")
    raw_emb = _graph_raw_features(graphs)
    print(f"  Shape: {raw_emb.shape}")
    projections['raw'] = run_tsne(raw_emb, perplexity=perplexity, seed=seed)
    print("  Done.")

    # phi1–phi4
    for phi in ['phi1', 'phi2', 'phi3', 'phi4']:
        print(f"Computing GVFA embeddings for {PHI_LABELS[phi]} ...")
        emb = _graph_gvfa_embeddings(graphs, phi=phi, D=D, num_layers=num_layers,
                                     normalize=normalize, pooling=pooling, seed=seed)
        print(f"  Shape: {emb.shape}")
        projections[phi] = run_tsne(emb, perplexity=perplexity, seed=seed)
        print("  Done.")

    print()
    print("Plotting ...")
    plot_tsne_comparison(
        projections, labels, class_names=class_names,
        node_size=node_size, save_path=save_path,
        title='Raw Features vs. GVFA t-SNE Projections — NCI1',
    )
    print("Complete.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GVFA t-SNE Projections on NCI1')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--node-size', type=int, default=14, help='Scatter point size')
    parser.add_argument('--save', type=str, default=None, help='Path to save figure')
    args = parser.parse_args()

    main(seed=args.seed, node_size=args.node_size, save_path=args.save)
