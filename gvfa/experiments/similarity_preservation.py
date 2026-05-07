"""
Similarity Preservation Under Graph Perturbation

Demonstrates the graceful degradation property of GVFA:
small structural changes → small changes in graph hypervector.

Uses a hand-crafted 10-node example graph with four progressive
perturbation levels.  The figure shows:
  Top row  – the graph at each perturbation stage (colour-coded edges)
  Bottom   – heatmap of cosine similarity to the original for all Φ variants

Run:
    python -m gvfa.experiments.similarity_preservation
    python -m gvfa.experiments.similarity_preservation --save figures/sim_preservation.png
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from torch.nn.functional import cosine_similarity

from gvfa import GVFA
from gvfa.data import random_projection


# ── Config ───────────────────────────────────────────────────────────────────
D = 5000
NUM_LAYERS = 3
NORMALIZE = 'sign'
POOLING = 'sum'

PHI_NAMES = ['phi1', 'phi2', 'phi3', 'phi4']

PHI_LABELS_SHORT = {
    'phi1': r'$\Phi_1$',
    'phi2': r'$\Phi_2$',
    'phi3': r'$\Phi_3$',
    'phi4': r'$\Phi_4$',
}

PHI_LABELS = {
    'phi1': r'$\Phi_1$:  $H_i \oplus \rho(F_i)$',
    'phi2': r'$\Phi_2$:  $H_i \oplus H_i \odot \rho(F_i)$',
    'phi3': r'$\Phi_3$:  $\rho(H_i \oplus F_i)$',
    'phi4': r'$\Phi_4$:  $\rho(H_i \oplus H_i \odot F_i)$',
}

# ── Hand-crafted example graph for the visualisation row ─────────────────
# A clean 10-node graph with 15 edges (resembles a small molecule).
# Fixed positions give a stable, readable layout across all panels.

EXAMPLE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),   # backbone chain
    (5, 0),                                      # close the ring
    (0, 6), (2, 7), (4, 8),                      # branches
    (6, 7), (7, 8),                               # branch links
    (1, 9), (3, 9), (5, 9),                       # hub through centre
    (6, 9),                                       # extra cross-link
]

EXAMPLE_POS = {
    0: (-1.0,  0.0),
    1: (-0.5,  0.85),
    2: ( 0.5,  0.85),
    3: ( 1.0,  0.0),
    4: ( 0.5, -0.85),
    5: (-0.5, -0.85),
    6: (-1.4,  0.85),
    7: ( 0.5,  1.6),
    8: ( 1.4, -0.85),
    9: ( 0.0,  0.0),
}

NUM_EXAMPLE_NODES = 10

# Each step: (title, edges_to_remove, edges_to_add)
# Progressive — each step adds MORE perturbation on top of the original.
PERTURBATION_STEPS = [
    ('Original\n15 edges',
     [],
     []),
    ('1 change',
     [(6, 9)],
     []),
    ('2 changes',
     [(6, 9)],
     [(3, 8)]),
    ('4 changes',
     [(6, 9), (5, 9)],
     [(3, 8), (0, 8)]),
    ('6 changes',
     [(6, 9), (5, 9), (6, 7)],
     [(3, 8), (0, 8), (1, 4)]),
    ('8 changes',
     [(6, 9), (5, 9), (6, 7), (1, 9)],
     [(3, 8), (0, 8), (1, 4), (2, 5)]),
    ('11 changes',
     [(6, 9), (5, 9), (6, 7), (1, 9), (4, 8), (7, 8)],
     [(3, 8), (0, 8), (1, 4), (2, 5), (0, 3)]),
]


# ── Edge-index helpers ───────────────────────────────────────────────────────

def _edge_index_from_edges(edges, num_nodes: int) -> torch.Tensor:
    """Build bidirectional [2, E] edge_index from an iterable of (u,v)."""
    src, dst = [], []
    for u, v in edges:
        src += [u, v]
        dst += [v, u]
    if len(src) == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


# ── Compute similarities on the example graph ────────────────────────────────

def compute_example_similarities(seed: int = 42) -> np.ndarray:
    """
    Run GVFA on the hand-crafted example graph at each perturbation step
    for each Φ variant.

    Returns:
        sims  – np.ndarray of shape (4, 4)  [phi × step]
                cosine similarity to the original graph embedding.
    """
    n = NUM_EXAMPLE_NODES
    # One-hot node features (identity matrix)
    node_features = torch.eye(n)
    h_proj = random_projection(node_features, D, seed=seed, normalize=True)

    # Build edge indices for each perturbation step
    orig_edges = set(EXAMPLE_EDGES)
    step_edge_indices = []
    for _, to_remove, to_add in PERTURBATION_STEPS:
        current = (orig_edges - set(to_remove)) | set(to_add)
        step_edge_indices.append(_edge_index_from_edges(current, n))

    sims = np.zeros((len(PHI_NAMES), len(PERTURBATION_STEPS)))

    for i, phi_name in enumerate(PHI_NAMES):
        model = GVFA(num_layers=NUM_LAYERS, phi=phi_name, normalize=NORMALIZE)
        orig_emb = model.forward_graph(
            h_proj, step_edge_indices[0], pooling=POOLING
        ).squeeze(0)

        for j, ei in enumerate(step_edge_indices):
            emb = model.forward_graph(h_proj, ei, pooling=POOLING).squeeze(0)
            sim = cosine_similarity(
                orig_emb.unsqueeze(0), emb.unsqueeze(0)
            ).item()
            sims[i, j] = sim

    return sims


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_perturbation_examples(ax_row: list) -> None:
    """
    Draw the hand-crafted example graph through progressive perturbation steps.
    Surviving edges = grey, removed = red dashed, added = green.
    """
    orig_set = set(EXAMPLE_EDGES)
    pos = EXAMPLE_POS
    n = NUM_EXAMPLE_NODES

    for ax, (title, to_remove, to_add) in zip(ax_row, PERTURBATION_STEPS):
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        ax.axis('off')

        removed = set(to_remove)
        added   = set(to_add)
        surviving = orig_set - removed

        # Full perturbed graph (for node positioning)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(surviving)
        G.add_edges_from(added)

        # Draw surviving original edges (grey)
        if surviving:
            nx.draw_networkx_edges(
                G, pos, edgelist=list(surviving), ax=ax,
                edge_color='#888888', width=1.8, alpha=0.65,
            )
        # Draw removed edges (red dashed – shown at original positions)
        if removed:
            G_rem = nx.Graph()
            G_rem.add_nodes_from(range(n))
            G_rem.add_edges_from(removed)
            nx.draw_networkx_edges(
                G_rem, pos, ax=ax,
                edge_color='#D55E00', width=2.0, alpha=0.85,
                style=(0, (4, 3)),  # dashed
            )
        # Draw added edges (green)
        if added:
            nx.draw_networkx_edges(
                G, pos, edgelist=list(added), ax=ax,
                edge_color='#009E73', width=2.0, alpha=0.85,
            )

        # Nodes on top
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=120,
            node_color='#0072B2', edgecolors='white', linewidths=1.0,
        )

        # Stats annotation
        n_rem = len(removed)
        n_add = len(added)
        if n_rem == 0 and n_add == 0:
            stats = f'{len(orig_set)} edges'
        else:
            parts = []
            if n_rem: parts.append(f'\u2212{n_rem}')
            if n_add: parts.append(f'+{n_add}')
            stats = '  '.join(parts) + ' edges'
        ax.text(0.5, -0.04, stats, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, color='#555555')


def plot_similarity_preservation(
    sims: np.ndarray,
    save_path: str = None,
) -> None:
    """
    Two-row figure:
      Top row  – example graph at each perturbation stage
      Bottom   – heatmap of cosine similarity (Φ vs perturbation level)
    """
    n_steps = len(PERTURBATION_STEPS)

    fig = plt.figure(figsize=(16, 6.5))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        2, n_steps, figure=fig,
        height_ratios=[1, 0.75],
        hspace=0.22, wspace=0.06,
        left=0.08, right=0.97, top=0.94, bottom=0.03,
    )

    # ── Top row: perturbation examples ────────────────────────────────────
    ax_top = [fig.add_subplot(gs[0, i]) for i in range(n_steps)]
    plot_perturbation_examples(ax_top)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#888888', lw=2, label='Kept'),
        Line2D([0], [0], color='#D55E00', lw=2, linestyle='dashed', label='Removed'),
        Line2D([0], [0], color='#009E73', lw=2, label='Added'),
    ]
    ax_top[-1].legend(
        handles=legend_elements, fontsize=8.5, loc='upper right',
        frameon=True, fancybox=True, framealpha=0.9, edgecolor='#cccccc',
        bbox_to_anchor=(1.02, 1.22),
    )

    # ── Bottom: heatmap ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :])

    # Colormap: dark red (low sim) → amber → yellow → white (1.0)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'sim',
        ['#8B0000', '#D55E00', '#E69F00', '#F0E442', '#FFFFCC', '#FFFFFF'],
        N=256,
    )
    # Fit colour range to actual data (with small padding)
    vmin = max(np.floor(sims.min() * 20) / 20 - 0.02, 0.0)  # round down
    vmax = 1.0
    im = ax.imshow(
        sims, cmap=cmap, aspect='auto',
        vmin=vmin, vmax=vmax,
    )

    # Annotate each cell with the similarity value
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            val = sims[i, j]
            # White text on dark cells, dark text on light cells
            frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 1.0
            text_col = 'white' if frac < 0.45 else '#333333'
            ax.text(
                j, i, f'{val:.4f}',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=text_col,
            )

    # Axes labels
    step_labels = [t.replace('\n', ' ') for t, _, _ in PERTURBATION_STEPS]
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(step_labels, fontsize=10)
    ax.set_yticks(range(len(PHI_NAMES)))
    ax.set_yticklabels(
        [PHI_LABELS[p] for p in PHI_NAMES], fontsize=10,
    )

    ax.set_title(
        'Cosine Similarity to Original Graph Hypervector',
        fontsize=13, fontweight='bold', pad=12,
    )

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Cosine Similarity', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Remove box spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)  # hide tick marks

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ── Entry point ──────────────────────────────────────────────────────────────

def main(seed: int = 42, save_path: str = None):
    print("=" * 60)
    print("Similarity Preservation Under Graph Perturbation")
    print("=" * 60)
    print()

    print("Computing GVFA embeddings on example graph ...")
    sims = compute_example_similarities(seed=seed)

    # Print results
    step_labels = [t.replace('\n', ' ') for t, _, _ in PERTURBATION_STEPS]
    for i, phi in enumerate(PHI_NAMES):
        print(f"\n{PHI_LABELS[phi]}:")
        for j, label in enumerate(step_labels):
            print(f"  {label:25s}  cosine sim = {sims[i, j]:.4f}")

    print("\nPlotting ...")
    plot_similarity_preservation(sims, save_path=save_path)
    print("Complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='GVFA Similarity Preservation Under Graph Perturbation',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save figure (e.g. figures/sim_preservation.png)')
    args = parser.parse_args()
    main(seed=args.seed, save_path=args.save)
