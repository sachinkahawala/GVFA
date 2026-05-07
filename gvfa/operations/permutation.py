"""
Permutation Operation: ρ

Circular shift for creating orthogonal variants of hypervectors.
"""

import torch


def permute(x: torch.Tensor, shift: int = 1) -> torch.Tensor:
    """
    Permutation via circular shift: ρ(x)

    Creates a new hypervector that is nearly orthogonal to x
    but can be reversed to recover x.

    Args:
        x: Hypervector [..., D]
        shift: Number of positions to shift (default: 1)

    Returns:
        Permuted hypervector [..., D]
    """
    return torch.roll(x, shifts=shift, dims=-1)
