"""
Normalization Functions: Λ

From paper Section 4.2, Equations 9-10
"""

import torch


def sign_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Sign normalization: Λ(x) = sign(x)

    Binarizes the hypervector to {-1, +1} values.
    Equation 10 from paper.

    Args:
        x: Input hypervector [..., D]

    Returns:
        Sign-normalized hypervector [..., D]
    """
    return torch.sign(x)


def clip_normalize(x: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """
    Clip normalization: Λ(x) = clip(x, -κ, κ)

    Hard tanh that clips values to [-κ, κ].
    Equation 9 from paper.

    Args:
        x: Input hypervector [..., D]
        kappa: Clipping threshold (default: 1.0)

    Returns:
        Clipped hypervector [..., D]
    """
    return torch.clamp(x, min=-kappa, max=kappa)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L2 normalization: Λ(x) = x / ||x||_2

    Normalizes hypervector to unit length.

    Args:
        x: Input hypervector [..., D]
        eps: Small value for numerical stability

    Returns:
        L2-normalized hypervector [..., D]
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return x / (norm + eps)
