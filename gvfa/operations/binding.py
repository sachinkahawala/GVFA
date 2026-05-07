"""
Binding Operation: ⊙

Circular convolution via FFT for binding hypervectors.
"""

import torch
from torch.fft import fft, ifft


def bind(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Binding via FFT circular convolution: x ⊙ y

    This is the core binding operation in VFA that creates
    a new hypervector representing the association of x and y.

    Args:
        x: First hypervector [..., D]
        y: Second hypervector [..., D]

    Returns:
        Bound hypervector [..., D]
    """
    return torch.real(ifft(fft(x, dim=-1) * fft(y, dim=-1), dim=-1))
