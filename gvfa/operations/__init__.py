"""
VFA Core Operations

- binding: ⊙ FFT circular convolution
- superposition: ⊕ addition
- permutation: ρ circular shift
"""

from .binding import bind
from .superposition import superpose
from .permutation import permute

__all__ = ['bind', 'superpose', 'permute']
