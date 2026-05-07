"""
Combine Functions: Φ1, Φ2, Φ3, Φ4

Four variants from paper Section 4.1.
Each combines a node's hypervector H_i with aggregated neighbors F(i).

Where:
  ⊕ = superposition (addition)
  ⊙ = binding (FFT circular convolution)
  ρ = permutation (circular shift)
"""

import torch
from ..operations import bind, permute


def phi1(h: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    Φ1: H_i ⊕ ρ(F(i))

    Permute aggregated neighbors, then add to self.

    Args:
        h: Node hypervector H_i [..., D]
        f: Aggregated neighbors F(i) [..., D]

    Returns:
        Combined hypervector [..., D]
    """
    return h + permute(f)


def phi2(h: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    Φ2: H_i ⊕ H_i ⊙ ρ(F(i))

    Bind self with permuted neighbors, add to self.

    Args:
        h: Node hypervector H_i [..., D]
        f: Aggregated neighbors F(i) [..., D]

    Returns:
        Combined hypervector [..., D]
    """
    return h + bind(h, permute(f))


def phi3(h: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    Φ3: ρ(H_i ⊕ F(i))

    Add self and neighbors, then permute.
    Best performer for node classification (Table 2).

    Args:
        h: Node hypervector H_i [..., D]
        f: Aggregated neighbors F(i) [..., D]

    Returns:
        Combined hypervector [..., D]
    """
    return permute(h + f)


def phi4(h: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    Φ4: ρ(H_i ⊕ H_i ⊙ F(i))

    Bind self with neighbors, add to self, then permute.

    Args:
        h: Node hypervector H_i [..., D]
        f: Aggregated neighbors F(i) [..., D]

    Returns:
        Combined hypervector [..., D]
    """
    return permute(h + bind(h, f))


# Mapping for convenient access
PHI_FUNCTIONS = {
    'phi1': phi1,
    'phi2': phi2,
    'phi3': phi3,
    'phi4': phi4,
}
