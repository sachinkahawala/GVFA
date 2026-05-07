"""
Superposition Operation: ⊕

Element-wise addition for bundling hypervectors.
"""

import torch
from typing import Sequence


def superpose(*vectors: torch.Tensor) -> torch.Tensor:
    """
    Superposition via element-wise addition: x ⊕ y ⊕ ...

    Bundles multiple hypervectors into a single representation
    that is similar to all inputs.

    Args:
        *vectors: Hypervectors to superpose [..., D]

    Returns:
        Superposed hypervector [..., D]
    """
    if len(vectors) == 0:
        raise ValueError("At least one vector required")
    if len(vectors) == 1:
        return vectors[0]

    result = vectors[0]
    for v in vectors[1:]:
        result = result + v
    return result
