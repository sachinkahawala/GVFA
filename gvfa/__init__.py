"""
Graph Vector Function Architecture (GVFA)

A zero-shot approach for graph representations using Hyperdimensional Computing.
Implements the methods described in "Graph Vector Function Architecture" paper.
"""

from .models import GVFA
from .operations import bind, superpose, permute
from .data import random_projection, S2VGraph

__version__ = "0.1.0"

__all__ = [
    'GVFA',
    'bind', 'superpose', 'permute',
    'random_projection', 'S2VGraph'
]
