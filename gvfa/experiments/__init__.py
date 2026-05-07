"""GVFA Experiments"""

from .node_classification import run_experiment as run_node_classification
from .node_classification_module import run_experiment as run_node_classification_module
from .graph_regression_zinc import run_experiment as run_graph_regression_zinc
from .tsne_projections import main as run_tsne_projections
from .tsne_graph_nci1 import main as run_tsne_graph_nci1
from .tsne_layer_evolution import main as run_tsne_layer_evolution
from .similarity_preservation import main as run_similarity_preservation

__all__ = [
    'run_node_classification',
    'run_node_classification_module',
    'run_graph_regression_zinc',
    'run_tsne_projections',
    'run_tsne_graph_nci1',
    'run_tsne_layer_evolution',
    'run_similarity_preservation',
]
