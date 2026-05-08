"""GVFA Experiments.

Imports of individual experiment runners are wrapped in try/except so that
``import gvfa.experiments.<x>`` works as long as ``<x>``'s own deps are
installed — without forcing the whole experiments suite (matplotlib,
torch-geometric, etc.) on a user who only wants to run one experiment.
"""

from typing import Any, Optional

_LAZY_RUNNERS = {
    "run_node_classification": ("node_classification", "run_experiment"),
    "run_node_classification_module": ("node_classification_module", "run_experiment"),
    "run_graph_regression_zinc": ("graph_regression_zinc", "run_experiment"),
    "run_tsne_projections": ("tsne_projections", "main"),
    "run_tsne_graph_nci1": ("tsne_graph_nci1", "main"),
    "run_tsne_layer_evolution": ("tsne_layer_evolution", "main"),
    "run_similarity_preservation": ("similarity_preservation", "main"),
}


def __getattr__(name: str) -> Any:
    """Resolve runner names lazily to avoid eager-importing heavy deps."""
    if name not in _LAZY_RUNNERS:
        raise AttributeError(f"module 'gvfa.experiments' has no attribute {name!r}")
    module_name, attr = _LAZY_RUNNERS[name]
    mod = __import__(f"gvfa.experiments.{module_name}", fromlist=[attr])
    value = getattr(mod, attr)
    globals()[name] = value
    return value


__all__ = list(_LAZY_RUNNERS)
