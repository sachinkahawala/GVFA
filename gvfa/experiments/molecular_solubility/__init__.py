"""
Molecular solubility experiment.

A configurable graph-regression pipeline for predicting LogS (or any
analogous solubility-like target) from SMILES. Composed of swappable
stages defined in :mod:`gvfa.chem`, driven by YAML configs under
``configs/``.

Entry points:

- :class:`gvfa.experiments.molecular_solubility.pipeline.MolecularSolubilityPipeline`
  — the orchestrator. Construct from a :class:`PipelineConfig` and call ``run()``.
- ``python -m gvfa.experiments.molecular_solubility.run --config <path>``
  — CLI for running a single config.
- ``python -m gvfa.experiments.molecular_solubility.ablations`` — runs every
  YAML in ``configs/`` and writes a single summary CSV.
"""

from .config import (
    PipelineConfig,
    DatasetConfig,
    SplitConfig,
    FeaturizerConfig,
    ScalerConfig,
    ProjectionConfig,
    EdgeBinderConfig,
    EncoderConfig,
    AugmentationConfig,
    PoolerConfig,
    SizeAwareConfig,
    HeadConfig,
    LoggingConfig,
    load_config,
    apply_overrides,
)
from .pipeline import MolecularSolubilityPipeline

__all__ = [
    "PipelineConfig",
    "DatasetConfig",
    "SplitConfig",
    "FeaturizerConfig",
    "ScalerConfig",
    "ProjectionConfig",
    "EdgeBinderConfig",
    "EncoderConfig",
    "AugmentationConfig",
    "PoolerConfig",
    "SizeAwareConfig",
    "HeadConfig",
    "LoggingConfig",
    "load_config",
    "apply_overrides",
    "MolecularSolubilityPipeline",
]
