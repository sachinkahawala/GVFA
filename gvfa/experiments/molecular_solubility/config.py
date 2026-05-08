"""
Pipeline configuration schema.

Each pipeline stage gets its own dataclass; :class:`PipelineConfig` is the
nested root. YAML files map 1:1 onto this schema — keys outside the schema
are ignored, missing keys fall back to the dataclass defaults, so an
ablation YAML only has to override the fields that differ from
``configs/base.yaml``.

The CLI also supports dotted-key overrides
(``--override encoder.num_layers=5``) on top of the loaded YAML.

Design notes:
- Optional sub-stages (edge binder, augmentation) use a string ``kind`` field
  with a sentinel value (``"none"`` or ``None``) instead of an
  ``Optional[Dataclass]``. This keeps the YAML loader trivial.
- All seeds default to ``42`` to match the legacy convention; the
  top-level ``seed`` field is meant for global reproducibility but each
  sub-stage can override it independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    kind: str = "csv"  # "csv" | "moleculenet"
    csv_path: Optional[str] = None
    test_csv_path: Optional[str] = None  # if set, used as the test set instead of splitting
    smiles_col: str = "smiles_canon"
    target_col: str = "LogS"
    moleculenet_name: Optional[str] = None  # "ESOL" | "FreeSolv" | "Lipo"
    moleculenet_root: str = "./data"


@dataclass
class SplitConfig:
    kind: str = "explicit"  # "explicit" | "random" | "scaffold" | "kfold"
    train_ratio: float = 0.8
    k: int = 5
    seed: int = 42


@dataclass
class FeaturizerConfig:
    atom_features: List[str] = field(
        default_factory=lambda: ["atomic_number", "degree", "valence_electrons", "hybridization", "hbond_flags"]
    )
    bond_features: Optional[List[str]] = None  # None disables bond featurizer
    descriptors: bool = False  # also compute molecule-level RDKit descriptors
    descriptor_names: Optional[List[str]] = None  # None → DEFAULT_DESCRIPTOR_NAMES
    embed_seed: int = 0xF00D


@dataclass
class ScalerConfig:
    """Bounded per-column scaler. ``enabled=False`` skips the scaler entirely."""
    enabled: bool = False
    binary_cols: List[int] = field(default_factory=list)
    minmax_cols: List[int] = field(default_factory=list)
    tanh_cols: Dict[int, float] = field(default_factory=dict)
    use_extended_atom_defaults: bool = False
    use_default_bond_cols: bool = False


@dataclass
class ProjectionConfig:
    """Projections for atom and (optionally) bond features.

    Atom and bond projections are configured independently — different seeds
    so the matrices are uncorrelated, and potentially different ``D`` if you
    want a different bond hypervector size.
    """
    D: int = 5000
    kind: str = "gaussian"  # "gaussian" | "orthogonal"
    seed: int = 42
    sign_normalize: bool = True
    scaler: ScalerConfig = field(default_factory=ScalerConfig)

    bond_D: Optional[int] = None  # None → mirror D
    bond_kind: Optional[str] = None  # None → mirror kind
    bond_seed: int = 43
    bond_sign_normalize: Optional[bool] = None  # None → mirror sign_normalize
    bond_scaler: ScalerConfig = field(default_factory=ScalerConfig)


@dataclass
class EdgeBinderConfig:
    kind: str = "none"  # "none" | "hadamard" | "circular"


@dataclass
class EncoderConfig:
    num_layers: int = 3
    phi: str = "phi3"
    normalize: str = "sign"  # "sign" | "clip" | "l2" | "none"
    kappa: float = 1.0


@dataclass
class AugmentationConfig:
    kind: Optional[str] = None  # None | "reservoir" | "sigma_pi" | "reservoir_sigma_pi"
    hop_decay: float = 0.85
    sigma_pi_orders: List[int] = field(default_factory=lambda: [0, 1, 2])
    sigma_pi_bind_kind: str = "hadamard"
    normalize: bool = True


@dataclass
class PoolerConfig:
    kind: str = "sum"  # "sum" | "mean"


@dataclass
class HeadConfig:
    kind: str = "ridge"  # "ridge" | "ridgecv" | "kernel_ridge" | "xgboost" | "random_forest"
    alpha: float = 1.0
    alphas: List[float] = field(
        default_factory=lambda: [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    )
    kernel_gamma: Optional[float] = None
    # XGBoost / RF
    n_estimators: int = 2000
    max_depth: int = 7
    learning_rate: float = 0.05


@dataclass
class LoggingConfig:
    out_dir: str = "results/molecular_solubility"
    run_name: str = "default"


@dataclass
class PipelineConfig:
    seed: int = 42
    descriptors_only: bool = False  # traditional baseline: skip GVFA, use descriptors as features
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    featurizer: FeaturizerConfig = field(default_factory=FeaturizerConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    edge_binder: EdgeBinderConfig = field(default_factory=EdgeBinderConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    pooler: PoolerConfig = field(default_factory=PoolerConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# YAML loading + dotted-key overrides
# ---------------------------------------------------------------------------


def _instantiate(cls, data: Any) -> Any:
    """Recursively instantiate a dataclass from a (possibly nested) dict."""
    if data is None or not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        return data
    type_map = {f.name: f.type for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in type_map:
            # Silently ignore unknown keys — lets older configs survive when
            # a field is removed and avoids strict-mode pain in YAML.
            continue
        ftype = type_map[key]
        # Handle nested dataclasses. We resolve string annotations against the
        # module globals where the dataclass is defined.
        nested_cls = _resolve_dataclass(ftype, cls)
        if nested_cls is not None:
            kwargs[key] = _instantiate(nested_cls, value)
        else:
            kwargs[key] = _coerce_scalar(value, ftype)
    return cls(**kwargs)


def _resolve_dataclass(ftype, owner_cls):
    """Return the concrete dataclass for a field type annotation, or None."""
    if isinstance(ftype, type) and is_dataclass(ftype):
        return ftype
    if isinstance(ftype, str):
        # Forward-reference (PEP 563). Resolve against this module's globals.
        ns = vars(__import__(owner_cls.__module__, fromlist=[owner_cls.__name__]))
        resolved = ns.get(ftype)
        if isinstance(resolved, type) and is_dataclass(resolved):
            return resolved
    return None


def _coerce_scalar(value: Any, ftype) -> Any:
    """Best-effort coercion for keys whose YAML representation needs it.

    YAML loads dict-key integers as strings (``{1: 0.5}`` becomes
    ``{"1": 0.5}`` in some cases). For the ``tanh_cols`` field that
    matters; we coerce dict keys to int when the field type is
    ``Dict[int, ...]``. This is intentionally minimal — a full
    type-coercion layer would obscure errors.
    """
    if isinstance(value, dict):
        # heuristic: if keys look numeric, coerce them
        if all(isinstance(k, str) and k.lstrip("-").isdigit() for k in value):
            return {int(k): v for k, v in value.items()}
    return value


def load_config(path: str) -> PipelineConfig:
    """Load a YAML file at ``path`` and return a :class:`PipelineConfig`.

    Empty / missing fields fall back to dataclass defaults.
    """
    import yaml

    text = Path(path).read_text()
    data = yaml.safe_load(text) or {}
    return _instantiate(PipelineConfig, data)


def apply_overrides(cfg: PipelineConfig, overrides: List[str]) -> PipelineConfig:
    """Apply ``--override key.path=value`` strings to a config in-place.

    Numeric / boolean values are inferred from the literal; everything else
    stays a string. List values can be passed as JSON-style ``[1, 2, 3]``.
    """
    import json

    def _parse(v: str) -> Any:
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        if v.lower() in {"null", "none"}:
            return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        if v.startswith(("[", "{")):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return v

    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key.path=value, got: {ov!r}")
        key, raw = ov.split("=", 1)
        parsed = _parse(raw)
        path = key.split(".")
        target = cfg
        for part in path[:-1]:
            target = getattr(target, part)
        setattr(target, path[-1], parsed)
    return cfg
