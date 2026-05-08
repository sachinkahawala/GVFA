"""End-to-end pipeline tests on a tiny subset."""

import os
import tempfile

import pandas as pd
import pytest

from gvfa.experiments.molecular_solubility import (
    PipelineConfig,
    MolecularSolubilityPipeline,
    apply_overrides,
    load_config,
)


SOLUBILITY_TRAIN = "data/solubility/train.csv"
SOLUBILITY_TEST = "data/solubility/test.csv"


def _have_solubility_csvs() -> bool:
    return os.path.exists(SOLUBILITY_TRAIN) and os.path.exists(SOLUBILITY_TEST)


@pytest.fixture(scope="module")
def small_csvs(tmp_path_factory):
    if not _have_solubility_csvs():
        pytest.skip("solubility CSVs not present under data/solubility/")
    train = pd.read_csv(SOLUBILITY_TRAIN).head(150)
    test = pd.read_csv(SOLUBILITY_TEST).head(40)
    out = tmp_path_factory.mktemp("solubility")
    train_path = out / "train.csv"
    test_path = out / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return str(train_path), str(test_path)


def _base_cfg(train_path: str, test_path: str, run_name: str, out_dir: str) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.dataset.csv_path = train_path
    cfg.dataset.test_csv_path = test_path
    cfg.dataset.smiles_col = "smiles_canon"
    cfg.dataset.target_col = "LogS"
    cfg.projection.D = 256
    cfg.logging.out_dir = out_dir
    cfg.logging.run_name = run_name
    return cfg


def test_pipeline_base_runs_to_completion(small_csvs, tmp_path):
    train_path, test_path = small_csvs
    cfg = _base_cfg(train_path, test_path, "base", str(tmp_path))
    res = MolecularSolubilityPipeline(cfg).run()
    assert "rmse_mean" in res
    assert res["rmse_mean"] > 0
    assert (tmp_path / "base.json").exists()
    assert (tmp_path / "summary.csv").exists()


def test_pipeline_edge_aware_differs_from_base(small_csvs, tmp_path):
    train_path, test_path = small_csvs
    base_cfg = _base_cfg(train_path, test_path, "base", str(tmp_path))
    edge_cfg = _base_cfg(train_path, test_path, "edge", str(tmp_path))
    edge_cfg.featurizer.bond_features = ["bond_type", "is_conjugated", "in_ring", "bond_length_3d"]
    edge_cfg.edge_binder.kind = "hadamard"

    base_res = MolecularSolubilityPipeline(base_cfg).run()
    edge_res = MolecularSolubilityPipeline(edge_cfg).run()
    # Different setups should not yield numerically identical metrics
    assert base_res["rmse_mean"] != edge_res["rmse_mean"]


def test_pipeline_descriptors_only_runs(small_csvs, tmp_path):
    train_path, test_path = small_csvs
    cfg = _base_cfg(train_path, test_path, "traditional", str(tmp_path))
    cfg.descriptors_only = True
    cfg.head.kind = "ridge"  # ridge is faster than xgboost for the smoke check
    res = MolecularSolubilityPipeline(cfg).run()
    assert res["rmse_mean"] > 0


def test_yaml_round_trip_and_overrides():
    """Loading base.yaml + overriding fields produces a coherent config."""
    cfg = load_config("gvfa/experiments/molecular_solubility/configs/base.yaml")
    assert cfg.encoder.phi == "phi3"
    assert cfg.head.kind == "ridge"
    cfg = apply_overrides(cfg, ["projection.D=1024", "edge_binder.kind=hadamard"])
    assert cfg.projection.D == 1024
    assert cfg.edge_binder.kind == "hadamard"


def test_binder_switch_changes_output(small_csvs, tmp_path):
    """Hadamard ↔ circular binding must produce numerically different metrics
    (the binder is actually wired into the encoder, not a no-op)."""
    train_path, test_path = small_csvs
    h_cfg = _base_cfg(train_path, test_path, "had", str(tmp_path))
    h_cfg.featurizer.bond_features = ["bond_type", "is_conjugated", "in_ring", "bond_length_3d"]
    h_cfg.edge_binder.kind = "hadamard"

    c_cfg = _base_cfg(train_path, test_path, "circ", str(tmp_path))
    c_cfg.featurizer.bond_features = ["bond_type", "is_conjugated", "in_ring", "bond_length_3d"]
    c_cfg.edge_binder.kind = "circular"

    h_res = MolecularSolubilityPipeline(h_cfg).run()
    c_res = MolecularSolubilityPipeline(c_cfg).run()
    assert h_res["rmse_mean"] != c_res["rmse_mean"]
