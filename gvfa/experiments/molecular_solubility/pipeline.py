"""
Molecular solubility pipeline orchestrator.

A single :class:`MolecularSolubilityPipeline` composes the 10 swappable
stages described in the plan:

  Dataset → Splitter → MolFeaturizer → Projection → EdgeBinder (opt) →
  Encoder → Augmentation (opt) → Pooler → Head → Evaluator

Each stage's behavior is driven by :class:`gvfa.experiments.molecular_solubility.config.PipelineConfig`.
The pipeline threads ``train_stats`` through projections so the test set is
transformed using only training-time statistics (no leakage).

A ``descriptors_only`` config flag short-circuits the GVFA path entirely:
the pipeline computes molecule-level RDKit descriptors and feeds them
straight to the regression head — that's the "traditional baseline"
ablation, and it shares all the splitting / logging / evaluation code with
the GVFA variants.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ...chem import (
    AtomFeaturizer,
    BondFeaturizer,
    DescriptorFeaturizer,
    DEFAULT_DESCRIPTOR_NAMES,
    BoundedScaler,
    EdgeAwareGVFA,
    EdgeBinder,
    MolGraph,
    MoleculeNetDataset,
    RandomProjection,
    Reservoir,
    SigmaPi,
    SolubilityDataset,
    default_bond_scaler,
    default_extended_atom_scaler,
    index_subset,
    kfold_split,
    random_split,
    scaffold_split,
    smiles_to_mol_graph,
)
from .config import PipelineConfig


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

SmilesTarget = Tuple[str, float]
TrainStats = Dict[str, Any]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class MolecularSolubilityPipeline:
    """Composes the configured stages and runs a single train/test pass."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Stage 1: Dataset
    # ------------------------------------------------------------------

    def _load_dataset(self) -> Tuple[List[SmilesTarget], Optional[List[SmilesTarget]]]:
        """Load (train_or_full, optional_explicit_test). When the explicit test
        path is set, the splitter is bypassed."""
        ds_cfg = self.config.dataset
        if ds_cfg.kind == "csv":
            if not ds_cfg.csv_path:
                raise ValueError("dataset.csv_path is required when kind='csv'")
            train = SolubilityDataset(
                ds_cfg.csv_path, smiles_col=ds_cfg.smiles_col, target_col=ds_cfg.target_col
            )
            test = None
            if ds_cfg.test_csv_path:
                test = SolubilityDataset(
                    ds_cfg.test_csv_path,
                    smiles_col=ds_cfg.smiles_col,
                    target_col=ds_cfg.target_col,
                )
            return list(train), (list(test) if test is not None else None)
        if ds_cfg.kind == "moleculenet":
            if not ds_cfg.moleculenet_name:
                raise ValueError("dataset.moleculenet_name is required when kind='moleculenet'")
            full = MoleculeNetDataset(
                name=ds_cfg.moleculenet_name, root=ds_cfg.moleculenet_root
            )
            return list(full), None
        raise ValueError(f"Unknown dataset kind: {ds_cfg.kind!r}")

    # ------------------------------------------------------------------
    # Stage 2: Splitter
    # ------------------------------------------------------------------

    def _split(
        self, full: List[SmilesTarget], explicit_test: Optional[List[SmilesTarget]]
    ) -> List[Tuple[List[SmilesTarget], List[SmilesTarget]]]:
        """Returns a list of (train, test) — multiple folds for kfold; one
        otherwise. With ``explicit_test`` set, the input ``full`` is treated
        as the train set verbatim and the splitter is bypassed."""
        if explicit_test is not None:
            return [(full, explicit_test)]
        sp = self.config.split
        n = len(full)
        if sp.kind == "explicit":
            raise ValueError(
                "split.kind='explicit' but no test_csv_path / moleculenet test was provided"
            )
        if sp.kind == "random":
            folds = random_split(n, train_ratio=sp.train_ratio, seed=sp.seed)
        elif sp.kind == "scaffold":
            smiles = [smi for smi, _ in full]
            folds = scaffold_split(smiles, train_ratio=sp.train_ratio, seed=sp.seed)
        elif sp.kind == "kfold":
            folds = kfold_split(n, k=sp.k, seed=sp.seed)
        else:
            raise ValueError(f"Unknown split kind: {sp.kind!r}")
        return [(index_subset(full, tr), index_subset(full, te)) for tr, te in folds]

    # ------------------------------------------------------------------
    # Stage 3: Featurization (SMILES → MolGraph)
    # ------------------------------------------------------------------

    def _build_featurizers(self) -> Tuple[AtomFeaturizer, Optional[BondFeaturizer]]:
        feat = self.config.featurizer
        atom = AtomFeaturizer(feature_set=tuple(feat.atom_features))
        bond = (
            BondFeaturizer(feature_set=tuple(feat.bond_features), embed_seed=feat.embed_seed)
            if feat.bond_features
            else None
        )
        return atom, bond

    def _featurize(
        self,
        items: Sequence[SmilesTarget],
        atom: AtomFeaturizer,
        bond: Optional[BondFeaturizer],
    ) -> List[MolGraph]:
        graphs: List[MolGraph] = []
        for smi, y in items:
            g = smiles_to_mol_graph(smi, target=y, atom_featurizer=atom, bond_featurizer=bond)
            if g is not None:
                graphs.append(g)
        return graphs

    # ------------------------------------------------------------------
    # Stages 4–5: Scaling + Projection (atom and bond, independently)
    # ------------------------------------------------------------------

    def _build_scaler(
        self, scfg, *, is_bond: bool = False
    ) -> Optional[BoundedScaler]:
        if not scfg.enabled:
            return None
        if scfg.use_extended_atom_defaults and not is_bond:
            return default_extended_atom_scaler()
        if scfg.use_default_bond_cols and is_bond:
            return default_bond_scaler()
        return BoundedScaler(
            binary_cols=list(scfg.binary_cols),
            minmax_cols=list(scfg.minmax_cols),
            tanh_cols={int(k): float(v) for k, v in scfg.tanh_cols.items()},
        )

    def _project_atoms(
        self,
        train_graphs: List[MolGraph],
        test_graphs: List[MolGraph],
    ) -> TrainStats:
        pcfg = self.config.projection
        # Stack train atom features
        X_train = torch.cat([g.x_atom for g in train_graphs], dim=0)
        scaler = self._build_scaler(pcfg.scaler, is_bond=False)
        if scaler is not None:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
        proj = RandomProjection(
            D=pcfg.D, kind=pcfg.kind, seed=pcfg.seed, sign_normalize=pcfg.sign_normalize
        )
        proj.fit(X_train)

        for g in train_graphs:
            x = g.x_atom if scaler is None else scaler.transform(g.x_atom)
            g.h_atom = proj.transform(x)
        for g in test_graphs:
            x = g.x_atom if scaler is None else scaler.transform(g.x_atom)
            g.h_atom = proj.transform(x)
        return {"atom_scaler": scaler, "atom_projection": proj}

    def _project_bonds(
        self,
        train_graphs: List[MolGraph],
        test_graphs: List[MolGraph],
    ) -> TrainStats:
        # Bonds are optional. If none of the graphs has x_bond we silently skip.
        if not train_graphs or train_graphs[0].x_bond is None:
            return {"bond_scaler": None, "bond_projection": None}
        pcfg = self.config.projection
        bond_D = pcfg.bond_D if pcfg.bond_D is not None else pcfg.D
        bond_kind = pcfg.bond_kind if pcfg.bond_kind is not None else pcfg.kind
        bond_sign = (
            pcfg.bond_sign_normalize
            if pcfg.bond_sign_normalize is not None
            else pcfg.sign_normalize
        )
        E_train = torch.cat([g.x_bond for g in train_graphs if g.x_bond is not None], dim=0)
        scaler = self._build_scaler(pcfg.bond_scaler, is_bond=True)
        if scaler is not None:
            scaler.fit(E_train)
            E_train = scaler.transform(E_train)
        proj = RandomProjection(
            D=bond_D, kind=bond_kind, seed=pcfg.bond_seed, sign_normalize=bond_sign
        )
        proj.fit(E_train)
        for g in train_graphs + test_graphs:
            if g.x_bond is None:
                continue
            x = g.x_bond if scaler is None else scaler.transform(g.x_bond)
            g.h_bond = proj.transform(x)
        return {"bond_scaler": scaler, "bond_projection": proj}

    # ------------------------------------------------------------------
    # Stages 6–8: Encode + Augment + Pool → graph-level embedding
    # ------------------------------------------------------------------

    def _build_encoder(self) -> EdgeAwareGVFA:
        enc = self.config.encoder
        eb_cfg = self.config.edge_binder
        edge_binder = (
            None
            if eb_cfg.kind in (None, "none", "off", "disabled")
            else EdgeBinder(kind=eb_cfg.kind)
        )
        return EdgeAwareGVFA(
            edge_binder=edge_binder,
            num_layers=enc.num_layers,
            phi=enc.phi,
            normalize=enc.normalize,
            kappa=enc.kappa,
        )

    def _build_augmentation(self):
        a = self.config.augmentation
        if a.kind in (None, "none", "off"):
            return None
        if a.kind == "reservoir":
            return ("reservoir", Reservoir(hop_decay=a.hop_decay, normalize=a.normalize))
        if a.kind == "sigma_pi":
            return (
                "sigma_pi",
                SigmaPi(orders=tuple(a.sigma_pi_orders), bind_kind=a.sigma_pi_bind_kind, normalize=a.normalize),
            )
        if a.kind == "reservoir_sigma_pi":
            return (
                "reservoir_sigma_pi",
                (
                    Reservoir(hop_decay=a.hop_decay, normalize=a.normalize),
                    SigmaPi(orders=tuple(a.sigma_pi_orders), bind_kind=a.sigma_pi_bind_kind, normalize=a.normalize),
                ),
            )
        raise ValueError(f"Unknown augmentation kind: {a.kind!r}")

    def _encode_and_pool(
        self,
        graphs: List[MolGraph],
        encoder: EdgeAwareGVFA,
        augmentation,
    ) -> np.ndarray:
        """Encode every graph and pool to a single graph-level vector. Returns ``[num_graphs, F]``."""
        pooling = self.config.pooler.kind
        rows: List[np.ndarray] = []
        for g in graphs:
            edge_h = g.h_bond  # may be None if no bond featurizer
            if augmentation is None:
                # default: concat all levels via the encoder's standard path
                graph_repr = encoder.forward_graph(
                    g.h_atom, g.edge_index, edge_h=edge_h, pooling=pooling
                )
            else:
                # pull per-level node embeddings, apply augmentation, then pool
                levels = encoder.forward_levels(g.h_atom, g.edge_index, edge_h=edge_h)
                kind, payload = augmentation
                if kind == "reservoir":
                    F1 = payload(levels)
                elif kind == "sigma_pi":
                    # sigma-pi alone operates on a single hypervector — use the
                    # final level as F1.
                    F1 = payload(levels[-1])
                else:  # reservoir_sigma_pi
                    res, sp = payload
                    F1 = sp(res(levels))
                graph_repr = (
                    F1.sum(dim=0, keepdim=True) if pooling == "sum" else F1.mean(dim=0, keepdim=True)
                )
            rows.append(graph_repr.detach().cpu().numpy().reshape(-1))
        return np.vstack(rows)

    # ------------------------------------------------------------------
    # Stage 9: Head + evaluation
    # ------------------------------------------------------------------

    def _fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler

        head = self.config.head
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if head.kind == "ridge":
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=head.alpha)
        elif head.kind == "ridgecv":
            from sklearn.linear_model import RidgeCV
            reg = RidgeCV(alphas=list(head.alphas))
        elif head.kind == "kernel_ridge":
            from sklearn.kernel_ridge import KernelRidge
            reg = KernelRidge(alpha=head.alpha, kernel="rbf", gamma=head.kernel_gamma)
        elif head.kind == "xgboost":
            import xgboost as xgb
            reg = xgb.XGBRegressor(
                n_estimators=head.n_estimators,
                max_depth=head.max_depth,
                learning_rate=head.learning_rate,
                random_state=self.config.seed,
                n_jobs=-1,
            )
        elif head.kind == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            reg = RandomForestRegressor(
                n_estimators=head.n_estimators,
                max_depth=head.max_depth or None,
                random_state=self.config.seed,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown head kind: {head.kind!r}")

        reg.fit(X_train_s, y_train)
        return reg.predict(X_test_s)

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        # Pearson r
        if y_true.std() > 0 and y_pred.std() > 0:
            pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            pearson = float("nan")
        return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson}

    # ------------------------------------------------------------------
    # Descriptor-only baseline
    # ------------------------------------------------------------------

    def _descriptor_baseline(
        self, train: List[SmilesTarget], test: List[SmilesTarget]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        names = self.config.featurizer.descriptor_names or list(DEFAULT_DESCRIPTOR_NAMES)
        dfeat = DescriptorFeaturizer(names)

        def featurize(items: List[SmilesTarget]) -> Tuple[np.ndarray, np.ndarray]:
            X_rows: List[np.ndarray] = []
            y_rows: List[float] = []
            for smi, y in items:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                X_rows.append(dfeat.transform(mol).numpy())
                y_rows.append(float(y))
            return np.vstack(X_rows), np.asarray(y_rows, dtype=np.float64)

        dfeat.fit([])  # initialize calculator
        X_train, y_train = featurize(train)
        X_test, y_test = featurize(test)
        return X_train, y_train, X_test, y_test

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run a single train/test pass (or a fold-averaged pass for k-fold)."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        full, explicit_test = self._load_dataset()
        folds = self._split(full, explicit_test)
        fold_metrics: List[Dict[str, float]] = []

        t0 = time.time()
        for i, (train, test) in enumerate(folds):
            if self.config.descriptors_only:
                X_train, y_train, X_test, y_test = self._descriptor_baseline(train, test)
            else:
                atom_feat, bond_feat = self._build_featurizers()
                train_graphs = self._featurize(train, atom_feat, bond_feat)
                test_graphs = self._featurize(test, atom_feat, bond_feat)
                self._project_atoms(train_graphs, test_graphs)
                self._project_bonds(train_graphs, test_graphs)
                encoder = self._build_encoder()
                augmentation = self._build_augmentation()
                X_train = self._encode_and_pool(train_graphs, encoder, augmentation)
                X_test = self._encode_and_pool(test_graphs, encoder, augmentation)
                y_train = np.array([float(g.target.item()) for g in train_graphs])
                y_test = np.array([float(g.target.item()) for g in test_graphs])

            y_pred = self._fit_predict(X_train, y_train, X_test)
            m = self._metrics(y_test, y_pred)
            m["fold"] = i
            m["n_train"] = int(len(y_train))
            m["n_test"] = int(len(y_test))
            fold_metrics.append(m)
        elapsed = time.time() - t0

        # Aggregate across folds
        keys = ["rmse", "mae", "r2", "pearson"]
        agg = {f"{k}_mean": float(np.mean([m[k] for m in fold_metrics])) for k in keys}
        agg.update({f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in keys})

        result: Dict[str, Any] = {
            "run_name": self.config.logging.run_name,
            "elapsed_sec": elapsed,
            "folds": fold_metrics,
            **agg,
        }

        # Persist
        out_dir = Path(self.config.logging.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_path = out_dir / f"{self.config.logging.run_name}.json"
        run_path.write_text(json.dumps({"config": asdict(self.config), "result": result}, indent=2))
        # Append a row to the experiment-wide summary CSV
        self._append_summary(out_dir, result)
        return result

    @staticmethod
    def _append_summary(out_dir: Path, result: Dict[str, Any]) -> None:
        path = out_dir / "summary.csv"
        new = not path.exists()
        scalar_keys = [k for k, v in result.items() if not isinstance(v, list)]
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(scalar_keys)
            writer.writerow([result[k] for k in scalar_keys])
