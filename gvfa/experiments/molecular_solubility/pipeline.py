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
    MultiStatPool,
    RandomProjection,
    Reservoir,
    SigmaPi,
    SizeAwarePost,
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

    def _log(self, message: str) -> None:
        if self.config.logging.verbose:
            print(f"[{self.config.logging.run_name}] {message}", flush=True)

    def _progress_interval(self) -> int:
        return int(self.config.logging.progress_interval or 0)

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
        split_name: str,
    ) -> List[MolGraph]:
        total = len(items)
        interval = self._progress_interval()
        t0 = time.time()
        self._log(f"{split_name}: featurizing {total} molecules")
        graphs: List[MolGraph] = []
        for i, (smi, y) in enumerate(items, start=1):
            g = smiles_to_mol_graph(smi, target=y, atom_featurizer=atom, bond_featurizer=bond)
            if g is not None:
                graphs.append(g)
            if interval > 0 and (i % interval == 0 or i == total):
                elapsed = time.time() - t0
                self._log(
                    f"{split_name}: featurized {i}/{total} molecules "
                    f"({len(graphs)} valid, {elapsed:.1f}s)"
                )
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
        seed_override: Optional[int] = None,
    ) -> TrainStats:
        pcfg = self.config.projection
        seed = pcfg.seed if seed_override is None else seed_override
        # Stack train atom features
        X_train = torch.cat([g.x_atom for g in train_graphs], dim=0)
        test_nodes = sum(g.num_nodes for g in test_graphs)
        self._log(
            "projection/atoms: "
            f"train_nodes={X_train.shape[0]} test_nodes={test_nodes} "
            f"in_dim={X_train.shape[1]} D={pcfg.D} kind={pcfg.kind} "
            f"seed={seed} sign={pcfg.sign_normalize}"
        )
        scaler = self._build_scaler(pcfg.scaler, is_bond=False)
        if scaler is not None:
            self._log("projection/atoms: fitting bounded scaler")
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
        proj = RandomProjection(
            D=pcfg.D, kind=pcfg.kind, seed=seed, sign_normalize=pcfg.sign_normalize
        )
        proj.fit(X_train)

        for g in train_graphs:
            x = g.x_atom if scaler is None else scaler.transform(g.x_atom)
            g.h_atom = proj.transform(x)
        for g in test_graphs:
            x = g.x_atom if scaler is None else scaler.transform(g.x_atom)
            g.h_atom = proj.transform(x)
        self._log("projection/atoms: done")
        return {"atom_scaler": scaler, "atom_projection": proj}

    def _project_bonds(
        self,
        train_graphs: List[MolGraph],
        test_graphs: List[MolGraph],
        seed_override: Optional[int] = None,
    ) -> TrainStats:
        # Bonds are optional. If none of the graphs has x_bond we silently skip.
        if not train_graphs or train_graphs[0].x_bond is None:
            self._log("projection/bonds: skipped (no bond features)")
            return {"bond_scaler": None, "bond_projection": None}
        pcfg = self.config.projection
        bond_D = pcfg.bond_D if pcfg.bond_D is not None else pcfg.D
        bond_kind = pcfg.bond_kind if pcfg.bond_kind is not None else pcfg.kind
        bond_sign = (
            pcfg.bond_sign_normalize
            if pcfg.bond_sign_normalize is not None
            else pcfg.sign_normalize
        )
        # Per-seed sweep: shift bond_seed by the same delta as atom_seed so the two
        # projection matrices stay decorrelated across seeds.
        if seed_override is None:
            bond_seed = pcfg.bond_seed
        else:
            bond_seed = pcfg.bond_seed + (seed_override - pcfg.seed)
        E_train = torch.cat([g.x_bond for g in train_graphs if g.x_bond is not None], dim=0)
        test_edges = sum(
            int(g.x_bond.shape[0]) for g in test_graphs if g.x_bond is not None
        )
        self._log(
            "projection/bonds: "
            f"train_edges={E_train.shape[0]} test_edges={test_edges} "
            f"in_dim={E_train.shape[1]} D={bond_D} kind={bond_kind} "
            f"seed={bond_seed} sign={bond_sign}"
        )
        scaler = self._build_scaler(pcfg.bond_scaler, is_bond=True)
        if scaler is not None:
            self._log("projection/bonds: fitting bounded scaler")
            scaler.fit(E_train)
            E_train = scaler.transform(E_train)
        proj = RandomProjection(
            D=bond_D, kind=bond_kind, seed=bond_seed, sign_normalize=bond_sign
        )
        proj.fit(E_train)
        for g in train_graphs + test_graphs:
            if g.x_bond is None:
                continue
            x = g.x_bond if scaler is None else scaler.transform(g.x_bond)
            g.h_bond = proj.transform(x)
        self._log("projection/bonds: done")
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

    def _build_multi_stat_pool(self) -> Optional[MultiStatPool]:
        pcfg = self.config.pooler
        if pcfg.kind != "multi_stat":
            return None
        return MultiStatPool(bind_kind=pcfg.multi_stat_bind_kind)

    def _build_size_aware(self) -> Optional[SizeAwarePost]:
        sa = self.config.pooler.size_aware
        if sa.scale == "none" and not sa.append_size:
            return None
        return SizeAwarePost(
            scale=sa.scale,
            append_size=sa.append_size,
            append_size_kind=sa.append_size_kind,
        )

    def _per_node_repr(
        self,
        g: MolGraph,
        encoder: EdgeAwareGVFA,
        augmentation,
    ) -> torch.Tensor:
        """Compute the per-node representation ``[N, D']`` to be pooled.

        Without an augmentation, this is the encoder's standard concat-all-levels
        output ``[N, D · num_layers]``. With an augmentation (``reservoir``,
        ``sigma_pi``, ``reservoir_sigma_pi``), this is the augmentation's
        ``[N, D]`` output.
        """
        edge_h = g.h_bond  # None if no bond featurizer
        if augmentation is None:
            return encoder.forward(
                g.h_atom, g.edge_index, edge_h=edge_h, return_all_levels=True
            )
        levels = encoder.forward_levels(g.h_atom, g.edge_index, edge_h=edge_h)
        kind, payload = augmentation
        if kind == "reservoir":
            return payload(levels)
        if kind == "sigma_pi":
            return payload(levels[-1])
        if kind == "reservoir_sigma_pi":
            res, sp = payload
            return sp(res(levels))
        raise ValueError(f"Unknown augmentation kind: {kind!r}")

    def _encode_and_pool(
        self,
        graphs: List[MolGraph],
        encoder: EdgeAwareGVFA,
        augmentation,
        multi_stat: Optional[MultiStatPool],
        size_aware: Optional[SizeAwarePost],
        split_name: str,
    ) -> np.ndarray:
        """Encode + pool every graph. Returns ``[num_graphs, F]``."""
        pooling = self.config.pooler.kind
        rows: List[np.ndarray] = []
        sizes: List[int] = []
        total = len(graphs)
        interval = self._progress_interval()
        t0 = time.time()
        self._log(f"{split_name}: encoding/pooling {total} graphs")
        for i, g in enumerate(graphs, start=1):
            F_v = self._per_node_repr(g, encoder, augmentation)
            if multi_stat is not None:
                graph_repr = multi_stat(F_v)
            elif pooling == "sum":
                graph_repr = F_v.sum(dim=0, keepdim=True)
            elif pooling == "mean":
                graph_repr = F_v.mean(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unknown pooler kind: {pooling!r}")
            rows.append(graph_repr.detach().cpu().numpy().reshape(-1))
            sizes.append(g.num_nodes)
            if interval > 0 and (i % interval == 0 or i == total):
                elapsed = time.time() - t0
                self._log(f"{split_name}: encoded {i}/{total} graphs ({elapsed:.1f}s)")
        X = np.vstack(rows)
        if size_aware is not None:
            self._log(f"{split_name}: applying size-aware post-processing")
            X_t = torch.from_numpy(X).float()
            sizes_t = torch.tensor(sizes, dtype=torch.float32)
            X_t = size_aware(X_t, sizes_t)
            X = X_t.numpy()
        self._log(f"{split_name}: embedding matrix shape={X.shape}")
        return X

    # ------------------------------------------------------------------
    # Stage 9: Head + evaluation
    # ------------------------------------------------------------------

    def _fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        head = self.config.head
        self._log(
            "head: "
            f"kind={head.kind} standardize={head.standardize} "
            f"X_train={X_train.shape} X_test={X_test.shape}"
        )
        if head.standardize:
            from sklearn.preprocessing import StandardScaler

            self._log("head: fitting StandardScaler")
            scaler = StandardScaler()
            X_train_fit = scaler.fit_transform(X_train)
            X_test_fit = scaler.transform(X_test)
        else:
            X_train_fit = X_train
            X_test_fit = X_test

        if head.kind == "ridge":
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=head.alpha)
        elif head.kind == "ridgecv":
            from sklearn.linear_model import RidgeCV
            alphas = self._head_alphas()
            kwargs = {"alphas": alphas}
            if head.ridgecv_cv is not None:
                kwargs["cv"] = head.ridgecv_cv
            if head.ridgecv_scoring is not None:
                kwargs["scoring"] = head.ridgecv_scoring
            self._log(
                "head/ridgecv: "
                f"alphas={len(alphas)} cv={head.ridgecv_cv} scoring={head.ridgecv_scoring}"
            )
            reg = RidgeCV(**kwargs)
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

        t0 = time.time()
        self._log("head: fitting regressor")
        reg.fit(X_train_fit, y_train)
        self._log(f"head: fit complete ({time.time() - t0:.1f}s), predicting")
        pred = reg.predict(X_test_fit)
        self._log("head: prediction complete")
        return pred

    def _head_alphas(self) -> np.ndarray:
        """Return RidgeCV alphas from either an explicit list or logspace config."""
        head = self.config.head
        if head.alphas_logspace is None:
            return np.asarray(list(head.alphas), dtype=np.float64)
        spec = head.alphas_logspace
        start = float(spec.get("start_exp", -4))
        stop = float(spec.get("stop_exp", 2))
        num = int(spec.get("num", 50))
        if num < 1:
            raise ValueError("head.alphas_logspace.num must be >= 1")
        return np.logspace(start, stop, num)

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        err = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
        std_err = float(np.std(err, ddof=0))
        # Pearson r
        if y_true.std() > 0 and y_pred.std() > 0:
            pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            pearson_r = float("nan")
        pearson_r2 = float(pearson_r * pearson_r) if not np.isnan(pearson_r) else float("nan")
        return {
            "rmse": rmse,
            "std_err": std_err,
            "mae": mae,
            "r2": r2,
            "pearson": pearson_r,
            "pearson_r": pearson_r,
            "pearson_r2": pearson_r2,
        }

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

        def featurize(items: List[SmilesTarget], split_name: str) -> Tuple[np.ndarray, np.ndarray]:
            total = len(items)
            interval = self._progress_interval()
            t0 = time.time()
            self._log(f"{split_name}: computing RDKit descriptors for {total} molecules")
            X_rows: List[np.ndarray] = []
            y_rows: List[float] = []
            for i, (smi, y) in enumerate(items, start=1):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                X_rows.append(dfeat.transform(mol).numpy())
                y_rows.append(float(y))
                if interval > 0 and (i % interval == 0 or i == total):
                    elapsed = time.time() - t0
                    self._log(
                        f"{split_name}: descriptors {i}/{total} "
                        f"({len(X_rows)} valid, {elapsed:.1f}s)"
                    )
            return np.vstack(X_rows), np.asarray(y_rows, dtype=np.float64)

        dfeat.fit([])  # initialize calculator
        X_train, y_train = featurize(train, "train")
        X_test, y_test = featurize(test, "test")
        return X_train, y_train, X_test, y_test

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run a single train/test pass (or a fold/seed-averaged pass).

        With ``config.seeds`` set, the dataset and featurization are computed
        once per fold; only the projection, encode, head, and metrics steps
        are repeated per seed. This makes multi-seed sweeps fast — featurization
        (especially 3D conformer generation) is the dominant cost.
        """
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self._log("loading dataset")
        full, explicit_test = self._load_dataset()
        self._log(
            f"dataset loaded: primary={len(full)} "
            f"explicit_test={len(explicit_test) if explicit_test is not None else 0}"
        )
        folds = self._split(full, explicit_test)
        seed_list = self.config.seeds if self.config.seeds else [self.config.seed]
        self._log(f"prepared {len(folds)} fold(s), seeds={seed_list}")
        fold_metrics: List[Dict[str, float]] = []

        t0 = time.time()
        for fold_i, (train, test) in enumerate(folds):
            self._log(
                f"fold {fold_i + 1}/{len(folds)}: "
                f"train={len(train)} test={len(test)}"
            )
            if self.config.descriptors_only:
                # Descriptor baseline: single deterministic featurization, no per-seed loop.
                X_train, y_train, X_test, y_test = self._descriptor_baseline(train, test)
                for seed in seed_list:
                    self._log(f"fold {fold_i + 1}: seed={seed} descriptor head/eval")
                    np.random.seed(seed)
                    y_pred = self._fit_predict(X_train, y_train, X_test)
                    m = self._metrics(y_test, y_pred)
                    self._log(
                        f"fold {fold_i + 1}: seed={seed} "
                        f"RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f}"
                    )
                    m["fold"] = fold_i
                    m["seed"] = int(seed)
                    m["n_train"] = int(len(y_train))
                    m["n_test"] = int(len(y_test))
                    fold_metrics.append(m)
                continue

            # GVFA path: featurize once per fold, then loop seeds.
            atom_feat, bond_feat = self._build_featurizers()
            train_graphs = self._featurize(train, atom_feat, bond_feat, "train")
            test_graphs = self._featurize(test, atom_feat, bond_feat, "test")
            y_train = np.array([float(g.target.item()) for g in train_graphs])
            y_test = np.array([float(g.target.item()) for g in test_graphs])
            self._log(
                f"fold {fold_i + 1}: graph counts train={len(train_graphs)} "
                f"test={len(test_graphs)}"
            )

            for seed in seed_list:
                seed_t0 = time.time()
                self._log(f"fold {fold_i + 1}: seed={seed} start")
                torch.manual_seed(seed)
                np.random.seed(seed)
                # Projections mutate g.h_atom / g.h_bond — reset between seeds.
                for g in train_graphs + test_graphs:
                    g.h_atom = None
                    g.h_bond = None
                self._project_atoms(train_graphs, test_graphs, seed_override=seed)
                self._project_bonds(train_graphs, test_graphs, seed_override=seed)
                encoder = self._build_encoder()
                augmentation = self._build_augmentation()
                multi_stat = self._build_multi_stat_pool()
                size_aware = self._build_size_aware()
                X_train = self._encode_and_pool(
                    train_graphs, encoder, augmentation, multi_stat, size_aware, "train"
                )
                X_test = self._encode_and_pool(
                    test_graphs, encoder, augmentation, multi_stat, size_aware, "test"
                )
                y_pred = self._fit_predict(X_train, y_train, X_test)
                m = self._metrics(y_test, y_pred)
                self._log(
                    f"fold {fold_i + 1}: seed={seed} done "
                    f"RMSE={m['rmse']:.4f} STD_err={m['std_err']:.4f} "
                    f"MAE={m['mae']:.4f} R2={m['r2']:.4f} "
                    f"Pearson_R2={m['pearson_r2']:.4f} "
                    f"({time.time() - seed_t0:.1f}s)"
                )
                m["fold"] = fold_i
                m["seed"] = int(seed)
                m["n_train"] = int(len(y_train))
                m["n_test"] = int(len(y_test))
                fold_metrics.append(m)
        elapsed = time.time() - t0
        self._log(f"all folds complete ({elapsed:.1f}s)")

        # Aggregate across folds
        keys = ["rmse", "std_err", "mae", "r2", "pearson", "pearson_r", "pearson_r2"]
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
        self._log(f"wrote result JSON: {run_path}")
        self._log(f"appended summary CSV: {out_dir / 'summary.csv'}")
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
