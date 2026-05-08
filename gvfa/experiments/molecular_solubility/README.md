# Molecular Solubility Experiment

A configurable graph-regression pipeline for predicting LogS (or any
solubility-like target) from SMILES.

The pipeline reuses the base GVFA encoder (`gvfa.models.GVFA`) and
hyperdimensional operations (`gvfa.operations.{bind, superpose, permute}`)
unchanged. Chemistry-specific stages live in `gvfa.chem.*`. Ablations are
expressed as YAML configs that map 1:1 onto the dataclass schema in
`config.py`.

## Pipeline

```
SMILES + target
   ↓ Dataset           (CSV loader / MoleculeNet)
   ↓ Splitter          (random / scaffold / kfold / explicit train+test files)
   ↓ MolFeaturizer     (atom features, optional bond features, optional 3D conformer)
   ↓ Projection        (Bounded scaler? + Random projection: gaussian | orthogonal)
   ↓ EdgeBinder        (off | hadamard | circular)
   ↓ Encoder           (EdgeAwareGVFA — base GVFA when no binder is configured)
   ↓ Augmentation      (off | reservoir | sigma_pi | reservoir_sigma_pi)
   ↓ Pooler            (sum | mean)
   ↓ Head              (ridge | ridgecv | kernel_ridge | xgboost | random_forest)
   ↓ Evaluator         (RMSE / MAE / R² / Pearson) → JSON + summary.csv
```

The `descriptors_only: true` config flag short-circuits everything between
MolFeaturizer and Head, replacing them with a 96-column RDKit descriptor
extractor — that's the "traditional baseline" ablation.

## Data layout

The pipeline expects CSV files at:

```
data/solubility/
├── train.csv
├── test.csv
└── novel_test.csv         # optional, used by some ablations
```

The repo's `.gitignore` deliberately excludes `data/`, so these are not
checked in. Populate them locally — for the in-house solubility dataset,
copy them from `Molecular_Solubility/GVFA/final_data/` and
`Molecular_Solubility/GVFA_with_edge/final_data/`.

The default column names are `smiles_canon` and `LogS`; override via the
`dataset.smiles_col` and `dataset.target_col` config fields.

For public benchmarks (ESOL, FreeSolv, Lipo) set
`dataset.kind: moleculenet` and `dataset.moleculenet_name: ESOL` instead.

## Running a single config

```bash
python -m gvfa.experiments.molecular_solubility.run \
    --config gvfa/experiments/molecular_solubility/configs/base.yaml
```

Override individual fields without editing the YAML:

```bash
python -m gvfa.experiments.molecular_solubility.run \
    --config gvfa/experiments/molecular_solubility/configs/edge.yaml \
    --override projection.D=10000 \
    --override encoder.num_layers=5
```

Each run writes:
- `results/molecular_solubility/<run_name>.json` — full config + metrics
- `results/molecular_solubility/summary.csv`     — one row appended

## Running every ablation

```bash
python -m gvfa.experiments.molecular_solubility.ablations
```

Loads every YAML under `configs/`, runs each, and prints a summary table.

## Adding a new ablation

1. Drop a new `configs/my_ablation.yaml` that overrides whatever fields differ
   from `base.yaml`. Unspecified fields fall back to dataclass defaults.
2. Run it: `python -m gvfa.experiments.molecular_solubility.run --config configs/my_ablation.yaml`.
3. (Optional) The ablation sweep picks it up automatically next run.

## Ablations bundled

| YAML                      | Encoder path                                           | Head    |
|---------------------------|--------------------------------------------------------|---------|
| `base.yaml`               | atoms-only, gaussian, GVFA                             | ridge   |
| `edge.yaml`               | + bonds, hadamard binder                               | ridge   |
| `binding_circular.yaml`   | edge.yaml with circular FFT binder                     | ridge   |
| `binding_hadamard.yaml`   | explicit hadamard twin of binding_circular             | ridge   |
| `edge_orthogonal.yaml`    | edge.yaml + orthogonal projection                      | ridge   |
| `edge_bounded.yaml`       | extended atoms + bonds + bounded scaling + orthogonal  | ridgecv |
| `sigma_pi.yaml`           | edge_bounded + Sigma-Pi expansion (orders 0,1,2)       | ridgecv |
| `reservoir.yaml`          | edge_bounded + reservoir tap-buffer + Sigma-Pi (4 hops)| ridgecv |
| `traditional_baseline.yaml`| RDKit descriptors → XGBoost (no GVFA)                 | xgboost |

## Parity with the legacy code

| Variant                                    | Legacy MAE | New MAE | Legacy R² | New R² |
|--------------------------------------------|------------|---------|-----------|--------|
| Base GVFA (D=2000, 5 layers, atoms, XGB)   | 0.518      | 0.510   | 0.880     | 0.884  |
| Traditional baseline (96 descriptors, XGB) | 0.418      | 0.439   | 0.923     | 0.915  |

Both within ±5% on every metric. (Legacy `Molecular_Solubility/GVFA/GVFA_main.py:95`
mislabels `mean_squared_error` as "RMSE" — the column reported as
`RMSE=0.502` in `Traditional_features/Result/Summary.csv` is actually MSE;
the new pipeline computes `RMSE = sqrt(MSE)` properly.)

To reproduce the parity check:

```bash
# Base GVFA (5 layers, XGB, D=2000)
python -m gvfa.experiments.molecular_solubility.run \
    --config configs/base.yaml \
    --override projection.D=2000 \
    --override encoder.num_layers=5 \
    --override head.kind=xgboost \
    --override head.n_estimators=2000 \
    --override head.max_depth=7 \
    --override head.learning_rate=0.03

# Traditional baseline
python -m gvfa.experiments.molecular_solubility.run \
    --config configs/traditional_baseline.yaml
```

## How this relates to the legacy code

The legacy `Molecular_Solubility/{GVFA, GVFA_update, GVFA_with_edge,
Traditional_features}` folders each implemented one of the above ablations
as a near-copy of the full pipeline. Every novel idea (orthogonal
projection, bounded scaling, edge binding, Sigma-Pi, reservoir) is now a
swappable stage in this pipeline; the old folders are kept as a read-only
reference until numerical parity is confirmed.

The mapping from legacy classes to new primitives is documented in
`/Users/sachinkahawala/.claude/plans/i-have-the-original-peaceful-hollerith.md`.
