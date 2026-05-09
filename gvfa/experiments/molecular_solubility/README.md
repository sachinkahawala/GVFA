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
   ↓ Pooler            (sum | mean | multi_stat)
   ↓ Size-aware post   (optional: 1/N^p scaling + appended size feature)
   ↓ Head              (ridge | ridgecv | kernel_ridge | xgboost | random_forest)
   ↓ Evaluator         (RMSE / STD_err / MAE / R² / Pearson / Pearson²) → JSON + summary.csv
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

| YAML                          | Encoder path                                            | Head    |
|-------------------------------|---------------------------------------------------------|---------|
| `base.yaml`                   | atoms-only, gaussian, GVFA                              | ridge   |
| `edge.yaml`                   | + bonds, hadamard binder                                | ridge   |
| `binding_circular.yaml`       | edge.yaml with circular FFT binder                      | ridge   |
| `binding_hadamard.yaml`       | explicit hadamard twin of binding_circular              | ridge   |
| `edge_orthogonal.yaml`        | edge.yaml + orthogonal projection                       | ridge   |
| `edge_bounded.yaml`           | extended atoms + bonds + bounded scaling + orthogonal   | ridgecv |
| `sigma_pi.yaml`               | edge_bounded + Sigma-Pi expansion (orders 0,1,2)        | ridgecv |
| `reservoir.yaml`              | edge_bounded + reservoir tap-buffer + Sigma-Pi (4 hops) | ridgecv |
| `traditional_baseline.yaml`   | RDKit descriptors → XGBoost (no GVFA)                   | xgboost |
| `best3.yaml`                  | Best3 sequence: raw orthogonal projection, phi1+sign, edge=circular, reservoir+Sigma-Pi[0,1], multi_stat, sqrt_n + raw size append | ridgecv |
| `best3_no_size_aware.yaml`    | best3.yaml with `pooler.size_aware` disabled (ablation) | ridgecv |
| `best3_clean_sequence.yaml`   | Fixed one-seed wiring check on the legacy solubility_novel CSVs, using modular SMILES chemistry | ridgecv |

## New knobs added by the Best3 port

- **`pooler.kind: multi_stat`** — readout = `concat[mean(F_v) | max(F_v) | mean(bind(F_v, F_v))]`
  per graph → `[num_graphs, 3·D]`. Reuses `gvfa.operations.bind` for the
  squared-mean term. The two non-squared stats match a standard mean+max
  pool; the third adds an HV second-order term.
- **`pooler.size_aware`** — per-graph post-pool transforms:
  - `scale: none | sqrt_n | n | n_pow_1_5` — divides each row by `N^p`.
  - `append_size: bool`, `append_size_kind: raw | log1p_over_log10` —
    optionally append a size column. Use `raw` to mirror Best3, the
    log1p form to keep the column comparable in magnitude to bipolar HVs.
- **`seeds: [int, ...]`** (top level) — when set, the pipeline runs
  one (project → encode → pool → head → metrics) pass per seed,
  reusing the (expensive) featurization across seeds. Output reports
  mean ± std across seeds. Falls back to single-seed when omitted.
- **Head controls** — `head.standardize` toggles `StandardScaler` before the
  regression head. `head.alphas_logspace` can generate RidgeCV grids such as
  `np.logspace(-4, 2, 50)`, and `head.ridgecv_cv` / `head.ridgecv_scoring`
  expose the legacy `cv=5, scoring="neg_mean_squared_error"` setup.

## Best3 sequence audit

Use `best3_clean_sequence.yaml` when you want to check the modular operation
order against the intended legacy Best3 recipe without doing a hyperparameter
search:

```bash
python -m gvfa.experiments.molecular_solubility.run \
    --config gvfa/experiments/molecular_solubility/configs/best3_clean_sequence.yaml \
    --override logging.out_dir=/private/tmp/gvfa_sequence_check
```

The modular path intentionally keeps chemically correct SMILES-derived atom
and bond features. Exact metric parity with `GVFA_with_edge` is not expected:
the legacy graph builder reconstructs an RDKit molecule from atomic numbers
and connectivity, which makes all reconstructed bonds single bonds and also
double-counts the directed edge list when computing the degree feature. To
inspect those differences directly:

```bash
python -m gvfa.experiments.molecular_solubility.compare_legacy_featurizers --limit 5
```

## Phi mapping for the legacy `equation` field

The legacy `models/graphcnnVSA_Binding_FULL.py:next_layer_eps` switches between
two per-layer formulas via the `equation` arg:

| Legacy `(equation, delta)` | Per-layer formula                          | Base-library equivalent          |
|----------------------------|--------------------------------------------|----------------------------------|
| `(10, 0)`                  | `sign(aggregate(roll(h)) + h)`             | **`phi1` + `sign`-normalize**    |
| `(10, 1)`                  | `sign(bind(h, aggregate(roll(h))) + h)`    | **`phi2` + `sign`-normalize**    |
| `(10, 2)`                  | `sign(bind(h, agg(roll(h))) + h + agg(roll(h)))` | (no stock phi — custom combine) |
| `(11, 0)`                  | `sign(roll(aggregate(h) + h))`             | **`phi3` + `sign`-normalize**    |
| `(11, 1)`                  | `sign(roll(bind(h, aggregate(h)) + h))`    | **`phi4` + `sign`-normalize**    |
| `(11, 2)`                  | `sign(roll(bind(h, agg(h)) + h + agg(h)))` | (no stock phi — custom combine) |

Two identities make the equation=10 and equation=11 columns map onto the
same four phi functions:

1. **Aggregation commutes with cyclic shift.** `aggregate(roll(h)) = roll(aggregate(h))`
   because aggregation is a sum, and `roll` is a linear operator.

2. **Circular FFT bind commutes with cyclic shift on either argument.**
   By the frequency-shift identity `fft(roll(x, k)) = fft(x) · exp(-2πik/N)`,
   the phase factor pulls out of the FFT product, giving
   `bind(a, roll(b, k)) = roll(bind(a, b), k) = bind(roll(a, k), b)`.
   *This identity is specific to circular FFT bind — Hadamard bind
   does NOT commute with `roll`.*

So `equation=10` (rotate `h` BEFORE aggregating, no rotation at the end)
produces the same algebraic expression as `equation=11` (no pre-rotation,
rotate the FINAL sum) — modulo the `+h` term, which is what shifts each
`equation=10` row to a one-lower phi index than its `equation=11` twin
(`(10,δ) → phi(δ+1)` ... `(11,δ) → phi(δ+3)`).

`delta=2` is a 3-term combine `(bind(h,f) + h + f)` that doesn't appear
in the four-phi catalog — using it would require a custom combine
function or a small extension to `gvfa.models.combine`.

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
reference for operation-sequence audits and metric comparisons.

The mapping from legacy classes to new primitives is documented in
`/Users/sachinkahawala/.claude/plans/i-have-the-original-peaceful-hollerith.md`.
