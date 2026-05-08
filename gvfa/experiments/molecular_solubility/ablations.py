"""
Ablation sweep runner.

Runs every YAML config under ``configs/`` (or a user-supplied directory) and
aggregates the per-run metrics into a single comparison table.

Usage:
    python -m gvfa.experiments.molecular_solubility.ablations
    python -m gvfa.experiments.molecular_solubility.ablations --configs-dir <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .config import apply_overrides, load_config
from .pipeline import MolecularSolubilityPipeline


def main(argv: List[str] | None = None) -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Run every YAML in configs/ and aggregate results."
    )
    parser.add_argument(
        "--configs-dir",
        default=str(here / "configs"),
        help="Directory containing ablation YAML files.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Override applied to every run (e.g. logging.out_dir=...).",
    )
    args = parser.parse_args(argv)

    configs_dir = Path(args.configs_dir)
    yamls = sorted(configs_dir.glob("*.yaml"))
    if not yamls:
        print(f"No YAML files under {configs_dir}", file=sys.stderr)
        return 1

    print(f"Running {len(yamls)} ablation(s) from {configs_dir}")
    rows: List[dict] = []
    for path in yamls:
        cfg = load_config(str(path))
        cfg = apply_overrides(cfg, args.override)
        # Force run_name from filename so each run gets a unique JSON / CSV row
        cfg.logging.run_name = path.stem
        print(f"\n=== {path.name} ===")
        pipeline = MolecularSolubilityPipeline(cfg)
        result = pipeline.run()
        rows.append(result)
        print(
            f"  RMSE: {result['rmse_mean']:.4f}  MAE: {result['mae_mean']:.4f}  "
            f"R²: {result['r2_mean']:.4f}"
        )

    print("\n=== Summary ===")
    print(f"{'run':<32} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    for r in rows:
        print(
            f"{r['run_name']:<32} {r['rmse_mean']:>10.4f} {r['mae_mean']:>10.4f} {r['r2_mean']:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
