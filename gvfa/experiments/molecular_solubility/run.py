"""
CLI for a single molecular-solubility run.

Usage:
    python -m gvfa.experiments.molecular_solubility.run --config <yaml>
        [--override key.path=value ...]

The YAML maps onto :class:`PipelineConfig`; overrides are applied on top.
Results are written to ``{logging.out_dir}/{logging.run_name}.json`` and a
row is appended to ``{logging.out_dir}/summary.csv``.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from .config import apply_overrides, load_config
from .pipeline import MolecularSolubilityPipeline


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single molecular-solubility configuration."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML config file."
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Override a config field (can be repeated).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    pipeline = MolecularSolubilityPipeline(cfg)
    result = pipeline.run()

    print(f"\n[{result['run_name']}] elapsed={result['elapsed_sec']:.1f}s")
    print(
        f"  RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}    "
        f"MAE: {result['mae_mean']:.4f} ± {result['mae_std']:.4f}    "
        f"R²:  {result['r2_mean']:.4f} ± {result['r2_std']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
