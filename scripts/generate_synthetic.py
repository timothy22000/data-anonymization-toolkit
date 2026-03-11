#!/usr/bin/env python3
"""CLI entry point for synthetic data generation.

Usage:
    python scripts/generate_synthetic.py \\
        --config config/example_synthetic.yaml

    python scripts/generate_synthetic.py \\
        --config config/example_synthetic.yaml \\
        --dev --validate
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic.config import load_synthetic_config
from synthetic.generator import ColumnProfiler, DataPreparer
from synthetic.synthesizers import PhaseTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_synthetic",
        description="Generate synthetic data using multi-phase conditional SDV models.",
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Dev mode: small sample, fewer epochs.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run quality checks after generation.",
    )
    parser.add_argument(
        "--n-rows", type=int, default=None,
        help="Override number of synthetic rows to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Load config
    config = load_synthetic_config(args.config)

    # Apply CLI overrides
    if args.dev:
        config.dev = True
        config.sample_frac = min(config.sample_frac, 0.01)
        config.epochs = min(config.epochs, 50)
        if config.n_rows == 0 or config.n_rows > 50_000:
            config.n_rows = 50_000
        log.info("Dev mode: sample_frac=%.2f, epochs=%d, n_rows=%d",
                 config.sample_frac, config.epochs, config.n_rows)
    if args.n_rows is not None:
        config.n_rows = args.n_rows
    if args.seed is not None:
        config.seed = args.seed
    if args.validate:
        config.validate = True

    # Load input data
    log.info("Loading input data from %s", config.input_path)
    t0 = time.perf_counter()
    df = pd.read_csv(config.input_path, low_memory=False)
    log.info("Loaded %d rows x %d columns in %.1fs",
             len(df), len(df.columns), time.perf_counter() - t0)

    # Sample if needed
    if config.sample_frac < 1.0:
        n_sample = max(1, int(len(df) * config.sample_frac))
        df = df.sample(n=n_sample, random_state=config.seed).reset_index(drop=True)
        log.info("Sampled %d rows for training (%.1f%%)",
                 len(df), config.sample_frac * 100)

    # Profile columns
    profiler = ColumnProfiler(config)
    profiles = profiler.profile(df)
    log.info("Profiled %d columns (%d sparse)",
             len(profiles),
             sum(1 for v in profiles.values() if v.get("is_sparse")))

    # Train and generate
    trainer = PhaseTrainer(config)
    t_start = time.perf_counter()
    synthetic_df = trainer.run_all_phases(df)
    elapsed = time.perf_counter() - t_start
    log.info("Generated %d synthetic rows in %.1fs", len(synthetic_df), elapsed)

    # Save output
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output_path, index=False)
    log.info("Saved synthetic data to %s", output_path)

    # Optional validation
    if config.validate:
        log.info("Running quality validation...")
        try:
            from validation.quality import run_quality_checks

            # Auto-detect column types
            cat_cols = synthetic_df.select_dtypes(include=["object"]).columns.tolist()
            num_cols = synthetic_df.select_dtypes(include=["number"]).columns.tolist()
            quality_config = {
                "categorical_cols": cat_cols,
                "numerical_cols": num_cols,
            }
            report = run_quality_checks(df, synthetic_df, quality_config)
            log.info("Quality score: %.3f", report.overall_score)
        except ImportError:
            log.warning("validation package not available; skipping quality checks")


if __name__ == "__main__":
    main()
