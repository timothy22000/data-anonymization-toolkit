#!/usr/bin/env python3
"""CLI entry point for quality and privacy validation.

Usage:
    python scripts/validate.py \\
        --real data/anonymized.csv \\
        --synthetic data/synthetic.csv \\
        --output reports/quality_report.md
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.quality import run_quality_checks
from validation.privacy_metrics import (
    membership_inference_auc,
    duplicate_class_rate,
    new_row_rate,
    nearest_neighbor_distance,
)
from reporting.quality_report import QualityReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate",
        description="Run quality and privacy validation on synthetic data.",
    )
    parser.add_argument(
        "--real", "-r", required=True,
        help="Path to real/anonymized CSV.",
    )
    parser.add_argument(
        "--synthetic", "-s", required=True,
        help="Path to synthetic CSV.",
    )
    parser.add_argument(
        "--output", "-o", default="reports",
        help="Output directory for reports (default: reports/).",
    )
    parser.add_argument(
        "--sample", type=int, default=50_000,
        help="Max rows to sample for expensive checks (default: 50000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Load data
    log.info("Loading real data from %s", args.real)
    real_df = pd.read_csv(args.real, low_memory=False)
    log.info("Loading synthetic data from %s", args.synthetic)
    synth_df = pd.read_csv(args.synthetic, low_memory=False)

    # Sample if needed
    if len(real_df) > args.sample:
        real_df = real_df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
    if len(synth_df) > args.sample:
        synth_df = synth_df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    log.info("Real: %d rows x %d cols | Synthetic: %d rows x %d cols",
             len(real_df), len(real_df.columns),
             len(synth_df), len(synth_df.columns))

    # Auto-detect column types
    cat_cols = real_df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = real_df.select_dtypes(include=["number"]).columns.tolist()

    # Quality checks
    log.info("Running quality checks...")
    quality_config = {
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
    }
    quality_report = run_quality_checks(real_df, synth_df, quality_config)
    log.info("Quality score: %.3f", quality_report.overall_score)

    # Privacy metrics
    log.info("Computing privacy metrics...")
    privacy_results = {}

    mia = membership_inference_auc(real_df, synth_df, num_cols, seed=args.seed)
    privacy_results["mia_auc"] = mia
    log.info("MIA AUC: %.3f", mia)

    dcr = duplicate_class_rate(real_df, synth_df)
    privacy_results["dcr"] = dcr
    log.info("Duplicate class rate: %.4f", dcr)

    nrr = new_row_rate(real_df, synth_df)
    privacy_results["new_row_rate"] = nrr
    log.info("New row rate: %.4f", nrr)

    nn_dist = nearest_neighbor_distance(real_df, synth_df, num_cols, seed=args.seed)
    privacy_results["nn_distances"] = nn_dist
    log.info("Nearest neighbor mean distance: %.4f", nn_dist.get("mean_distance", 0))

    # Generate reports
    reporter = QualityReportGenerator(output_dir=args.output)

    q_path = reporter.generate_quality_report(quality_report)
    log.info("Quality report saved to %s", q_path)

    p_path = reporter.generate_privacy_report(privacy_results)
    log.info("Privacy report saved to %s", p_path)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Quality score:       {quality_report.overall_score:.3f}")
    print(f"  MIA AUC:             {mia:.3f}  {'PASS' if mia < 0.6 else 'FAIL'}")
    print(f"  Duplicate class rate: {dcr:.4f}  {'PASS' if dcr < 0.01 else 'FAIL'}")
    print(f"  New row rate:        {nrr:.4f}  {'PASS' if nrr > 0.9 else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
