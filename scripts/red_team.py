#!/usr/bin/env python3
"""CLI entry point for red team adversarial testing.

Usage:
    python scripts/red_team.py \\
        --anonymized data/anonymized.csv \\
        --config config/example_red_team.yaml \\
        --output reports/red_team_report.md

    python scripts/red_team.py \\
        --anonymized data/synthetic.csv \\
        --original data/anonymized.csv \\
        --config config/example_red_team.yaml \\
        --mode synthetic
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.red_team import (
    RedTeamConfig,
    DataLoader,
    RedTeamRunner,
    load_red_team_config,
    Attack_Uniqueness,
    Attack_TemporalLinkage,
    Attack_Fingerprints,
    Attack_OutlierReidentification,
    Attack_DistributionSkew,
    Attack_NullPatternLinkage,
    Attack_RareComboLinkage,
    Attack_NumericPrecision,
    Attack_NumericRatio,
    Attack_CompoundEntity,
)
from reporting.quality_report import QualityReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# All available attack classes
ALL_ATTACKS = [
    Attack_Uniqueness,
    Attack_TemporalLinkage,
    Attack_Fingerprints,
    Attack_OutlierReidentification,
    Attack_DistributionSkew,
    Attack_NullPatternLinkage,
    Attack_RareComboLinkage,
    Attack_NumericPrecision,
    Attack_NumericRatio,
    Attack_CompoundEntity,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="red_team",
        description="Run adversarial attacks against anonymized/synthetic data.",
    )
    parser.add_argument(
        "--anonymized", "-a", required=True,
        help="Path to anonymized or synthetic CSV to test.",
    )
    parser.add_argument(
        "--original", default=None,
        help="Path to original CSV (required for some attacks).",
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to red team YAML configuration.",
    )
    parser.add_argument(
        "--mode", default=None, choices=["anonymized", "synthetic"],
        help="Override mode from config.",
    )
    parser.add_argument(
        "--output", "-o", default="reports",
        help="Output directory for the report (default: reports/).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Override max_sample from config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Load config
    config = load_red_team_config(args.config)

    # Apply CLI overrides
    if args.mode is not None:
        config.mode = args.mode
    if args.sample is not None:
        config.max_sample = args.sample

    # Load data
    loader = DataLoader(config)

    log.info("Loading anonymized/synthetic data from %s", args.anonymized)
    df_anon = loader.load_sample(args.anonymized)
    log.info("Loaded %d rows x %d columns", len(df_anon), len(df_anon.columns))

    df_orig = None
    if args.original:
        log.info("Loading original data from %s", args.original)
        df_orig = loader.load_sample(args.original)
        log.info("Loaded %d rows x %d columns", len(df_orig), len(df_orig.columns))

    # Register and run attacks
    runner = RedTeamRunner(config)
    for attack_cls in ALL_ATTACKS:
        runner.register_attack(attack_cls)

    log.info("Running %d attacks (mode=%s, profile=%s)...",
             len(ALL_ATTACKS), config.mode, config.profile)
    results = runner.run_all(df_anon, df_orig)

    # Generate report
    report_md = runner.generate_report(results)
    reporter = QualityReportGenerator(output_dir=args.output)
    report_path = reporter.generate_red_team_report(results)
    log.info("Red team report saved to %s", report_path)

    # Print summary
    passed = sum(1 for r in results if r.passed or r.skipped)
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)

    print("\n" + "=" * 60)
    print("RED TEAM SUMMARY")
    print("=" * 60)
    print(f"  Total attacks: {len(results)}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {failed}")
    print(f"  Skipped:       {skipped}")
    print("=" * 60)

    if failed > 0:
        print("\nFailed attacks:")
        for r in results:
            if not r.passed and not r.skipped:
                print(f"  [{r.severity}] {r.name}: score={r.score:.2f}")
                for rec in r.recommendations:
                    print(f"    -> {rec}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
