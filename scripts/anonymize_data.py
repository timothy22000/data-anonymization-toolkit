#!/usr/bin/env python3
"""CLI entry point for the anonymization pipeline.

Usage:
    python scripts/anonymize_data.py \\
        --input data/raw.csv \\
        --output data/anonymized.csv \\
        --config config/example_anonymization.yaml

    python scripts/anonymize_data.py \\
        --input data/raw.csv \\
        --output data/anonymized.csv \\
        --config config/example_anonymization.yaml \\
        --profile public --seed 99
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anonymization.config import load_config
from anonymization.anonymizer import AnonymizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anonymize_data",
        description="Run the config-driven anonymization pipeline on a CSV dataset.",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path for anonymized output CSV.",
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--profile", default=None,
        help="Override config profile ('ml' or 'public').",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--k-target", type=int, default=None,
        help="Override k-anonymity target.",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Process only a random sample of N rows (for testing).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.profile is not None:
        config.profile = args.profile
    if args.seed is not None:
        config.seed = args.seed
    if args.k_target is not None:
        config.k_target = args.k_target

    # Load data
    log.info("Loading input data from %s", args.input)
    t0 = time.perf_counter()
    df = pd.read_csv(args.input, low_memory=False)
    log.info("Loaded %d rows x %d columns in %.1fs",
             len(df), len(df.columns), time.perf_counter() - t0)

    # Optional sampling
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=config.seed).reset_index(drop=True)
        log.info("Sampled %d rows for processing", len(df))

    # Run pipeline
    pipeline = AnonymizationPipeline(config)
    result = pipeline.run(df)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    log.info("Saved anonymized data to %s (%d rows x %d columns)",
             output_path, len(result), len(result.columns))


if __name__ == "__main__":
    main()
