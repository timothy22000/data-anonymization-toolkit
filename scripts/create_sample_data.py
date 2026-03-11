#!/usr/bin/env python3
"""Generate a realistic sample CSV for demonstrating the anonymization pipeline.

Creates a 200-row dataset with columns that match the example YAML configs,
including direct identifiers (PII), quasi-identifiers, numeric targets,
date columns, and fingerprint/operational columns.

Usage:
    python scripts/create_sample_data.py
    python scripts/create_sample_data.py --rows 500 --output data/custom_sample.csv
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DEFAULT_ROWS = 200
DEFAULT_OUTPUT = "data/sample.csv"

REGIONS = [
    "North", "South", "East", "West", "Central",
    "Northeast", "Southeast", "Northwest", "Southwest", "Midlands",
]

GENDERS = ["M", "F", "Other"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]

MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"]
MARITAL_WEIGHTS = [0.35, 0.45, 0.12, 0.08]

SOURCE_SYSTEMS = ["SYS_A", "SYS_B"]


def generate_sample_data(n_rows: int = DEFAULT_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Generate a realistic sample dataset.

    Args:
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns matching the example YAML configs.
    """
    rng = np.random.default_rng(seed)

    # --- Direct identifiers (PII) ---
    full_names = [f"Person_{i + 1:03d}" for i in range(n_rows)]
    emails = [f"person_{i + 1:03d}@example.com" for i in range(n_rows)]
    phones = [f"555-{i + 1:04d}" for i in range(n_rows)]
    national_ids = [f"{rng.integers(100, 999)}-{rng.integers(10, 99)}-{rng.integers(1000, 9999)}" for _ in range(n_rows)]

    # --- Quasi-identifiers ---
    ages = rng.integers(18, 81, size=n_rows)
    genders = rng.choice(GENDERS, size=n_rows, p=GENDER_WEIGHTS)
    regions = rng.choice(REGIONS, size=n_rows, p=[0.15, 0.12, 0.10, 0.10, 0.13, 0.08, 0.09, 0.07, 0.06, 0.10])
    marital_statuses = rng.choice(MARITAL_STATUSES, size=n_rows, p=MARITAL_WEIGHTS)

    # --- Numeric targets ---
    incomes = np.round(rng.lognormal(mean=10.8, sigma=0.5, size=n_rows), 2)
    incomes = np.clip(incomes, 20000, 150000)

    balances = np.round(rng.normal(loc=15000, scale=8000, size=n_rows), 2)
    balances = np.clip(balances, -5000, 50000)

    credit_scores = rng.integers(300, 851, size=n_rows)

    transaction_amounts = np.round(rng.exponential(scale=250, size=n_rows), 2)
    transaction_amounts = np.clip(transaction_amounts, 1.0, 5000.0)

    # --- Date columns ---
    base_date = datetime(2024, 1, 1)
    created_dates = [
        (base_date + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d")
        for _ in range(n_rows)
    ]
    last_login_dates = [
        (base_date + timedelta(days=int(rng.integers(180, 365)))).strftime("%Y-%m-%d")
        for _ in range(n_rows)
    ]

    # --- Fingerprint / operational columns ---
    source_system_codes = rng.choice(SOURCE_SYSTEMS, size=n_rows)
    internal_batch_ids = [f"BATCH-{rng.integers(1000, 9999)}" for _ in range(n_rows)]

    # --- Sprinkle in some nulls (realistic) ---
    null_indices_marital = rng.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    null_indices_balance = rng.choice(n_rows, size=int(n_rows * 0.03), replace=False)

    marital_list = list(marital_statuses)
    for idx in null_indices_marital:
        marital_list[idx] = None

    balances_list = list(balances)
    for idx in null_indices_balance:
        balances_list[idx] = np.nan

    df = pd.DataFrame({
        "full_name": full_names,
        "email": emails,
        "phone_number": phones,
        "national_id": national_ids,
        "age": ages,
        "gender": genders,
        "region": regions,
        "marital_status": marital_list,
        "income": incomes,
        "balance": balances_list,
        "credit_score": credit_scores,
        "transaction_amount": transaction_amounts,
        "created_date": created_dates,
        "last_login_date": last_login_dates,
        "source_system_code": source_system_codes,
        "internal_batch_id": internal_batch_ids,
    })

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a sample CSV dataset for anonymization demos."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help=f"Number of rows (default: {DEFAULT_ROWS})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_sample_data(n_rows=args.rows, seed=args.seed)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} rows x {len(df.columns)} columns")
    print(f"Saved to: {output_path}")
    print(f"\nColumns: {', '.join(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
