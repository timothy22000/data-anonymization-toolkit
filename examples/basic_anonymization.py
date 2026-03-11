"""Basic anonymization example.

Demonstrates how to use the AnonymizationPipeline programmatically.
"""

import pandas as pd

from anonymization.config import load_config
from anonymization.anonymizer import AnonymizationPipeline


def main():
    # --- Option 1: Load config from YAML ---
    config = load_config("config/example_simple_anonymization.yaml")

    # Load your data
    df = pd.read_csv("data/input.csv")

    # Run the full pipeline
    pipeline = AnonymizationPipeline(config)
    anonymized = pipeline.run(df)

    # Save result
    anonymized.to_csv("data/anonymized.csv", index=False)
    print(f"Anonymized: {len(anonymized)} rows x {len(anonymized.columns)} columns")

    # --- Option 2: Run individual steps ---
    # pipeline = AnonymizationPipeline(config)
    # df = pipeline.drop_columns(df)
    # df = pipeline.round_floats(df, decimals=2)
    # ... etc.


if __name__ == "__main__":
    main()
