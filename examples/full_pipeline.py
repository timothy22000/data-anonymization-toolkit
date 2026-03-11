"""Full pipeline example: anonymize -> generate synthetic -> validate -> red team.

Demonstrates the end-to-end workflow using all toolkit components.
"""

import pandas as pd

from anonymization.config import load_config
from anonymization.anonymizer import AnonymizationPipeline
from synthetic.config import load_synthetic_config
from synthetic.synthesizers import PhaseTrainer
from validation.quality import run_quality_checks
from validation.privacy_metrics import (
    membership_inference_auc,
    duplicate_class_rate,
    new_row_rate,
)
from reporting.quality_report import QualityReportGenerator


def main():
    # Step 1: Anonymize
    print("=" * 50)
    print("STEP 1: Anonymization")
    print("=" * 50)
    anon_config = load_config("config/example_anonymization.yaml")
    df = pd.read_csv("data/input.csv")
    pipeline = AnonymizationPipeline(anon_config)
    anonymized = pipeline.run(df)
    anonymized.to_csv("data/anonymized.csv", index=False)
    print(f"Anonymized: {len(anonymized)} rows\n")

    # Step 2: Generate synthetic data
    print("=" * 50)
    print("STEP 2: Synthetic generation")
    print("=" * 50)
    synth_config = load_synthetic_config("config/example_synthetic.yaml")
    trainer = PhaseTrainer(synth_config)
    synthetic = trainer.run_all_phases(anonymized)
    synthetic.to_csv("data/synthetic.csv", index=False)
    print(f"Generated: {len(synthetic)} synthetic rows\n")

    # Step 3: Validate
    print("=" * 50)
    print("STEP 3: Validation")
    print("=" * 50)
    cat_cols = anonymized.select_dtypes(include=["object"]).columns.tolist()
    num_cols = anonymized.select_dtypes(include=["number"]).columns.tolist()

    quality_report = run_quality_checks(
        anonymized, synthetic,
        {"categorical_cols": cat_cols, "numerical_cols": num_cols},
    )
    print(f"Quality score: {quality_report.overall_score:.3f}")

    mia = membership_inference_auc(anonymized, synthetic, num_cols)
    dcr = duplicate_class_rate(anonymized, synthetic)
    nrr = new_row_rate(anonymized, synthetic)
    print(f"MIA AUC: {mia:.3f} | DCR: {dcr:.4f} | New row rate: {nrr:.4f}\n")

    # Step 4: Generate reports
    print("=" * 50)
    print("STEP 4: Reports")
    print("=" * 50)
    reporter = QualityReportGenerator(output_dir="reports")
    q_path = reporter.generate_quality_report(quality_report)
    p_path = reporter.generate_privacy_report({
        "mia_auc": mia, "dcr": dcr, "new_row_rate": nrr,
    })
    print(f"Quality report: {q_path}")
    print(f"Privacy report: {p_path}")


if __name__ == "__main__":
    main()
