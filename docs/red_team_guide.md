# Red Team Adversarial Testing Guide

## Overview

The red team module simulates adversarial attacks against anonymized and synthetic data to validate privacy guarantees. It implements 10 automated attacks covering re-identification, fingerprinting, and linkage risks.

## Threat Model

The adversary is assumed to have:
- **Auxiliary knowledge**: External datasets containing some of the same individuals
- **Quasi-identifier knowledge**: Publicly available attributes (age, region, gender, etc.)
- **Statistical knowledge**: Understanding of the original data's distributions
- **Technical capability**: Ability to run computational attacks

## Attack Inventory

### Critical Severity

| Attack | Description | Defence |
|--------|-------------|---------|
| **Uniqueness** | Tests 2/3/4-way QI combinations for unique records | k-anonymity enforcement |

### High Severity

| Attack | Description | Defence |
|--------|-------------|---------|
| **Temporal Linkage** | Checks if dates can link to external calendars | Date perturbation |
| **Fingerprints** | Scans for residual source-identifying patterns | Fingerprint scrubbing |
| **Outlier Re-identification** | Finds statistical outliers identifiable in external data | Noise injection + capping |
| **Rare Combo Linkage** | Finds rare value combinations violating k-anonymity | QI generalization |
| **Compound Entity** | Checks if multi-attribute records create unique fingerprints | Broader generalization |

### Medium Severity

| Attack | Description | Defence |
|--------|-------------|---------|
| **Distribution Skew** | Detects near-identical pre/post distributions | Noise + equalization |
| **Null Pattern Linkage** | Checks if null patterns across columns are unique | Null normalization |
| **Numeric Precision** | Checks if values retain too many decimal places | Rounding |
| **Numeric Ratio** | Checks if column ratios reconstruct originals | Independent noise |

## Running Attacks

```bash
# Against anonymized data
python scripts/red_team.py \
    --anonymized data/anonymized.csv \
    --config config/example_red_team.yaml

# Against synthetic data
python scripts/red_team.py \
    --anonymized data/synthetic.csv \
    --original data/anonymized.csv \
    --config config/example_red_team.yaml \
    --mode synthetic
```

## Interpreting Results

Each attack returns:
- **Status**: PASS / FAIL / SKIP
- **Score**: 0.0 (total failure) to 1.0 (perfect pass)
- **Severity**: CRITICAL / HIGH / MEDIUM / LOW
- **Recommendations**: Specific actions to fix failures

A dataset should pass all CRITICAL and HIGH severity attacks before release.

## Custom Attacks

Extend `BaseAttack` to add domain-specific attacks:

```python
from validation.red_team import BaseAttack, AttackResult

class MyCustomAttack(BaseAttack):
    name = "Custom_Attack"
    severity = "HIGH"

    def run(self, df_anon, df_orig=None):
        # Your attack logic here
        return AttackResult(
            name=self.name,
            passed=True,
            score=1.0,
            details={"info": "attack details"},
        )
```
