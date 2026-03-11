# Anonymization Guide

## Overview

The anonymization pipeline transforms a raw dataset into a privacy-protected version by applying a sequence of 10 steps. Each step targets a different aspect of re-identification risk.

## Step-by-Step Guide

### Step 1: Drop Direct Identifiers

Direct identifiers are columns that uniquely identify an individual on their own : names, email addresses, phone numbers, government IDs, etc. These are dropped entirely as they cannot be meaningfully anonymized.

Configure in YAML:
```yaml
columns:
  drop:
    direct_ids: [full_name, email, phone, national_id]
```

### Step 2: Round Floats

High-precision floating-point values can serve as quasi-identifiers. Rounding reduces the effective cardinality of numeric columns.

### Step 3: Generalize Quasi-Identifiers

Quasi-identifiers (QIs) are columns that, in combination, could re-identify individuals. Four generalization strategies are available:

- **Banding**: Group continuous values into ranges (e.g., age → 5-year bands)
- **Rounding**: Round to nearest step (e.g., income → nearest 5000)
- **Top-N**: Keep the N most common categories, replace rest with "Other"
- **Capping**: Clip values to min/max bounds

### Step 4: Scrub Fingerprints

Scan all string columns for patterns that could identify the data source : organization names, internal codes, system identifiers. Matches are replaced with generic tokens.

### Step 5: Equalize Categoricals (Optional)

If a categorical column has a skewed distribution that matches known external information, subsample to flatten the distribution. This defeats distribution-matching attacks.

### Step 6: Perturb Dates

Date columns are shifted by a deterministic offset derived from HMAC-SHA256. The same record always gets the same offset, ensuring consistency if the pipeline is re-run.

### Step 7: Inject Numeric Noise

Two noise types are supported:
- **Multiplicative** (N(1, σ)): Preserves sign and rough magnitude
- **Laplacian** (Lap(0, b)): Heavier tails, aligned with differential privacy concepts

### Step 8: Normalize Null Patterns

Replace various null representations ("", "N/A", "NULL", "None", NaN) with a single token. This prevents null-pattern fingerprinting across columns.

### Step 9: Enforce k-Anonymity

Iteratively suppress records in the smallest equivalence classes until every QI group combination has at least k records.

### Step 10: Prepare Output

Apply column renaming, sort, and format for release.

## Choosing a Profile

| Aspect | ml (internal) | public (external) |
|--------|---------------|-------------------|
| k-anonymity | 5 | 20 |
| Noise level | ±2% | ±5% |
| Date perturbation | ±7 days | ±30 days |
| Float precision | 2 dp | 0 dp |
| Use case | Internal ML training | Public data release |
