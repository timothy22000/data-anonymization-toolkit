# Anonymization Methodology

## Overview

This document describes the anonymization methodology applied to the [DATASET_NAME] dataset. The goal is to produce a privacy-protected version suitable for [PURPOSE] while preserving analytical utility.

## Pipeline Steps

### 1. Direct Identifier Removal
All columns containing direct personal identifiers (names, contact details, government IDs) were removed entirely.

### 2. Float Rounding
Numeric values were rounded to reduce precision that could aid re-identification.

### 3. Quasi-Identifier Generalization
Columns that could be combined to identify individuals were generalised:
- **Banding**: Continuous values grouped into ranges (e.g., age 25 → "20-29")
- **Top-N capping**: Rare categories replaced with "Other"
- **Rounding**: Values rounded to the nearest step

### 4. Fingerprint Scrubbing
String columns were scanned for patterns that could identify the data source (organisation names, internal codes, system identifiers).

### 5. Categorical Equalization
Where applicable, the distribution of categorical columns was flattened to prevent distribution-based re-identification.

### 6. Date Perturbation
Date columns were shifted by a deterministic, HMAC-derived offset (consistent per record) to prevent calendar-based linkage attacks.

### 7. Numeric Noise Injection
Numeric target columns received calibrated noise:
- **Multiplicative noise**: N(1, σ) scaling preserves relative magnitudes
- **Laplacian noise**: Additive perturbation for differential-privacy-style protection

### 8. Null Pattern Normalization
Inconsistent null representations (empty strings, "N/A", "NULL", etc.) were unified to prevent null-pattern-based fingerprinting.

### 9. k-Anonymity Enforcement
Multi-stage enforcement ensured every combination of quasi-identifiers appears in at least k records:
- **Target k**: [K_VALUE]
- **Method**: Iterative suppression of smallest equivalence classes
- **QI groups checked**: [LIST_QI_GROUPS]

### 10. Output Preparation
Columns were renamed to remove internal naming conventions, and the final dataset was sorted and formatted for release.

## Privacy Parameters

| Parameter | Value |
|-----------|-------|
| Profile | [PROFILE] |
| k-anonymity target | [K_VALUE] |
| Numeric noise (σ) | [NOISE_PCT] |
| Date perturbation (max days) | [MAX_DAYS] |
| Random seed | [SEED] |

## Validation

The anonymized dataset was validated using:
- **Statistical quality checks**: Marginal distributions, correlation fidelity, cross-tabulation divergence
- **Privacy metrics**: Membership inference AUC, duplicate class rate, nearest-neighbour distances
- **Adversarial red team**: [N_ATTACKS] automated attacks testing re-identification, fingerprinting, and linkage risks
