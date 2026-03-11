# Dataset Card : [DATASET_NAME]

## Dataset Description

- **Created by**: [ORGANISATION]
- **Purpose**: [PURPOSE]
- **Language**: English
- **License**: [LICENSE]

## Dataset Summary

[Brief description of the dataset, its origin, and intended use.]

## Dataset Structure

### Data Instances

Each row represents [DESCRIPTION_OF_RECORD].

### Data Fields

See the [Data Dictionary](data_dictionary.md) for full column descriptions.

### Data Splits

| Split | Rows | Description |
|-------|------|-------------|
| Full | [N_ROWS] | Complete anonymized dataset |

## Privacy Protection

This dataset has been processed through a multi-stage anonymization pipeline:

1. Direct identifiers removed
2. Quasi-identifiers generalised (banding, top-N capping, rounding)
3. Numeric values perturbed with calibrated noise
4. Date values shifted by deterministic offsets
5. k-anonymity enforced (k = [K_VALUE])
6. Organisational fingerprints scrubbed
7. Validated via adversarial red team testing

### Privacy Metrics

| Metric | Value | Threshold |
|--------|-------|-----------|
| k-anonymity | [K_VALUE] | ≥ [K_TARGET] |
| Membership inference AUC | [MIA_AUC] | < 0.60 |
| Duplicate class rate | [DCR] | < 0.01 |
| New row rate | [NRR] | > 0.90 |

## Considerations

### Intended Uses
- [List intended analytical uses]

### Out-of-Scope Uses
- Re-identification of individuals
- Linkage with external datasets
- Precision analysis at the individual record level

### Limitations
- Numeric values include noise and should not be treated as exact
- Rare categories have been suppressed to "Other"
- Date precision is limited due to perturbation
