# Data Dictionary : [DATASET_NAME]

## Overview

| Property | Value |
|----------|-------|
| Dataset name | [DATASET_NAME] |
| Rows | [N_ROWS] |
| Columns | [N_COLUMNS] |
| Anonymization profile | [PROFILE] |
| k-anonymity | [K_VALUE] |
| Date created | [DATE] |

## Column Descriptions

| Column | Type | Description | Anonymization |
|--------|------|-------------|---------------|
| [column_1] | categorical | [Description] | Generalised (top-N) |
| [column_2] | numerical | [Description] | Noise injected (±X%) |
| [column_3] | date | [Description] | Perturbed (±N days) |
| [column_4] | categorical | [Description] | Banded |
| ... | ... | ... | ... |

## Removed Columns

The following column categories were removed during anonymization:

- **Direct identifiers**: Names, contact details, government IDs
- **Organisational fingerprints**: Internal codes, system identifiers
- **Operational columns**: ETL timestamps, checksums, debug flags

## Value Encoding

| Value | Meaning |
|-------|---------|
| UNKNOWN | Missing or suppressed value |
| Other | Rare category (below top-N threshold) |
| [lo - hi] | Banded numeric range |

## Usage Notes

- This dataset has been anonymized and should not be linked with external data sources
- All numeric values include calibrated noise : individual values should not be treated as exact
- Date values have been perturbed : temporal analysis at the day level is not meaningful
