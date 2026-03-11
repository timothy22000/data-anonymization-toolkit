# Quality & Privacy Validation Report

**Dataset**: [DATASET_NAME]
**Date**: [DATE]
**Profile**: [PROFILE]

## Summary

| Check | Score | Status |
|-------|-------|--------|
| Overall quality | [SCORE] | [PASS/FAIL] |
| Marginal distributions | [SCORE] | [PASS/FAIL] |
| Correlation fidelity | [SCORE] | [PASS/FAIL] |
| Cross-tabulation | [SCORE] | [PASS/FAIL] |
| Membership inference | [AUC] | [PASS/FAIL] |
| Duplicate class rate | [DCR] | [PASS/FAIL] |
| New row rate | [NRR] | [PASS/FAIL] |

## Marginal Distribution Tests

| Column | Type | Test | p-value | Status |
|--------|------|------|---------|--------|
| [col] | categorical | Chi-square | [p] | [PASS/FAIL] |
| [col] | numerical | KS | [p] | [PASS/FAIL] |

## Correlation Fidelity

Mean absolute difference between real and synthetic correlation matrices: [VALUE]

## Red Team Results

| Attack | Severity | Score | Status |
|--------|----------|-------|--------|
| [attack_name] | [severity] | [score] | [PASS/FAIL] |

## Recommendations

- [List any recommendations based on failed checks]
