# Synthetic Data Modelling & Generation Guide

## Overview

The synthetic generation module creates entirely new records that preserve the statistical properties of the original data without containing any real individual's information. This is achieved through multi-phase conditional generation using SDV (Synthetic Data Vault).

## Multi-Phase Approach

Rather than training a single model on all columns simultaneously, the pipeline splits columns into phases. Each phase is trained on a subset of columns, conditioned on outputs from prior phases.

```
Phase 1: Demographics    → Train model → Generate
                                              ↓ (condition)
Phase 2: Financial       → Train model → Generate
                                              ↓ (condition)
Phase 3: Behavioural     → Train model → Generate
                                              ↓ (condition)
Phase 4: Sparse fields   → Train model → Generate
```

This approach:
- Captures inter-phase dependencies through conditioning
- Handles sparse columns (high null rates) in dedicated phases
- Allows different synthesizer strategies per phase

## Synthesizer Strategies

| Strategy | Best For | Speed | Quality |
|----------|----------|-------|---------|
| `copula` (GaussianCopula) | Continuous data, fast training | Fast | Good |
| `ctgan` (CTGAN) | Mixed categorical/numeric | Slow | Better |
| `tvae` (TVAE) | Complex distributions | Medium | Better |
| `hybrid` | Auto-select per phase | Varies | Best |

## Configuration

```yaml
phases:
  - name: demographics
    columns: [age, gender, region]
    strategy: copula
    sparse: false
  - name: financial
    columns: [income, balance]
    strategy: ctgan
    sparse: false
    condition_on: [demographics]  # condition on prior phases
```

## Column Profiling

The `ColumnProfiler` automatically detects:
- **sdtype**: numerical, categorical, boolean, datetime
- **Sparsity**: columns with >50% null values
- **Cardinality**: number of unique values

Override auto-detection with `column_sdtypes` in config.

## Dev Mode

Use `--dev` for quick iteration:
- Sample fraction: 1%
- Epochs: 50
- Rows: 50,000

This runs in minutes rather than hours on large datasets.
