 # Data Anonymization Toolkit

A config-driven Python toolkit for anonymizing tabular datasets, generating synthetic data, and validating privacy guarantees through adversarial testing.

## Features

- **10-step anonymization pipeline** : Drop PII, generalise quasi-identifiers, inject calibrated noise, enforce k-anonymity
- **Config-driven** : All column names and parameters defined in YAML, no hardcoded references
- **Synthetic data modelling & generation** : Multi-phase conditional generation using SDV (GaussianCopula, CTGAN, TVAE)
- **Statistical quality validation** : Chi-square, Kolmogorov-Smirnov, correlation fidelity, Jensen-Shannon divergence
- **Privacy metrics** : Membership inference AUC, duplicate class rate, nearest-neighbour distances
- **Adversarial red team** : 10 automated attacks testing re-identification, fingerprinting, and linkage risks
- **Report generation** : PDF privacy guides, markdown quality/privacy/red-team reports
- **Two built-in profiles** : "ml" (lighter, k=5) for internal use, "public" (stronger, k=20) for external release

## How It Works

The toolkit implements three pillars of data protection:

```
                        ┌─────────────────┐
                        │   Raw Dataset    │
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   ANONYMIZATION PIPELINE │
                    │                          │
                    │  1. Drop direct IDs       │
                    │  2. Round floats           │
                    │  3. Generalise QIs         │
                    │  4. Scrub fingerprints     │
                    │  5. Equalise categoricals  │
                    │  6. Perturb dates          │
                    │  7. Inject numeric noise   │
                    │  8. Normalise nulls        │
                    │  9. Enforce k-anonymity    │
                    │ 10. Prepare output         │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
    ┌─────────────────┐ ┌───────────────┐ ┌─────────────────┐
    │ Anonymised Data  │ │ Synthetic Data│ │   Validation    │
    │   (direct use)   │ │   Modelling   │ │  & Red Team     │
    └─────────────────┘ └───────────────┘ └─────────────────┘
```

### Anonymization

The pipeline applies a sequence of transformations to reduce re-identification risk:

- **Quasi-identifier generalization**: Continuous values are banded (e.g., age 25 → "20-29"), rare categories are collapsed to "Other", and numeric values are rounded to coarser granularity.
- **Noise injection**: Multiplicative Gaussian noise preserves relative magnitudes while preventing exact-value matching. Laplacian noise provides differential-privacy-style protection.
- **Date perturbation**: HMAC-based deterministic offsets ensure consistency across re-runs while preventing calendar-based linkage.
- **k-Anonymity enforcement**: Iterative suppression ensures every combination of quasi-identifiers appears in at least k records.

### Synthetic Data Modelling & Generation

Rather than training a single model on all columns simultaneously, the pipeline splits columns into semantic phases. Each phase trains its own generative model on a column subset, conditioned on outputs from prior phases. This preserves complex inter-column dependencies while allowing different synthesizer strategies per group.

```
Phase 1: Demographics    → Train model → Generate
                                            ↓ (condition)
Phase 2: Financial       → Train model → Generate
                                            ↓ (condition)
Phase 3: Behavioural     → Train model → Generate
                                            ↓ (condition)
Phase 4: Sparse fields   → Train model → Generate
```

**Why multi-phase?**
- Captures inter-phase dependencies through conditioning without a single monolithic model
- Handles sparse columns (high null rates) in dedicated phases
- Allows different synthesizer strategies per phase : use fast models for simple distributions, deep learning for complex ones

**Supported synthesizer models:**

| Model | Approach | Best For |
|-------|----------|----------|
| **GaussianCopula** | Fits marginal distributions + Gaussian copula for joint dependencies | Fast training, continuous-heavy data |
| **CTGAN** | Conditional GAN with mode-specific normalisation | Mixed categorical/numeric, complex distributions |
| **TVAE** | Variational autoencoder with triplet loss | Complex multi-modal distributions |
| **Hybrid** | Per-phase strategy selection : defaults to copula, overridable per phase | Production use : match strategy to column characteristics |

**Column profiling** auto-detects each column's sdtype (numerical, categorical, datetime) and sparsity, with optional `column_sdtypes` overrides in config.

**Key synthetic configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | `copula` | Global synthesizer: copula, ctgan, tvae, hybrid |
| `epochs` | 300 | Training epochs (CTGAN/TVAE only) |
| `batch_size` | 500 | Mini-batch size (CTGAN/TVAE only) |
| `n_rows` | 0 | Target synthetic rows (0 = match input size) |
| `phases[].condition_on` | all prior | Which prior phases to condition on |
| `column_sdtypes` | auto | Override auto-detected column types |
| `sparse_threshold` | 0.5 | Null rate above which a column is marked sparse |

### Adversarial Validation

Ten automated attacks probe the anonymized/synthetic output for residual privacy risks: uniqueness profiling, temporal linkage, fingerprint detection, outlier re-identification, distribution skew, null-pattern linkage, rare-combination linkage, numeric precision leaks, ratio preservation, and compound-entity fingerprinting.

## Quick Start

Sample data is included so you can see the anonymization effect immediately:

```bash
# Compare before and after (included in the repo)
head data/sample.csv           # 500 rows, 16 columns : includes PII, dates, financials
head data/sample_anonymized.csv # 339 rows, 10 columns : PII dropped, values generalised

# Or regenerate from scratch
python scripts/create_sample_data.py
python scripts/anonymize_data.py \
    --input data/sample.csv \
    --output data/anonymized.csv \
    --config config/example_simple_anonymization.yaml
```

**What changes:**
| Before | After |
|--------|-------|
| `full_name`, `email`, `phone_number`, `national_id` | Dropped (direct identifiers) |
| `source_system_code`, `internal_batch_id` | Dropped (fingerprints) |
| `age: 51` | `age: 50 - 60` (banded) |
| `region: Northeast` | `region: Other` (top-N capping) |
| `income: 40617.24` | `income: 40021.53` (noise injected) |
| `created_date: 2024-03-12` | `created_date: 2024-03-09` (perturbed) |
| Rows where QI groups have <5 members | Suppressed (k-anonymity) |

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/data-anonymization-toolkit.git
cd data-anonymization-toolkit

# Create conda environment
conda env create -f environment.yml
conda activate data-anonymization-toolkit

# Or install with pip
pip install -e .                # core dependencies
pip install -e ".[synthetic]"   # + SDV for synthetic data modelling
pip install -e ".[reporting]"   # + ReportLab for PDF guides
pip install -e ".[all]"         # everything
```

### Verify Installation

```bash
python -c "from anonymization import AnonymizationPipeline; print('OK')"
python scripts/anonymize_data.py --help
```

## Usage

### CLI

```bash
# Anonymize a dataset
python scripts/anonymize_data.py \
    --input data/raw.csv \
    --output data/anonymized.csv \
    --config config/example_anonymization.yaml

# Generate synthetic data
python scripts/generate_synthetic.py \
    --config config/example_synthetic.yaml

# Validate quality and privacy
python scripts/validate.py \
    --real data/anonymized.csv \
    --synthetic data/synthetic.csv

# Run adversarial red team
python scripts/red_team.py \
    --anonymized data/anonymized.csv \
    --config config/example_red_team.yaml

# Generate privacy guide
python scripts/generate_guide.py --output reports/ --format pdf
```

### Python API

```python
import pandas as pd
from anonymization.config import load_config
from anonymization.anonymizer import AnonymizationPipeline

# Load config and data
config = load_config("config/example_simple_anonymization.yaml")
df = pd.read_csv("data/sample.csv")

# Run full pipeline
pipeline = AnonymizationPipeline(config)
anonymized = pipeline.run(df)

# Or run individual steps
pipeline = AnonymizationPipeline(config)
df = pipeline.drop_columns(df)
df = pipeline.round_floats(df, decimals=2)
# ... etc.
```

## Configuration

All behaviour is controlled through YAML configuration files. See `config/` for examples.

### Key Parameters

| Parameter | Default (ml) | Default (public) | Description |
|-----------|-------------|-------------------|-------------|
| `k_target` | 5 | 20 | Minimum k-anonymity group size |
| `numeric_noise_pct` | 0.02 | 0.05 | Multiplicative noise std dev |
| `date_max_days` | 7 | 30 | Max date perturbation (days) |
| `numeric_round_to` | 2 | 0 | Decimal places for float rounding |

### QI Generalization Methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| `band` | `width` | Fixed-width numeric bands (e.g., 10-year age bands) |
| `round_to` | `step` | Round to nearest step value |
| `top_n` | `n` | Keep N most frequent categories, replace rest with "Other" |
| `cap` | `lower`, `upper` | Clip values to bounds |

## Project Structure

```
data-anonymization-toolkit/
├── anonymization/          # Core anonymization pipeline
│   ├── anonymizer.py       # AnonymizationPipeline orchestrator
│   ├── config.py           # YAML config loader
│   ├── generalization.py   # QI generalization strategies
│   ├── noise.py            # Noise injection + date perturbation
│   ├── fingerprint.py      # Fingerprint scrubbing
│   └── k_anonymity.py      # k-anonymity enforcement
├── synthetic/              # Synthetic data modelling & generation
│   ├── config.py           # SyntheticConfig + YAML loader
│   ├── generator.py        # ColumnProfiler, DataPreparer
│   └── synthesizers.py     # PhaseTrainer (SDV wrapper)
├── validation/             # Quality & privacy validation
│   ├── quality.py          # Statistical quality checks
│   ├── privacy_metrics.py  # MIA, DCR, nearest-neighbour
│   └── red_team.py         # 10 adversarial attacks + runner
├── reporting/              # Report generation
│   ├── privacy_guide.py    # PDF/Markdown privacy guide
│   └── quality_report.py   # Quality/privacy/red-team reports
├── config/                 # Example YAML configurations
├── scripts/                # CLI entry points
├── templates/              # Document templates
└── examples/               # Usage examples
```

## API Reference

### `AnonymizationPipeline`

```python
from anonymization.anonymizer import AnonymizationPipeline

pipeline = AnonymizationPipeline(config)
result = pipeline.run(df)                    # Full 10-step pipeline
df = pipeline.drop_columns(df)               # Step 1
df = pipeline.round_floats(df, decimals=2)   # Step 2
df = pipeline.equalize_categorical(df)       # Step 5
df = pipeline.normalize_null_patterns(df)    # Step 8
df = pipeline.prepare_output(df)             # Step 10
```

### `PhaseTrainer`

```python
from synthetic.synthesizers import PhaseTrainer

trainer = PhaseTrainer(config)
synthetic_df = trainer.run_all_phases(df)     # Full multi-phase generation
model = trainer.train_phase(df, phase_cfg, metadata)  # Single phase
output = trainer.generate_phase(model, n_rows)
```

### `RedTeamRunner`

```python
from validation.red_team import RedTeamRunner, Attack_Uniqueness

runner = RedTeamRunner(config)
runner.register_attack(Attack_Uniqueness)
results = runner.run_all(df_anon, df_orig)
report = runner.generate_report(results)
```

### Quality & Privacy

```python
from validation.quality import run_quality_checks
from validation.privacy_metrics import membership_inference_auc

quality = run_quality_checks(real_df, synth_df, config)
mia_auc = membership_inference_auc(real_df, synth_df, numerical_cols)
```
