<h1 align="center">Data Anonymization Toolkit</h1>

<p align="center">
  <em>Config-driven anonymisation, synthetic data generation, and adversarial privacy validation for tabular datasets.</em>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-blue?style=flat">
  <a href="https://mirai-analytics.com"><img alt="Built by MIRAI" src="https://img.shields.io/badge/Built%20by-MIRAI%20Analytics-0F172A?style=flat"></a>
</p>

---

## Why this exists

Most real-world analytics work is gated on a single, recurring blocker: **moving sensitive tabular data across organisational, regulatory, or vendor boundaries**. The default options are unsatisfying.

- *Hand-rolled anonymisation scripts* are brittle, hard to audit, and rarely tested against re-identification attacks.
- *Off-the-shelf synthetic data tools* generate plausible-looking rows but don't quantify the residual privacy risk.
- *Differential privacy libraries* are powerful but require expertise most analytics teams don't have on hand.

This toolkit closes that gap. It runs a **10-step pipeline** that anonymises real tabular data, generates statistically faithful synthetic data, and stress-tests the result against a **10-attack adversarial red team** - all from a single YAML config. The output is a privacy-validated dataset plus a reproducible audit report you can hand to a DPO, security reviewer, or regulator.

Originally built for [MIRAI Analytics](https://mirai-analytics.com) consulting engagements where clients needed to share data with vendors, partners, or research collaborators without triggering full GDPR-grade controls on every transfer.

---

## What it does

```
Raw data ──> [Anonymisation] ──> [Synthetic generation] ──> [Validation] ──> Release-ready dataset + audit report
                  │                       │                       │
                  ▼                       ▼                       ▼
            PII removal,            SDV-based modelling      Statistical fidelity
            quasi-identifier        (GaussianCopula,         + privacy red team
            generalisation,         CTGAN, TVAE)             (10 attacks)
            k-anonymity, noise
```

### Highlights

| Capability | What it gives you |
|---|---|
| **10-step anonymisation pipeline** | PII removal, quasi-identifier generalisation, calibrated noise injection, k-anonymity enforcement |
| **YAML-driven configuration** | No hardcoded column references. One config per dataset; reusable across engagements |
| **Synthetic data via SDV** | Multi-phase conditional generation with GaussianCopula, CTGAN, and TVAE backends |
| **Statistical fidelity validation** | Chi-square, Kolmogorov-Smirnov, correlation-structure tests against the source |
| **Privacy metrics** | Membership inference AUC, duplicate class rate, nearest-neighbour distance distributions |
| **Adversarial red team** | 10 automated attacks testing re-identification, attribute disclosure, and linkage risk |
| **Two release profiles** | `ml` (k=5, lighter, for internal model training) and `public` (k=20, stronger, for external release) |
| **Auto-generated reports** | PDF guides and markdown audit trails - ready for DPO / security review |

---

## Quick start

```bash
# Install
git clone https://github.com/timothy22000/data-anonymization-toolkit
cd data-anonymization-toolkit
conda env create -f environment.yml && conda activate data-anon

# Generate the bundled sample data
python scripts/create_sample_data.py

# 1. Anonymise
python scripts/anonymize_data.py \
  --input data/raw.csv \
  --output data/anonymized.csv \
  --config config/example_simple_anonymization.yaml \
  --profile ml --seed 42

# 2. Generate synthetic data
python scripts/generate_synthetic.py \
  --input data/raw.csv \
  --output data/synthetic.csv \
  --config config/example_synthetic.yaml

# 3. Validate fidelity + privacy
python scripts/validate.py \
  --real data/raw.csv \
  --released data/anonymized.csv

# 4. Adversarial red team
python scripts/red_team.py \
  --real data/raw.csv \
  --released data/anonymized.csv \
  --config config/example_red_team.yaml

# 5. Build the audit guide
python scripts/generate_guide.py --output reports/audit.pdf
```

### Minimal anonymisation config

```yaml
# config/example_simple_anonymization.yaml
profile: ml
seed: 42
k_target: 5

columns:
  drop:
    direct_ids:   [full_name, email, phone_number, national_id]
    fingerprints: [source_system_code, internal_batch_id]
  quasi_identifiers:
    - column: age
      method: band
      params: {width: 10}
    - column: region
      method: top_n
      params: {n: 6}
  numeric_targets:
    - column: income
      noise_type: multiplicative
      noise_pct: 0.05
      round_to: 2
  date_columns:
    - created_date
```

For larger releases, switch `profile: public` (default `k_target: 20`) and use `config/example_anonymization.yaml` as a starting template.

---

## Project structure

```
anonymization/   # PII removal, generalisation, noise, k-anonymity enforcement
synthetic/       # SDV-based synthetic data generation
validation/      # Statistical fidelity + privacy validation
reporting/       # PDF + markdown audit output
config/          # YAML configs (anonymisation, synthetic, red team)
scripts/         # CLI entry points (anonymize, generate_synthetic, validate, red_team, generate_guide)
templates/       # Report templates
examples/        # End-to-end sample runs
data/            # Sample datasets (synthetic - no real data)
docs/            # Methodology notes + design rationale
```

---

## Methods reference

### Anonymisation methods supported

| Method | When to use |
|---|---|
| Direct identifier removal | Mandatory for any release |
| Quasi-identifier generalisation (hierarchies) | Reduces re-identification risk while preserving analytic utility |
| k-anonymity enforcement | Default privacy baseline; tunable per profile |
| Calibrated noise injection | For numeric quasi-identifiers (configurable epsilon) |
| Suppression | For records that can't reach k without over-generalising |

### Synthetic data backends

| Backend | Best for |
|---|---|
| **GaussianCopula** | Fast, interpretable, captures marginal + correlation structure |
| **CTGAN** | Better for mixed continuous/categorical with complex joint distributions |
| **TVAE** | Strong for high-cardinality categoricals; smoother latent space |

### Red team attacks

The validation suite runs 10 adversarial attacks against the released dataset, including membership inference, attribute disclosure, linkage attacks against plausible auxiliary datasets, and nearest-neighbour-based identification. Each attack outputs an AUC or success-rate metric that's included in the audit report.

---

## Status

Active, used in production MIRAI Analytics engagements. Public release is intentionally minimal so client-specific extensions stay private; the core pipeline, validation suite, and red team are open source under MIT.

If you're using this in your own work, an issue or PR is very welcome.

---

## Citation

If you use this toolkit in research, please cite:

```bibtex
@software{sumhonmun_data_anon_toolkit,
  author = {Sum Hon Mun, Timothy},
  title  = {Data Anonymization Toolkit: config-driven anonymisation, synthetic data generation, and adversarial privacy validation for tabular datasets},
  year   = {2025},
  url    = {https://github.com/timothy22000/data-anonymization-toolkit}
}
```

---

## About

Built and maintained by **[Timothy Sum Hon Mun](https://www.linkedin.com/in/timothysumhonmun)** at **[MIRAI Analytics](https://mirai-analytics.com)**.

PhD in medical imaging AI (Institute of Cancer Research). 4+ years commercial ML at Ageas Insurance. 11 peer-reviewed publications across MICCAI, AAAI, ISMRM, and MIDL.

- 🤗 [HuggingFace](https://huggingface.co/t22000t)
- 💼 [LinkedIn](https://www.linkedin.com/in/timothysumhonmun)
- 🌐 [MIRAI Analytics](https://mirai-analytics.com)

---

## License

MIT - see [LICENSE](LICENSE).
