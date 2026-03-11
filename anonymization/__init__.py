"""Anonymization toolkit : config-driven data anonymization pipeline.

This package provides a composable, YAML-driven pipeline for transforming
sensitive datasets into anonymized outputs suitable for research, testing,
or public release.  All behaviour is controlled through configuration; no
column names, thresholds, or business rules are hard-coded in Python.

Quick start
-----------
::

    from anonymization import AnonymizationPipeline, AnonymizationConfig, load_config
    import pandas as pd

    config = load_config("config/default.yaml")
    pipeline = AnonymizationPipeline(config)
    anonymized_df = pipeline.run(pd.read_csv("data/raw.csv"))
    anonymized_df.to_csv("data/anonymized.csv", index=False)

Package layout
--------------
``anonymizer``
    :class:`AnonymizationPipeline` : orchestrates the full 10-step pipeline.
``config``
    :class:`AnonymizationConfig` dataclass and :func:`load_config` YAML loader.
``generalization``
    QI generalisation strategies: :func:`generalize_quasi_identifiers`.
``noise``
    Noise injection: :func:`add_numeric_noise`, :func:`perturb_dates`.
``fingerprint``
    Pattern-based scrubbing: :func:`scrub_fingerprints`.
``k_anonymity``
    k-anonymity enforcement: :func:`enforce_k_anonymity`.
"""

from .anonymizer import AnonymizationPipeline
from .config import AnonymizationConfig, load_config
from .fingerprint import scrub_fingerprints
from .generalization import generalize_quasi_identifiers
from .k_anonymity import enforce_k_anonymity
from .noise import add_numeric_noise, perturb_dates

__all__ = [
    # Pipeline
    "AnonymizationPipeline",
    # Config
    "AnonymizationConfig",
    "load_config",
    # Generalization
    "generalize_quasi_identifiers",
    # Noise
    "add_numeric_noise",
    "perturb_dates",
    # Fingerprints
    "scrub_fingerprints",
    # k-anonymity
    "enforce_k_anonymity",
]
