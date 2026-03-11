"""Synthetic data generation toolkit : config-driven multi-phase synthesis pipeline.

This package provides a composable, YAML-driven pipeline for generating high-fidelity
synthetic datasets from real source data using SDV-based synthesizers.  All behaviour
is controlled through configuration; no column names, thresholds, or domain-specific
rules are hard-coded in Python.

Quick start
-----------
::

    from synthetic import SyntheticConfig, load_synthetic_config
    from synthetic import ColumnProfiler, DataPreparer, PhaseTrainer
    import pandas as pd

    config = load_synthetic_config("config/synthetic.yaml")
    trainer = PhaseTrainer(config)
    source_df = pd.read_csv(config.input_path)
    synthetic_df = trainer.run_all_phases(source_df)
    synthetic_df.to_csv(config.output_path, index=False)

Package layout
--------------
``config``
    :class:`SyntheticConfig` dataclass and :func:`load_synthetic_config` YAML loader.
``generator``
    :class:`ColumnProfiler` : column introspection and sdtype detection.
    :class:`DataPreparer` : phase-scoped data selection, null handling, and
    conditioning-column merging.
``synthesizers``
    :class:`PhaseTrainer` : SDV synthesizer training, row generation, and
    multi-phase orchestration.
"""

from .config import SyntheticConfig, load_synthetic_config
from .generator import ColumnProfiler, DataPreparer
from .synthesizers import PhaseTrainer

__all__ = [
    # Config
    "SyntheticConfig",
    "load_synthetic_config",
    # Generator
    "ColumnProfiler",
    "DataPreparer",
    # Synthesizers
    "PhaseTrainer",
]
