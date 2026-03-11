"""Configuration loading and dataclass definitions for the synthetic data pipeline.

Supports YAML-based configuration with per-phase strategy overrides and per-column
sdtype overrides.  A ``dev`` flag provides fast iteration with small samples and
reduced epoch counts.

Example YAML
------------
::

    strategy: hybrid
    n_rows: 10000
    seed: 42
    epochs: 300
    batch_size: 500
    dev: false
    validate: false
    save_model: false
    load_model: ""
    report_dir: reports
    sparse_threshold: 0.5
    phases:
      - name: demographics
        columns: [col_a, col_b, col_c]
        strategy: copula
        sparse: false
      - name: financials
        columns: [col_d, col_e]
        strategy: ctgan
        sparse: false
    column_sdtypes:
      col_a: numerical
      col_b: categorical
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

_VALID_STRATEGIES: frozenset[str] = frozenset({"copula", "ctgan", "tvae", "hybrid"})


@dataclass
class SyntheticConfig:
    """Holds all configuration values for a single synthetic-data generation run.

    Attributes:
        input_path: Path to the source CSV (used as training data).
        output_path: Destination path for the generated synthetic CSV.
        strategy: Global synthesizer strategy to use when a phase does not specify
            its own.  One of ``"copula"``, ``"ctgan"``, ``"tvae"``, or ``"hybrid"``.
        n_rows: Number of synthetic rows to generate.  ``0`` means match the row
            count of the (sampled) input data.
        sample_frac: Fraction of source rows to use during training.  ``1.0``
            means use all rows.
        epochs: Number of training epochs for deep-learning-based synthesizers
            (CTGAN, TVAE).  Ignored by GaussianCopula.
        batch_size: Mini-batch size for CTGAN / TVAE training.
        seed: Global random seed for reproducibility.
        dev: When ``True``, activate development mode: training uses a small
            sample (up to 1 000 rows) and a reduced epoch count (10) regardless
            of other settings.
        validate: When ``True``, run statistical quality checks after generation
            and emit a report.
        save_model: When ``True``, serialize each trained synthesizer to disk
            inside ``report_dir``.
        load_model: Path to a previously saved synthesizer directory.  When
            non-empty, training is skipped and the serialized model is loaded.
        report_dir: Directory in which to write quality reports and saved models.
        phases: Ordered list of phase configuration dicts.  Each dict must
            contain ``name`` and ``columns``; ``strategy`` and ``sparse`` are
            optional per-phase overrides.
        column_sdtypes: Mapping from column name to SDV sdtype string (e.g.
            ``"numerical"``, ``"categorical"``, ``"datetime"``).  Columns not
            listed here are auto-detected by :class:`~synthetic.generator.ColumnProfiler`.
        sparse_threshold: Null rate above which a column is classified as
            *sparse*.  Sparse columns receive special handling during data
            preparation (e.g. null-rate preservation).
    """

    input_path: str
    output_path: str
    strategy: str = "copula"
    n_rows: int = 0
    sample_frac: float = 1.0
    epochs: int = 300
    batch_size: int = 500
    seed: int = 42
    dev: bool = False
    validate: bool = False
    save_model: bool = False
    load_model: str = ""
    report_dir: str = "reports"
    phases: list[dict[str, Any]] = field(default_factory=list)
    column_sdtypes: dict[str, str] = field(default_factory=dict)
    sparse_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                f"Valid values: {sorted(_VALID_STRATEGIES)}"
            )
        if not 0.0 < self.sample_frac <= 1.0:
            raise ValueError(
                f"sample_frac must be in (0.0, 1.0]; got {self.sample_frac}"
            )
        if not 0.0 <= self.sparse_threshold <= 1.0:
            raise ValueError(
                f"sparse_threshold must be in [0.0, 1.0]; got {self.sparse_threshold}"
            )
        if self.n_rows < 0:
            raise ValueError(
                f"n_rows must be >= 0 (use 0 to match input size); got {self.n_rows}"
            )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_synthetic_config(yaml_path: str | Path) -> SyntheticConfig:
    """Load and validate a synthetic-data configuration from a YAML file.

    Expected top-level YAML structure::

        strategy: hybrid          # copula | ctgan | tvae | hybrid
        n_rows: 10000             # 0 = match source row count
        sample_frac: 1.0          # fraction of source rows to use for training
        epochs: 300
        batch_size: 500
        seed: 42
        dev: false
        validate: false
        save_model: false
        load_model: ""
        report_dir: reports
        sparse_threshold: 0.5

        phases:
          - name: group_a
            columns: [col_x, col_y, col_z]
            strategy: copula      # optional per-phase override
            sparse: false         # optional per-phase flag
          - name: group_b
            columns: [col_p, col_q]
            strategy: ctgan

        column_sdtypes:
          col_x: numerical
          col_y: categorical
          col_z: datetime

    The ``input_path`` and ``output_path`` keys are required.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A fully populated :class:`SyntheticConfig` instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If ``input_path`` or ``output_path`` are absent.
        ValueError: If any field value falls outside its valid range.
        yaml.YAMLError: If the file cannot be parsed.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    logger.info("Loading synthetic configuration from %s", yaml_path)

    with yaml_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    missing = [k for k in ("input_path", "output_path") if k not in raw]
    if missing:
        raise KeyError(
            f"Required key(s) missing from config: {missing}"
        )

    config = SyntheticConfig(
        input_path=str(raw["input_path"]),
        output_path=str(raw["output_path"]),
        strategy=str(raw.get("strategy", "copula")),
        n_rows=int(raw.get("n_rows", 0)),
        sample_frac=float(raw.get("sample_frac", 1.0)),
        epochs=int(raw.get("epochs", 300)),
        batch_size=int(raw.get("batch_size", 500)),
        seed=int(raw.get("seed", 42)),
        dev=bool(raw.get("dev", False)),
        validate=bool(raw.get("validate", False)),
        save_model=bool(raw.get("save_model", False)),
        load_model=str(raw.get("load_model", "")),
        report_dir=str(raw.get("report_dir", "reports")),
        phases=list(raw.get("phases", [])),
        column_sdtypes=dict(raw.get("column_sdtypes", {})),
        sparse_threshold=float(raw.get("sparse_threshold", 0.5)),
    )

    logger.info(
        "Synthetic config loaded : strategy=%s  n_rows=%d  phases=%d  seed=%d",
        config.strategy,
        config.n_rows,
        len(config.phases),
        config.seed,
    )
    return config
