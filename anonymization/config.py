"""Configuration loading and dataclass definitions for the anonymization pipeline.

Supports YAML-based configuration with two built-in profiles:
  - "ml"     : lighter anonymization suitable for internal ML workloads (k=5)
  - "public" : stronger anonymization for public data releases (k=20)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profile defaults
# ---------------------------------------------------------------------------

_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "ml": {
        "seed": 42,
        "k_target": 5,
        "date_max_days": 7,
        "numeric_noise_pct": 0.02,
        "numeric_round_to": 2,
        "skip_categorical_eq": False,
    },
    "public": {
        "seed": 42,
        "k_target": 20,
        "date_max_days": 30,
        "numeric_noise_pct": 0.05,
        "numeric_round_to": 0,
        "skip_categorical_eq": True,
    },
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnonymizationConfig:
    """Holds all configuration values for a single anonymization run.

    Attributes:
        profile: Named preset ("ml" or "public") used to supply defaults.
        seed: Global random seed for reproducibility.
        k_target: Minimum group size required for k-anonymity.
        date_max_days: Maximum absolute perturbation in days for date columns.
        numeric_noise_pct: Fractional noise level for multiplicative noise
            (e.g. 0.02 = ±2 %).
        numeric_round_to: Number of decimal places for the final rounding step
            applied to all floats in the output.
        skip_categorical_eq: When True, the equalize_categorical step is
            skipped even if a target column is present in the config.
        drop_direct_ids: Column names to drop as direct identifiers.
        drop_fingerprints: Column names that act as fingerprints and must be
            dropped.
        drop_operational: Operational / technical columns to drop.
        quasi_identifiers: QI generalisation configs : list of dicts with
            keys ``column``, ``method``, and ``params``.
        numeric_targets: Noise injection configs : list of dicts with keys
            ``column``, ``noise_type``, ``noise_pct``, and ``round_to``.
        date_columns: Names of date-like columns subject to perturbation.
        rename_map: Optional mapping ``{old_name: new_name}`` applied to the
            output DataFrame.
        fingerprint_patterns: List of dicts with keys ``pattern``,
            ``replacement``, and optionally ``case_insensitive``.
        sparse_groups: Column names whose rare values are suppressed before
            k-anonymity enforcement.
        qi_groups: List of column-name lists; each inner list is an
            independent QI group checked for k-anonymity.
        equalize_column: Column name whose distribution should be equalised.
            Empty string means the step is skipped.
        null_token: String token used to represent missing values in the
            final output.
    """

    profile: str = "ml"
    seed: int = 42
    k_target: int = 5
    date_max_days: int = 7
    numeric_noise_pct: float = 0.02
    numeric_round_to: int = 2
    skip_categorical_eq: bool = False

    # Column lists
    drop_direct_ids: list[str] = field(default_factory=list)
    drop_fingerprints: list[str] = field(default_factory=list)
    drop_operational: list[str] = field(default_factory=list)

    # Structured configs
    quasi_identifiers: list[dict[str, Any]] = field(default_factory=list)
    numeric_targets: list[dict[str, Any]] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    rename_map: dict[str, str] = field(default_factory=dict)
    fingerprint_patterns: list[dict[str, Any]] = field(default_factory=list)
    sparse_groups: list[str] = field(default_factory=list)
    qi_groups: list[list[str]] = field(default_factory=list)

    # Optional steps
    equalize_column: str = ""
    null_token: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(yaml_path: str | Path) -> AnonymizationConfig:
    """Load and validate an anonymization configuration from a YAML file.

    The YAML file may specify a ``profile`` key to inherit defaults.  Any
    explicit keys present in the YAML will override the profile defaults.

    Expected top-level YAML structure::

        profile: ml          # or "public"
        seed: 42
        k_target: 5
        date_max_days: 7
        numeric_noise_pct: 0.02
        numeric_round_to: 2
        skip_categorical_eq: false
        equalize_column: ""
        null_token: UNKNOWN

        columns:
          drop:
            direct_ids: [col_a, col_b]
            fingerprints: [col_c]
            operational: [col_d, col_e]
          quasi_identifiers:
            - column: age
              method: band
              params: {width: 5}
            - column: region
              method: top_n
              params: {n: 10}
          numeric_targets:
            - column: amount
              noise_type: multiplicative
              noise_pct: 0.02
              round_to: 2
          date_columns: [created_at, updated_at]
          rename_map:
            internal_id: record_id

        fingerprint_patterns:
          - pattern: "\\b\\d{9}\\b"
            replacement: "[REDACTED]"
            case_insensitive: false

        sparse_groups: [postal_code]

        qi_groups:
          - [age_band, region, gender]
          - [postal_code, age_band]

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A fully populated :class:`AnonymizationConfig` instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the profile name is not recognised.
        yaml.YAMLError: If the file cannot be parsed.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    logger.info("Loading configuration from %s", yaml_path)

    with yaml_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    profile = raw.get("profile", "ml")
    if profile not in _PROFILE_DEFAULTS:
        raise ValueError(
            f"Unknown profile '{profile}'. "
            f"Valid profiles: {list(_PROFILE_DEFAULTS)}"
        )

    # Start from profile defaults, then overlay YAML values
    defaults = _PROFILE_DEFAULTS[profile].copy()
    scalar_keys = {
        "seed",
        "k_target",
        "date_max_days",
        "numeric_noise_pct",
        "numeric_round_to",
        "skip_categorical_eq",
        "equalize_column",
        "null_token",
    }
    scalars: dict[str, Any] = {k: raw.get(k, defaults.get(k)) for k in scalar_keys}

    # Parse nested columns section
    columns: dict[str, Any] = raw.get("columns", {})
    drop_section: dict[str, Any] = columns.get("drop", {})

    config = AnonymizationConfig(
        profile=profile,
        seed=scalars["seed"],
        k_target=scalars["k_target"],
        date_max_days=scalars["date_max_days"],
        numeric_noise_pct=float(scalars["numeric_noise_pct"]),
        numeric_round_to=int(scalars["numeric_round_to"]),
        skip_categorical_eq=bool(scalars["skip_categorical_eq"]),
        equalize_column=str(scalars.get("equalize_column") or ""),
        null_token=str(scalars.get("null_token") or "UNKNOWN"),
        drop_direct_ids=_to_str_list(drop_section.get("direct_ids", [])),
        drop_fingerprints=_to_str_list(drop_section.get("fingerprints", [])),
        drop_operational=_to_str_list(drop_section.get("operational", [])),
        quasi_identifiers=list(columns.get("quasi_identifiers", [])),
        numeric_targets=list(columns.get("numeric_targets", [])),
        date_columns=_to_str_list(columns.get("date_columns", [])),
        rename_map=dict(columns.get("rename_map") or {}),
        fingerprint_patterns=list(raw.get("fingerprint_patterns", [])),
        sparse_groups=_to_str_list(raw.get("sparse_groups", [])),
        qi_groups=[list(g) for g in raw.get("qi_groups", [])],
    )

    logger.info(
        "Config loaded : profile=%s  k_target=%d  seed=%d",
        config.profile,
        config.k_target,
        config.seed,
    )
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_str_list(value: Any) -> list[str]:
    """Coerce a YAML value to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]
