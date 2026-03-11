"""Noise injection functions for numeric and date columns.

Provides multiplicative Gaussian noise, additive Laplacian noise, and
HMAC-based deterministic date perturbation.  All functions operate on
pandas Series or DataFrames and return new objects : inputs are never
mutated.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numeric noise
# ---------------------------------------------------------------------------


def add_multiplicative_noise(
    series: pd.Series,
    pct: float,
    seed: int,
) -> pd.Series:
    """Multiply each value by a factor drawn from N(1, pct).

    Each observation *x* is transformed to ``x * (1 + N(0, pct))``, which
    preserves the sign and rough magnitude of the original while adding
    relative noise proportional to *pct*.

    Args:
        series: Numeric series to perturb.
        pct: Standard deviation of the multiplicative noise factor.
            E.g. ``0.05`` means approximately ±5 % noise (1 std dev).
        seed: Random seed for reproducibility.

    Returns:
        New numeric series with multiplicative noise applied.

    Raises:
        ValueError: If *pct* is negative.
    """
    if pct < 0:
        raise ValueError(f"pct must be >= 0, got {pct!r}")

    rng = np.random.default_rng(seed)
    numeric = pd.to_numeric(series, errors="coerce")
    factors = 1.0 + rng.normal(0.0, pct, size=len(numeric))
    result = numeric * factors
    logger.debug(
        "add_multiplicative_noise: column='%s'  pct=%s  seed=%d",
        series.name,
        pct,
        seed,
    )
    return result


def add_laplacian_noise(
    series: pd.Series,
    scale: float,
    seed: int,
) -> pd.Series:
    """Add zero-mean Laplacian noise with the given *scale* parameter.

    The Laplace distribution has heavier tails than the Gaussian, making
    it a common choice for differential-privacy-inspired perturbation.

    Args:
        series: Numeric series to perturb.
        scale: Scale (b) parameter of the Laplace distribution.  Larger
            values produce more noise.  Must be positive.
        seed: Random seed for reproducibility.

    Returns:
        New numeric series with additive Laplacian noise.

    Raises:
        ValueError: If *scale* is not positive.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale!r}")

    rng = np.random.default_rng(seed)
    numeric = pd.to_numeric(series, errors="coerce")
    noise = rng.laplace(loc=0.0, scale=scale, size=len(numeric))
    result = numeric + noise
    logger.debug(
        "add_laplacian_noise: column='%s'  scale=%s  seed=%d",
        series.name,
        scale,
        seed,
    )
    return result


def add_numeric_noise(
    df: pd.DataFrame,
    targets_config: list[dict[str, Any]],
    seed: int,
) -> pd.DataFrame:
    """Apply config-driven noise to multiple numeric columns.

    Each entry in *targets_config* specifies a column and the noise
    parameters to apply.  Different columns receive independent noise
    streams derived from the global *seed*.

    Config dict schema::

        column:     str    # column name in df
        noise_type: str    # "multiplicative" or "laplacian"
        noise_pct:  float  # for multiplicative : std dev fraction
        scale:      float  # for laplacian : scale parameter (optional)
        round_to:   int    # decimal places for final rounding (optional)

    When ``noise_type`` is ``"multiplicative"``, the ``noise_pct`` field is
    used as the standard deviation.  When ``noise_type`` is ``"laplacian"``,
    ``scale`` is used (falls back to ``noise_pct`` if ``scale`` is absent).

    Args:
        df: Source DataFrame.
        targets_config: List of per-column noise configuration dicts.
        seed: Base random seed; each column's seed is derived from this.

    Returns:
        New DataFrame with noisy numeric columns.
    """
    df = df.copy()

    for idx, cfg in enumerate(targets_config):
        col: str = cfg["column"]
        noise_type: str = cfg.get("noise_type", "multiplicative")
        noise_pct: float = float(cfg.get("noise_pct", 0.0))
        round_to: int | None = cfg.get("round_to")
        col_seed = seed + idx  # deterministic per-column seed

        if col not in df.columns:
            logger.warning(
                "add_numeric_noise: column '%s' not found : skipping", col
            )
            continue

        logger.info(
            "Injecting noise into '%s'  type=%s  seed=%d",
            col,
            noise_type,
            col_seed,
        )

        if noise_type == "multiplicative":
            df[col] = add_multiplicative_noise(df[col], pct=noise_pct, seed=col_seed)
        elif noise_type == "laplacian":
            scale: float = float(cfg.get("scale", noise_pct))
            df[col] = add_laplacian_noise(df[col], scale=scale, seed=col_seed)
        else:
            raise ValueError(
                f"Unknown noise_type '{noise_type}' for column '{col}'. "
                "Valid types: multiplicative, laplacian"
            )

        if round_to is not None:
            df[col] = df[col].round(int(round_to))

    return df


# ---------------------------------------------------------------------------
# Date perturbation
# ---------------------------------------------------------------------------


def perturb_dates(
    df: pd.DataFrame,
    date_columns: list[str],
    key_columns: list[str],
    max_days: int,
    seed: int,
) -> pd.DataFrame:
    """Perturb date columns by a consistent, HMAC-derived offset.

    For each row, a deterministic integer offset (in days) in the range
    ``[-max_days, +max_days]`` is computed from the *key_columns* values
    using HMAC-SHA256 with the *seed* as the key.  The same composite key
    always maps to the same offset, so re-running the pipeline on the same
    data produces identical perturbations.

    Args:
        df: Source DataFrame.
        date_columns: Names of columns containing date-like values.
        key_columns: Columns whose combined value forms the HMAC key
            (e.g. a row identifier or composite natural key).  If empty,
            the row index is used.
        max_days: Maximum absolute perturbation in calendar days.
        seed: Integer used as the HMAC secret key (converted to bytes).

    Returns:
        New DataFrame with perturbed date columns.  Non-date values in the
        specified columns are coerced to NaT.

    Raises:
        ValueError: If *max_days* is negative.
    """
    if max_days < 0:
        raise ValueError(f"max_days must be >= 0, got {max_days!r}")

    df = df.copy()
    secret = str(seed).encode()

    def _hmac_offset(row: pd.Series) -> int:
        """Derive a deterministic day offset from *row* using HMAC."""
        if key_columns:
            composite = "|".join(str(row[c]) for c in key_columns)
        else:
            composite = str(row.name)
        digest = hmac.new(secret, composite.encode(), hashlib.sha256).digest()
        raw = int.from_bytes(digest[:4], "big")
        span = 2 * max_days + 1
        return (raw % span) - max_days

    for col in date_columns:
        if col not in df.columns:
            logger.warning("perturb_dates: column '%s' not found : skipping", col)
            continue

        parsed = pd.to_datetime(df[col], errors="coerce")
        offsets = df.apply(_hmac_offset, axis=1)
        deltas = pd.to_timedelta(offsets, unit="D")
        df[col] = (parsed + deltas).dt.strftime("%Y-%m-%d")
        logger.info(
            "Perturbed dates in '%s'  max_days=%d", col, max_days
        )

    return df
