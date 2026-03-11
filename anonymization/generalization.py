"""Quasi-identifier (QI) generalisation strategies.

Each public function accepts a pandas Series and returns a transformed Series.
The config-driven entry point :func:`generalize_quasi_identifiers` dispatches
to the correct strategy based on the ``method`` key in each QI config dict.

Supported methods
-----------------
``band``
    Bin a continuous column into fixed-width numeric bands.
``round_to``
    Round to the nearest ``step`` (works for both ints and floats).
``top_n``
    Keep the *N* most frequent categories; replace the rest with a label.
``cap``
    Clip values to ``[lower, upper]`` bounds.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual strategies
# ---------------------------------------------------------------------------


def band(series: pd.Series, width: int | float) -> pd.Series:
    """Bin continuous values into fixed-width numeric bands.

    Each value *v* is mapped to the string ``"[lo, hi)"`` where
    ``lo = floor(v / width) * width`` and ``hi = lo + width``.

    Args:
        series: Numeric series to generalise.
        width: Band width.  Must be positive.

    Returns:
        Object series of band labels, preserving the original index.

    Raises:
        ValueError: If *width* is not positive.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([1, 14, 27, 35])
        >>> band(s, 10).tolist()
        ['0 - 10', '10 - 20', '20 - 30', '30 - 40']
    """
    if width <= 0:
        raise ValueError(f"band width must be positive, got {width!r}")

    numeric = pd.to_numeric(series, errors="coerce")
    lo = (numeric // width) * width
    hi = lo + width
    result = lo.astype(str) + " - " + hi.astype(str)
    # Preserve NaN positions
    result[numeric.isna()] = pd.NA
    logger.debug("band: column='%s'  width=%s", series.name, width)
    return result


def round_to(series: pd.Series, step: int | float) -> pd.Series:
    """Round each value to the nearest *step*.

    Args:
        series: Numeric series to round.
        step: Rounding granularity.  Must be positive.

    Returns:
        Numeric series rounded to the nearest ``step``, preserving dtype
        where possible.

    Raises:
        ValueError: If *step* is not positive.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([1, 4, 7, 12])
        >>> round_to(s, 5).tolist()
        [0, 5, 5, 10]
    """
    if step <= 0:
        raise ValueError(f"round_to step must be positive, got {step!r}")

    numeric = pd.to_numeric(series, errors="coerce")
    result = (numeric / step).round() * step
    logger.debug("round_to: column='%s'  step=%s", series.name, step)
    return result


def top_n(
    series: pd.Series,
    n: int,
    other_label: str = "Other",
) -> pd.Series:
    """Keep the *N* most frequent categories; replace the rest.

    Args:
        series: Categorical or string series.
        n: Number of top categories to retain.
        other_label: Replacement label for less-frequent values.

    Returns:
        Series with rare values replaced by *other_label*.

    Raises:
        ValueError: If *n* is less than 1.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series(["a", "a", "b", "b", "c", "d"])
        >>> top_n(s, 2).tolist()
        ['a', 'a', 'b', 'b', 'Other', 'Other']
    """
    if n < 1:
        raise ValueError(f"top_n requires n >= 1, got {n!r}")

    top_values = series.value_counts().nlargest(n).index
    result = series.where(series.isin(top_values), other=other_label)
    logger.debug(
        "top_n: column='%s'  n=%d  retained=%s",
        series.name,
        n,
        list(top_values),
    )
    return result


def cap(series: pd.Series, lower: int | float, upper: int | float) -> pd.Series:
    """Clip values to the closed interval ``[lower, upper]``.

    Args:
        series: Numeric series to cap.
        lower: Minimum allowed value (inclusive).
        upper: Maximum allowed value (inclusive).

    Returns:
        Series with values clipped to ``[lower, upper]``.

    Raises:
        ValueError: If *lower* is greater than *upper*.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([-5, 0, 50, 200])
        >>> cap(s, 0, 100).tolist()
        [0, 0, 50, 100]
    """
    if lower > upper:
        raise ValueError(
            f"cap requires lower <= upper, got lower={lower!r} upper={upper!r}"
        )

    result = pd.to_numeric(series, errors="coerce").clip(lower=lower, upper=upper)
    logger.debug("cap: column='%s'  lower=%s  upper=%s", series.name, lower, upper)
    return result


# ---------------------------------------------------------------------------
# Dispatch mapping
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[str, Any] = {
    "band": band,
    "round_to": round_to,
    "top_n": top_n,
    "cap": cap,
}


# ---------------------------------------------------------------------------
# Config-driven entry point
# ---------------------------------------------------------------------------


def generalize_quasi_identifiers(
    df: pd.DataFrame,
    qi_configs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Apply config-driven QI generalisation to a DataFrame.

    For each entry in *qi_configs* the function looks up the column in *df*,
    selects the strategy by ``method`` name, and replaces the column in-place
    (on a copy of the DataFrame).

    Config dict schema::

        column: str          # column name in df
        method: str          # one of: band, round_to, top_n, cap
        params: dict         # keyword arguments forwarded to the strategy

    Args:
        df: Source DataFrame.
        qi_configs: List of QI config dicts.

    Returns:
        New DataFrame with generalised QI columns.

    Raises:
        KeyError: If a configured column is not present in *df*.
        ValueError: If an unknown ``method`` is specified.
    """
    df = df.copy()

    for cfg in qi_configs:
        col: str = cfg["column"]
        method: str = cfg["method"]
        params: dict[str, Any] = cfg.get("params") or {}

        if col not in df.columns:
            logger.warning(
                "generalize_quasi_identifiers: column '%s' not found : skipping", col
            )
            continue

        strategy = _STRATEGY_MAP.get(method)
        if strategy is None:
            raise ValueError(
                f"Unknown generalisation method '{method}'. "
                f"Valid methods: {list(_STRATEGY_MAP)}"
            )

        logger.info(
            "Generalising QI column '%s' with method='%s' params=%s",
            col,
            method,
            params,
        )
        df[col] = strategy(df[col], **params)

    return df
