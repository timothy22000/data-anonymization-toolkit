"""Fingerprint scrubbing: scan string columns for sensitive patterns.

A "fingerprint" is any value or substring that could uniquely identify an
individual or entity even after direct identifiers have been dropped :
examples include account numbers, national ID patterns, email addresses,
or phone numbers.

This module scans all object/string columns in a DataFrame and replaces
any substring matching a configured regex with a replacement token.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def scrub_fingerprints(
    df: pd.DataFrame,
    patterns: list[dict[str, Any]],
) -> pd.DataFrame:
    """Scan all string columns and replace fingerprint pattern matches.

    Each pattern in *patterns* is applied to every ``object`` (string)
    column in *df*.  Matches are replaced with ``replacement``.

    Pattern config dict schema::

        pattern:          str   # regular expression to match
        replacement:      str   # replacement string (may include backrefs)
        case_insensitive: bool  # optional, default False

    Args:
        df: Source DataFrame.
        patterns: List of pattern config dicts.

    Returns:
        New DataFrame with fingerprints scrubbed from all string columns.
        Non-string columns are left untouched.

    Raises:
        re.error: If a configured regex pattern is invalid.
    """
    if not patterns:
        logger.debug("scrub_fingerprints: no patterns configured : skipping")
        return df.copy()

    df = df.copy()

    # Compile patterns once to avoid repeated compilation per cell
    compiled: list[tuple[re.Pattern[str], str]] = []
    for cfg in patterns:
        raw_pattern: str = cfg["pattern"]
        replacement: str = cfg.get("replacement", "[REDACTED]")
        flags = re.IGNORECASE if cfg.get("case_insensitive", False) else 0
        try:
            rx = re.compile(raw_pattern, flags)
        except re.error as exc:
            raise re.error(
                f"Invalid fingerprint pattern {raw_pattern!r}: {exc}"
            ) from exc
        compiled.append((rx, replacement))
        logger.debug(
            "Compiled fingerprint pattern: %r  replacement=%r  flags=%d",
            raw_pattern,
            replacement,
            flags,
        )

    # Identify string columns
    string_cols = [col for col in df.columns if df[col].dtype == object]
    if not string_cols:
        logger.info("scrub_fingerprints: no string columns found : nothing to scrub")
        return df

    logger.info(
        "Scrubbing fingerprints across %d string column(s) with %d pattern(s)",
        len(string_cols),
        len(compiled),
    )

    for col in string_cols:
        series = df[col].astype(str)
        for rx, replacement in compiled:
            series = series.str.replace(rx, replacement, regex=True)
        # Restore original NaN positions
        df[col] = series.where(df[col].notna(), other=pd.NA)

    return df
