"""k-anonymity checking and enforcement.

k-anonymity guarantees that every combination of quasi-identifier (QI)
values appears in at least *k* rows, preventing re-identification by
matching against external datasets.

This module provides three public functions:

- :func:`suppress_rare_values` : replace individual values that appear
  fewer than *min_count* times with a replacement token.
- :func:`check_k_anonymity` : report whether a QI group satisfies
  k-anonymity and return a rich statistics dict.
- :func:`enforce_k_anonymity` : multi-stage iterative enforcement that
  brings the dataset into compliance across all configured QI groups.

Enforcement strategy
--------------------
For each QI group the enforcement applies the following stages:

1. **Rare-value suppression** : replace per-column values whose frequency
   is below *k_target* with the replacement token (see
   :func:`suppress_rare_values`).
2. **k-anonymity check** : if the group already satisfies *k_target* after
   stage 1, no further action is taken for that group.
3. **Iterative row suppression** : rows whose QI combination still appears
   fewer than *k_target* times are removed.  The loop repeats until the
   constraint is met or *max_iterations* is reached.

The suppression rate (fraction of rows removed) is logged as a warning
when it exceeds 5 %.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_REPLACEMENT: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def suppress_rare_values(
    df: pd.DataFrame,
    columns: list[str],
    min_count: int,
    replacement: str = _DEFAULT_REPLACEMENT,
) -> pd.DataFrame:
    """Replace values appearing fewer than *min_count* times with *replacement*.

    Only ``object`` (string) columns are modified; numeric columns are left
    untouched.  Columns listed in *columns* that are not present in *df* are
    silently skipped.

    Args:
        df: Source DataFrame.
        columns: Column names to check for rare values.
        min_count: Frequency threshold.  Values with a count strictly less
            than *min_count* are replaced.
        replacement: Token substituted for rare values.  Defaults to
            ``"UNKNOWN"``.

    Returns:
        New DataFrame with rare values replaced.

    Raises:
        ValueError: If *min_count* is less than 1.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"region": ["A", "A", "A", "B", "C"]})
        >>> suppress_rare_values(df, ["region"], min_count=2)["region"].tolist()
        ['A', 'A', 'A', 'UNKNOWN', 'UNKNOWN']
    """
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count!r}")

    df = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.debug("suppress_rare_values: column '%s' not found : skipping", col)
            continue

        if df[col].dtype != object:
            logger.debug(
                "suppress_rare_values: column '%s' is not string dtype : skipping", col
            )
            continue

        counts = df[col].value_counts(dropna=False)
        rare = counts[counts < min_count].index.tolist()

        if rare:
            logger.info(
                "suppress_rare_values: replacing %d rare value(s) in '%s' with %r",
                len(rare),
                col,
                replacement,
            )
            df[col] = df[col].where(~df[col].isin(rare), other=replacement)
        else:
            logger.debug(
                "suppress_rare_values: no rare values found in '%s'", col
            )

    return df


def check_k_anonymity(
    df: pd.DataFrame,
    qi_columns: list[str],
    k: int,
) -> dict[str, Any]:
    """Report k-anonymity statistics for a set of quasi-identifier columns.

    Groups the DataFrame by *qi_columns* and measures the size of each
    equivalence class.  Returns a statistics dictionary rather than a simple
    boolean so callers can make informed decisions about remediation.

    Args:
        df: DataFrame to check.  Must contain all columns in *qi_columns*.
        qi_columns: Column names that together form the quasi-identifier
            group.
        k: Minimum required equivalence-class size.

    Returns:
        Dictionary with the following keys:

        ``min_k`` (int)
            Size of the smallest equivalence class.
        ``median_k`` (float)
            Median equivalence-class size.
        ``n_violations`` (int)
            Number of equivalence classes smaller than *k*.
        ``pct_violations`` (float)
            Fraction of equivalence classes that violate the constraint,
            as a value in [0, 1].
        ``passed`` (bool)
            ``True`` iff every equivalence class has size >= *k*.

    Raises:
        ValueError: If *k* < 1 or any column in *qi_columns* is missing
            from *df*.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"age_band": ["20-30"] * 5 + ["30-40"] * 3})
        >>> result = check_k_anonymity(df, ["age_band"], k=4)
        >>> result["passed"]
        False
        >>> result["n_violations"]
        1
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")

    missing = [c for c in qi_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"check_k_anonymity: columns not found in DataFrame: {missing}"
        )

    if df.empty or not qi_columns:
        return {
            "min_k": 0,
            "median_k": 0.0,
            "n_violations": 0,
            "pct_violations": 0.0,
            "passed": True,
        }

    group_sizes = df.groupby(qi_columns, dropna=False).size()
    n_groups = len(group_sizes)
    n_violations = int((group_sizes < k).sum())

    result: dict[str, Any] = {
        "min_k": int(group_sizes.min()),
        "median_k": float(group_sizes.median()),
        "n_violations": n_violations,
        "pct_violations": n_violations / n_groups if n_groups > 0 else 0.0,
        "passed": n_violations == 0,
    }

    logger.debug(
        "check_k_anonymity: qi=%s  k=%d  min_k=%d  n_violations=%d  passed=%s",
        qi_columns,
        k,
        result["min_k"],
        n_violations,
        result["passed"],
    )
    return result


def enforce_k_anonymity(
    df: pd.DataFrame,
    qi_groups: list[list[str]],
    k_target: int,
    max_iterations: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Enforce k-anonymity across multiple QI groups via staged suppression.

    The function processes each QI group independently through the following
    stages:

    1. Suppress rare individual column values (frequency < *k_target*)
       by replacing them with the ``"UNKNOWN"`` token.
    2. Check whether the group already satisfies *k_target* after stage 1.
       If so, advance to the next group without row suppression.
    3. Iteratively remove rows whose QI combination still appears fewer
       than *k_target* times, repeating until the constraint is satisfied
       or *max_iterations* is reached.

    Args:
        df: Source DataFrame.
        qi_groups: List of QI groups; each inner list is a set of column
            names checked together as an equivalence-class definition.
        k_target: Required minimum equivalence-class size.  Must be >= 1.
        max_iterations: Maximum number of row-suppression passes per group
            before giving up.  Defaults to 50.

    Returns:
        A two-element tuple ``(result_df, report)`` where:

        ``result_df`` (pd.DataFrame)
            Anonymized DataFrame satisfying k-anonymity for all groups
            (to the extent possible within *max_iterations*).

        ``report`` (dict)
            Top-level keys:

            ``rows_in`` (int)
                Row count before enforcement.
            ``rows_out`` (int)
                Row count after enforcement.
            ``suppression_rate`` (float)
                Fraction of rows removed, in [0, 1].
            ``groups`` (dict)
                Per-group statistics.  Keys are stringified versions of each
                QI group list.  Values are dicts with:

                ``iterations`` (int)
                    Number of row-suppression iterations performed.
                ``converged`` (bool)
                    ``True`` if k-anonymity was achieved within the iteration
                    limit.
                ``rows_suppressed`` (int)
                    Rows removed for this group.
                ``final_check`` (dict)
                    The :func:`check_k_anonymity` result after enforcement.

    Raises:
        ValueError: If *k_target* < 1 or *max_iterations* < 1.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "region": ["A"] * 10 + ["B"] * 10 + ["C"] * 2,
        ...     "value": range(22),
        ... })
        >>> result_df, report = enforce_k_anonymity(df, [["region"]], k_target=5)
        >>> report["groups"]["['region']"]["converged"]
        True
    """
    if k_target < 1:
        raise ValueError(f"k_target must be >= 1, got {k_target!r}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")

    if not qi_groups:
        logger.info("enforce_k_anonymity: no QI groups configured : skipping")
        report: dict[str, Any] = {
            "rows_in": len(df),
            "rows_out": len(df),
            "suppression_rate": 0.0,
            "groups": {},
        }
        return df.copy(), report

    rows_in = len(df)
    df = df.copy()

    # Stage 1 : suppress rare individual column values across all QI columns
    all_qi_cols: list[str] = list({col for group in qi_groups for col in group})
    df = suppress_rare_values(df, columns=all_qi_cols, min_count=k_target)

    group_reports: dict[str, Any] = {}

    # Stage 2 & 3 : per-group iterative row suppression
    keep_mask = pd.Series(True, index=df.index)

    for qi_cols in qi_groups:
        group_key = str(qi_cols)
        present = [c for c in qi_cols if c in df.columns]

        if not present:
            logger.warning(
                "enforce_k_anonymity: none of %s found in DataFrame : skipping group",
                qi_cols,
            )
            group_reports[group_key] = {
                "iterations": 0,
                "converged": True,
                "rows_suppressed": 0,
                "final_check": {
                    "min_k": 0,
                    "median_k": 0.0,
                    "n_violations": 0,
                    "pct_violations": 0.0,
                    "passed": True,
                },
            }
            continue

        rows_before_group = int(keep_mask.sum())
        stats = check_k_anonymity(df[keep_mask], present, k_target)

        if stats["passed"]:
            logger.info(
                "QI group %s already satisfies k=%d after value suppression",
                present,
                k_target,
            )
            group_reports[group_key] = {
                "iterations": 0,
                "converged": True,
                "rows_suppressed": 0,
                "final_check": stats,
            }
            continue

        logger.info(
            "Applying iterative row suppression for QI group %s  k=%d",
            present,
            k_target,
        )

        iteration = 0
        converged = False

        for iteration in range(1, max_iterations + 1):
            active_df = df[keep_mask]
            group_sizes = active_df.groupby(present, dropna=False).transform("size")
            violating_idx = active_df.index[group_sizes < k_target]

            if len(violating_idx) == 0:
                converged = True
                break

            keep_mask[violating_idx] = False
            logger.debug(
                "Group %s  iteration=%d  suppressed=%d rows this pass",
                present,
                iteration,
                len(violating_idx),
            )

        if not converged:
            logger.warning(
                "enforce_k_anonymity: group %s did not converge within "
                "%d iteration(s) : %d rows may still violate k=%d",
                present,
                max_iterations,
                int(keep_mask.sum()),
                k_target,
            )

        rows_after_group = int(keep_mask.sum())
        group_suppressed = rows_before_group - rows_after_group
        final_stats = check_k_anonymity(df[keep_mask], present, k_target)

        group_reports[group_key] = {
            "iterations": iteration,
            "converged": converged,
            "rows_suppressed": group_suppressed,
            "final_check": final_stats,
        }

        logger.info(
            "Group %s enforcement done: converged=%s  iterations=%d  "
            "rows_suppressed=%d  final_min_k=%d",
            present,
            converged,
            iteration,
            group_suppressed,
            final_stats["min_k"],
        )

    df = df[keep_mask].reset_index(drop=True)
    rows_out = len(df)
    suppression_rate = (rows_in - rows_out) / rows_in if rows_in > 0 else 0.0

    report = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "suppression_rate": suppression_rate,
        "groups": group_reports,
    }

    log_fn = logger.warning if suppression_rate > 0.05 else logger.info
    log_fn(
        "k-anonymity enforcement complete: %d/%d rows retained "
        "(suppressed %d, rate=%.2f%%)",
        rows_out,
        rows_in,
        rows_in - rows_out,
        suppression_rate * 100,
    )

    return df, report
