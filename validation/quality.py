"""
validation.quality
==================

Statistical quality checks for comparing a real dataset against a synthetic
counterpart.  All checks are generic and operate on any tabular DataFrame.

Typical usage
-------------
>>> from validation.quality import run_quality_checks
>>> report = run_quality_checks(real_df, synth_df, config)
>>> print(report.overall_score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    """Aggregated results from all statistical quality checks.

    Attributes:
        marginal_scores: Mapping of column name to p-value from the
            marginal-distribution test (chi-square for categoricals,
            Kolmogorov-Smirnov for numericals).  Higher p-value indicates
            the synthetic distribution is closer to real.
        correlation_fidelity: Mean absolute difference between the real and
            synthetic Pearson correlation matrices.  0.0 is perfect fidelity.
        cross_tab_divergences: Mapping of ``"col_a__col_b"`` to
            Jensen-Shannon divergence (0.0–1.0).  Lower is better.
        overall_score: Composite score in [0, 1] where 1.0 is perfect
            statistical fidelity.  Derived from the individual checks.
        details: Arbitrary extra metadata produced by individual checks.
    """

    marginal_scores: dict[str, float] = field(default_factory=dict)
    correlation_fidelity: float = 0.0
    cross_tab_divergences: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def compare_marginals(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> dict[str, float]:
    """Compare per-column marginal distributions between real and synthetic data.

    Categorical columns are compared via the chi-square goodness-of-fit test.
    Numerical columns are compared via the two-sample Kolmogorov-Smirnov test.

    Args:
        real_df: Ground-truth DataFrame.
        synth_df: Synthetic DataFrame to evaluate.
        categorical_cols: Column names to treat as categorical.
        numerical_cols: Column names to treat as numerical.

    Returns:
        Mapping of column name to p-value.  A p-value close to 1.0 indicates
        the synthetic column is statistically indistinguishable from real.
        A p-value below 0.05 suggests a significant distributional difference.
    """
    scores: dict[str, float] = {}

    for col in categorical_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            logger.warning("Column %r not found in both DataFrames; skipping.", col)
            continue

        real_counts = real_df[col].value_counts()
        synth_counts = synth_df[col].value_counts()

        # Align on the union of categories so chi-square receives paired arrays.
        all_cats = real_counts.index.union(synth_counts.index)
        real_aligned = real_counts.reindex(all_cats, fill_value=0)
        synth_aligned = synth_counts.reindex(all_cats, fill_value=0)

        # Avoid chi-square on all-zero expected frequencies.
        if real_aligned.sum() == 0 or synth_aligned.sum() == 0:
            logger.warning("Column %r has zero total counts; skipping chi-square.", col)
            scores[col] = 0.0
            continue

        # Build a 2-row contingency table: [real_frequencies, synth_frequencies].
        contingency = np.array([real_aligned.values, synth_aligned.values])
        try:
            _, p_value, _, _ = chi2_contingency(contingency)
        except ValueError as exc:
            logger.warning("Chi-square failed for column %r: %s", col, exc)
            p_value = 0.0

        scores[col] = float(p_value)
        logger.debug("Categorical %r: chi-square p=%.4f", col, p_value)

    for col in numerical_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            logger.warning("Column %r not found in both DataFrames; skipping.", col)
            continue

        real_vals = real_df[col].dropna().values
        synth_vals = synth_df[col].dropna().values

        if len(real_vals) == 0 or len(synth_vals) == 0:
            logger.warning("Column %r is all-NaN in one DataFrame; skipping KS.", col)
            scores[col] = 0.0
            continue

        ks_stat, p_value = ks_2samp(real_vals, synth_vals)
        scores[col] = float(p_value)
        logger.debug(
            "Numerical %r: KS statistic=%.4f, p=%.4f", col, ks_stat, p_value
        )

    return scores


def compare_correlations(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numerical_cols: list[str],
) -> float:
    """Compare Pearson correlation matrices between real and synthetic data.

    Args:
        real_df: Ground-truth DataFrame.
        synth_df: Synthetic DataFrame to evaluate.
        numerical_cols: Column names to include in the correlation comparison.
            Columns absent from either DataFrame are silently dropped.

    Returns:
        Mean absolute difference between corresponding entries in the two
        correlation matrices.  0.0 means perfect correlation fidelity; 1.0
        means maximally different.
    """
    present_cols = [
        c
        for c in numerical_cols
        if c in real_df.columns and c in synth_df.columns
    ]
    if len(present_cols) < 2:
        logger.warning(
            "Fewer than 2 numerical columns available for correlation comparison."
        )
        return 0.0

    real_corr = real_df[present_cols].corr()
    synth_corr = synth_df[present_cols].corr()

    diff = (real_corr - synth_corr).abs()
    # Exclude the diagonal (always 0 for both matrices).
    np.fill_diagonal(diff.values, np.nan)
    mean_abs_diff = float(np.nanmean(diff.values))

    logger.debug("Correlation matrix mean absolute difference: %.4f", mean_abs_diff)
    return mean_abs_diff


def cross_tab_divergence(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    col_pairs: list[tuple[str, str]],
) -> dict[str, float]:
    """Compute Jensen-Shannon divergence for cross-tabulations of column pairs.

    The cross-tabulation of each pair is normalised to a joint probability
    distribution before computing JS divergence.

    Args:
        real_df: Ground-truth DataFrame.
        synth_df: Synthetic DataFrame to evaluate.
        col_pairs: List of ``(col_a, col_b)`` tuples to cross-tabulate.

    Returns:
        Mapping of ``"col_a__col_b"`` to JS divergence in [0, 1].  A value
        of 0.0 means the joint distributions are identical.
    """
    divergences: dict[str, float] = {}

    for col_a, col_b in col_pairs:
        key = f"{col_a}__{col_b}"

        missing = [
            c
            for c in (col_a, col_b)
            if c not in real_df.columns or c not in synth_df.columns
        ]
        if missing:
            logger.warning("Columns %s missing from DataFrames; skipping pair.", missing)
            divergences[key] = float("nan")
            continue

        real_ct = pd.crosstab(real_df[col_a], real_df[col_b])
        synth_ct = pd.crosstab(synth_df[col_a], synth_df[col_b])

        # Align indices and columns on the union of observed values.
        all_rows = real_ct.index.union(synth_ct.index)
        all_cols = real_ct.columns.union(synth_ct.columns)
        real_ct = real_ct.reindex(index=all_rows, columns=all_cols, fill_value=0)
        synth_ct = synth_ct.reindex(index=all_rows, columns=all_cols, fill_value=0)

        # Flatten to 1-D probability vectors.
        real_prob = real_ct.values.flatten().astype(float)
        synth_prob = synth_ct.values.flatten().astype(float)

        real_prob_norm = real_prob / real_prob.sum() if real_prob.sum() > 0 else real_prob
        synth_prob_norm = (
            synth_prob / synth_prob.sum() if synth_prob.sum() > 0 else synth_prob
        )

        js_div = float(jensenshannon(real_prob_norm, synth_prob_norm))
        divergences[key] = js_div
        logger.debug("Cross-tab JS divergence for %r: %.4f", key, js_div)

    return divergences


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_quality_checks(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    config: dict[str, Any],
) -> QualityReport:
    """Orchestrate all statistical quality checks and return a QualityReport.

    The ``config`` dict is expected to contain the following keys (all
    optional; missing keys are treated as empty):

    * ``categorical_cols`` : list of categorical column names.
    * ``numerical_cols`` : list of numerical column names.
    * ``cross_tab_pairs`` : list of ``[col_a, col_b]`` pairs to cross-tabulate.

    The ``overall_score`` is computed as follows:

    1. **Marginal score** : fraction of columns whose marginal p-value ≥ 0.05
       (i.e. we do *not* reject the null that distributions are the same).
    2. **Correlation score** : ``1 - mean_abs_diff``, clipped to [0, 1].
    3. **Cross-tab score** : mean of ``(1 - JS divergence)`` across all pairs,
       or 1.0 if no pairs were configured.
    4. ``overall_score = mean(marginal_score, correlation_score, cross_tab_score)``

    Args:
        real_df: Ground-truth DataFrame.
        synth_df: Synthetic DataFrame to evaluate.
        config: Dictionary of quality-check parameters (see above).

    Returns:
        A populated :class:`QualityReport` instance.
    """
    categorical_cols: list[str] = config.get("categorical_cols", [])
    numerical_cols: list[str] = config.get("numerical_cols", [])
    cross_tab_pairs: list[tuple[str, str]] = [
        tuple(pair) for pair in config.get("cross_tab_pairs", [])
    ]

    logger.info(
        "Running quality checks: %d categorical, %d numerical, %d cross-tab pairs.",
        len(categorical_cols),
        len(numerical_cols),
        len(cross_tab_pairs),
    )

    # --- Marginal distributions ---
    marginal_scores = compare_marginals(
        real_df, synth_df, categorical_cols, numerical_cols
    )

    # --- Correlation fidelity ---
    correlation_fidelity = compare_correlations(real_df, synth_df, numerical_cols)

    # --- Cross-tab divergences ---
    cross_tab_divs = cross_tab_divergence(real_df, synth_df, cross_tab_pairs)

    # --- Overall score ---
    if marginal_scores:
        passing = sum(1 for p in marginal_scores.values() if p >= 0.05)
        marginal_score = passing / len(marginal_scores)
    else:
        marginal_score = 1.0

    correlation_score = float(np.clip(1.0 - correlation_fidelity, 0.0, 1.0))

    valid_divs = [v for v in cross_tab_divs.values() if not np.isnan(v)]
    cross_tab_score = float(np.mean([1.0 - d for d in valid_divs])) if valid_divs else 1.0

    overall_score = float(np.mean([marginal_score, correlation_score, cross_tab_score]))

    logger.info(
        "Quality check complete. Overall score: %.3f "
        "(marginal=%.3f, correlation=%.3f, cross_tab=%.3f)",
        overall_score,
        marginal_score,
        correlation_score,
        cross_tab_score,
    )

    return QualityReport(
        marginal_scores=marginal_scores,
        correlation_fidelity=correlation_fidelity,
        cross_tab_divergences=cross_tab_divs,
        overall_score=overall_score,
        details={
            "marginal_score": marginal_score,
            "correlation_score": correlation_score,
            "cross_tab_score": cross_tab_score,
            "n_real": len(real_df),
            "n_synth": len(synth_df),
        },
    )
