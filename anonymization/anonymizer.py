"""Pipeline orchestrator: runs the full anonymization workflow.

The :class:`AnonymizationPipeline` class sequences ten transformation steps,
each driven entirely by the values in an :class:`~anonymization.config.AnonymizationConfig`.
No column names, thresholds, or business rules are hard-coded; everything
flows from the YAML configuration.

Pipeline steps
--------------
1.  **drop_columns** : remove direct IDs, fingerprint columns, and
    operational columns listed in the config.
2.  **round_floats** : round all float columns to the configured number of
    decimal places (overridden per column via ``numeric_targets`` config).
3.  **generalize_quasi_identifiers** : apply QI generalisation strategies
    (band, round_to, top_n, cap) from the config.
4.  **scrub_fingerprints** : scan string columns for regex fingerprint
    patterns and replace matches.
5.  **equalize_categorical** : resample a categorical column so that all
    categories appear with equal frequency (optional; skipped when
    ``equalize_column`` is empty or ``skip_categorical_eq`` is True).
6.  **perturb_dates** : apply HMAC-based date perturbation.
7.  **add_numeric_noise** : inject multiplicative or Laplacian noise into
    numeric target columns.
8.  **normalize_null_patterns** : replace all null-like string tokens
    (``"nan"``, ``"none"``, ``"null"``, ``"na"``, ``"n/a"``, ``""``) with
    a uniform null token from the config.
9.  **enforce_k_anonymity** : suppress rows or values to satisfy the
    k-anonymity requirement across all configured QI groups.
10. **prepare_output** : rename columns, shuffle rows, reorder columns
    alphabetically, and return the final DataFrame.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import AnonymizationConfig
from .fingerprint import scrub_fingerprints
from .generalization import generalize_quasi_identifiers
from .k_anonymity import enforce_k_anonymity
from .noise import add_numeric_noise, perturb_dates

logger = logging.getLogger(__name__)

# String values that are treated as null in the normalisation step
_NULL_TOKENS: frozenset[str] = frozenset({"nan", "none", "null", "na", "n/a", ""})


class AnonymizationPipeline:
    """Orchestrates the full anonymization pipeline.

    All behaviour is controlled by the supplied :class:`AnonymizationConfig`;
    no column names, thresholds, or business logic are hard-coded in this
    class.

    Args:
        config: Fully populated :class:`~anonymization.config.AnonymizationConfig`.

    Attributes:
        config: The active configuration instance.

    Example::

        from anonymization.config import load_config
        from anonymization.anonymizer import AnonymizationPipeline
        import pandas as pd

        cfg = load_config("config/default.yaml")
        pipeline = AnonymizationPipeline(cfg)
        anon_df = pipeline.run(pd.read_csv("data/raw.csv"))
    """

    def __init__(self, config: AnonymizationConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full 10-step anonymization pipeline.

        Each step is timed individually.  A summary table is printed to
        stdout after all steps complete, showing the wall-clock time consumed
        by each step and the final row/column counts.

        The input DataFrame is never mutated; each step operates on a copy.

        Args:
            df: Raw source DataFrame.

        Returns:
            Anonymized DataFrame ready for release.
        """
        cfg = self.config
        n_rows_in = len(df)
        n_cols_in = len(df.columns)
        logger.info(
            "AnonymizationPipeline starting : profile=%s  rows=%d  cols=%d",
            cfg.profile,
            n_rows_in,
            n_cols_in,
        )

        timings: list[tuple[str, float]] = []

        def _timed(
            name: str,
            fn: Callable[[pd.DataFrame], pd.DataFrame],
            data: pd.DataFrame,
        ) -> pd.DataFrame:
            logger.info("--- Step: %s ---", name)
            t0 = time.perf_counter()
            result = fn(data)
            elapsed = time.perf_counter() - t0
            timings.append((name, elapsed))
            logger.debug(
                "Step '%s' complete: rows=%d  cols=%d  elapsed=%.3fs",
                name,
                len(result),
                len(result.columns),
                elapsed,
            )
            return result

        df = _timed("drop_columns", self.drop_columns, df)
        df = _timed("round_floats", self.round_floats, df)
        df = _timed(
            "generalize_quasi_identifiers",
            lambda d: generalize_quasi_identifiers(d, cfg.quasi_identifiers),
            df,
        )
        df = _timed(
            "scrub_fingerprints",
            lambda d: scrub_fingerprints(d, cfg.fingerprint_patterns),
            df,
        )
        df = _timed("equalize_categorical", self.equalize_categorical, df)
        df = _timed(
            "perturb_dates",
            lambda d: perturb_dates(
                d,
                date_columns=cfg.date_columns,
                key_columns=[],
                max_days=cfg.date_max_days,
                seed=cfg.seed,
            ),
            df,
        )
        df = _timed(
            "add_numeric_noise",
            lambda d: add_numeric_noise(d, cfg.numeric_targets, seed=cfg.seed),
            df,
        )
        df = _timed("normalize_null_patterns", self.normalize_null_patterns, df)

        # enforce_k_anonymity returns (df, report); unwrap inside the closure
        def _k_anon_step(data: pd.DataFrame) -> pd.DataFrame:
            result_df, report = enforce_k_anonymity(
                data,
                qi_groups=cfg.qi_groups,
                k_target=cfg.k_target,
            )
            logger.info(
                "k-anonymity report: rows_in=%d  rows_out=%d  rate=%.2f%%",
                report["rows_in"],
                report["rows_out"],
                report["suppression_rate"] * 100,
            )
            return result_df

        df = _timed("enforce_k_anonymity", _k_anon_step, df)
        df = _timed("prepare_output", self.prepare_output, df)

        total = sum(t for _, t in timings)
        _print_step_summary(
            timings, total, n_rows_in, len(df), n_cols_in, len(df.columns)
        )

        logger.info(
            "AnonymizationPipeline complete : rows_in=%d  rows_out=%d  "
            "cols_in=%d  cols_out=%d  total_time=%.3fs",
            n_rows_in,
            len(df),
            n_cols_in,
            len(df.columns),
            total,
        )
        return df

    # ------------------------------------------------------------------
    # Step implementations (public : usable standalone)
    # ------------------------------------------------------------------

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1: Drop direct IDs, fingerprint columns, and operational columns.

        Columns listed in the config that are absent from *df* are skipped
        with a warning rather than raising an error.

        Args:
            df: Source DataFrame.

        Returns:
            DataFrame with configured columns removed.
        """
        cfg = self.config
        to_drop = (
            cfg.drop_direct_ids + cfg.drop_fingerprints + cfg.drop_operational
        )
        present = [c for c in to_drop if c in df.columns]
        absent = [c for c in to_drop if c not in df.columns]

        if absent:
            logger.warning(
                "drop_columns: configured columns not found and skipped: %s",
                absent,
            )

        if present:
            logger.info(
                "drop_columns: removing %d column(s): %s", len(present), present
            )
            df = df.drop(columns=present)
        else:
            logger.info("drop_columns: no columns to drop")

        return df

    def round_floats(self, df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
        """Step 2: Round all float columns to a configured number of decimal places.

        The default decimal count comes from ``config.numeric_round_to``.
        Per-column overrides in ``config.numeric_targets`` take precedence
        over both the *decimals* argument and the config default.

        Args:
            df: Source DataFrame.
            decimals: Fallback decimal count when neither a column override
                nor ``config.numeric_round_to`` is available.

        Returns:
            DataFrame with float columns rounded.
        """
        cfg = self.config
        default_dp = (
            cfg.numeric_round_to if cfg.numeric_round_to is not None else decimals
        )
        float_cols = (
            df.select_dtypes(include=["float64", "float32", "float"])
            .columns.tolist()
        )

        # Build per-column overrides from numeric_targets
        overrides: dict[str, int] = {
            tgt["column"]: int(tgt["round_to"])
            for tgt in cfg.numeric_targets
            if "round_to" in tgt and tgt.get("column")
        }

        for col in float_cols:
            dp = overrides.get(col, default_dp)
            df[col] = df[col].round(dp)

        logger.info(
            "round_floats: rounded %d float column(s) "
            "(default=%d d.p., %d column-level override(s))",
            len(float_cols),
            default_dp,
            len(overrides),
        )
        return df

    def equalize_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 5: Under-sample so a categorical column has equal class sizes.

        Uses the *minimum* category count so no synthetic rows are introduced.
        Skipped when ``config.equalize_column`` is empty or
        ``config.skip_categorical_eq`` is ``True``.

        Args:
            df: Source DataFrame.

        Returns:
            DataFrame with equalised class distribution, or the original
            DataFrame when the step is skipped.
        """
        cfg = self.config
        col = cfg.equalize_column

        if not col or cfg.skip_categorical_eq:
            logger.info("equalize_categorical: skipped")
            return df

        if col not in df.columns:
            logger.warning(
                "equalize_categorical: column '%s' not found : skipping", col
            )
            return df

        counts = df[col].value_counts()
        min_count = int(counts.min())
        logger.info(
            "equalize_categorical: column='%s'  categories=%d  target_size=%d",
            col,
            len(counts),
            min_count,
        )

        rng = np.random.default_rng(cfg.seed)
        frames: list[pd.DataFrame] = []

        for category, group in df.groupby(col):
            sampled = group.sample(
                n=min_count,
                replace=False,
                random_state=int(rng.integers(0, 2**31)),
            )
            frames.append(sampled)
            logger.debug(
                "equalize_categorical: category=%r  original=%d  sampled=%d",
                category,
                len(group),
                len(sampled),
            )

        return pd.concat(frames).reset_index(drop=True)

    def normalize_null_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 8: Replace null-like string tokens with a uniform null token.

        The following case-insensitive representations of missing values are
        replaced: ``"nan"``, ``"none"``, ``"null"``, ``"na"``, ``"n/a"``,
        and the empty string ``""``.  The replacement comes from
        ``config.null_token``.  Only ``object`` (string) columns are
        processed.

        Args:
            df: Source DataFrame.

        Returns:
            DataFrame with normalised null representations.
        """
        cfg = self.config
        token = cfg.null_token
        string_cols = [c for c in df.columns if df[c].dtype == object]

        for col in string_cols:
            mask = df[col].astype(str).str.strip().str.lower().isin(_NULL_TOKENS)
            if mask.any():
                df[col] = df[col].where(~mask, other=token)

        logger.info(
            "normalize_null_patterns: processed %d string column(s)  null_token=%r",
            len(string_cols),
            token,
        )
        return df

    def prepare_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 10: Apply rename map, shuffle rows, and sort columns alphabetically.

        Column renaming uses ``config.rename_map``; only keys present in *df*
        are applied.  Rows are shuffled deterministically using ``config.seed``.
        Columns are sorted alphabetically for a stable output schema.

        Args:
            df: Source DataFrame.

        Returns:
            Final output DataFrame with renamed, sorted columns and shuffled rows.
        """
        cfg = self.config

        rename_map = {k: v for k, v in cfg.rename_map.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info("prepare_output: renamed %d column(s)", len(rename_map))
        else:
            logger.debug("prepare_output: no column renames to apply")

        df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
        df = df.reindex(sorted(df.columns), axis=1)

        logger.info("prepare_output: rows=%d  cols=%d", len(df), len(df.columns))
        return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_step_summary(
    timings: list[tuple[str, float]],
    total: float,
    rows_in: int,
    rows_out: int,
    cols_in: int,
    cols_out: int,
) -> None:
    """Print a formatted timing summary table to stdout.

    Args:
        timings: List of ``(step_name, elapsed_seconds)`` pairs in order.
        total: Total wall-clock time for the full pipeline in seconds.
        rows_in: Input row count.
        rows_out: Output row count.
        cols_in: Input column count.
        cols_out: Output column count.
    """
    width = 50
    print()
    print("=" * width)
    print("  AnonymizationPipeline : Step Timings")
    print("=" * width)

    for step, elapsed in timings:
        pct = (elapsed / total * 100) if total > 0 else 0.0
        print(f"  {step:<36}  {elapsed:>6.3f}s  ({pct:>4.1f}%)")

    print("-" * width)
    print(f"  {'Total':<36}  {total:>6.3f}s")
    print("=" * width)
    print(
        f"  Rows : {rows_in:>10,}  ->  {rows_out:>10,}"
        f"  ({rows_in - rows_out:+,})"
    )
    print(
        f"  Cols : {cols_in:>10,}  ->  {cols_out:>10,}"
        f"  ({cols_in - cols_out:+,})"
    )
    print("=" * width)
    print()
