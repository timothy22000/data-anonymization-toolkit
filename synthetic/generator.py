"""Column profiling and data preparation for the synthetic data pipeline.

Two public classes are provided:

* :class:`ColumnProfiler` : introspects a DataFrame and returns per-column
  metadata (sdtype, null rate, cardinality, sparsity flag).
* :class:`DataPreparer` : selects and conditions the data for a single
  synthesis phase, merging outputs from prior phases when conditioning is
  required.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import SyntheticConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Values that, when the only values present in a column, indicate a boolean
# column (case-insensitive string comparison after coercion).
_BOOL_LIKE_VALUES: frozenset[str] = frozenset(
    {"true", "false", "0", "1", "y", "n"}
)

# SDV sdtype to use for boolean-classified columns.
_SDTYPE_BOOLEAN: str = "categorical"

# Default sdtype when nothing else can be determined.
_SDTYPE_CATEGORICAL: str = "categorical"
_SDTYPE_NUMERICAL: str = "numerical"
_SDTYPE_DATETIME: str = "datetime"


# ---------------------------------------------------------------------------
# ColumnProfiler
# ---------------------------------------------------------------------------


class ColumnProfiler:
    """Scans a DataFrame and produces per-column metadata for SDV.

    Configuration overrides in :attr:`SyntheticConfig.column_sdtypes` take
    precedence over auto-detected types.

    Args:
        config: Populated :class:`~synthetic.config.SyntheticConfig`.

    Example::

        profiler = ColumnProfiler(config)
        profile = profiler.profile(df)
        # {"col_a": {"sdtype": "numerical", "null_rate": 0.0, ...}, ...}
    """

    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Scan every column in *df* and return per-column metadata.

        For each column the returned dict contains:

        * ``sdtype`` (str) : SDV semantic type: ``"numerical"``,
          ``"categorical"``, or ``"datetime"``.
        * ``null_rate`` (float) : fraction of rows where the value is null.
        * ``n_unique`` (int) : number of distinct non-null values.
        * ``is_sparse`` (bool) : ``True`` when ``null_rate`` exceeds
          :attr:`~synthetic.config.SyntheticConfig.sparse_threshold`.

        Column-level sdtype overrides from
        :attr:`~synthetic.config.SyntheticConfig.column_sdtypes` are applied
        after auto-detection.

        Args:
            df: Source DataFrame to profile.

        Returns:
            Dictionary mapping column name to its metadata dict.
        """
        result: dict[str, dict[str, Any]] = {}
        n_rows = len(df)

        for col in df.columns:
            series = df[col]
            null_rate = float(series.isna().mean()) if n_rows > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))
            is_sparse = null_rate > self._config.sparse_threshold

            sdtype = self._detect_sdtype(series, n_unique)

            # Apply config override last so it always wins.
            if col in self._config.column_sdtypes:
                override = self._config.column_sdtypes[col]
                logger.debug(
                    "ColumnProfiler: sdtype override for '%s': %s -> %s",
                    col,
                    sdtype,
                    override,
                )
                sdtype = override

            result[col] = {
                "sdtype": sdtype,
                "null_rate": null_rate,
                "n_unique": n_unique,
                "is_sparse": is_sparse,
            }

        logger.info(
            "ColumnProfiler.profile: scanned %d column(s) across %d row(s)",
            len(result),
            n_rows,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_sdtype(series: pd.Series, n_unique: int) -> str:
        """Infer an SDV sdtype for a single Series.

        Detection order:
        1. Datetime dtype -> ``"datetime"``.
        2. Numeric dtype -> ``"numerical"``.
        3. Boolean-like (at most 2 distinct values drawn from a well-known
           boolean vocabulary) -> ``"categorical"``.
        4. Fallback -> ``"categorical"``.

        Args:
            series: Column data (any dtype).
            n_unique: Pre-computed distinct non-null value count.

        Returns:
            SDV sdtype string.
        """
        # 1. Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return _SDTYPE_DATETIME

        # 2. Numeric
        if pd.api.types.is_numeric_dtype(series):
            return _SDTYPE_NUMERICAL

        # 3. Boolean-like: <= 2 unique values all within the boolean vocabulary
        if n_unique <= 2:
            non_null = series.dropna()
            coerced = {str(v).strip().lower() for v in non_null.unique()}
            if coerced.issubset(_BOOL_LIKE_VALUES):
                return _SDTYPE_BOOLEAN

        # 4. Default
        return _SDTYPE_CATEGORICAL


# ---------------------------------------------------------------------------
# DataPreparer
# ---------------------------------------------------------------------------


class DataPreparer:
    """Selects and conditions data for a single synthesis phase.

    Each synthesis phase operates on a subset of the source columns.  When
    conditional generation is required, columns from previously generated
    phases can be merged in as conditioning context.

    Args:
        config: Populated :class:`~synthetic.config.SyntheticConfig`.

    Example::

        preparer = DataPreparer(config)
        phase_df = preparer.prepare(source_df, phase_config)
        conditioned_df = preparer.add_conditioning_columns(
            phase_df, prior_outputs, phase_config
        )
    """

    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def prepare(
        self,
        df: pd.DataFrame,
        phase_config: dict[str, Any],
    ) -> pd.DataFrame:
        """Select and clean columns for a single phase.

        Steps performed:

        1. Restrict to the columns listed in ``phase_config["columns"]``.
        2. Warn and skip any columns absent from *df*.
        3. Cast object columns that contain only numeric-looking values to
           ``float64`` (best-effort; non-convertible columns are left as-is).
        4. In dev mode, sub-sample to at most 1 000 rows.

        Args:
            df: Full source DataFrame (all columns).
            phase_config: Phase configuration dict.  Must contain a
                ``"columns"`` key whose value is a list of column names.

        Returns:
            A new DataFrame containing only the requested columns, ready for
            SDV training.

        Raises:
            ValueError: If ``phase_config`` does not contain a ``"columns"``
                key, or if none of the requested columns exist in *df*.
        """
        if "columns" not in phase_config:
            raise ValueError(
                "phase_config must contain a 'columns' key; "
                f"received keys: {list(phase_config)}"
            )

        requested: list[str] = list(phase_config["columns"])
        present = [c for c in requested if c in df.columns]
        absent = [c for c in requested if c not in df.columns]

        if absent:
            logger.warning(
                "DataPreparer.prepare [phase=%s]: column(s) not found in "
                "source DataFrame and will be skipped: %s",
                phase_config.get("name", "<unnamed>"),
                absent,
            )

        if not present:
            raise ValueError(
                f"Phase '{phase_config.get('name', '<unnamed>')}' lists columns "
                f"{requested}, but none of them exist in the DataFrame "
                f"(available: {list(df.columns[:10])}"
                f"{'...' if len(df.columns) > 10 else ''})."
            )

        phase_df = df[present].copy()

        # Best-effort numeric coercion for object columns
        for col in phase_df.select_dtypes(include=["object"]).columns:
            converted = pd.to_numeric(phase_df[col], errors="coerce")
            # Only apply coercion when no additional nulls are introduced
            # (i.e. the column contained clean numeric strings throughout).
            original_null_count = int(phase_df[col].isna().sum())
            new_null_count = int(converted.isna().sum())
            if new_null_count == original_null_count:
                phase_df[col] = converted
                logger.debug(
                    "DataPreparer.prepare: coerced '%s' from object to float64",
                    col,
                )

        # Dev-mode row limit
        if self._config.dev and len(phase_df) > 1_000:
            phase_df = phase_df.sample(
                n=1_000,
                random_state=self._config.seed,
            ).reset_index(drop=True)
            logger.info(
                "DataPreparer.prepare [dev mode]: sub-sampled to 1 000 rows "
                "for phase '%s'",
                phase_config.get("name", "<unnamed>"),
            )

        logger.info(
            "DataPreparer.prepare [phase=%s]: selected %d column(s), %d row(s)",
            phase_config.get("name", "<unnamed>"),
            len(present),
            len(phase_df),
        )
        return phase_df

    def add_conditioning_columns(
        self,
        df: pd.DataFrame,
        prior_outputs: dict[str, pd.DataFrame],
        phase_config: dict[str, Any],
    ) -> pd.DataFrame:
        """Merge columns from prior phase outputs into *df* for conditional generation.

        Each DataFrame in *prior_outputs* is aligned to *df* by row position.
        Columns that already exist in *df* are not overwritten.  Row counts
        must match; if a prior output has a different length it is skipped
        with a warning.

        Args:
            df: Phase DataFrame produced by :meth:`prepare`.
            prior_outputs: Mapping from phase name to the synthetic DataFrame
                produced by that phase.
            phase_config: Phase configuration dict.  When a ``"condition_on"``
                key is present, only phases listed there are merged; otherwise
                all prior outputs are merged.

        Returns:
            A new DataFrame that is *df* augmented with conditioning columns.
            The original *df* is not mutated.
        """
        if not prior_outputs:
            return df

        condition_on: list[str] | None = phase_config.get("condition_on")
        sources = (
            {k: v for k, v in prior_outputs.items() if k in condition_on}
            if condition_on is not None
            else prior_outputs
        )

        if not sources:
            return df

        result = df.copy()
        n_rows = len(df)

        for phase_name, prior_df in sources.items():
            if len(prior_df) != n_rows:
                logger.warning(
                    "DataPreparer.add_conditioning_columns: prior phase '%s' "
                    "has %d rows but current phase has %d rows -- skipping",
                    phase_name,
                    len(prior_df),
                    n_rows,
                )
                continue

            new_cols = [c for c in prior_df.columns if c not in result.columns]
            if not new_cols:
                logger.debug(
                    "DataPreparer.add_conditioning_columns: no new columns "
                    "from phase '%s' (all already present)",
                    phase_name,
                )
                continue

            result = pd.concat(
                [
                    result.reset_index(drop=True),
                    prior_df[new_cols].reset_index(drop=True),
                ],
                axis=1,
            )
            logger.info(
                "DataPreparer.add_conditioning_columns [phase=%s]: merged "
                "%d conditioning column(s) from prior phase '%s'",
                phase_config.get("name", "<unnamed>"),
                len(new_cols),
                phase_name,
            )

        return result
