"""SDV-backed synthesizer training and multi-phase generation orchestration.

All SDV imports are wrapped in a single ``try/except ImportError`` block so the
package remains importable when SDV is not installed.  The public :class:`PhaseTrainer`
class raises a descriptive :exc:`ImportError` at the point of use rather than at
import time.

Public class
------------
:class:`PhaseTrainer` : trains one SDV synthesizer per phase, generates rows,
and orchestrates the full multi-phase pipeline with conditioning support.

Module flag
-----------
``SDV_AVAILABLE`` (bool) : ``True`` when SDV is importable.  Callers may
inspect this flag to branch without triggering an exception.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import SyntheticConfig
from .generator import ColumnProfiler, DataPreparer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDV import : entire block in a single try/except
# ---------------------------------------------------------------------------

SDV_AVAILABLE: bool = False

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import (
        GaussianCopulaSynthesizer,
        CTGANSynthesizer,
        TVAESynthesizer,
    )

    SDV_AVAILABLE = True
    logger.debug("SDV imported successfully")
except ImportError:
    logger.warning(
        "SDV is not installed.  Install it with:  pip install sdv\n"
        "PhaseTrainer will raise ImportError when called."
    )

# ---------------------------------------------------------------------------
# Strategy routing table
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[str, str] = {
    "copula": "GaussianCopulaSynthesizer",
    "ctgan": "CTGANSynthesizer",
    "tvae": "TVAESynthesizer",
    # "hybrid" uses copula as the global default; individual phases may
    # specify a different strategy via their own "strategy" key.
    "hybrid": "GaussianCopulaSynthesizer",
}


def _require_sdv() -> None:
    """Raise a descriptive ImportError if SDV is not available at call time."""
    if not SDV_AVAILABLE:
        raise ImportError(
            "The 'sdv' package is required for synthetic data generation but "
            "is not installed.  Install it with:  pip install sdv"
        )


# ---------------------------------------------------------------------------
# PhaseTrainer
# ---------------------------------------------------------------------------


class PhaseTrainer:
    """Trains SDV synthesizers and generates synthetic data per phase.

    A multi-phase pipeline allows different synthesizers and column subsets
    to be used for different semantic groups of columns.  Outputs from earlier
    phases can be passed as conditioning context to later phases through
    :class:`~synthetic.generator.DataPreparer`.

    Args:
        config: Fully populated :class:`~synthetic.config.SyntheticConfig`.

    Example::

        from synthetic import SyntheticConfig, PhaseTrainer
        import pandas as pd

        config = SyntheticConfig(
            input_path="data/source.csv",
            output_path="data/synthetic.csv",
            strategy="copula",
            phases=[
                {"name": "group_a", "columns": ["col_x", "col_y"]},
                {"name": "group_b", "columns": ["col_p", "col_q"]},
            ],
        )
        trainer = PhaseTrainer(config)
        source_df = pd.read_csv(config.input_path)
        synthetic_df = trainer.run_all_phases(source_df)
    """

    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config
        self._profiler = ColumnProfiler(config)
        self._preparer = DataPreparer(config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_phase(
        self,
        df: pd.DataFrame,
        phase_config: dict[str, Any],
        metadata: Any,
    ) -> Any:
        """Train an SDV synthesizer for a single phase.

        The synthesizer class is chosen from the phase-level ``"strategy"``
        key when present; otherwise the global
        :attr:`~synthetic.config.SyntheticConfig.strategy` is used.  In dev
        mode the epoch count is capped at 10 regardless of configuration.

        Args:
            df: Training DataFrame containing only the columns relevant to
                this phase (already prepared by
                :class:`~synthetic.generator.DataPreparer`).
            phase_config: Phase configuration dict.  Optional key
                ``"strategy"`` overrides the global strategy for this phase.
            metadata: A fitted SDV :class:`SingleTableMetadata` instance
                whose column definitions cover all columns present in *df*.

        Returns:
            A fitted SDV synthesizer instance : one of
            ``GaussianCopulaSynthesizer``, ``CTGANSynthesizer``, or
            ``TVAESynthesizer``.

        Raises:
            ImportError: If SDV is not installed.
            ValueError: If the resolved strategy is not a recognised value.
        """
        _require_sdv()

        strategy = str(
            phase_config.get("strategy", self._config.strategy)
        ).lower()

        if strategy not in _STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy '{strategy}' for phase "
                f"'{phase_config.get('name', '<unnamed>')}'.  "
                f"Valid values: {sorted(_STRATEGY_MAP)}"
            )

        synthesizer_class_name = _STRATEGY_MAP[strategy]
        epochs = 10 if self._config.dev else self._config.epochs

        logger.info(
            "PhaseTrainer.train_phase [phase=%s]: strategy=%s  "
            "synthesizer=%s  rows=%d  epochs=%s",
            phase_config.get("name", "<unnamed>"),
            strategy,
            synthesizer_class_name,
            len(df),
            epochs if synthesizer_class_name != "GaussianCopulaSynthesizer" else "n/a",
        )

        synthesizer: Any

        if synthesizer_class_name == "GaussianCopulaSynthesizer":
            synthesizer = GaussianCopulaSynthesizer(metadata)
        elif synthesizer_class_name == "CTGANSynthesizer":
            synthesizer = CTGANSynthesizer(
                metadata,
                epochs=epochs,
                batch_size=self._config.batch_size,
            )
        elif synthesizer_class_name == "TVAESynthesizer":
            synthesizer = TVAESynthesizer(
                metadata,
                epochs=epochs,
                batch_size=self._config.batch_size,
            )
        else:  # pragma: no cover : guarded exhaustively by _STRATEGY_MAP above
            raise ValueError(
                f"Unhandled synthesizer class: {synthesizer_class_name}"
            )

        synthesizer.fit(df)

        logger.info(
            "PhaseTrainer.train_phase [phase=%s]: training complete",
            phase_config.get("name", "<unnamed>"),
        )
        return synthesizer

    def generate_phase(
        self,
        synthesizer: Any,
        n_rows: int,
        conditions: Any | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic rows from a fitted SDV synthesizer.

        Args:
            synthesizer: A fitted SDV synthesizer instance returned by
                :meth:`train_phase`.
            n_rows: Number of rows to generate.
            conditions: Optional SDV ``Condition`` object for conditional
                sampling.  When ``None``, rows are sampled unconditionally.

        Returns:
            DataFrame of *n_rows* synthetic rows.

        Raises:
            ImportError: If SDV is not installed.
        """
        _require_sdv()

        logger.info(
            "PhaseTrainer.generate_phase: generating %d row(s)%s",
            n_rows,
            " with conditions" if conditions is not None else "",
        )

        synthetic_df: pd.DataFrame

        if conditions is not None:
            synthetic_df = synthesizer.sample_remaining_columns(
                known_columns=conditions,
                max_tries_per_batch=100,
            )
        else:
            synthetic_df = synthesizer.sample(num_rows=n_rows)

        logger.info(
            "PhaseTrainer.generate_phase: generated %d row(s)", len(synthetic_df)
        )
        return synthetic_df

    def run_all_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Orchestrate multi-phase synthetic data generation end-to-end.

        For each phase in :attr:`~synthetic.config.SyntheticConfig.phases`
        (in declaration order):

        1. Prepare data : select phase columns; apply dev-mode sub-sampling.
        2. Add conditioning columns from all prior phase outputs.
        3. Build SDV :class:`SingleTableMetadata` from the column profile.
        4. Train a synthesizer via :meth:`train_phase`.
        5. Determine the target row count (``config.n_rows`` or match source).
        6. Generate synthetic rows via :meth:`generate_phase`.
        7. Store the output for conditioning subsequent phases.

        After all phases are complete, the per-phase DataFrames are
        concatenated column-wise (``axis=1``).  Duplicate column names
        (e.g. conditioning columns shared across phases) are deduplicated
        by keeping the first occurrence.

        Args:
            df: Full source DataFrame used for training.

        Returns:
            Synthetic DataFrame whose columns are the union of all phase
            column sets, in phase order.

        Raises:
            ImportError: If SDV is not installed.
            ValueError: If :attr:`~synthetic.config.SyntheticConfig.phases`
                is empty.
        """
        _require_sdv()

        if not self._config.phases:
            raise ValueError(
                "No phases defined in SyntheticConfig.phases.  "
                "At minimum one phase with a 'columns' list is required."
            )

        # Optional global sub-sampling of the source data before any training
        train_df = df
        if self._config.sample_frac < 1.0:
            n_sample = max(1, int(len(df) * self._config.sample_frac))
            train_df = df.sample(
                n=n_sample,
                random_state=self._config.seed,
            ).reset_index(drop=True)
            logger.info(
                "run_all_phases: source sampled to %d row(s) "
                "(sample_frac=%.3f)",
                len(train_df),
                self._config.sample_frac,
            )

        phase_outputs: dict[str, pd.DataFrame] = {}

        for phase_config in self._config.phases:
            phase_name: str = phase_config.get("name", "<unnamed>")
            logger.info("--- Phase: %s ---", phase_name)

            # Step 1: prepare data for this phase
            phase_df = self._preparer.prepare(train_df, phase_config)

            # Step 2: merge conditioning columns from prior phases
            phase_df = self._preparer.add_conditioning_columns(
                phase_df, phase_outputs, phase_config
            )

            # Step 3: build SDV metadata from column profile
            metadata = self._build_metadata(phase_df)

            # Step 4: train synthesizer
            synthesizer = self.train_phase(phase_df, phase_config, metadata)

            # Step 5: determine target row count
            n_rows = (
                self._config.n_rows if self._config.n_rows > 0 else len(phase_df)
            )

            # Step 6: generate synthetic rows
            synthetic_phase_df = self.generate_phase(synthesizer, n_rows)

            # Step 7: store output for conditioning downstream phases
            phase_outputs[phase_name] = synthetic_phase_df

            logger.info(
                "Phase '%s' complete: %d row(s) x %d col(s)",
                phase_name,
                len(synthetic_phase_df),
                len(synthetic_phase_df.columns),
            )

        # Concatenate all phase outputs column-wise
        final_df = self._concat_phases(phase_outputs)

        logger.info(
            "run_all_phases complete: %d row(s) x %d col(s) across %d phase(s)",
            len(final_df),
            len(final_df.columns),
            len(self._config.phases),
        )
        return final_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_metadata(self, df: pd.DataFrame) -> Any:
        """Construct an SDV SingleTableMetadata instance for *df*.

        Uses :class:`~synthetic.generator.ColumnProfiler` to resolve each
        column's sdtype and registers every column individually so that
        config-level overrides (from
        :attr:`~synthetic.config.SyntheticConfig.column_sdtypes`) are always
        respected.

        Args:
            df: DataFrame whose columns should be registered.

        Returns:
            A configured :class:`sdv.metadata.SingleTableMetadata` instance.
        """
        metadata = SingleTableMetadata()
        profile = self._profiler.profile(df)

        for col, info in profile.items():
            if col not in df.columns:
                continue
            metadata.add_column(col, sdtype=info["sdtype"])

        logger.debug(
            "_build_metadata: registered %d column(s)", len(profile)
        )
        return metadata

    @staticmethod
    def _concat_phases(
        phase_outputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Concatenate per-phase DataFrames column-wise.

        Duplicate column names introduced by conditioning merges are
        deduplicated by keeping the first occurrence (i.e. the phase that
        owns the column).

        Args:
            phase_outputs: Ordered mapping from phase name to the synthetic
                DataFrame produced for that phase.

        Returns:
            Single DataFrame combining all phase columns, with a clean
            integer index.
        """
        frames = list(phase_outputs.values())
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
        return combined.reset_index(drop=True)
