"""
validation.red_team
====================

Adversarial attack framework for testing anonymisation and synthetic data
quality.  Every attack is dataset-agnostic: column names are supplied via
:class:`RedTeamConfig` (loaded from YAML) rather than hardcoded.

Typical usage
-------------
>>> from validation.red_team import RedTeamConfig, RedTeamRunner, load_red_team_config
>>> config = load_red_team_config("config/red_team.yaml")
>>> runner = RedTeamRunner(config)
>>> results = runner.run_all(df_anon, df_orig=df_real)
>>> print(runner.generate_report(results))

YAML structure expected by ``load_red_team_config``
-----------------------------------------------------
::

    profile: ml            # optional; default "ml"
    mode: anonymized       # "anonymized" or "synthetic"
    k_target: 5
    max_sample: 50000
    random_seed: 42

    qi_columns:
      - column_a
      - column_b

    fingerprint_patterns:
      - "\\\\b\\\\d{9}\\\\b"   # regex strings

    numeric_targets:
      - column_c
      - column_d
"""

from __future__ import annotations

import abc
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result / config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AttackResult:
    """Summary of a single adversarial attack run.

    Attributes:
        name: Human-readable name of the attack.
        passed: ``True`` when the dataset survives the attack (no finding).
        severity: Risk level : one of ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``,
            or ``"LOW"``.
        score: Continuous risk score in [0.0, 1.0].  Interpretation is
            attack-specific, but higher always means higher risk.
        details: Arbitrary dict of attack-specific metrics and diagnostics.
        recommendations: List of actionable remediation strings.
        elapsed_seconds: Wall-clock time taken by the attack.
        skipped: ``True`` when the attack was inapplicable and not executed.
        skip_reason: Human-readable explanation when ``skipped`` is ``True``.
    """

    name: str
    passed: bool
    severity: str = "HIGH"
    score: float = 0.0
    details: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class RedTeamConfig:
    """Configuration for the red-team adversarial evaluation.

    Attributes:
        profile: Named profile string used for documentation and logging.
        mode: Either ``"anonymized"`` (df_orig required for diff checks) or
            ``"synthetic"`` (df_orig used when available).
        k_target: Minimum group size required for k-anonymity.  Attacks that
            check group sizes use this threshold.
        max_sample: Maximum number of rows drawn from each DataFrame before
            running attacks.  Set to 0 to disable sampling.
        random_seed: Global seed for all random operations.
        qi_columns: Quasi-identifier column names used in linkage attacks.
        fingerprint_patterns: List of regex pattern strings to scan for
            residual fingerprints.
        numeric_targets: Numeric column names subject to precision and ratio
            attacks.
    """

    profile: str = "ml"
    mode: str = "anonymized"
    k_target: int = 5
    max_sample: int = 50_000
    random_seed: int = 42
    qi_columns: list[str] = field(default_factory=list)
    fingerprint_patterns: list[str] = field(default_factory=list)
    numeric_targets: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_red_team_config(yaml_path: str | Path) -> RedTeamConfig:
    """Load a :class:`RedTeamConfig` from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A populated :class:`RedTeamConfig` instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ImportError: If PyYAML is not installed.
        yaml.YAMLError: If the file cannot be parsed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load a red-team config.") from exc

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Red-team config not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return RedTeamConfig(
        profile=str(raw.get("profile", "ml")),
        mode=str(raw.get("mode", "anonymized")),
        k_target=int(raw.get("k_target", 5)),
        max_sample=int(raw.get("max_sample", 50_000)),
        random_seed=int(raw.get("random_seed", 42)),
        qi_columns=[str(c) for c in raw.get("qi_columns", [])],
        fingerprint_patterns=[str(p) for p in raw.get("fingerprint_patterns", [])],
        numeric_targets=[str(c) for c in raw.get("numeric_targets", [])],
    )


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


class DataLoader:
    """Utility for loading DataFrames from common tabular file formats.

    Supports CSV and Parquet via file extension.  Chunked sampling is
    provided for very large files so that memory pressure is bounded.
    """

    _SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".parquet", ".pq"})

    def load(self, path: str | Path, nrows: int | None = None) -> pd.DataFrame:
        """Load a DataFrame from ``path``.

        Args:
            path: Path to a CSV or Parquet file.
            nrows: If set, read at most this many rows.  Supported natively
                for CSV; for Parquet the full file is read then truncated.

        Returns:
            Loaded DataFrame.

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        ext = path.suffix.lower()
        if ext not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {sorted(self._SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".csv":
            return pd.read_csv(path, nrows=nrows)

        df = pd.read_parquet(path)
        if nrows is not None:
            df = df.iloc[:nrows]
        return df

    def load_sample(
        self,
        path: str | Path,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Load a representative sample from a potentially large file.

        For CSV files sampling is done via chunked reading to avoid loading
        the entire file into memory.  For Parquet the full file is loaded
        (Parquet already supports efficient column-level I/O) before sampling.

        Args:
            path: Path to a CSV or Parquet file.
            n: Number of rows to sample.  When ``None`` the full file is
               returned (equivalent to :meth:`load`).

        Returns:
            Sampled DataFrame.
        """
        path = Path(path)
        ext = path.suffix.lower()

        if n is None:
            return self.load(path)

        if ext != ".csv":
            df = self.load(path)
            return df.sample(n=min(n, len(df)), random_state=0) if len(df) > n else df

        # Chunked reservoir sampling for large CSVs.
        chunks: list[pd.DataFrame] = []
        total_read = 0
        chunksize = max(n * 2, 10_000)

        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunks.append(chunk)
            total_read += len(chunk)
            if total_read >= n * 5:
                break

        if not chunks:
            return pd.DataFrame()

        combined = pd.concat(chunks, ignore_index=True)
        return (
            combined.sample(n=min(n, len(combined)), random_state=0)
            if len(combined) > n
            else combined
        )


# ---------------------------------------------------------------------------
# Base attack
# ---------------------------------------------------------------------------


class BaseAttack(abc.ABC):
    """Abstract base class for all adversarial attacks.

    Subclasses must set class attributes ``name``, ``severity``, and
    ``applies_to_synthetic``, and implement :meth:`run`.

    Class attributes:
        name: Short human-readable identifier for the attack.
        severity: Risk level : ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``, or
            ``"LOW"``.
        applies_to_synthetic: When ``True`` the attack is relevant for both
            anonymised and synthetic data.  When ``False`` it only applies to
            anonymised data (``config.mode == "anonymized"``).
    """

    name: str = "BaseAttack"
    severity: str = "HIGH"
    applies_to_synthetic: bool = True

    def __init__(self, config: RedTeamConfig, loader: DataLoader) -> None:
        """Initialise the attack with shared config and a data loader.

        Args:
            config: Red-team configuration instance.
            loader: :class:`DataLoader` instance used if file-based loading
                is needed within the attack.
        """
        self.config = config
        self.loader = loader

    @abc.abstractmethod
    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        """Execute the attack and return a result.

        Args:
            df_anon: Anonymised or synthetic DataFrame under evaluation.
            df_orig: Original (real) DataFrame for diff-based attacks.
                May be ``None`` when not available.

        Returns:
            An :class:`AttackResult` describing the outcome.
        """

    def execute(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        """Wrapper around :meth:`run` that handles timing, errors, and skip logic.

        The attack is automatically skipped when:

        - ``applies_to_synthetic`` is ``False`` and ``config.mode`` is
          ``"synthetic"``.

        On any unhandled exception from :meth:`run` an :class:`AttackResult`
        with ``passed=False`` and the exception message in ``details`` is
        returned so the runner can continue with remaining attacks.

        Args:
            df_anon: Anonymised or synthetic DataFrame under evaluation.
            df_orig: Original DataFrame.  May be ``None``.

        Returns:
            An :class:`AttackResult` with ``elapsed_seconds`` populated.
        """
        if not self.applies_to_synthetic and self.config.mode == "synthetic":
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                skipped=True,
                skip_reason=f"{self.name} is not applicable in synthetic mode.",
            )

        t0 = time.perf_counter()
        try:
            result = self.run(df_anon, df_orig=df_orig)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.exception("Attack %r raised an unexpected error.", self.name)
            return AttackResult(
                name=self.name,
                passed=False,
                severity=self.severity,
                score=1.0,
                details={"error": str(exc)},
                recommendations=[
                    "Investigate the error and re-run the attack after correcting the data."
                ],
                elapsed_seconds=elapsed,
            )

        result.elapsed_seconds = time.perf_counter() - t0
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Return a reproducible row sample of at most ``max_rows``."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def _present_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return only those column names that actually exist in ``df``."""
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Attack 1: Uniqueness
# ---------------------------------------------------------------------------


class Attack_Uniqueness(BaseAttack):
    """Test quasi-identifier column combinations for unique records.

    Checks 2-way, 3-way, and 4-way combinations of ``qi_columns``.  A record
    that is unique on a QI combination is trivially re-identifiable if an
    adversary has even partial external knowledge.

    Severity: CRITICAL
    """

    name = "Attack_Uniqueness"
    severity = "CRITICAL"
    applies_to_synthetic = True

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        from itertools import combinations

        qi_cols = _present_cols(df_anon, self.config.qi_columns)
        if not qi_cols:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                skipped=True,
                skip_reason="No qi_columns present in DataFrame.",
            )

        df = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        n_total = len(df)
        if n_total == 0:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={"n_rows": 0},
                recommendations=[],
            )

        combo_stats: dict[str, dict[str, Any]] = {}
        worst_unique_rate = 0.0

        for combo_size in (2, 3, 4):
            if len(qi_cols) < combo_size:
                continue
            for combo in combinations(qi_cols, combo_size):
                key = "__".join(combo)
                group_sizes = df.groupby(list(combo), dropna=False).size()
                unique_records = int((group_sizes == 1).sum())
                unique_rate = unique_records / n_total
                combo_stats[key] = {
                    "combo_size": combo_size,
                    "unique_records": unique_records,
                    "unique_rate": round(unique_rate, 4),
                    "n_groups": int(len(group_sizes)),
                    "min_group_size": int(group_sizes.min()),
                }
                if unique_rate > worst_unique_rate:
                    worst_unique_rate = unique_rate

        # Per-column k-anonymity violations.
        single_col_violations: dict[str, int] = {}
        for col in qi_cols:
            group_sizes = df.groupby(col, dropna=False).size()
            below_k = int((group_sizes < self.config.k_target).sum())
            if below_k:
                single_col_violations[col] = below_k

        passed = worst_unique_rate < 0.01 and not single_col_violations
        recommendations: list[str] = []
        if worst_unique_rate >= 0.01:
            recommendations.append(
                f"Reduce uniqueness ({worst_unique_rate:.1%} of rows are unique on some "
                f"QI combination).  Apply stronger generalisation or suppression."
            )
        if single_col_violations:
            cols_str = ", ".join(single_col_violations.keys())
            recommendations.append(
                f"Columns {cols_str} have groups below k={self.config.k_target}.  "
                f"Apply k-anonymity enforcement."
            )

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=round(worst_unique_rate, 4),
            details={
                "combo_stats": combo_stats,
                "k_target": self.config.k_target,
                "k_violations_by_column": single_col_violations,
                "worst_unique_rate": round(worst_unique_rate, 4),
                "n_rows_evaluated": n_total,
            },
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Attack 2: Temporal Linkage
# ---------------------------------------------------------------------------


class Attack_TemporalLinkage(BaseAttack):
    """Check if date columns can link records to external calendars.

    Detects date/datetime columns retaining full precision (exact day or
    finer).  Exact dates combined with other attributes can enable linkage
    to external event calendars or news records.

    Severity: HIGH
    """

    name = "Attack_TemporalLinkage"
    severity = "HIGH"
    applies_to_synthetic = True

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        date_cols: list[str] = []
        for col in df_anon.columns:
            if pd.api.types.is_datetime64_any_dtype(df_anon[col]):
                date_cols.append(col)
            elif df_anon[col].dtype == object:
                sample = df_anon[col].dropna().head(200)
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        date_cols.append(col)
                except Exception:
                    pass

        if not date_cols:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={"date_columns_found": []},
                recommendations=[],
            )

        df = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        findings: dict[str, dict[str, Any]] = {}
        max_risk = 0.0

        for col in date_cols:
            parsed = pd.to_datetime(df[col], errors="coerce")
            n_valid = int(parsed.notna().sum())
            if n_valid == 0:
                continue

            n_unique_dates = int(parsed.dt.date.nunique())
            n_with_time = int(
                (parsed.dt.time != pd.Timestamp("00:00:00").time()).sum()
            )
            exact_day_rate = n_unique_dates / max(n_valid, 1)
            sub_day_rate = n_with_time / max(n_valid, 1)
            risk = min(exact_day_rate * (1.0 + sub_day_rate), 1.0)

            findings[col] = {
                "n_valid": n_valid,
                "n_unique_dates": n_unique_dates,
                "exact_day_rate": round(exact_day_rate, 4),
                "sub_day_precision_rate": round(sub_day_rate, 4),
                "risk_score": round(risk, 4),
            }
            if risk > max_risk:
                max_risk = risk

        passed = max_risk < 0.5
        recommendations: list[str] = []
        if not passed:
            recommendations.append(
                "Truncate or perturb date columns to reduce temporal precision.  "
                "Replace exact dates with month/year bands or add calendar noise "
                "of several days."
            )
        if any(v["sub_day_precision_rate"] > 0.1 for v in findings.values()):
            recommendations.append(
                "Sub-day timestamps detected.  Strip time components or round to "
                "the nearest hour or day."
            )

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=round(max_risk, 4),
            details={"date_columns": findings},
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Attack 3: Fingerprints
# ---------------------------------------------------------------------------


class Attack_Fingerprints(BaseAttack):
    """Scan for residual fingerprint patterns from config.

    Each pattern in ``config.fingerprint_patterns`` is compiled as a regular
    expression and applied to every string column.  Matches indicate that
    identifying information has survived the anonymisation step.

    Severity: HIGH
    """

    name = "Attack_Fingerprints"
    severity = "HIGH"
    applies_to_synthetic = True

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        patterns = self.config.fingerprint_patterns
        if not patterns:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                skipped=True,
                skip_reason="No fingerprint_patterns configured.",
            )

        compiled: list[tuple[str, re.Pattern]] = []
        for pat in patterns:
            try:
                compiled.append((pat, re.compile(pat)))
            except re.error as exc:
                logger.warning("Invalid fingerprint pattern %r: %s", pat, exc)

        if not compiled:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                skipped=True,
                skip_reason="All fingerprint_patterns failed to compile.",
            )

        df = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        str_cols = df.select_dtypes(include="object").columns.tolist()

        match_summary: dict[str, dict[str, int]] = {}
        total_matches = 0

        for col in str_cols:
            col_series = df[col].astype(str)
            for raw_pat, rx in compiled:
                n_matches = int(col_series.str.contains(rx, na=False).sum())
                if n_matches > 0:
                    if col not in match_summary:
                        match_summary[col] = {}
                    match_summary[col][raw_pat] = n_matches
                    total_matches += n_matches

        passed = total_matches == 0
        score = min(total_matches / max(len(df), 1), 1.0)

        recommendations: list[str] = []
        if not passed:
            recommendations.append(
                f"Residual fingerprint patterns found in {len(match_summary)} column(s).  "
                f"Apply pattern redaction or replacement before releasing data."
            )

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=round(score, 4),
            details={
                "total_matches": total_matches,
                "columns_with_matches": match_summary,
                "patterns_checked": len(compiled),
                "string_columns_scanned": len(str_cols),
            },
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Attack 4: Outlier Reidentification
# ---------------------------------------------------------------------------


class Attack_OutlierReidentification(BaseAttack):
    """Find statistical outliers that could be re-identified.

    For each numeric column the IQR method is used to identify extreme
    outliers (beyond 3 × IQR from the quartiles).  Outlier records are
    likely to be unique and thus re-identifiable even after mild
    anonymisation.

    Severity: HIGH
    """

    name = "Attack_OutlierReidentification"
    severity = "HIGH"
    applies_to_synthetic = True

    _IQR_MULTIPLIER = 3.0

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        num_cols = df_anon.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={"numeric_columns_found": 0},
                recommendations=[],
            )

        df = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        n_total = len(df)
        col_stats: dict[str, dict[str, Any]] = {}
        outlier_mask = pd.Series(False, index=df.index)

        for col in num_cols:
            vals = df[col].dropna()
            if len(vals) < 10:
                continue
            q1 = float(vals.quantile(0.25))
            q3 = float(vals.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0.0:
                continue
            lower = q1 - self._IQR_MULTIPLIER * iqr
            upper = q3 + self._IQR_MULTIPLIER * iqr
            col_outliers = df[col].notna() & ((df[col] < lower) | (df[col] > upper))
            n_outliers = int(col_outliers.sum())
            outlier_rate = n_outliers / n_total if n_total else 0.0
            col_stats[col] = {
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
                "n_outliers": n_outliers,
                "outlier_rate": round(outlier_rate, 4),
            }
            outlier_mask = outlier_mask | col_outliers

        total_outlier_rows = int(outlier_mask.sum())
        overall_rate = total_outlier_rows / n_total if n_total else 0.0
        passed = overall_rate < 0.01

        recommendations: list[str] = []
        if not passed:
            recommendations.append(
                f"{total_outlier_rows} rows ({overall_rate:.1%}) are extreme outliers "
                f"on at least one numeric column.  Suppress or cap extreme values "
                f"before release."
            )

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=round(overall_rate, 4),
            details={
                "per_column": col_stats,
                "total_outlier_rows": total_outlier_rows,
                "overall_outlier_rate": round(overall_rate, 4),
                "n_rows_evaluated": n_total,
                "iqr_multiplier": self._IQR_MULTIPLIER,
            },
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Attack 5: Distribution Skew
# ---------------------------------------------------------------------------


class Attack_DistributionSkew(BaseAttack):
    """Compare pre/post distributions for information leakage.

    Computes the Jensen-Shannon divergence between the original and anonymised
    marginal distributions for each numeric column.  A very low divergence
    means the anonymised column is almost indistinguishable from the original,
    which can leak distributional information about individuals.

    Requires ``df_orig``.

    Severity: MEDIUM
    """

    name = "Attack_DistributionSkew"
    severity = "MEDIUM"
    applies_to_synthetic = True

    _BINS = 50

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        if df_orig is None:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                skipped=True,
                skip_reason="df_orig is required for distribution comparison.",
            )

        try:
            from scipy.spatial.distance import jensenshannon
        except ImportError:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                skipped=True,
                skip_reason="scipy is required for Jensen-Shannon divergence.",
            )

        num_cols = [
            c
            for c in df_anon.select_dtypes(include=[np.number]).columns
            if c in df_orig.columns
        ]
        if not num_cols:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={"numeric_columns_compared": 0},
                recommendations=[],
            )

        df_a = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        df_o = _sample(df_orig, self.config.max_sample, self.config.random_seed)

        col_stats: dict[str, dict[str, float]] = {}
        js_scores: list[float] = []

        for col in num_cols:
            a_vals = df_a[col].dropna().values.astype(float)
            o_vals = df_o[col].dropna().values.astype(float)
            if len(a_vals) < 5 or len(o_vals) < 5:
                continue

            combined_min = min(a_vals.min(), o_vals.min())
            combined_max = max(a_vals.max(), o_vals.max())
            if combined_min == combined_max:
                continue

            bins = np.linspace(combined_min, combined_max, self._BINS + 1)
            a_hist, _ = np.histogram(a_vals, bins=bins, density=False)
            o_hist, _ = np.histogram(o_vals, bins=bins, density=False)

            a_prob = a_hist / a_hist.sum() if a_hist.sum() > 0 else a_hist.astype(float)
            o_prob = o_hist / o_hist.sum() if o_hist.sum() > 0 else o_hist.astype(float)

            js = float(jensenshannon(o_prob, a_prob))
            col_stats[col] = {"js_divergence": round(js, 4)}
            js_scores.append(js)

        if not js_scores:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={"per_column": col_stats},
                recommendations=[],
            )

        mean_js = float(np.mean(js_scores))
        # Very low divergence (<0.05) means the output is almost identical to input.
        very_low = [c for c, v in col_stats.items() if v["js_divergence"] < 0.05]
        passed = len(very_low) == 0

        recommendations: list[str] = []
        if very_low:
            recommendations.append(
                f"Columns {very_low} have near-identical distributions to the original "
                f"(JS divergence < 0.05).  Increase noise or generalisation."
            )

        # Score: inverted so low JS (high leakage risk) maps to high score.
        score = round(1.0 - min(mean_js / 0.5, 1.0), 4)

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=score,
            details={
                "per_column": col_stats,
                "mean_js_divergence": round(mean_js, 4),
                "low_divergence_columns": very_low,
            },
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Attack 6: Null Pattern Linkage
# ---------------------------------------------------------------------------


class Attack_NullPatternLinkage(BaseAttack):
    """Check if null patterns across columns create unique signatures.

    The pattern of which cells are null for each row is encoded as a boolean
    tuple.  If many rows have a unique null pattern, an adversary knowing
    which fields are missing for a specific individual can narrow the search
    space dramatically.

    Severity: MEDIUM
    """

    name = "Attack_NullPatternLinkage"
    severity = "MEDIUM"
    applies_to_synthetic = True

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        df = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        n_total = len(df)
        if n_total == 0:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={},
                recommendations=[],
            )

        null_matrix = df.isnull()
        cols_with_nulls = null_matrix.columns[null_matrix.any()].tolist()

        if not cols_with_nulls:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={"columns_with_nulls": 0, "unique_null_patterns": 0},
                recommendations=[],
            )

        null_patterns = null_matrix[cols_with_nulls].apply(
            lambda row: tuple(row), axis=1
        )
        pattern_counts = null_patterns.value_counts()
        unique_patterns = int((pattern_counts == 1).sum())
        unique_rate = unique_patterns / n_total

        passed = unique_rate < 0.05
        recommendations: list[str] = []
        if not passed:
            recommendations.append(
                f"{unique_rate:.1%} of rows have unique null patterns across "
                f"{len(cols_with_nulls)} column(s).  Impute rare null combinations "
                f"or suppress sparse columns."
            )

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=round(unique_rate, 4),
            details={
                "columns_with_nulls": len(cols_with_nulls),
                "unique_null_patterns": unique_patterns,
                "unique_null_pattern_rate": round(unique_rate, 4),
                "total_distinct_patterns": int(len(pattern_counts)),
                "n_rows_evaluated": n_total,
            },
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Attack 7: Rare Combo Linkage
# ---------------------------------------------------------------------------


class Attack_RareComboLinkage(BaseAttack):
    """Find rare value combinations that violate k-anonymity.

    All pairwise combinations of ``qi_columns`` present in the DataFrame are
    checked.  Any combination whose smallest group is below ``k_target``
    represents a k-anonymity violation and a potential re-identification
    vector.

    Severity: HIGH
    """

    name = "Attack_RareComboLinkage"
    severity = "HIGH"
    applies_to_synthetic = True

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        from itertools import combinations

        qi_cols = _present_cols(df_anon, self.config.qi_columns)
        if len(qi_cols) < 2:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                skipped=True,
                skip_reason=(
                    f"Need at least 2 qi_columns in DataFrame; found {len(qi_cols)}."
                ),
            )

        df = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        n_total = len(df)
        violation_summary: dict[str, dict[str, Any]] = {}
        max_violation_rate = 0.0

        for col_a, col_b in combinations(qi_cols, 2):
            group_sizes = df.groupby([col_a, col_b], dropna=False).size()
            below_k = group_sizes[group_sizes < self.config.k_target]
            if len(below_k) == 0:
                continue
            n_affected = int(below_k.sum())
            violation_rate = n_affected / n_total if n_total else 0.0
            key = f"{col_a}__{col_b}"
            violation_summary[key] = {
                "n_groups_below_k": int(len(below_k)),
                "n_records_affected": n_affected,
                "violation_rate": round(violation_rate, 4),
                "min_group_size": int(below_k.min()),
            }
            if violation_rate > max_violation_rate:
                max_violation_rate = violation_rate

        passed = len(violation_summary) == 0
        recommendations: list[str] = []
        if not passed:
            recommendations.append(
                f"{len(violation_summary)} column pair(s) violate "
                f"k={self.config.k_target} anonymity.  Apply generalisation or "
                f"suppression to those QI pairs."
            )

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=round(max_violation_rate, 4),
            details={
                "k_target": self.config.k_target,
                "violating_pairs": violation_summary,
                "n_violating_pairs": len(violation_summary),
                "n_rows_evaluated": n_total,
            },
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Attack 8: Numeric Precision
# ---------------------------------------------------------------------------


class Attack_NumericPrecision(BaseAttack):
    """Check if numeric values retain too many decimal places.

    Excessive decimal precision can act as a fingerprint: if the original
    value is stored with many significant digits and the anonymised version
    retains those digits exactly, the value is essentially unchanged and
    potentially re-identifiable.

    Operates on ``numeric_targets`` (falls back to all numeric columns).

    Severity: MEDIUM
    """

    name = "Attack_NumericPrecision"
    severity = "MEDIUM"
    applies_to_synthetic = False  # most relevant for anonymised data

    _MAX_SAFE_DECIMALS = 2

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        num_cols = _present_cols(df_anon, self.config.numeric_targets)
        if not num_cols:
            num_cols = df_anon.select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={"numeric_columns_checked": 0},
                recommendations=[],
            )

        df = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        col_stats: dict[str, dict[str, Any]] = {}
        flagged: list[str] = []

        for col in num_cols:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            str_vals = vals.astype(str)
            decimal_places = str_vals.apply(
                lambda v: len(v.split(".")[1].rstrip("0")) if "." in v else 0
            )
            median_dp = float(decimal_places.median())
            max_dp = int(decimal_places.max())
            pct_excessive = float(
                (decimal_places > self._MAX_SAFE_DECIMALS).mean()
            )
            col_stats[col] = {
                "median_decimal_places": round(median_dp, 1),
                "max_decimal_places": max_dp,
                "pct_exceeding_threshold": round(pct_excessive, 4),
                "threshold": self._MAX_SAFE_DECIMALS,
            }
            if pct_excessive > 0.1:
                flagged.append(col)

        passed = len(flagged) == 0
        score = len(flagged) / max(len(num_cols), 1)

        recommendations: list[str] = []
        if not passed:
            recommendations.append(
                f"Columns {flagged} retain more than {self._MAX_SAFE_DECIMALS} decimal "
                f"places in >10% of rows.  Round to {self._MAX_SAFE_DECIMALS} d.p. or "
                f"apply multiplicative noise before rounding."
            )

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=round(score, 4),
            details={
                "per_column": col_stats,
                "flagged_columns": flagged,
                "max_safe_decimals": self._MAX_SAFE_DECIMALS,
            },
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Attack 9: Numeric Ratio
# ---------------------------------------------------------------------------


class Attack_NumericRatio(BaseAttack):
    """Check if ratios between numeric columns are preserved.

    If the ratio between two numeric columns is nearly constant in the
    anonymised data (low coefficient of variation), an adversary who knows
    one value can reconstruct the other.  This attack flags column pairs
    whose ratio distribution has a coefficient of variation below a threshold.

    Requires ``df_orig`` for context; can run with ``df_anon`` alone to
    detect near-constant ratios in the output.

    Severity: MEDIUM
    """

    name = "Attack_NumericRatio"
    severity = "MEDIUM"
    applies_to_synthetic = False

    # Ratio CV below this value is considered suspiciously constant.
    _LOW_CV_THRESHOLD = 0.05

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        if df_orig is None:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                skipped=True,
                skip_reason="df_orig is required for ratio comparison.",
            )

        from itertools import combinations

        num_cols = _present_cols(df_anon, self.config.numeric_targets)
        if len(num_cols) < 2:
            num_cols = df_anon.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={"numeric_columns_checked": len(num_cols)},
                recommendations=[],
            )

        # Restrict to columns present in both DataFrames.
        num_cols = [c for c in num_cols if c in df_orig.columns]

        df_a = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        df_o = _sample(df_orig, self.config.max_sample, self.config.random_seed)

        ratio_stats: dict[str, dict[str, Any]] = {}

        for col_a, col_b in combinations(num_cols, 2):
            orig_b = df_o[col_b].replace(0, np.nan)
            orig_ratio = (df_o[col_a] / orig_b).dropna()

            anon_b = df_a[col_b].replace(0, np.nan)
            anon_ratio = (df_a[col_a] / anon_b).dropna()

            if len(orig_ratio) < 10 or len(anon_ratio) < 10:
                continue

            orig_mean = float(orig_ratio.mean())
            anon_mean = float(anon_ratio.mean())
            anon_std = float(anon_ratio.std())

            diff_in_mean = abs(orig_mean - anon_mean)
            ref_mean = max(abs(orig_mean), 1e-9)
            relative_diff = diff_in_mean / ref_mean

            # Low CV of the anonymised ratio signals a near-constant relationship.
            anon_cv = anon_std / max(abs(anon_mean), 1e-9)

            key = f"{col_a}__{col_b}"
            ratio_stats[key] = {
                "orig_ratio_mean": round(orig_mean, 6),
                "anon_ratio_mean": round(anon_mean, 6),
                "relative_mean_diff": round(relative_diff, 4),
                "anon_ratio_cv": round(anon_cv, 4),
                "flagged": anon_cv < self._LOW_CV_THRESHOLD,
            }

        flagged = [k for k, v in ratio_stats.items() if v["flagged"]]
        passed = len(flagged) == 0
        score = len(flagged) / max(len(ratio_stats), 1) if ratio_stats else 0.0

        recommendations: list[str] = []
        if not passed:
            recommendations.append(
                f"{len(flagged)} column pair(s) have near-constant ratios (CV < "
                f"{self._LOW_CV_THRESHOLD}): {flagged}.  Apply independent noise to "
                f"each column to break ratio preservation."
            )

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=round(score, 4),
            details={
                "ratio_pairs": ratio_stats,
                "flagged_pairs": flagged,
                "low_cv_threshold": self._LOW_CV_THRESHOLD,
            },
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Attack 10: Compound Entity
# ---------------------------------------------------------------------------


class Attack_CompoundEntity(BaseAttack):
    """Check if compound attributes create unique entity fingerprints.

    Even when each quasi-identifier column is individually generalised, the
    combination of all QI columns can produce a high joint cardinality that
    acts as a unique entity fingerprint.  This attack measures the joint
    uniqueness rate and compares it against what would be expected under
    column independence.

    Operates on ``qi_columns``.

    Severity: HIGH
    """

    name = "Attack_CompoundEntity"
    severity = "HIGH"
    applies_to_synthetic = True

    def run(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> AttackResult:
        qi_cols = _present_cols(df_anon, self.config.qi_columns)
        if len(qi_cols) < 2:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                skipped=True,
                skip_reason=(
                    f"Need at least 2 qi_columns; found {len(qi_cols)}."
                ),
            )

        df = _sample(df_anon, self.config.max_sample, self.config.random_seed)
        n_total = len(df)
        if n_total == 0:
            return AttackResult(
                name=self.name,
                passed=True,
                severity=self.severity,
                score=0.0,
                details={},
                recommendations=[],
            )

        individual_card: dict[str, int] = {
            col: int(df[col].nunique(dropna=False)) for col in qi_cols
        }

        joint_unique = int(
            df[qi_cols].astype(str).apply(tuple, axis=1).nunique()
        )
        joint_unique_rate = joint_unique / n_total

        # Expected cardinality under column independence (product of marginals).
        expected_joint = min(
            int(np.prod([v for v in individual_card.values()])),
            n_total,
        )
        amplification = joint_unique / max(expected_joint, 1)

        passed = joint_unique_rate < 0.05

        recommendations: list[str] = []
        if not passed:
            recommendations.append(
                f"{joint_unique_rate:.1%} of records are jointly unique across all "
                f"QI columns.  Reduce individual column cardinality via top-N grouping "
                f"or suppress rare cross-column combinations."
            )
        if amplification > 2.0:
            recommendations.append(
                f"Joint cardinality ({joint_unique}) is {amplification:.1f}x higher "
                f"than expected under independence.  Strong compound entity fingerprints "
                f"may persist despite individual-column anonymisation."
            )

        return AttackResult(
            name=self.name,
            passed=passed,
            severity=self.severity,
            score=round(joint_unique_rate, 4),
            details={
                "individual_cardinalities": individual_card,
                "joint_unique_records": joint_unique,
                "joint_unique_rate": round(joint_unique_rate, 4),
                "expected_joint_cardinality": expected_joint,
                "amplification_factor": round(amplification, 2),
                "n_rows_evaluated": n_total,
            },
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_DEFAULT_ATTACK_CLASSES: list[Type[BaseAttack]] = [
    Attack_Uniqueness,
    Attack_TemporalLinkage,
    Attack_Fingerprints,
    Attack_OutlierReidentification,
    Attack_DistributionSkew,
    Attack_NullPatternLinkage,
    Attack_RareComboLinkage,
    Attack_NumericPrecision,
    Attack_NumericRatio,
    Attack_CompoundEntity,
]

_SEVERITY_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}


class RedTeamRunner:
    """Orchestrates all registered attacks and emits summary reports.

    By default all 10 built-in attacks are registered.  Additional attacks
    can be appended via :meth:`register_attack`.

    Args:
        config: :class:`RedTeamConfig` instance driving all attacks.
    """

    def __init__(self, config: RedTeamConfig) -> None:
        self.config = config
        self._loader = DataLoader()
        self._attack_classes: list[Type[BaseAttack]] = list(_DEFAULT_ATTACK_CLASSES)

    def register_attack(self, attack_class: Type[BaseAttack]) -> None:
        """Append a custom attack class to the execution list.

        Args:
            attack_class: A concrete subclass of :class:`BaseAttack`.

        Raises:
            TypeError: If ``attack_class`` is not a subclass of
                :class:`BaseAttack`.
        """
        if not (
            isinstance(attack_class, type) and issubclass(attack_class, BaseAttack)
        ):
            raise TypeError(
                f"attack_class must be a subclass of BaseAttack, "
                f"got {attack_class!r}."
            )
        self._attack_classes.append(attack_class)
        logger.debug("Registered attack: %s", attack_class.name)

    def run_all(
        self,
        df_anon: pd.DataFrame,
        df_orig: pd.DataFrame | None = None,
    ) -> list[AttackResult]:
        """Execute every registered attack and return the results.

        Attacks are run sequentially.  Errors within individual attacks are
        caught and returned as failed results (see :meth:`BaseAttack.execute`).

        Args:
            df_anon: Anonymised or synthetic DataFrame under evaluation.
            df_orig: Original (real) DataFrame.  Pass ``None`` when not
                available; diff-based attacks will be skipped automatically.

        Returns:
            List of :class:`AttackResult` instances, one per registered
            attack, in registration order.
        """
        results: list[AttackResult] = []
        logger.info(
            "RedTeamRunner: running %d attacks (mode=%s, profile=%s).",
            len(self._attack_classes),
            self.config.mode,
            self.config.profile,
        )
        for attack_cls in self._attack_classes:
            attack = attack_cls(self.config, self._loader)
            logger.debug("Executing %s ...", attack_cls.name)
            result = attack.execute(df_anon, df_orig=df_orig)
            results.append(result)
            status = (
                "SKIPPED"
                if result.skipped
                else ("PASS" if result.passed else "FAIL")
            )
            logger.info(
                "  %-35s  [%s]  severity=%-8s  score=%.4f  (%.2fs)",
                result.name,
                status,
                result.severity,
                result.score,
                result.elapsed_seconds,
            )
        return results

    def generate_report(self, results: list[AttackResult]) -> str:
        """Produce a Markdown summary report from a list of attack results.

        Args:
            results: Output of :meth:`run_all`.

        Returns:
            A Markdown-formatted string suitable for writing to a file or
            printing to the terminal.
        """
        total = len(results)
        skipped = sum(1 for r in results if r.skipped)
        executed = total - skipped
        passed = sum(1 for r in results if not r.skipped and r.passed)
        failed = executed - passed

        lines: list[str] = [
            "# Red-Team Adversarial Attack Report",
            "",
            (
                f"**Profile:** {self.config.profile}  |  "
                f"**Mode:** {self.config.mode}  |  "
                f"**k-target:** {self.config.k_target}"
            ),
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total attacks | {total} |",
            f"| Executed | {executed} |",
            f"| Skipped | {skipped} |",
            f"| Passed | {passed} |",
            f"| Failed | {failed} |",
            "",
        ]

        # Sort by severity then by pass/fail (failures first within each severity).
        sorted_results = sorted(
            results,
            key=lambda r: (
                _SEVERITY_ORDER.get(r.severity, 99),
                0 if (not r.passed and not r.skipped) else 1,
            ),
        )

        lines += [
            "## Results",
            "",
            "| Attack | Severity | Status | Score | Time (s) |",
            "|--------|----------|--------|-------|----------|",
        ]
        for r in sorted_results:
            if r.skipped:
                status_icon = "SKIP"
            elif r.passed:
                status_icon = "PASS"
            else:
                status_icon = "FAIL"
            lines.append(
                f"| {r.name} | {r.severity} | {status_icon} | "
                f"{r.score:.4f} | {r.elapsed_seconds:.2f} |"
            )

        lines.append("")

        # Detailed findings for failed attacks only.
        failures = [r for r in results if not r.passed and not r.skipped]
        if failures:
            lines += ["## Findings and Recommendations", ""]
            for r in failures:
                lines += [
                    f"### {r.name}  ({r.severity})",
                    "",
                    f"**Score:** {r.score:.4f}",
                    "",
                ]
                if r.recommendations:
                    lines += ["**Recommendations:**", ""]
                    for rec in r.recommendations:
                        lines.append(f"- {rec}")
                    lines.append("")
                if r.details:
                    lines += ["**Details:**", ""]
                    lines.append("```")
                    for k, v in r.details.items():
                        lines.append(f"{k}: {v}")
                    lines.append("```")
                    lines.append("")
        else:
            lines += [
                "## Findings and Recommendations",
                "",
                "No failures detected.  All executed attacks passed.",
                "",
            ]

        skipped_results = [r for r in results if r.skipped]
        if skipped_results:
            lines += ["## Skipped Attacks", ""]
            for r in skipped_results:
                lines.append(f"- **{r.name}**: {r.skip_reason}")
            lines.append("")

        return "\n".join(lines)
