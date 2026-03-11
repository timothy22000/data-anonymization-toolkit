"""
reporting.quality_report
========================

Generates Markdown reports from structured validation result objects.

Three report types are supported:

``generate_quality_report``
    Summarises a :class:`~validation.quality.QualityReport` with a composite
    score, per-column marginal results, correlation fidelity, and cross-tab
    divergence.

``generate_privacy_report``
    Summarises membership-inference AUC, distance-to-closest-record score,
    new-row rate, and nearest-neighbour distance statistics.

``generate_red_team_report``
    Summarises adversarial attack results with a severity table, per-failure
    details, and actionable recommendations.

Typical usage
-------------
::

    from reporting.quality_report import QualityReportGenerator
    from validation.quality import run_quality_checks

    gen = QualityReportGenerator(output_dir="reports")

    quality_report = run_quality_checks(real_df, synth_df, config)
    path = gen.generate_quality_report(quality_report)

    path = gen.generate_privacy_report(privacy_results)
    path = gen.generate_red_team_report(attack_results)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds used for pass / fail assessments
# ---------------------------------------------------------------------------

_MIA_AUC_THRESHOLD = 0.55
_DCR_THRESHOLD = 0.50
_NEW_ROW_RATE_THRESHOLD = 0.99
_QUALITY_PASS_THRESHOLD = 0.70
_MARGINAL_P_THRESHOLD = 0.05
_CROSS_TAB_PASS_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# QualityReportGenerator
# ---------------------------------------------------------------------------


class QualityReportGenerator:
    """Generate Markdown validation reports from structured result objects.

    Args:
        output_dir: Directory where reports will be written.
            Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Quality report
    # ------------------------------------------------------------------

    def generate_quality_report(
        self,
        quality_report: Any,
        output_path: str | None = None,
    ) -> str:
        """Generate a statistical quality report from a QualityReport object.

        The report includes the composite overall score, a per-column table
        of marginal distribution test results, a correlation fidelity
        assessment, cross-tab Jensen-Shannon divergences, and an overall
        pass/fail verdict.

        Args:
            quality_report: A :class:`~validation.quality.QualityReport`
                instance (or any object with matching attributes).
            output_path: Optional explicit output file path.  When ``None``
                a timestamped file is created inside ``output_dir``.

        Returns:
            Absolute path to the written Markdown file as a string.
        """
        out_path = self._resolve_path(output_path, "quality_report")

        overall = float(quality_report.overall_score)
        details: dict[str, Any] = quality_report.details or {}

        passed = overall >= _QUALITY_PASS_THRESHOLD
        verdict = "PASS" if passed else "FAIL"

        lines: list[str] = []
        lines += _header("Statistical Quality Report")
        lines += [
            f"**Overall Score:** {overall:.3f}  ",
            f"**Assessment:** {verdict}  ",
            f"**Threshold:** {_QUALITY_PASS_THRESHOLD:.2f}",
            "",
        ]

        if details:
            lines.append("### Component Scores")
            lines.append("")
            lines.append("| Component | Score |")
            lines.append("|-----------|-------|")
            for key in ("marginal_score", "correlation_score", "cross_tab_score"):
                if key in details:
                    label = key.replace("_", " ").title()
                    lines.append(f"| {label} | {details[key]:.3f} |")
            if "n_real" in details and "n_synth" in details:
                lines.append("")
                lines.append(
                    f"_Dataset sizes: real = {details['n_real']:,}  "
                    f"/ synthetic = {details['n_synth']:,}_"
                )
            lines.append("")

        # Marginal distribution results
        marginal_scores: dict[str, float] = quality_report.marginal_scores or {}
        if marginal_scores:
            lines.append("### Marginal Distribution Results")
            lines.append("")
            lines.append(
                "_p-value from chi-square (categorical) or KS test (numerical). "
                f"p >= {_MARGINAL_P_THRESHOLD} = PASS._"
            )
            lines.append("")
            lines.append("| Column | p-value | Assessment |")
            lines.append("|--------|---------|------------|")
            for col, p in sorted(marginal_scores.items()):
                assessment = "PASS" if p >= _MARGINAL_P_THRESHOLD else "FAIL"
                lines.append(f"| `{col}` | {p:.4f} | {assessment} |")
            lines.append("")

        # Correlation fidelity
        corr_fidelity = float(quality_report.correlation_fidelity)
        lines.append("### Correlation Fidelity")
        lines.append("")
        lines.append(
            f"Mean absolute difference between real and synthetic correlation matrices: "
            f"**{corr_fidelity:.4f}**"
        )
        lines.append(
            "_0.0 = identical correlations.  "
            "Values below 0.05 indicate very high fidelity._"
        )
        lines.append("")

        # Cross-tab divergences
        cross_tab: dict[str, float] = quality_report.cross_tab_divergences or {}
        if cross_tab:
            lines.append("### Cross-Tab Jensen-Shannon Divergences")
            lines.append("")
            lines.append(
                f"_JS divergence in [0, 1].  "
                f"Values below {_CROSS_TAB_PASS_THRESHOLD} = PASS._"
            )
            lines.append("")
            lines.append("| Column Pair | JS Divergence | Assessment |")
            lines.append("|-------------|---------------|------------|")
            for pair_key, js in sorted(cross_tab.items()):
                col_a, col_b = pair_key.split("__", 1) if "__" in pair_key else (pair_key, "")
                pair_label = f"`{col_a}` x `{col_b}`" if col_b else f"`{col_a}`"
                if js != js:  # NaN check
                    lines.append(f"| {pair_label} | N/A | N/A |")
                else:
                    assessment = "PASS" if js < _CROSS_TAB_PASS_THRESHOLD else "FAIL"
                    lines.append(f"| {pair_label} | {js:.4f} | {assessment} |")
            lines.append("")

        # Final verdict
        lines += _verdict_block(passed)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Quality report written to %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Privacy report
    # ------------------------------------------------------------------

    def generate_privacy_report(
        self,
        privacy_results: dict[str, Any],
        output_path: str | None = None,
    ) -> str:
        """Generate a privacy metric summary report.

        The report covers membership-inference AUC, distance-to-closest-record
        (DCR) score, new-row rate, and nearest-neighbour distance statistics.

        Args:
            privacy_results: Dictionary of privacy metric values.  Expected
                keys (all optional; missing keys are shown as ``N/A``):

                * ``mia_auc`` : float, membership-inference AUC score.
                * ``dcr`` : float, distance-to-closest-record score.
                * ``new_row_rate`` : float, fraction of novel synthetic rows.
                * ``nn_distances`` : array-like of nearest-neighbour distances,
                  or a dict with keys ``mean``, ``median``, ``p5``, ``p95``.

            output_path: Optional explicit output file path.

        Returns:
            Absolute path to the written Markdown file as a string.
        """
        out_path = self._resolve_path(output_path, "privacy_report")

        mia_auc: float | None = privacy_results.get("mia_auc")
        dcr: float | None = privacy_results.get("dcr")
        new_row_rate: float | None = privacy_results.get("new_row_rate")
        nn_distances = privacy_results.get("nn_distances")

        lines: list[str] = []
        lines += _header("Privacy Metrics Report")

        # MIA AUC
        lines.append("### Membership Inference Attack (MIA) AUC")
        lines.append("")
        lines.append(
            "_Measures how well an adversary can distinguish real from synthetic "
            "records.  AUC close to 0.5 = good privacy; AUC close to 1.0 = poor privacy._"
        )
        lines.append("")
        if mia_auc is not None:
            mia_passed = mia_auc <= _MIA_AUC_THRESHOLD
            lines.append(f"| Metric | Value | Threshold | Assessment |")
            lines.append(f"|--------|-------|-----------|------------|")
            lines.append(
                f"| MIA AUC | {mia_auc:.4f} | <= {_MIA_AUC_THRESHOLD} "
                f"| {'PASS' if mia_passed else 'FAIL'} |"
            )
        else:
            lines.append("_MIA AUC not computed._")
        lines.append("")

        # DCR
        lines.append("### Distance to Closest Record (DCR)")
        lines.append("")
        lines.append(
            "_Fraction of synthetic records whose nearest real neighbour is "
            "sufficiently distant.  Higher scores indicate more generative diversity._"
        )
        lines.append("")
        if dcr is not None:
            dcr_passed = dcr >= _DCR_THRESHOLD
            lines.append("| Metric | Value | Threshold | Assessment |")
            lines.append("|--------|-------|-----------|------------|")
            lines.append(
                f"| DCR Score | {dcr:.4f} | >= {_DCR_THRESHOLD} "
                f"| {'PASS' if dcr_passed else 'FAIL'} |"
            )
        else:
            lines.append("_DCR score not computed._")
        lines.append("")

        # New-row rate
        lines.append("### New Row Rate")
        lines.append("")
        lines.append(
            "_Fraction of synthetic rows that are entirely novel (no exact "
            "match in the real dataset).  Should be close to 1.0._"
        )
        lines.append("")
        if new_row_rate is not None:
            nrr_passed = new_row_rate >= _NEW_ROW_RATE_THRESHOLD
            lines.append("| Metric | Value | Threshold | Assessment |")
            lines.append("|--------|-------|-----------|------------|")
            lines.append(
                f"| New Row Rate | {new_row_rate:.4f} | >= {_NEW_ROW_RATE_THRESHOLD} "
                f"| {'PASS' if nrr_passed else 'FAIL'} |"
            )
        else:
            lines.append("_New row rate not computed._")
        lines.append("")

        # Nearest-neighbour distances
        lines.append("### Nearest-Neighbour Distance Distribution")
        lines.append("")
        lines.append(
            "_Distribution of Euclidean distances from each synthetic record to "
            "its closest real record (after z-score normalisation).  "
            "A distribution far from zero indicates diverse, non-memorised output._"
        )
        lines.append("")
        if nn_distances is not None:
            nn_stats = _summarise_distances(nn_distances)
            if nn_stats:
                lines.append("| Statistic | Value |")
                lines.append("|-----------|-------|")
                for stat_name, stat_val in nn_stats.items():
                    lines.append(f"| {stat_name} | {stat_val:.4f} |")
        else:
            lines.append("_Nearest-neighbour distances not computed._")
        lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Privacy report written to %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Red-team report
    # ------------------------------------------------------------------

    def generate_red_team_report(
        self,
        attack_results: list[Any],
        output_path: str | None = None,
    ) -> str:
        """Generate a red-team adversarial testing report.

        The report contains a summary table of all attacks with their
        severity and pass/fail status, detailed information for each
        failed attack, and a consolidated list of recommendations.

        Args:
            attack_results: List of :class:`~validation.red_team.AttackResult`
                instances (or any objects with the attributes ``attack_name``,
                ``severity``, ``passed``, ``score``, ``details``, and
                ``recommendation``).
            output_path: Optional explicit output file path.

        Returns:
            Absolute path to the written Markdown file as a string.
        """
        out_path = self._resolve_path(output_path, "red_team_report")

        lines: list[str] = []
        lines += _header("Red Team Attack Report")

        total = len(attack_results)
        passed_count = sum(1 for r in attack_results if getattr(r, "passed", False))
        failed_count = total - passed_count

        lines += [
            f"**Attacks run:** {total}  ",
            f"**Passed:** {passed_count}  ",
            f"**Failed:** {failed_count}",
            "",
        ]

        # Summary table
        lines.append("### Attack Summary")
        lines.append("")
        lines.append("| Attack | Severity | Score | Status |")
        lines.append("|--------|----------|-------|--------|")

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_results = sorted(
            attack_results,
            key=lambda r: (
                severity_order.get(
                    str(getattr(r, "severity", "low")).lower(), 99
                ),
                not getattr(r, "passed", True),
            ),
        )

        for result in sorted_results:
            name = getattr(result, "attack_name", "Unknown")
            severity = str(getattr(result, "severity", "N/A")).capitalize()
            score = getattr(result, "score", None)
            passed = getattr(result, "passed", None)
            score_str = f"{score:.4f}" if score is not None else "N/A"
            status_str = "PASS" if passed else "FAIL"
            lines.append(f"| {name} | {severity} | {score_str} | {status_str} |")

        lines.append("")

        # Failed attack details
        failed = [r for r in sorted_results if not getattr(r, "passed", True)]
        if failed:
            lines.append("### Failed Attack Details")
            lines.append("")
            for result in failed:
                name = getattr(result, "attack_name", "Unknown")
                severity = str(getattr(result, "severity", "N/A")).capitalize()
                score = getattr(result, "score", None)
                details = getattr(result, "details", {}) or {}
                recommendation = getattr(result, "recommendation", "") or ""

                lines.append(f"#### {name}")
                lines.append("")
                lines.append(f"- **Severity:** {severity}")
                if score is not None:
                    lines.append(f"- **Score:** {score:.4f}")

                if details:
                    lines.append("- **Details:**")
                    for key, val in details.items():
                        lines.append(f"  - `{key}`: {val}")

                if recommendation:
                    lines.append(f"- **Recommendation:** {recommendation}")

                lines.append("")
        else:
            lines.append("_All attacks passed.  No failed attack details to report._")
            lines.append("")

        # Consolidated recommendations
        recommendations = [
            getattr(r, "recommendation", "")
            for r in attack_results
            if not getattr(r, "passed", True)
            and getattr(r, "recommendation", "")
        ]
        unique_recs = list(dict.fromkeys(recommendations))

        if unique_recs:
            lines.append("### Recommendations")
            lines.append("")
            lines.append(
                "_The following actions are recommended to address failed attacks:_"
            )
            lines.append("")
            for rec in unique_recs:
                lines.append(f"- {rec}")
            lines.append("")

        overall_passed = failed_count == 0
        lines += _verdict_block(overall_passed)

        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Red team report written to %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, output_path: str | None, default_stem: str) -> Path:
        """Return a concrete output Path, creating parent directories as needed.

        Args:
            output_path: Caller-supplied path string, or ``None``.
            default_stem: Basename stem used when *output_path* is ``None``.

        Returns:
            Resolved :class:`~pathlib.Path` ending in ``.md``.
        """
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{default_stem}_{timestamp}.md"
        return self._output_dir / filename


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


def _header(title: str) -> list[str]:
    """Return a standard Markdown report header block.

    Args:
        title: Human-readable title for the report.

    Returns:
        List of Markdown lines.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return [
        f"# {title}",
        "",
        f"_Generated: {timestamp}_",
        "",
        "---",
        "",
    ]


def _verdict_block(passed: bool) -> list[str]:
    """Return a bolded overall pass/fail block.

    Args:
        passed: ``True`` if the overall assessment is a pass.

    Returns:
        List of Markdown lines.
    """
    verdict = "PASS" if passed else "FAIL"
    message = (
        "All checks passed.  The dataset meets the required privacy and quality standards."
        if passed
        else (
            "One or more checks failed.  Review the sections above and apply the "
            "recommended remediation steps before releasing the dataset."
        )
    )
    return [
        "---",
        "",
        f"## Overall Assessment: {verdict}",
        "",
        message,
        "",
    ]


def _summarise_distances(nn_distances: Any) -> dict[str, float]:
    """Extract summary statistics from a nearest-neighbour distance input.

    Accepts either an array-like (numpy array or list) or a pre-computed
    summary dict with keys ``mean``, ``median``, ``p5``, ``p95``.

    Args:
        nn_distances: Distance values or summary statistics dict.

    Returns:
        Mapping of statistic name to float value.  Empty dict on error.
    """
    if isinstance(nn_distances, dict):
        return {
            k: float(v)
            for k, v in nn_distances.items()
            if v is not None
        }

    try:
        import numpy as np

        arr = np.asarray(nn_distances, dtype=float)
        if arr.size == 0:
            return {}
        return {
            "Mean": float(arr.mean()),
            "Median": float(np.median(arr)),
            "5th Percentile": float(np.percentile(arr, 5)),
            "95th Percentile": float(np.percentile(arr, 95)),
        }
    except Exception as exc:
        logger.warning("Could not summarise nearest-neighbour distances: %s", exc)
        return {}
