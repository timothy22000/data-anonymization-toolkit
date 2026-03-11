"""
reporting
=========

Output generators for privacy and quality assessment reports.

This package produces human-readable artefacts from the structured result
objects emitted by the ``validation`` package.  Two generators are provided:

``PrivacyGuideGenerator``
    Produces a self-contained PDF (or PNG, or Markdown fallback) that
    explains each privacy-preserving technique employed by the pipeline,
    illustrated with diagrams drawn from ReportLab primitives.

``QualityReportGenerator``
    Produces Markdown reports for statistical quality results, privacy
    metric assessments, and red-team attack summaries.

Quick start
-----------
::

    from reporting import PrivacyGuideGenerator, QualityReportGenerator
    from validation.quality import run_quality_checks

    # Generate the privacy guide
    guide = PrivacyGuideGenerator(output_dir="reports")
    path = guide.generate_guide(title="Data Privacy Guide", output_format="pdf")

    # Generate a quality report
    report_gen = QualityReportGenerator(output_dir="reports")
    quality_report = run_quality_checks(real_df, synth_df, config)
    report_path = report_gen.generate_quality_report(quality_report)
"""

from .privacy_guide import PrivacyGuideGenerator
from .quality_report import QualityReportGenerator

__all__ = [
    "PrivacyGuideGenerator",
    "QualityReportGenerator",
]
