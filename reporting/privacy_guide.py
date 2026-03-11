"""
reporting.privacy_guide
=======================

Generates a self-contained privacy-technique guide as a PDF, PNG, or
Markdown file.  Each section explains one anonymization concept and is
illustrated with a diagram drawn from ReportLab primitives.

If ReportLab is not installed the generator falls back to a Markdown file
so that the rest of the pipeline never hard-fails on a missing optional
dependency.

Typical usage
-------------
::

    from reporting.privacy_guide import PrivacyGuideGenerator

    gen = PrivacyGuideGenerator(output_dir="reports")
    path = gen.generate_guide(title="Data Privacy Guide", output_format="pdf")
    print(f"Guide written to {path}")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional ReportLab import
# ---------------------------------------------------------------------------

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        HRFlowable,
        PageBreak,
    )
    from reportlab.graphics.shapes import (
        Drawing,
        Rect,
        String,
        Line,
        Arrow,
        Group,
    )
    from reportlab.graphics import renderPDF, renderPM
    from reportlab.graphics.charts.textlabels import Label

    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    REPORTLAB_AVAILABLE = False
    logger.warning(
        "ReportLab is not installed.  PrivacyGuideGenerator will fall back to "
        "Markdown output.  Install with: pip install reportlab"
    )

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

OutputFormat = Literal["pdf", "png", "md"]

# ---------------------------------------------------------------------------
# Colour palette (used only when ReportLab is available)
# ---------------------------------------------------------------------------

_PALETTE = {
    "header_bg": "#2C3E50",
    "header_fg": "#FFFFFF",
    "accent": "#2980B9",
    "light_bg": "#ECF0F1",
    "border": "#BDC3C7",
    "positive": "#27AE60",
    "negative": "#E74C3C",
    "arrow": "#7F8C8D",
    "text": "#2C3E50",
}


# ---------------------------------------------------------------------------
# Section content definitions
# ---------------------------------------------------------------------------

_SECTIONS: list[dict] = [
    {
        "title": "k-Anonymity",
        "body": (
            "k-Anonymity ensures that every combination of quasi-identifier "
            "values (such as age band, region, or gender) appears in at least "
            "k records.  This prevents an adversary from singling out an "
            "individual by matching against publicly available reference data.  "
            "Records in groups smaller than k are either suppressed or merged "
            "into broader categories until the threshold is met."
        ),
        "diagram": "k_anonymity",
    },
    {
        "title": "Noise Injection",
        "body": (
            "Numerical columns are perturbed by adding carefully calibrated "
            "random noise before release.  Two strategies are supported: "
            "multiplicative noise, which scales each value by a factor drawn "
            "from a Gaussian distribution, and additive Laplacian noise, which "
            "adds a zero-mean offset whose magnitude follows the heavier-tailed "
            "Laplace distribution.  Both approaches preserve aggregate statistics "
            "while making individual records harder to re-identify."
        ),
        "diagram": "noise_injection",
    },
    {
        "title": "Quasi-Identifier Generalisation",
        "body": (
            "Quasi-identifiers are columns that are not directly identifying but "
            "can become identifying in combination (e.g. age, postcode, "
            "occupation).  Generalisation replaces precise values with coarser "
            "representations: continuous values are binned into ranges, rare "
            "categorical values are collapsed into an 'Other' bucket, and "
            "numeric values are rounded to a specified granularity.  Precision "
            "is traded for privacy."
        ),
        "diagram": "generalisation",
    },
    {
        "title": "Synthetic Data Generation",
        "body": (
            "A generative model learns the statistical structure of the source "
            "dataset : marginal distributions, correlations, and conditional "
            "dependencies : and then samples entirely new records from that "
            "learned distribution.  Because synthetic records are statistically "
            "representative without corresponding to any real individual, they "
            "can be shared more freely than anonymized originals."
        ),
        "diagram": "synthetic_data",
    },
    {
        "title": "Red Team Testing",
        "body": (
            "Red team testing subjects the released dataset to a battery of "
            "simulated adversarial attacks : including membership inference, "
            "nearest-neighbour distance analysis, and duplicate-row detection : "
            "and measures how much private information each attack can recover.  "
            "Attacks that succeed beyond an acceptable threshold trigger a "
            "recommendation to strengthen the anonymization pipeline before "
            "release."
        ),
        "diagram": "red_team",
    },
]


# ---------------------------------------------------------------------------
# PrivacyGuideGenerator
# ---------------------------------------------------------------------------


class PrivacyGuideGenerator:
    """Generate a privacy-technique guide document.

    The guide describes each anonymization technique used in the pipeline,
    accompanied by illustrative diagrams.  Output format can be PDF, PNG,
    or Markdown (the last being the fallback when ReportLab is unavailable).

    Args:
        output_dir: Directory where the generated file will be written.
            Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_guide(
        self,
        title: str = "Data Privacy Guide",
        output_format: OutputFormat = "pdf",
    ) -> str:
        """Generate the privacy guide document.

        Args:
            title: Document title shown on the cover page.
            output_format: One of ``"pdf"``, ``"png"``, or ``"md"``.
                PNG renders each page as a raster image.  If ReportLab is
                not available, the format is silently overridden to ``"md"``.

        Returns:
            Absolute path to the generated file as a string.

        Raises:
            ValueError: If *output_format* is not one of the supported values.
        """
        if output_format not in {"pdf", "png", "md"}:
            raise ValueError(
                f"Unsupported output_format {output_format!r}. "
                "Choose from: pdf, png, md"
            )

        if not REPORTLAB_AVAILABLE and output_format in {"pdf", "png"}:
            logger.warning(
                "ReportLab unavailable : falling back to Markdown output."
            )
            output_format = "md"

        if output_format == "md":
            return self._generate_markdown(title)
        return self._generate_pdf(title, output_format)

    # ------------------------------------------------------------------
    # PDF / PNG generation (ReportLab path)
    # ------------------------------------------------------------------

    def _generate_pdf(self, title: str, output_format: OutputFormat) -> str:
        """Build and write the PDF (or PNG) guide using ReportLab."""
        stem = _safe_filename(title)
        out_path = self._output_dir / f"{stem}.{output_format}"

        styles = _build_styles()
        story = []

        # Cover / title
        story.append(Spacer(1, 3 * cm))
        story.append(Paragraph(title, styles["cover_title"]))
        story.append(Spacer(1, 0.5 * cm))
        story.append(
            Paragraph(
                "A conceptual overview of data privacy techniques",
                styles["cover_subtitle"],
            )
        )
        story.append(Spacer(1, 1 * cm))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor(_PALETTE["accent"])))
        story.append(PageBreak())

        for section in _SECTIONS:
            # Section heading
            story.append(Paragraph(section["title"], styles["section_heading"]))
            story.append(Spacer(1, 0.3 * cm))

            # Body text
            story.append(Paragraph(section["body"], styles["body_text"]))
            story.append(Spacer(1, 0.5 * cm))

            # Diagram
            drawing = _build_diagram(section["diagram"])
            story.append(drawing)
            story.append(Spacer(1, 0.5 * cm))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=1,
                    color=colors.HexColor(_PALETTE["border"]),
                )
            )
            story.append(Spacer(1, 0.8 * cm))

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=title,
            author="Data Anonymization Toolkit",
        )
        doc.build(story)

        if output_format == "png":
            # renderPM needs the PDF; re-render each page as PNG instead.
            # For simplicity we render the drawing objects directly to PNG.
            # Full multi-page PNG output requires pdf2image; here we produce
            # a single composite PNG of all diagrams.
            png_path = self._output_dir / f"{stem}.png"
            composite = _build_composite_drawing(_SECTIONS)
            renderPM.drawToFile(composite, str(png_path), fmt="PNG")
            logger.info("Privacy guide PNG written to %s", png_path)
            return str(png_path)

        logger.info("Privacy guide PDF written to %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Markdown fallback
    # ------------------------------------------------------------------

    def _generate_markdown(self, title: str) -> str:
        """Write a Markdown version of the privacy guide."""
        stem = _safe_filename(title)
        out_path = self._output_dir / f"{stem}.md"

        lines: list[str] = [f"# {title}\n"]
        lines.append(
            "_A conceptual overview of data privacy techniques_\n"
        )
        lines.append("---\n")

        for section in _SECTIONS:
            lines.append(f"## {section['title']}\n")
            lines.append(f"{section['body']}\n")
            lines.append(_markdown_diagram(section["diagram"]))
            lines.append("\n---\n")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Privacy guide Markdown written to %s", out_path)
        return str(out_path)


# ---------------------------------------------------------------------------
# ReportLab style sheet
# ---------------------------------------------------------------------------


def _build_styles() -> dict:
    """Construct a mapping of named ParagraphStyle objects."""
    base = getSampleStyleSheet()

    cover_title = ParagraphStyle(
        "cover_title",
        parent=base["Title"],
        fontSize=28,
        textColor=colors.HexColor(_PALETTE["header_bg"]),
        spaceAfter=12,
        alignment=1,  # centre
    )
    cover_subtitle = ParagraphStyle(
        "cover_subtitle",
        parent=base["Normal"],
        fontSize=14,
        textColor=colors.HexColor(_PALETTE["accent"]),
        alignment=1,
    )
    section_heading = ParagraphStyle(
        "section_heading",
        parent=base["Heading1"],
        fontSize=16,
        textColor=colors.HexColor(_PALETTE["header_bg"]),
        spaceBefore=6,
        spaceAfter=4,
        borderPad=4,
    )
    body_text = ParagraphStyle(
        "body_text",
        parent=base["Normal"],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor(_PALETTE["text"]),
    )

    return {
        "cover_title": cover_title,
        "cover_subtitle": cover_subtitle,
        "section_heading": section_heading,
        "body_text": body_text,
    }


# ---------------------------------------------------------------------------
# Diagram builders
# ---------------------------------------------------------------------------


def _build_diagram(diagram_key: str) -> "Drawing":
    """Dispatch to the appropriate diagram builder.

    Args:
        diagram_key: Internal identifier for the diagram type.

    Returns:
        A ReportLab ``Drawing`` instance sized to fit A4 page width.
    """
    builders = {
        "k_anonymity": _diagram_k_anonymity,
        "noise_injection": _diagram_noise_injection,
        "generalisation": _diagram_generalisation,
        "synthetic_data": _diagram_synthetic_data,
        "red_team": _diagram_red_team,
    }
    builder = builders.get(diagram_key, _diagram_placeholder)
    return builder()


def _diagram_k_anonymity() -> "Drawing":
    """Table diagram illustrating k-anonymity grouping (k=3)."""
    w, h = 480, 160
    d = Drawing(w, h)

    col_widths = [100, 80, 80, 80, 80]
    headers = ["Record", "Age Band", "Region", "Gender", "Group Size"]
    rows = [
        ["R1", "30-40", "North", "M", "3"],
        ["R2", "30-40", "North", "M", "3"],
        ["R3", "30-40", "North", "M", "3"],
        ["R4", "40-50", "South", "F", "3"],
        ["R5", "40-50", "South", "F", "3"],
        ["R6", "40-50", "South", "F", "3"],
    ]

    row_height = 20
    x_start = 10
    y_start = h - 30

    # Header row
    x = x_start
    for i, (header, cw) in enumerate(zip(headers, col_widths)):
        d.add(
            Rect(
                x, y_start - row_height, cw, row_height,
                fillColor=colors.HexColor(_PALETTE["header_bg"]),
                strokeColor=colors.white,
                strokeWidth=1,
            )
        )
        d.add(
            String(
                x + cw / 2, y_start - row_height + 5,
                header,
                fontSize=8,
                fillColor=colors.white,
                textAnchor="middle",
            )
        )
        x += cw

    # Data rows
    group_colours = [
        colors.HexColor("#D6EAF8"),
        colors.HexColor("#D5F5E3"),
    ]
    for row_idx, row in enumerate(rows):
        x = x_start
        y = y_start - (row_idx + 2) * row_height
        fill = group_colours[row_idx // 3]
        for col_idx, (cell, cw) in enumerate(zip(row, col_widths)):
            is_size_col = col_idx == 4
            d.add(
                Rect(
                    x, y, cw, row_height,
                    fillColor=colors.HexColor(_PALETTE["positive"]) if is_size_col else fill,
                    strokeColor=colors.HexColor(_PALETTE["border"]),
                    strokeWidth=0.5,
                )
            )
            d.add(
                String(
                    x + cw / 2, y + 5,
                    cell,
                    fontSize=8,
                    fillColor=colors.white if is_size_col else colors.HexColor(_PALETTE["text"]),
                    textAnchor="middle",
                )
            )
            x += cw

    # Legend
    d.add(
        String(
            x_start, 8,
            "Shading shows groups where every QI combination appears >= k times  (k = 3)",
            fontSize=7,
            fillColor=colors.HexColor(_PALETTE["arrow"]),
        )
    )
    return d


def _diagram_noise_injection() -> "Drawing":
    """Bar-chart-style diagram comparing original vs. noisy values."""
    w, h = 480, 160
    d = Drawing(w, h)

    labels = ["A", "B", "C", "D", "E"]
    originals = [100, 200, 150, 300, 250]
    # Simulated noisy values (deterministic for reproducibility)
    mult_noisy = [95, 214, 138, 312, 262]
    lap_noisy = [107, 188, 161, 287, 241]

    bar_group_w = 70
    bar_w = 16
    x_origin = 60
    y_origin = 25
    scale = 0.35

    max_val = max(originals) * scale + y_origin

    # Axis
    d.add(Line(x_origin, y_origin, x_origin, max_val + 20,
               strokeColor=colors.HexColor(_PALETTE["border"]), strokeWidth=1))
    d.add(Line(x_origin, y_origin, x_origin + len(labels) * bar_group_w + 20, y_origin,
               strokeColor=colors.HexColor(_PALETTE["border"]), strokeWidth=1))

    colour_orig = colors.HexColor(_PALETTE["accent"])
    colour_mult = colors.HexColor(_PALETTE["positive"])
    colour_lap = colors.HexColor(_PALETTE["negative"])

    for i, (lbl, orig, mult, lap) in enumerate(
        zip(labels, originals, mult_noisy, lap_noisy)
    ):
        x_group = x_origin + i * bar_group_w + 10

        for j, (val, col) in enumerate(
            [(orig, colour_orig), (mult, colour_mult), (lap, colour_lap)]
        ):
            bx = x_group + j * (bar_w + 2)
            bh = val * scale
            d.add(
                Rect(bx, y_origin, bar_w, bh,
                     fillColor=col, strokeColor=colors.white, strokeWidth=0.5)
            )

        d.add(String(x_group + bar_w + 1, y_origin - 10, lbl,
                     fontSize=8, textAnchor="middle",
                     fillColor=colors.HexColor(_PALETTE["text"])))

    # Legend
    legend_x = x_origin
    legend_y = h - 18
    for col, label in [
        (colour_orig, "Original"),
        (colour_mult, "Multiplicative noise"),
        (colour_lap, "Laplacian noise"),
    ]:
        d.add(Rect(legend_x, legend_y, 10, 10, fillColor=col, strokeColor=colors.white))
        d.add(String(legend_x + 13, legend_y + 2, label, fontSize=7,
                     fillColor=colors.HexColor(_PALETTE["text"])))
        legend_x += 120

    return d


def _diagram_generalisation() -> "Drawing":
    """Arrow diagram showing banding and rounding transformations."""
    w, h = 480, 140
    d = Drawing(w, h)

    examples = [
        ("Precise value", "Generalised value", "Method"),
        ("37", "30 - 40", "Band (width=10)"),
        ("0.8731", "0.87", "Round to 2 d.p."),
        ("Rural East", "Other", "Top-N (N=5)"),
        ("1200", "1000 - 1500", "Band (width=500)"),
    ]

    col_w = [130, 150, 130]
    x_start = 10
    row_h = 22

    for row_idx, (orig, gen, method) in enumerate(examples):
        y = h - (row_idx + 1) * row_h - 5
        x = x_start
        is_header = row_idx == 0
        for col_idx, (cell, cw) in enumerate(zip([orig, gen, method], col_w)):
            d.add(
                Rect(
                    x, y, cw, row_h,
                    fillColor=colors.HexColor(_PALETTE["header_bg"]) if is_header
                    else (colors.HexColor("#FEF9E7") if col_idx == 0
                          else colors.HexColor("#EAF4FB") if col_idx == 1
                          else colors.HexColor(_PALETTE["light_bg"])),
                    strokeColor=colors.HexColor(_PALETTE["border"]),
                    strokeWidth=0.5,
                )
            )
            d.add(
                String(
                    x + cw / 2, y + 6,
                    cell,
                    fontSize=8,
                    fillColor=colors.white if is_header else colors.HexColor(_PALETTE["text"]),
                    textAnchor="middle",
                )
            )
            # Arrow between first and second column
            if col_idx == 0 and not is_header:
                arrow_x = x + cw
                d.add(
                    Line(arrow_x, y + row_h / 2,
                         arrow_x + col_w[1] - 10, y + row_h / 2,
                         strokeColor=colors.HexColor(_PALETTE["accent"]),
                         strokeWidth=1.5)
                )
            x += cw

    return d


def _diagram_synthetic_data() -> "Drawing":
    """Flow diagram: real data -> model -> synthetic data."""
    w, h = 480, 120
    d = Drawing(w, h)

    box_w, box_h = 100, 50
    y_centre = h / 2 - box_h / 2
    centres_x = [40, 190, 340]
    labels = [["Real", "Dataset"], ["Generative", "Model"], ["Synthetic", "Dataset"]]
    fill_colours = [
        colors.HexColor(_PALETTE["accent"]),
        colors.HexColor(_PALETTE["header_bg"]),
        colors.HexColor(_PALETTE["positive"]),
    ]

    for i, (cx, lbl_lines, fill) in enumerate(
        zip(centres_x, labels, fill_colours)
    ):
        d.add(
            Rect(cx, y_centre, box_w, box_h,
                 fillColor=fill, strokeColor=colors.white,
                 strokeWidth=1.5, rx=6, ry=6)
        )
        for j, line in enumerate(lbl_lines):
            d.add(
                String(
                    cx + box_w / 2,
                    y_centre + box_h / 2 - (j - 0.5) * 12,
                    line,
                    fontSize=9,
                    fillColor=colors.white,
                    textAnchor="middle",
                )
            )

        if i < 2:
            arrow_x_start = cx + box_w + 2
            arrow_x_end = centres_x[i + 1] - 2
            mid_y = y_centre + box_h / 2
            d.add(
                Line(arrow_x_start, mid_y, arrow_x_end, mid_y,
                     strokeColor=colors.HexColor(_PALETTE["arrow"]),
                     strokeWidth=2)
            )
            # Arrow head
            d.add(
                Line(arrow_x_end - 8, mid_y - 5, arrow_x_end, mid_y,
                     strokeColor=colors.HexColor(_PALETTE["arrow"]), strokeWidth=2)
            )
            d.add(
                Line(arrow_x_end - 8, mid_y + 5, arrow_x_end, mid_y,
                     strokeColor=colors.HexColor(_PALETTE["arrow"]), strokeWidth=2)
            )

    # Step labels below arrows
    step_labels = ["Learns distributions", "Samples new rows"]
    step_x = [140, 292]
    for sx, sl in zip(step_x, step_labels):
        d.add(
            String(sx, y_centre - 18, sl,
                   fontSize=7, textAnchor="middle",
                   fillColor=colors.HexColor(_PALETTE["arrow"]))
        )

    return d


def _diagram_red_team() -> "Drawing":
    """Diagram showing attack categories and pass/fail outcomes."""
    w, h = 480, 150
    d = Drawing(w, h)

    attacks = [
        ("Membership\nInference", "AUC < 0.55", True),
        ("Nearest\nNeighbour", "DCR > 0.5", True),
        ("Duplicate\nRow", "Rate < 1%", False),
        ("Attribute\nInference", "Accuracy < 60%", True),
    ]

    box_w, box_h = 90, 55
    spacing = 110
    y_box = h - box_h - 20

    for i, (name, criterion, passed) in enumerate(attacks):
        x = 10 + i * spacing
        fill = colors.HexColor(_PALETTE["positive"]) if passed else colors.HexColor(_PALETTE["negative"])

        d.add(
            Rect(x, y_box, box_w, box_h,
                 fillColor=fill, strokeColor=colors.white,
                 strokeWidth=1.5, rx=4, ry=4)
        )
        for j, line in enumerate(name.split("\n")):
            d.add(
                String(x + box_w / 2, y_box + box_h - 15 - j * 12, line,
                       fontSize=8, fillColor=colors.white, textAnchor="middle")
            )
        d.add(
            String(x + box_w / 2, y_box + 6, criterion,
                   fontSize=7, fillColor=colors.white, textAnchor="middle")
        )

    # Legend
    for passed, label, lx in [
        (True, "PASS", 10),
        (False, "FAIL : tighten controls", 70),
    ]:
        fill = colors.HexColor(_PALETTE["positive"]) if passed else colors.HexColor(_PALETTE["negative"])
        d.add(Rect(lx, 5, 10, 10, fillColor=fill, strokeColor=colors.white))
        d.add(String(lx + 14, 7, label, fontSize=7,
                     fillColor=colors.HexColor(_PALETTE["text"])))

    return d


def _diagram_placeholder() -> "Drawing":
    """Fallback empty drawing for unrecognised diagram keys."""
    d = Drawing(480, 60)
    d.add(
        String(240, 30, "[Diagram not available]",
               fontSize=10, textAnchor="middle",
               fillColor=colors.HexColor(_PALETTE["border"]))
    )
    return d


def _build_composite_drawing(sections: list[dict]) -> "Drawing":
    """Stack all section diagrams into one tall Drawing for PNG export."""
    individual_h = 160
    total_h = individual_h * len(sections) + 10
    composite = Drawing(480, total_h)

    for i, section in enumerate(sections):
        drawing = _build_diagram(section["diagram"])
        y_offset = total_h - (i + 1) * individual_h
        for shape in drawing.contents:
            # Translate each shape down by y_offset
            shape_copy = shape.copy()
            shape_copy.y = getattr(shape_copy, "y", 0) + y_offset
            composite.add(shape_copy)

    return composite


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _markdown_diagram(diagram_key: str) -> str:
    """Return an ASCII-art approximation for Markdown output."""
    diagrams = {
        "k_anonymity": (
            "\n```\n"
            "| Record | Age Band | Region | Gender | Group Size |\n"
            "|--------|----------|--------|--------|------------|\n"
            "| R1     | 30-40    | North  | M      |     3      |\n"
            "| R2     | 30-40    | North  | M      |     3      |\n"
            "| R3     | 30-40    | North  | M      |     3      |\n"
            "| R4     | 40-50    | South  | F      |     3      |\n"
            "| R5     | 40-50    | South  | F      |     3      |\n"
            "| R6     | 40-50    | South  | F      |     3      |\n"
            "```\n"
            "_Every QI combination appears at least k=3 times._\n"
        ),
        "noise_injection": (
            "\n```\n"
            "Original: [100, 200, 150, 300, 250]\n"
            "Multiplicative (+/-5%): [95, 214, 138, 312, 262]\n"
            "Laplacian (scale=10):   [107, 188, 161, 287, 241]\n"
            "```\n"
        ),
        "generalisation": (
            "\n```\n"
            "37       --[band 10]--> 30 - 40\n"
            "0.8731   --[round 2]--> 0.87\n"
            "'Rural East' --[top-5]--> Other\n"
            "```\n"
        ),
        "synthetic_data": (
            "\n```\n"
            "[Real Dataset] --> (Generative Model) --> [Synthetic Dataset]\n"
            "                   learns distributions   samples new rows\n"
            "```\n"
        ),
        "red_team": (
            "\n```\n"
            "Attack                Criterion        Result\n"
            "--------------------  ---------------  ------\n"
            "Membership Inference  AUC < 0.55       PASS\n"
            "Nearest Neighbour     DCR > 0.5        PASS\n"
            "Duplicate Row         Rate < 1%        FAIL\n"
            "Attribute Inference   Accuracy < 60%   PASS\n"
            "```\n"
        ),
    }
    return diagrams.get(diagram_key, "\n_[Diagram not available]_\n")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _safe_filename(title: str) -> str:
    """Convert a human-readable title to a filesystem-safe stem."""
    stem = title.lower()
    stem = "".join(c if c.isalnum() or c in " _-" else "" for c in stem)
    stem = stem.replace(" ", "_")
    return stem or "privacy_guide"
