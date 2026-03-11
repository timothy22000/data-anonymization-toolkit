#!/usr/bin/env python3
"""CLI entry point for generating the privacy guide document.

Usage:
    python scripts/generate_guide.py --output reports/
    python scripts/generate_guide.py --output reports/ --format pdf
    python scripts/generate_guide.py --output reports/ --format md
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reporting.privacy_guide import PrivacyGuideGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_guide",
        description="Generate a privacy methodology guide (PDF or Markdown).",
    )
    parser.add_argument(
        "--output", "-o", default="reports",
        help="Output directory (default: reports/).",
    )
    parser.add_argument(
        "--format", "-f", default="pdf", choices=["pdf", "md", "png"],
        help="Output format (default: pdf).",
    )
    parser.add_argument(
        "--title", "-t", default="Data Privacy Guide",
        help="Title for the guide.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    generator = PrivacyGuideGenerator(output_dir=args.output)
    path = generator.generate_guide(title=args.title, output_format=args.format)
    log.info("Privacy guide saved to %s", path)
    print(f"Guide generated: {path}")


if __name__ == "__main__":
    main()
