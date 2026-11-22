"""Command-line entry point for document preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_intelligence.preprocessing.models import PreprocessingConfig
from document_intelligence.preprocessing.pipeline import preprocess_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/preprocessed"))
    parser.add_argument("--pdf-dpi", type=int, default=200)
    parser.add_argument("--max-long-edge", type=int, default=2400)
    parser.add_argument("--no-deskew", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PreprocessingConfig(
        pdf_dpi=args.pdf_dpi,
        max_long_edge=args.max_long_edge,
        auto_deskew=not args.no_deskew,
    )
    result = preprocess_document(args.source, args.output_dir, config)
    report_path = result.pages[0].output_path.parent / "preprocessing-report.json"
    report_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
