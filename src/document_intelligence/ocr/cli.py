"""Run Tesseract and the deterministic extraction baseline on page images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_intelligence.ocr.engine import TesseractConfig, TesseractEngine
from document_intelligence.ocr.rules import extract_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", nargs="+", type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--psm", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = TesseractEngine(
        TesseractConfig(language=args.language, page_segmentation_mode=args.psm)
    )
    ocr = engine.extract(args.pages)
    extraction = extract_document(ocr, document_id=args.document_id)
    payload = json.dumps(extraction.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
