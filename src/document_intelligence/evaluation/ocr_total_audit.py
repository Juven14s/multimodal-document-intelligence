"""Capture OCR words and replay total extraction against annotated receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import fmean

from document_intelligence.data.io import read_records
from document_intelligence.data.models import DatasetSplit
from document_intelligence.evaluation.benchmark import _image_path, _select_records
from document_intelligence.ocr.engine import (
    TesseractConfig,
    TesseractEngine,
    words_from_tesseract_data,
)
from document_intelligence.ocr.layout import reconstruct_lines
from document_intelligence.ocr.models import OCRDocument, OCRLine
from document_intelligence.ocr.rules import OCRRulesVersion, _amounts, extract_document
from document_intelligence.ocr.total_fallback import SECOND_PASS_VERSION, apply_second_pass
from document_intelligence.preprocessing.pipeline import preprocess_document


def _matching_lines(lines: list[OCRLine], expected: Decimal) -> list[str]:
    return [
        line.text for line in lines if any(value == expected for _, value in _amounts(line.text))
    ]


def _unfiltered_lines(ocr: OCRDocument, cache: Path) -> list[OCRLine]:
    lines = []
    for page in ocr.pages:
        raw_data = cache / f"tesseract-{page.page_number:04d}.json"
        words = words_from_tesseract_data(
            json.loads(raw_data.read_text(encoding="utf-8")),
            page_number=page.page_number,
            width=page.width,
            height=page.height,
            minimum_confidence=0.0,
        )
        lines.extend(reconstruct_lines(words))
    return lines


def _capture_ocr(engine: TesseractEngine, pages: list[Path], snapshot: Path) -> OCRDocument:
    result = engine.extract(pages, raw_output_dir=snapshot.parent)
    pending = snapshot.with_suffix(".tmp")
    pending.write_text(result.model_dump_json(), encoding="utf-8")
    pending.replace(snapshot)
    return result


def _read_second_pass(
    engine: TesseractEngine, primary: OCRDocument, cache: Path, *, replay_only: bool
) -> OCRDocument:
    snapshot = cache / "psm-11" / "ocr.json"
    if snapshot.is_file():
        return OCRDocument.model_validate_json(snapshot.read_text(encoding="utf-8"))
    if replay_only:
        raise FileNotFoundError(snapshot)
    pages = [
        cache / "pages" / page.image_path.parent.name / page.image_path.name
        for page in primary.pages
    ]
    return _capture_ocr(engine, pages, snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=[split.value for split in DatasetSplit], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-only", action="store_true", help="Require cached OCR output")
    parser.add_argument("--rules-version", type=OCRRulesVersion, default=OCRRulesVersion.V2)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--second-pass", action="store_true", help="Try PSM 11 for missing totals")
    args = parser.parse_args()
    if args.second_pass and args.rules_version is not OCRRulesVersion.V3:
        parser.error("--second-pass requires --rules-version ocr-rules-v3")

    manifest_path = args.manifest.resolve()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if args.expected_manifest_sha256 and manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError(f"Manifest checksum mismatch: {manifest_path}")
    records = _select_records(read_records(manifest_path), DatasetSplit(args.split), None)
    rows = []
    engine = TesseractEngine()
    alternate_engine = TesseractEngine(TesseractConfig(page_segmentation_mode=11))
    for record in records:
        expected = record.annotation.total
        if expected is None:
            continue
        image_path = _image_path(record, manifest_path)
        with image_path.open("rb") as image:
            digest = hashlib.file_digest(image, "sha256").hexdigest()
        if record.manifest.image_sha256 not in (None, digest):
            raise ValueError(f"Image does not match manifest: {image_path}")
        cache = args.output_dir / "ocr" / digest
        snapshot = cache / "ocr.json"
        if snapshot.is_file():
            ocr = OCRDocument.model_validate_json(snapshot.read_text(encoding="utf-8"))
        else:
            if args.replay_only:
                raise FileNotFoundError(snapshot)
            prep = preprocess_document(image_path, cache / "pages")
            ocr = _capture_ocr(engine, [page.output_path for page in prep.pages], snapshot)

        extraction = extract_document(
            ocr, document_id=record.manifest.document_id, rules_version=args.rules_version
        )
        second_pass = None
        if args.second_pass and extraction.fields.total.value is None:
            alternate = _read_second_pass(
                alternate_engine, ocr, cache, replay_only=args.replay_only
            )
            extraction = apply_second_pass(extraction, alternate)
            second_pass = {
                "latency_ms": alternate.latency_ms,
                "proposed_total": extraction.fields.total.value is not None,
                "reference_matching_lines": _matching_lines(alternate.lines, expected),
            }
        total = extraction.fields.total
        value = total.value if total else None
        matches = _matching_lines(ocr.lines, expected)
        raw_matches = matches or _matching_lines(_unfiltered_lines(ocr, cache), expected)
        rows.append({
            "document_id": record.manifest.document_id,
            "image_sha256": digest,
            "reference_total": str(expected),
            "predicted_total": str(value) if value is not None else None,
            "outcome": "absent" if value is None else "exact" if value == expected else "wrong",
            "evidence_lines": [item.text for item in total.evidence] if total else [],
            "reference_matching_lines": matches,
            "reference_found_before_confidence_filter": bool(raw_matches),
            "primary_ocr_latency_ms": ocr.latency_ms,
            "second_pass": second_pass,
            "requires_human_review": extraction.requires_human_review,
        })

    if not rows:
        raise ValueError(f"No annotated totals in split {args.split}")
    errors = [row for row in rows if row["outcome"] != "exact"]
    alternate_runs = [row["second_pass"] for row in rows if row["second_pass"] is not None]
    alternate_latencies = [run["latency_ms"] for run in alternate_runs]
    added_latency_ms = sum(alternate_latencies) / len(rows)
    rules_version = SECOND_PASS_VERSION if args.second_pass else args.rules_version.value
    summary = {
        "dataset": records[0].manifest.dataset.value,
        "split": args.split,
        "manifest_sha256": manifest_sha256,
        "tesseract_version": ocr.engine_version,
        "tesseract_config": engine.config.model_dump(),
        "rules_version": rules_version,
        "documents": len(rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "errors_reference_found_in_rule_input": sum(
            bool(row["reference_matching_lines"]) for row in errors
        ),
        "errors_reference_found_before_confidence_filter": sum(
            row["reference_found_before_confidence_filter"] for row in errors
        ),
        "mean_primary_ocr_latency_ms": fmean(row["primary_ocr_latency_ms"] for row in rows),
        "second_pass": {
            "config": alternate_engine.config.model_dump() if args.second_pass else None,
            "attempted": len(alternate_runs),
            "proposed_totals": sum(run["proposed_total"] for run in alternate_runs),
            "proposed_totals_requiring_human_review": sum(
                bool(row["second_pass"] and row["second_pass"]["proposed_total"])
                and row["requires_human_review"]
                for row in rows
            ),
            "mean_latency_ms_when_called": fmean(alternate_latencies) if alternate_latencies else 0,
            "mean_added_ocr_latency_ms_per_receipt": added_latency_ms,
        },
    }
    output = args.output_dir / f"{rules_version}.json"
    output.write_text(
        json.dumps({**summary, "receipts": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
