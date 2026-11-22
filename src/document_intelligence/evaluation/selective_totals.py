"""Evaluate selective totals using Tesseract captures and archived OCR v2 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from document_intelligence.data.io import read_records
from document_intelligence.data.models import CanonicalRecord, DatasetSplit
from document_intelligence.evaluation.benchmark import _read_predictions
from document_intelligence.ocr.engine import TesseractEngine
from document_intelligence.ocr.models import OCRDocument
from document_intelligence.ocr.total_selection import POLICY_ID, select_total_proposal
from document_intelligence.service import ExtractionResult


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _archived_pages(benchmark: Path, record: CanonicalRecord, job_id: str) -> list[Path]:
    if Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError(f"Invalid job identifier: {job_id}")
    job = benchmark / "inference/jobs" / job_id
    saved = ExtractionResult.model_validate_json((job / "result.json").read_text(encoding="utf-8"))
    prep = saved.preprocessing
    if prep.source_sha256 != record.manifest.image_sha256:
        raise ValueError(f"Archived source hash differs for {record.manifest.document_id}")
    pages = []
    for page in prep.pages:
        original = PurePosixPath(str(page.output_path).replace("\\", "/"))
        path = job / "pages" / original.parent.name / original.name
        if _digest(path) != page.output_sha256:
            raise ValueError(f"Archived page hash differs: {path}")
        pages.append(path)
    return pages


def _summarize(rows: list[dict], outcome_key: str) -> dict:
    counts = Counter(row[outcome_key] for row in rows)
    proposed = counts["exact"] + counts["wrong"]
    return {
        "exact": counts["exact"], "wrong": counts["wrong"], "blank": counts["blank"],
        "coverage": proposed / len(rows),
        "precision_when_shown": counts["exact"] / proposed if proposed else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", type=DatasetSplit, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-missing", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()

    manifest_sha256 = _digest(args.manifest)
    if args.expected_manifest_sha256 and args.expected_manifest_sha256 != manifest_sha256:
        raise ValueError("Manifest checksum mismatch")
    records = [
        record for record in read_records(args.manifest)
        if record.manifest.benchmark_split is args.split and record.annotation.total is not None
    ]
    if not records:
        raise ValueError(f"No annotated totals for split {args.split}")
    predictions, invalid = _read_predictions(args.benchmark / "predictions.jsonl")
    by_id = {prediction.document_id: prediction for prediction in predictions}
    if invalid or len(by_id) != len(predictions):
        raise ValueError("Invalid or duplicate benchmark predictions")
    jobs = {}
    if args.capture_missing:
        log = args.benchmark / "traces/inference.jsonl"
        for line in log.read_text(encoding="utf-8").splitlines():
            trace = json.loads(line)
            if trace["succeeded"]:
                jobs[trace["document_id"]] = trace["job_id"]

    def evaluate(record: CanonicalRecord) -> dict:
        entry = record.manifest
        prediction = by_id[entry.document_id]
        if prediction.inference.prompt_version != "ocr-rules-v2" or not entry.image_sha256:
            raise ValueError(f"Missing image hash or OCR v2 prediction: {entry.document_id}")
        snapshot = args.ocr_cache / entry.image_sha256 / "ocr.json"
        if snapshot.is_file():
            ocr = OCRDocument.model_validate_json(snapshot.read_text(encoding="utf-8"))
        else:
            if not args.capture_missing:
                raise FileNotFoundError(snapshot)
            pages = _archived_pages(args.benchmark, record, jobs[entry.document_id])
            ocr = TesseractEngine().extract(pages, raw_output_dir=snapshot.parent)
            pending = snapshot.with_suffix(".tmp")
            pending.write_text(ocr.model_dump_json(), encoding="utf-8")
            pending.replace(snapshot)

        selected = select_total_proposal(prediction, ocr)
        original = prediction.fields.total.value
        shown = selected.value if selected else None
        expected = record.annotation.total
        return {
            "document_id": entry.document_id,
            "image_sha256": entry.image_sha256,
            "baseline_outcome": (
                "blank" if original is None else "exact" if original == expected else "wrong"
            ),
            "outcome": "blank" if shown is None else "exact" if shown == expected else "wrong",
            "proposed_total": format(shown, "f") if shown is not None else None,
            "evidence": [item.text for item in selected.evidence] if selected else [],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        rows = list(workers.map(evaluate, records))

    summary = {
        "policy": POLICY_ID, "dataset": records[0].manifest.dataset.value, "split": args.split,
        "manifest_sha256": manifest_sha256, "document_count": len(rows),
        "baseline": _summarize(rows, "baseline_outcome"), "selective": _summarize(rows, "outcome"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**summary, "receipts": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
