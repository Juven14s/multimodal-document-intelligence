"""Run real OCR, HTTP upload and review against a small set of known documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

from document_intelligence.api.app import create_app
from document_intelligence.api.settings import APISettings
from document_intelligence.data.io import read_records
from document_intelligence.data.models import CanonicalField, DatasetSplit
from document_intelligence.evaluation.benchmark import _image_path
from document_intelligence.schemas import DocumentExtraction
from document_intelligence.service import ExtractionResult
from document_intelligence.ui.workbench import FIELD_NAMES, build_human_correction, case_to_view
from document_intelligence.uncertainty.review_queue import ReviewCase, ReviewStatus

MONEY_FIELDS = {"subtotal", "tax", "total", "balance_due", "quantity", "unit_price", "amount"}
ITEM_COLUMNS = ("description", "quantity", "unit_price", "amount")
REVIEWER = "demo-reference"


@dataclass(frozen=True)
class Example:
    document_id: str
    group: str
    path: Path
    pages: int
    fields: dict[str, str | None]
    document_type: str | None = None
    line_items: list[dict[str, str]] | None = None


def _examples(invoices: Path, manifest: Path | None) -> list[Example]:
    refs = json.loads((invoices / "references.json").read_text(encoding="utf-8"))
    examples = [
        Example(
            ref["id"], "invoice", invoices / ref["file"], ref["pages"], ref["fields"],
            ref["document_type"], ref["line_items"],
        )
        for ref in refs
    ]
    if manifest is None:
        return examples
    fields = {
        CanonicalField.MERCHANT_NAME,
        CanonicalField.MERCHANT_ADDRESS,
        CanonicalField.ISSUE_DATE,
        CanonicalField.TOTAL,
    }
    candidates = [
        record for record in read_records(manifest)
        if record.manifest.benchmark_split is DatasetSplit.TEST
        and fields <= record.annotation.annotated_fields
    ]
    rng = random.Random("document-flow-demo-v1")
    sampled = rng.sample(sorted(candidates, key=lambda record: record.manifest.document_id), 4)
    for record in sampled:
        fields_json = record.annotation.model_dump(mode="json")
        expected = {field.value: fields_json[field.value] for field in sorted(fields)}
        examples.append(Example(
            record.manifest.document_id, "receipt", _image_path(record, manifest), 1, expected,
        ))
    return examples


def _value(field: dict | None) -> str | None:
    return field["value"] if field else None


def _normalized(column: str, value: str) -> Decimal | str:
    if column in MONEY_FIELDS:
        return Decimal(value)
    return " ".join(value.casefold().split())


def _comparison(example: Example, extraction: DocumentExtraction) -> dict:
    fields = extraction.model_dump(mode="json")["fields"]
    rows = {}
    for name, expected in example.fields.items():
        actual = _value(fields[name])
        if expected is None:
            outcome = "expected_blank" if actual is None else "unexpected"
        elif actual is None:
            outcome = "missing"
        else:
            outcome = (
                "exact" if _normalized(name, actual) == _normalized(name, expected) else "wrong"
            )
        rows[name] = {"expected": expected, "actual": actual, "outcome": outcome}
    if example.document_type is not None:
        actual = extraction.document_type.value
        rows["document_type"] = {
            "expected": example.document_type, "actual": actual,
            "outcome": "exact" if actual == example.document_type else "wrong",
        }
    items_exact = None
    if example.line_items is not None:
        actual_items = [
            [_value(item[name]) for name in ITEM_COLUMNS] for item in fields["line_items"]
        ]
        expected_items = [[item[name] for name in ITEM_COLUMNS] for item in example.line_items]

        def normalize(items: list[list[str | None]]) -> list[list[Decimal | str | None]]:
            return [
                [_normalized(name, value) if value else None for name, value in zip(
                    ITEM_COLUMNS, row, strict=True
                )]
                for row in items
            ]

        items_exact = normalize(actual_items) == normalize(expected_items)
    return {"fields": rows, "line_items_exact": items_exact}


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_case(
    client: TestClient, example: Example, output: Path, *, extract_only: bool = False
) -> dict:
    case_output = output / "cases" / example.document_id
    case_output.mkdir(parents=True)
    media_type = "application/pdf" if example.path.suffix.lower() == ".pdf" else "image/png"
    with example.path.open("rb") as source:
        response = client.post(
            "/v1/documents/extract",
            files={"file": (example.path.name, source, media_type)},
            data={"document_id": example.document_id, "method": "ocr_rules"},
        )
    response.raise_for_status()
    initial = ExtractionResult.model_validate(response.json())
    if len(initial.preprocessing.pages) != example.pages:
        raise AssertionError(f"Page count mismatch: {example.document_id}")
    _write(case_output / "initial.json", initial.model_dump(mode="json"))
    before = _comparison(example, initial.extraction)
    data = {
        "document_id": example.document_id, "group": example.group,
        "input_file": example.path.name, "pages": len(initial.preprocessing.pages),
        "request_id": response.headers["x-request-id"], "http_status": response.status_code,
        "ocr_latency_ms": initial.extraction.inference.latency_ms,
        "rules_version": initial.extraction.inference.prompt_version,
        "initial_issues": [issue.code for issue in initial.extraction.validation_issues],
        "initial": before, "schema_valid": True,
    }
    if extract_only:
        return {
            **data, "review_status": "pending" if initial.review_case_id else None, "final": None,
        }

    claim = client.post("/v1/reviews/claim", json={"reviewer": REVIEWER})
    claim.raise_for_status()
    reviewed = None
    events: list[str] = []
    review_status = None
    final = initial.extraction
    if initial.review_case_id is not None:
        claimed = ReviewCase.model_validate(claim.json())
        if claimed.case_id != initial.review_case_id or claimed.prediction != initial.extraction:
            raise AssertionError(f"Review claim differs from original: {example.document_id}")
        view = case_to_view(claimed)
        if len(view.image_paths) != example.pages:
            raise AssertionError(f"Images unavailable in the review view: {example.document_id}")
        correction = build_human_correction(
            claimed.prediction,
            reviewer=REVIEWER,
            document_type=example.document_type or claimed.prediction.document_type.value,
            field_rows=[[name, example.fields.get(name, "")] for name in FIELD_NAMES],
            line_item_rows=[
                [item[name] for name in ITEM_COLUMNS] for item in example.line_items or []
            ],
        )
        for name in example.fields:
            prior = getattr(initial.extraction.fields, name)
            updated = getattr(correction.fields, name)
            previous_value = prior.value if prior else None
            if updated.value != previous_value and (updated.raw_text or updated.evidence):
                raise AssertionError(f"Changed field still has OCR evidence: {name}")
        resolved = client.post(
            f"/v1/reviews/{claimed.case_id}/resolve",
            json={"reviewer": REVIEWER, "correction": correction.model_dump(mode="json")},
        )
        resolved.raise_for_status()
        reviewed = ReviewCase.model_validate(resolved.json())
        if (
            reviewed.status is not ReviewStatus.CORRECTED
            or reviewed.prediction != initial.extraction
            or reviewed.correction != correction
        ):
            raise AssertionError(f"Review history lost the prediction: {example.document_id}")
        history = client.app.state.review_queue.history(claimed.case_id)
        events = [event.event_type for event in history]
        if events != ["enqueued", "claimed", "corrected"]:
            raise AssertionError(f"Unexpected review events: {events}")
        _write(case_output / "review.json", {
            "case": reviewed.model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in history],
        })
        review_status = reviewed.status.value
        final = correction
    elif claim.json() is not None:
        raise AssertionError("The isolated review queue contains an unrelated case")

    _write(case_output / "final.json", final.model_dump(mode="json"))
    final = DocumentExtraction.model_validate_json(
        (case_output / "final.json").read_text(encoding="utf-8")
    )
    return {
        **data,
        "review_status": review_status, "review_events": events,
        "final": _comparison(example, final),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invoices", type=Path, required=True)
    parser.add_argument("--sroie", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invoice-rules", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--expected-protocol", type=Path)
    args = parser.parse_args()
    if args.expected_protocol:
        frozen = json.loads(args.expected_protocol.read_text(encoding="utf-8"))["files"]
        for filename, digest in frozen.items():
            if Path(filename).name != filename:
                raise ValueError(f"Invalid fixture filename: {filename}")
            with (args.invoices / filename).open("rb") as fixture:
                if hashlib.file_digest(fixture, "sha256").hexdigest() != digest:
                    raise ValueError(f"Fixture checksum mismatch: {filename}")
    args.output.mkdir(parents=True, exist_ok=False)
    app = create_app(settings=APISettings(
        artifacts_dir=args.output / "runtime",
        review_database=args.output / "runtime/review.sqlite3",
        calibration_profile=args.calibration,
        ocr_invoice_rules=args.invoice_rules,
    ))
    with TestClient(app) as client:
        if not client.get("/ready").json()["methods"]["ocr_rules"]:
            raise RuntimeError("The OCR engine is unavailable in this environment")
        examples = _examples(args.invoices, args.sroie)
        rows = [
            _run_case(client, example, args.output, extract_only=args.extract_only)
            for example in examples
        ]

    group_counts = defaultdict(lambda: {"initial": Counter(), "final": Counter()})
    for row in rows:
        for stage in ("initial", "final"):
            if row[stage] is None:
                continue
            group_counts[row["group"]][stage].update(
                field["outcome"] for field in row[stage]["fields"].values()
            )
    summary = {
        "documents": len(rows), "pages": sum(row["pages"] for row in rows),
        "reviewed": sum(row["review_status"] == "corrected" for row in rows),
        "pending_review": sum(row["review_status"] == "pending" for row in rows),
        "json_schema_valid": sum(row["schema_valid"] for row in rows),
        "field_counts": dict(group_counts),
        "review_method": None if args.extract_only else (
            "Annotated values submitted through the product review workflow"
        ),
    }
    _write(args.output / "report.json", {**summary, "cases": rows})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
