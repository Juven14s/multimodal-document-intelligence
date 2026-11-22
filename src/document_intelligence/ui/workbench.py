"""UI-independent transformations for the human-review workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.schemas import (
    DocumentExtraction,
    DocumentType,
    ExtractionMethod,
    InferenceMetadata,
)
from document_intelligence.uncertainty.review_queue import ReviewCase, ReviewQueue
from document_intelligence.validation import validate_extraction

FIELD_NAMES = (
    "merchant_name",
    "merchant_address",
    "document_number",
    "issue_date",
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "total",
    "balance_due",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


class ReviewView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    prediction_json: str
    metadata_markdown: str
    image_paths: list[str]
    document_type: DocumentType
    field_rows: list[list[Any]]
    line_item_rows: list[list[Any]]
    reasons: list[str] = Field(default_factory=list)


def _display(value: Any) -> str:
    if value is None:
        return ""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def extraction_to_field_rows(extraction: DocumentExtraction) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for name in FIELD_NAMES:
        field = getattr(extraction.fields, name)
        rows.append(
            [
                name,
                _display(field.value) if field else "",
                round(field.confidence, 4) if field else 0.0,
                (field.raw_text or "") if field else "",
            ]
        )
    return rows


def extraction_to_line_rows(extraction: DocumentExtraction) -> list[list[Any]]:
    return [
        [
            _display(item.description.value),
            _display(item.quantity.value) if item.quantity else "",
            _display(item.unit_price.value) if item.unit_price else "",
            _display(item.amount.value),
        ]
        for item in extraction.fields.line_items
    ]


def resolve_source_images(source_path: Path | None) -> list[str]:
    if source_path is None:
        return []
    path = source_path.resolve()
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [str(path)]
    if path.is_dir():
        return [
            str(candidate)
            for candidate in sorted(path.iterdir())
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        ]
    return []


def _rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if hasattr(value, "values") and hasattr(value.values, "tolist"):
        return value.values.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return [list(row) for row in value]


def _human_field(value: Any, original: dict | None = None) -> dict:
    normalized = None if value is None or str(value).strip() == "" else str(value).strip()
    same_value = original is not None and normalized is not None and normalized == original["value"]
    return {
        "value": normalized,
        "raw_text": original.get("raw_text") if same_value else None,
        "confidence": 1.0,
        "evidence": original.get("evidence", []) if same_value else [],
    }


def build_human_correction(
    original: DocumentExtraction,
    *,
    reviewer: str,
    document_type: str,
    field_rows: Any,
    line_item_rows: Any,
) -> DocumentExtraction:
    if not reviewer.strip():
        raise ValueError("Reviewer name is required")
    payload = original.model_dump(mode="json")
    payload["document_type"] = document_type
    payload["classification_confidence"] = 1.0
    supplied_fields = {str(row[0]): row[1] for row in _rows(field_rows) if len(row) >= 2}
    for name in FIELD_NAMES:
        if name not in supplied_fields:
            continue
        existing = payload["fields"].get(name)
        payload["fields"][name] = _human_field(supplied_fields[name], existing)

    line_items = []
    for row in _rows(line_item_rows):
        padded = [*row, "", "", "", ""][:4]
        if not any(str(value).strip() for value in padded):
            continue
        description, quantity, unit_price, amount = padded
        line_items.append(
            {
                "description": _human_field(description),
                "quantity": _human_field(quantity) if str(quantity).strip() else None,
                "unit_price": _human_field(unit_price) if str(unit_price).strip() else None,
                "amount": _human_field(amount),
            }
        )
    payload["fields"]["line_items"] = line_items
    payload["inference"] = InferenceMetadata(
        method=ExtractionMethod.HUMAN,
        model_name=f"human:{reviewer.strip()}",
        latency_ms=0,
    ).model_dump(mode="json")
    payload["validation_issues"] = []
    payload["requires_human_review"] = False
    corrected = validate_extraction(DocumentExtraction.model_validate(payload))
    if corrected.requires_human_review:
        reasons = ", ".join(issue.code for issue in corrected.validation_issues if issue.blocking)
        raise ValueError(f"Correction still fails validation: {reasons}")
    return corrected


def case_to_view(case: ReviewCase) -> ReviewView:
    prediction = case.prediction
    issue_lines = "\n".join(
        f"- `{issue.code}` — {issue.message}"
        for issue in prediction.validation_issues
        if issue.blocking
    )
    metadata = (
        f"### Document `{case.document_id}`\n\n"
        f"**Priorité :** {case.priority}  \n"
        f"**Méthode :** `{prediction.inference.method.value}`  \n"
        f"**Modèle :** `{prediction.inference.model_name}`  \n"
        f"**Raisons de revue :**\n{issue_lines or '- Revue demandée'}"
    )
    return ReviewView(
        case_id=case.case_id,
        prediction_json=prediction.model_dump_json(),
        metadata_markdown=metadata,
        image_paths=resolve_source_images(case.source_path),
        document_type=prediction.document_type,
        field_rows=extraction_to_field_rows(prediction),
        line_item_rows=extraction_to_line_rows(prediction),
        reasons=case.reasons,
    )


class ReviewWorkbench:
    def __init__(self, queue: ReviewQueue) -> None:
        self.queue = queue

    def claim(self, reviewer: str) -> ReviewView | None:
        if not reviewer.strip():
            raise ValueError("Reviewer name is required")
        case = self.queue.claim_next(reviewer.strip())
        return case_to_view(case) if case else None

    def approve(self, case_id: str, reviewer: str) -> ReviewCase:
        return self.queue.resolve(case_id, reviewer.strip())

    def correct(
        self,
        case_id: str,
        reviewer: str,
        original_json: str,
        document_type: str,
        field_rows: Any,
        line_item_rows: Any,
    ) -> ReviewCase:
        original = DocumentExtraction.model_validate_json(original_json)
        correction = build_human_correction(
            original,
            reviewer=reviewer,
            document_type=document_type,
            field_rows=field_rows,
            line_item_rows=line_item_rows,
        )
        return self.queue.resolve(
            case_id,
            reviewer.strip(),
            correction=correction,
        )
