"""Conservative proposals from a second OCR pass for missing totals."""

from __future__ import annotations

import re
from decimal import Decimal

from document_intelligence.ocr.models import OCRDocument, OCRLine
from document_intelligence.ocr.rules import _amounts
from document_intelligence.schemas import (
    DocumentExtraction,
    Evidence,
    ExtractedField,
    ValidationIssue,
)
from document_intelligence.validation import validate_extraction

SECOND_PASS_VERSION = "ocr-rules-v3-psm11"
TOTAL_LABEL = re.compile(
    r"^\s*(?:grand\s+total|net\s+total|total(?:\s+(?:amount|amt))?|amount\s+due)\b"
    r"\s*(?:\((?:RM|MYR|RP|IDR|USD|EUR|GBP)\))?\s*[:=~]?\s*",
    re.IGNORECASE,
)
MONETARY_VALUE = re.compile(
    r"(?:(?:RM|MYR|RP|IDR|USD|EUR|GBP|[$€£])\s*)?"
    r"(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?|\d+[.,:]\s*\d{2})",
    re.IGNORECASE,
)


def _is_to_right_on_same_row(label: OCRLine, amount: OCRLine) -> bool:
    a, b = label.box, amount.box
    if a.page != b.page or a.x1 > b.x0:
        return False
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    return overlap >= 0.5 * min(a.y1 - a.y0, b.y1 - b.y0)


def _candidate_total(lines: list[OCRLine]) -> ExtractedField[Decimal] | None:
    candidates: dict[Decimal, ExtractedField[Decimal]] = {}
    amount_lines = [line for line in lines if MONETARY_VALUE.fullmatch(line.text.strip())]
    for label_line in lines:
        label = TOTAL_LABEL.match(label_line.text)
        if label is None:
            continue
        suffix = label_line.text[label.end() :].strip()
        if suffix:
            possible_amounts = [(suffix, label_line)]
        else:
            possible_amounts = [
                (line.text.strip(), line)
                for line in amount_lines
                if _is_to_right_on_same_row(label_line, line)
            ]
        for raw, amount_line in possible_amounts:
            if not MONETARY_VALUE.fullmatch(raw):
                continue
            amounts = _amounts(raw)
            if len(amounts) != 1:
                continue
            _, value = amounts[0]
            evidence_lines = [label_line]
            if amount_line is not label_line:
                evidence_lines.append(amount_line)
            confidence = min(line.confidence for line in evidence_lines)
            if value not in candidates or candidates[value].confidence < confidence:
                candidates[value] = ExtractedField(
                    value=value,
                    raw_text=raw,
                    confidence=confidence,
                    evidence=[Evidence(text=line.text, box=line.box) for line in evidence_lines],
                )
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def apply_second_pass(primary: DocumentExtraction, alternate: OCRDocument) -> DocumentExtraction:
    """Propose a missing total with explicit evidence and mandatory human review."""

    if primary.fields.total.value is not None:
        return primary

    candidate = _candidate_total(alternate.lines)
    if candidate is None:
        return primary

    fields = primary.fields.model_copy(update={"total": candidate})
    result = validate_extraction(primary.model_copy(update={"fields": fields}))
    result.validation_issues.append(
        ValidationIssue(
            code="total_from_second_ocr_pass",
            field="total",
            message="Total read by the alternate OCR pass requires human verification",
            blocking=True,
        )
    )
    result.requires_human_review = True
    return result
