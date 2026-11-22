"""Filter proposed totals for human-assisted entry."""

from decimal import Decimal

from document_intelligence.ocr.models import OCRDocument
from document_intelligence.ocr.total_fallback import _candidate_total
from document_intelligence.schemas import DocumentExtraction, ExtractedField

POLICY_ID = "total-unique-label-agreement-v1"


def select_total_proposal(
    extraction: DocumentExtraction, ocr: OCRDocument
) -> ExtractedField[Decimal] | None:
    proposed = extraction.fields.total.value
    if proposed is None:
        return None
    if any(issue.code == "inconsistent_total" for issue in extraction.validation_issues):
        return None
    candidate = _candidate_total(ocr.lines)
    if candidate is None or candidate.value != proposed:
        return None
    return candidate
