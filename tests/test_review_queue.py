from decimal import Decimal
from pathlib import Path

from document_intelligence.schemas import (
    DocumentExtraction,
    DocumentFields,
    DocumentType,
    ExtractedField,
    ExtractionMethod,
    InferenceMetadata,
    ValidationIssue,
)
from document_intelligence.uncertainty.review_queue import ReviewQueue, ReviewStatus


def extraction(total: str = "10.00") -> DocumentExtraction:
    return DocumentExtraction(
        document_id="doc-1",
        document_type=DocumentType.RECEIPT,
        classification_confidence=0.9,
        fields=DocumentFields(
            merchant_name=ExtractedField(value="Shop", confidence=0.9),
            total=ExtractedField(value=Decimal(total), confidence=0.6),
        ),
        inference=InferenceMetadata(
            method=ExtractionMethod.VLM_ZERO_SHOT,
            model_name="fake-vlm",
            prompt_version="v1",
            latency_ms=1,
        ),
        validation_issues=[
            ValidationIssue(
                code="low_field_confidence",
                message="Review total",
                field="total",
                blocking=True,
            )
        ],
        requires_human_review=True,
    )


def test_review_queue_preserves_original_and_audit_history(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path / "review.sqlite3")
    original = extraction()

    case = queue.enqueue(original, source_path=Path("doc.png"))
    duplicate = queue.enqueue(original)
    claimed = queue.claim_next("alice")
    corrected = extraction("10.60")
    resolved = queue.resolve(claimed.case_id, "alice", correction=corrected)

    assert duplicate.case_id == case.case_id
    assert claimed.status is ReviewStatus.IN_PROGRESS
    assert resolved.status is ReviewStatus.CORRECTED
    assert resolved.prediction.fields.total.value == Decimal("10.00")
    assert resolved.correction.fields.total.value == Decimal("10.60")
    assert [event.event_type for event in queue.history(case.case_id)] == [
        "enqueued",
        "claimed",
        "corrected",
    ]


def test_empty_queue_returns_none(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path / "review.sqlite3")

    assert queue.claim_next("alice") is None
