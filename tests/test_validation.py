from decimal import Decimal

from document_intelligence.schemas import (
    DocumentExtraction,
    DocumentFields,
    DocumentType,
    ExtractedField,
    ExtractionMethod,
    InferenceMetadata,
    LineItem,
)
from document_intelligence.validation import ReviewPolicy, validate_extraction


def make_extraction(*, total: str = "12.00", total_confidence: float = 0.95):
    return DocumentExtraction(
        document_id="cord-001",
        document_type=DocumentType.RECEIPT,
        classification_confidence=0.98,
        fields=DocumentFields(
            merchant_name=ExtractedField(value="Cafe", confidence=0.96),
            subtotal=ExtractedField(value=Decimal("10.00"), confidence=0.90),
            tax=ExtractedField(value=Decimal("2.00"), confidence=0.90),
            total=ExtractedField(
                value=Decimal(total), confidence=total_confidence
            ),
            line_items=[
                LineItem(
                    description=ExtractedField(value="Lunch", confidence=0.90),
                    amount=ExtractedField(value=Decimal("10.00"), confidence=0.90),
                )
            ],
        ),
        inference=InferenceMetadata(
            method=ExtractionMethod.VLM_ZERO_SHOT,
            model_name="test-model",
            latency_ms=100,
        ),
    )


def test_consistent_high_confidence_extraction_is_auto_accepted() -> None:
    result = validate_extraction(make_extraction())

    assert result.validation_issues == []
    assert result.requires_human_review is False


def test_inconsistent_total_routes_to_human_review() -> None:
    result = validate_extraction(make_extraction(total="13.00"))

    assert result.requires_human_review is True
    assert {issue.code for issue in result.validation_issues} == {"inconsistent_total"}


def test_low_required_field_confidence_routes_to_human_review() -> None:
    policy = ReviewPolicy(min_required_field_confidence=0.80)
    result = validate_extraction(make_extraction(total_confidence=0.60), policy)

    assert result.requires_human_review is True
    assert {issue.code for issue in result.validation_issues} == {"low_field_confidence"}
