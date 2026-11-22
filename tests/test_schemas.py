from decimal import Decimal

import pytest
from pydantic import ValidationError

from document_intelligence.schemas import (
    BoundingBox,
    DocumentExtraction,
    DocumentFields,
    DocumentType,
    ExtractedField,
    ExtractionMethod,
    InferenceMetadata,
)


def test_bounding_box_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(page=1, x0=0.8, y0=0.1, x1=0.2, y1=0.5)


def test_schema_serializes_normalized_values() -> None:
    extraction = DocumentExtraction(
        document_id="sroie-001",
        document_type=DocumentType.RECEIPT,
        classification_confidence=0.99,
        fields=DocumentFields(
            merchant_name=ExtractedField(value="Example Shop", confidence=0.95),
            currency=ExtractedField(value=" eur ", confidence=0.90),
            total=ExtractedField(value=Decimal("12.30"), raw_text="12,30", confidence=0.94),
        ),
        inference=InferenceMetadata(
            method=ExtractionMethod.OCR_RULES,
            model_name="tesseract",
            latency_ms=42,
        ),
    )

    payload = extraction.model_dump(mode="json")

    assert payload["fields"]["currency"]["value"] == "EUR"
    assert payload["fields"]["total"]["value"] == "12.30"
    assert DocumentExtraction.model_validate(payload) == extraction
