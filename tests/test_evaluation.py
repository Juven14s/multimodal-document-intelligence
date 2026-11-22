from decimal import Decimal

from document_intelligence.data.models import (
    CanonicalAnnotation,
    CanonicalField,
    CanonicalRecord,
    DatasetName,
    DatasetSplit,
    ManifestEntry,
)
from document_intelligence.evaluation.metrics import evaluate_predictions
from document_intelligence.schemas import (
    DocumentExtraction,
    DocumentFields,
    DocumentType,
    ExtractedField,
    ExtractionMethod,
    InferenceMetadata,
)


def make_record() -> CanonicalRecord:
    annotated = {CanonicalField.MERCHANT_NAME, CanonicalField.TOTAL}
    return CanonicalRecord(
        manifest=ManifestEntry(
            document_id="doc-1",
            dataset=DatasetName.SROIE,
            source_split="test",
            benchmark_split=DatasetSplit.TEST,
            source_key="1",
            image_width=100,
            image_height=200,
            ocr_region_count=0,
            annotated_fields=annotated,
        ),
        annotation=CanonicalAnnotation(
            merchant_name="Example Café",
            total=Decimal("12.30"),
            annotated_fields=annotated,
        ),
    )


def make_prediction() -> DocumentExtraction:
    return DocumentExtraction(
        document_id="doc-1",
        document_type=DocumentType.RECEIPT,
        classification_confidence=0.9,
        fields=DocumentFields(
            merchant_name=ExtractedField(value="EXAMPLE CAFE", confidence=0.9),
            total=ExtractedField(value=Decimal("12.30"), confidence=0.9),
        ),
        inference=InferenceMetadata(
            method=ExtractionMethod.OCR_RULES,
            model_name="test",
            latency_ms=1,
        ),
    )


def test_evaluation_respects_field_mask_and_normalization() -> None:
    result = evaluate_predictions([make_record()], [make_prediction()])

    assert set(result.fields) == {"merchant_name", "total"}
    assert result.fields["merchant_name"].exact_match == 1.0
    assert result.fields["merchant_name"].f1 == 1.0
    assert result.fields["total"].exact_match == 1.0
    assert result.prediction_coverage == 1.0
    assert result.valid_json_rate == 1.0
    assert result.selective.exact_documents == 1
    assert result.selective.document_exact_match_rate == 1.0
    assert result.selective.auto_accepted_documents == 1
    assert result.selective.auto_accepted_exact_match_rate == 1.0
    assert result.selective.reviewed_documents == 0


def test_missing_and_invalid_predictions_reduce_rates() -> None:
    result = evaluate_predictions([make_record()], [], invalid_prediction_count=1)

    assert result.prediction_coverage == 0.0
    assert result.valid_json_rate == 0.0
    assert result.fields["total"].exact_match == 0.0
    assert result.human_review_rate == 1.0
    assert result.selective.reviewed_documents == 1
    assert result.selective.review_precision == 1.0
    assert result.selective.error_detection_recall == 1.0
