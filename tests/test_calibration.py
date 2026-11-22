from decimal import Decimal

import pytest

from document_intelligence.data.models import (
    CanonicalAnnotation,
    CanonicalField,
    CanonicalRecord,
    DatasetName,
    DatasetSplit,
    ManifestEntry,
)
from document_intelligence.schemas import (
    DocumentExtraction,
    DocumentFields,
    DocumentType,
    ExtractedField,
    ExtractionMethod,
    InferenceMetadata,
)
from document_intelligence.uncertainty.calibration import (
    CalibrationConfig,
    apply_calibration,
    calibrate_predictions,
)


def record(document_id: str, split: DatasetSplit = DatasetSplit.VALIDATION):
    fields = {CanonicalField.TOTAL}
    return CanonicalRecord(
        manifest=ManifestEntry(
            document_id=document_id,
            dataset=DatasetName.SROIE,
            source_split=split.value,
            benchmark_split=split,
            source_key=document_id,
            image_width=100,
            image_height=200,
            ocr_region_count=0,
            annotated_fields=fields,
        ),
        annotation=CanonicalAnnotation(
            total=Decimal("10.00"),
            annotated_fields=fields,
        ),
    )


def prediction(document_id: str, value: str, confidence: float):
    return DocumentExtraction(
        document_id=document_id,
        document_type=DocumentType.RECEIPT,
        classification_confidence=0.9,
        fields=DocumentFields(
            merchant_name=ExtractedField(value="Shop", confidence=0.9),
            total=ExtractedField(value=Decimal(value), confidence=confidence),
        ),
        inference=InferenceMetadata(
            method=ExtractionMethod.OCR_RULES,
            model_name="test-ocr",
            prompt_version="rules-v1",
            latency_ms=1,
        ),
    )


def test_calibration_selects_lowest_threshold_meeting_target() -> None:
    records = [record(f"d{i}") for i in range(4)]
    predictions = [
        prediction("d0", "10.00", 0.9),
        prediction("d1", "10.00", 0.8),
        prediction("d2", "11.00", 0.7),
        prediction("d3", "12.00", 0.6),
    ]
    config = CalibrationConfig(
        target_auto_accept_accuracy=1.0,
        minimum_samples_per_field=4,
        minimum_accepted_per_field=2,
        reliability_bins=2,
    )

    profile = calibrate_predictions(records, predictions, config)

    total = profile.fields["total"]
    assert total.status == "calibrated"
    assert total.threshold == 0.8
    assert total.coverage == 0.5
    assert total.accepted_accuracy == 1.0


def test_calibration_routes_below_threshold_to_review() -> None:
    records = [record(f"d{i}") for i in range(4)]
    predictions = [
        prediction("d0", "10.00", 0.9),
        prediction("d1", "10.00", 0.8),
        prediction("d2", "11.00", 0.7),
        prediction("d3", "12.00", 0.6),
    ]
    profile = calibrate_predictions(
        records,
        predictions,
        CalibrationConfig(
            target_auto_accept_accuracy=1.0,
            minimum_samples_per_field=4,
            minimum_accepted_per_field=2,
        ),
    )

    result = apply_calibration(predictions[2], profile)

    assert result.requires_human_review is True
    assert "calibration_low_confidence" in {issue.code for issue in result.validation_issues}


def test_calibration_routes_missing_calibrated_field_to_review() -> None:
    records = [record(f"d{i}") for i in range(4)]
    predictions = [
        prediction("d0", "10.00", 0.9),
        prediction("d1", "10.00", 0.8),
        prediction("d2", "11.00", 0.7),
        prediction("d3", "12.00", 0.6),
    ]
    profile = calibrate_predictions(
        records,
        predictions,
        CalibrationConfig(
            target_auto_accept_accuracy=1.0,
            minimum_samples_per_field=4,
            minimum_accepted_per_field=2,
        ),
    )
    missing_total = predictions[0].model_copy(
        update={"fields": predictions[0].fields.model_copy(update={"total": None})}
    )

    result = apply_calibration(missing_total, profile)

    assert result.requires_human_review is True
    assert "calibration_missing_field" in {
        issue.code for issue in result.validation_issues
    }


def test_calibration_rejects_test_data() -> None:
    with pytest.raises(ValueError, match="validation records only"):
        calibrate_predictions([record("test", DatasetSplit.TEST)], [prediction("test", "10", 0.9)])
