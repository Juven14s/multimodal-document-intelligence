from decimal import Decimal
from pathlib import Path

import pytest

from document_intelligence.schemas import (
    DocumentExtraction,
    DocumentFields,
    DocumentType,
    Evidence,
    ExtractedField,
    ExtractionMethod,
    InferenceMetadata,
    LineItem,
    ValidationIssue,
)
from document_intelligence.ui.workbench import (
    ReviewWorkbench,
    build_human_correction,
    extraction_to_field_rows,
    resolve_source_images,
)
from document_intelligence.uncertainty.review_queue import ReviewQueue, ReviewStatus


def prediction() -> DocumentExtraction:
    return DocumentExtraction(
        document_id="doc-ui",
        document_type=DocumentType.RECEIPT,
        classification_confidence=0.9,
        fields=DocumentFields(
            merchant_name=ExtractedField(value="Shop", confidence=0.9),
            subtotal=ExtractedField(value=Decimal("10.00"), confidence=0.8),
            tax=ExtractedField(value=Decimal("1.00"), confidence=0.8),
            total=ExtractedField(value=Decimal("12.00"), confidence=0.6),
            line_items=[
                LineItem(
                    description=ExtractedField(value="Coffee", confidence=0.8),
                    amount=ExtractedField(value=Decimal("10.00"), confidence=0.8),
                )
            ],
        ),
        inference=InferenceMetadata(
            method=ExtractionMethod.VLM_ZERO_SHOT,
            model_name="fake-vlm",
            prompt_version="v1",
            latency_ms=1,
        ),
        validation_issues=[
            ValidationIssue(
                code="inconsistent_total",
                message="Total mismatch",
                field="total",
                blocking=True,
            )
        ],
        requires_human_review=True,
    )


def test_human_correction_parses_values_and_revalidates() -> None:
    original = prediction()
    rows = extraction_to_field_rows(original)
    for row in rows:
        if row[0] == "total":
            row[1] = "11.00"

    corrected = build_human_correction(
        original,
        reviewer="alice",
        document_type="receipt",
        field_rows=rows,
        line_item_rows=[["Coffee", "", "", "10.00"]],
    )

    assert corrected.fields.total.value == Decimal("11.00")
    assert corrected.fields.total.confidence == 1.0
    assert corrected.inference.method is ExtractionMethod.HUMAN
    assert corrected.inference.model_name == "human:alice"
    assert corrected.requires_human_review is False


def test_corrected_value_keeps_its_provenance_separate_from_original_ocr() -> None:
    original = prediction()
    original.fields.total.raw_text = "12.00"
    original.fields.total.evidence = [Evidence(text="OCR total 12.00")]
    original.fields.merchant_name.raw_text = "Shop"
    original.fields.merchant_name.evidence = [Evidence(text="Shop")]
    corrected = build_human_correction(
        original,
        reviewer="alice",
        document_type="receipt",
        field_rows=[["total", "11.00"], ["merchant_name", "Shop"]],
        line_item_rows=[["Coffee", "", "", "10.00"]],
    )

    assert corrected.fields.total.value == Decimal("11.00")
    assert corrected.fields.total.raw_text is None
    assert corrected.fields.total.evidence == []
    assert corrected.fields.merchant_name.evidence == [Evidence(text="Shop")]
    assert original.fields.total.evidence == [Evidence(text="OCR total 12.00")]


def test_invalid_human_correction_is_rejected() -> None:
    original = prediction()
    rows = extraction_to_field_rows(original)

    with pytest.raises(ValueError, match="still fails validation"):
        build_human_correction(
            original,
            reviewer="alice",
            document_type="receipt",
            field_rows=rows,
            line_item_rows=[["Coffee", "", "", "10.00"]],
        )


def test_workbench_claims_and_corrects_case(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image-placeholder")
    queue = ReviewQueue(tmp_path / "review.sqlite3")
    queue.enqueue(prediction(), source_path=image)
    workbench = ReviewWorkbench(queue)

    view = workbench.claim("alice")
    assert view is not None
    for row in view.field_rows:
        if row[0] == "total":
            row[1] = "11.00"
    resolved = workbench.correct(
        view.case_id,
        "alice",
        view.prediction_json,
        view.document_type.value,
        view.field_rows,
        view.line_item_rows,
    )

    assert view.image_paths == [str(image.resolve())]
    assert resolved.status is ReviewStatus.CORRECTED
    assert resolved.prediction.fields.total.value == Decimal("12.00")
    assert resolved.correction is not None
    assert resolved.correction.fields.total.value == Decimal("11.00")


def test_source_directory_returns_sorted_images(tmp_path: Path) -> None:
    (tmp_path / "page-2.png").write_bytes(b"2")
    (tmp_path / "page-1.jpg").write_bytes(b"1")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    assert resolve_source_images(tmp_path) == [
        str((tmp_path / "page-1.jpg").resolve()),
        str((tmp_path / "page-2.png").resolve()),
    ]
