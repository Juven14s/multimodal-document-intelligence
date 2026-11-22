import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from product_fixtures import FakeOCREngine, png_bytes

from document_intelligence.observability import InferenceTracer, JsonlTraceSink
from document_intelligence.schemas import ExtractionMethod
from document_intelligence.service import (
    DocumentIntelligenceService,
    InferenceFailure,
    ServiceConfig,
)
from document_intelligence.uncertainty.calibration import (
    CalibrationConfig,
    CalibrationProfile,
    FieldCalibration,
)
from document_intelligence.uncertainty.review_queue import ReviewQueue
from document_intelligence.vlm.models import GenerationResult


class FakeVLMBackend:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, image_paths, prompt, few_shot_turns=None) -> GenerationResult:
        assert image_paths
        return GenerationResult(
            model_id="fixture-vlm",
            raw_response=self.response,
            latency_ms=7,
            device="cpu",
        )


def vlm_payload() -> str:
    return json.dumps(
        {
            "document_type": "receipt",
            "classification_confidence": 0.95,
            "fields": {
                "merchant_name": {
                    "value": "Example Cafe",
                    "raw_text": "EXAMPLE CAFE",
                    "confidence": 0.95,
                },
                "total": {"value": "10.00", "raw_text": "10.00", "confidence": 0.95},
                "line_items": [],
            },
        }
    )


def test_service_runs_product_flow_and_writes_privacy_safe_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces" / "inference.jsonl"
    artifacts = tmp_path / "artifacts"
    queue = ReviewQueue(tmp_path / "review.sqlite3")
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=artifacts),
        review_queue=queue,
        tracer=InferenceTracer([JsonlTraceSink(trace_path)]),
        ocr_engine=FakeOCREngine(),
    )

    result = service.extract_bytes(
        png_bytes(),
        filename="../../Reçu client.png",
        method=ExtractionMethod.OCR_RULES,
        document_id="receipt-001",
    )

    assert result.extraction.fields.total.value == 10
    assert result.review_case_id is None
    assert result.preprocessing.pages[0].output_path.is_file()
    persisted_result = json.loads(
        (artifacts / "jobs" / result.job_id / "result.json").read_text(encoding="utf-8")
    )
    assert persisted_result["document_id"] == "receipt-001"
    assert persisted_result["extraction"]["fields"]["total"]["value"] == "10.00"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["document_id"] == "receipt-001"
    assert trace["succeeded"] is True
    assert "EXAMPLE CAFE" not in trace_path.read_text(encoding="utf-8")


def test_service_enqueues_uncertain_extraction(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path / "review.sqlite3")
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=tmp_path / "artifacts"),
        review_queue=queue,
        ocr_engine=FakeOCREngine(include_total=False),
    )

    result = service.extract_bytes(png_bytes(), filename="receipt.png")
    claimed = queue.claim_next("alice")

    assert result.extraction.requires_human_review is True
    assert result.review_case_id is not None
    assert claimed is not None
    assert claimed.source_path.is_dir()


def test_service_can_require_review_when_no_calibration_matches(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path / "review.sqlite3")
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=tmp_path / "artifacts", require_calibration=True),
        review_queue=queue,
        ocr_engine=FakeOCREngine(),
    )

    result = service.extract_bytes(png_bytes(), filename="receipt.png")

    assert result.review_case_id is not None
    assert result.extraction.requires_human_review
    assert any(
        issue.code == "calibration_profile_missing" for issue in result.extraction.validation_issues
    )


def test_service_applies_matching_calibration_before_review_routing(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path / "review.sqlite3")
    profile = CalibrationProfile(
        created_at=datetime.now(UTC),
        method=ExtractionMethod.OCR_RULES,
        model_name="fixture-ocr",
        prompt_version="ocr-rules-v2",
        validation_set_sha256="0" * 64,
        config=CalibrationConfig(),
        fields={
            "merchant_address": FieldCalibration(
                sample_count=20,
                correct_count=19,
                status="calibrated",
                threshold=1.0,
                accepted_count=5,
                coverage=0.25,
                accepted_accuracy=1.0,
                brier_score=0.1,
                expected_calibration_error=0.1,
            )
        },
    )
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=tmp_path / "artifacts"),
        review_queue=queue,
        ocr_engine=FakeOCREngine(),
        calibration_profile=profile,
    )

    result = service.extract_bytes(png_bytes(), filename="receipt.png")

    assert result.extraction.requires_human_review is True
    assert result.review_case_id is not None
    assert "calibration_missing_field" in {
        issue.code for issue in result.extraction.validation_issues
    }
    assert queue.claim_next("alice") is not None


@pytest.mark.parametrize("filename", ["receipt.exe", "no-extension", "receipt.txt"])
def test_service_rejects_unsupported_uploads(tmp_path: Path, filename: str) -> None:
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=tmp_path),
        ocr_engine=FakeOCREngine(),
    )

    with pytest.raises(ValueError, match="Unsupported document extension"):
        service.extract_bytes(png_bytes(), filename=filename)


def test_service_rejects_empty_and_oversized_uploads(tmp_path: Path) -> None:
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=tmp_path, max_upload_bytes=3),
        ocr_engine=FakeOCREngine(),
    )

    with pytest.raises(ValueError, match="empty"):
        service.extract_bytes(b"", filename="receipt.png")
    with pytest.raises(ValueError, match="maximum"):
        service.extract_bytes(b"1234", filename="receipt.png")


def test_service_persists_contract_valid_vlm_trace(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=artifacts),
        vlm_backend=FakeVLMBackend(vlm_payload()),
    )

    result = service.extract_bytes(
        png_bytes(),
        filename="receipt.png",
        method=ExtractionMethod.VLM_ZERO_SHOT,
        document_id="vlm-valid",
    )

    trace_path = next((artifacts / "jobs").glob("*/vlm-trace.json"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert result.extraction.fields.total.value == 10
    assert trace["document_id"] == "vlm-valid"
    assert trace["json_syntax_valid"] is True
    assert trace["valid_json"] is True


def test_service_persists_vlm_trace_before_raising_on_invalid_output(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=artifacts),
        vlm_backend=FakeVLMBackend("not json"),
    )

    with pytest.raises(InferenceFailure, match="no JSON object"):
        service.extract_bytes(
            png_bytes(),
            filename="receipt.png",
            method=ExtractionMethod.VLM_ZERO_SHOT,
            document_id="vlm-invalid",
        )

    trace_path = next((artifacts / "jobs").glob("*/vlm-trace.json"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["document_id"] == "vlm-invalid"
    assert trace["raw_response"] == "not json"
    assert trace["valid_json"] is False
