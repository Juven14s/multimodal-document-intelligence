import sys
from pathlib import Path

import uvicorn
from fastapi.testclient import TestClient
from product_fixtures import FakeOCREngine, png_bytes

from document_intelligence.api.app import create_app
from document_intelligence.api.cli import main as run_api
from document_intelligence.api.settings import APISettings
from document_intelligence.service import DocumentIntelligenceService, ServiceConfig
from document_intelligence.uncertainty.review_queue import ReviewQueue


def make_client(
    tmp_path: Path,
    *,
    include_total: bool = True,
    max_upload_bytes: int = 1024 * 1024,
) -> TestClient:
    queue = ReviewQueue(tmp_path / "review.sqlite3")
    service = DocumentIntelligenceService(
        config=ServiceConfig(
            artifacts_dir=tmp_path / "artifacts",
            max_upload_bytes=max_upload_bytes,
        ),
        review_queue=queue,
        ocr_engine=FakeOCREngine(include_total=include_total),
    )
    settings = APISettings(
        artifacts_dir=tmp_path / "artifacts",
        review_database=tmp_path / "review.sqlite3",
        max_upload_bytes=max_upload_bytes,
    )
    return TestClient(create_app(service=service, review_queue=queue, settings=settings))


def test_health_and_openapi_are_available(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/health", headers={"x-request-id": "test-request"})
        schema = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}
    assert response.headers["x-request-id"] == "test-request"
    assert schema.status_code == 200
    assert "/v1/documents/extract" in schema.json()["paths"]


def test_upload_returns_extraction(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/documents/extract",
            files={"file": ("receipt.png", png_bytes(), "image/png")},
            data={"method": "ocr_rules", "document_id": "api-doc-1"},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["document_id"] == "api-doc-1"
    assert payload["extraction"]["fields"]["total"]["value"] == "10.00"
    assert payload["review_case_id"] is None


def test_upload_validation_errors_are_explicit(tmp_path: Path) -> None:
    with make_client(tmp_path, max_upload_bytes=10) as client:
        too_large = client.post(
            "/v1/documents/extract",
            files={"file": ("receipt.png", b"x" * 11, "image/png")},
        )
        empty = client.post(
            "/v1/documents/extract",
            files={"file": ("receipt.png", b"", "image/png")},
        )

    assert too_large.status_code == 413
    assert empty.status_code == 422
    assert empty.json()["detail"] == "Uploaded document is empty"


def test_review_case_can_be_claimed_and_approved(tmp_path: Path) -> None:
    with make_client(tmp_path, include_total=False) as client:
        extraction = client.post(
            "/v1/documents/extract",
            files={"file": ("receipt.png", png_bytes(), "image/png")},
        ).json()
        claimed = client.post("/v1/reviews/claim", json={"reviewer": "alice"})
        case_id = claimed.json()["case_id"]
        resolved = client.post(
            f"/v1/reviews/{case_id}/resolve",
            json={"reviewer": "alice", "correction": None},
        )

    assert extraction["review_case_id"] == case_id
    assert claimed.status_code == 200
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "approved"


def test_review_export_requires_resolution_and_returns_correction(tmp_path: Path) -> None:
    with make_client(tmp_path, include_total=False) as client:
        initial = client.post(
            "/v1/documents/extract",
            files={"file": ("receipt.png", png_bytes(), "image/png")},
        ).json()
        case_id = initial["review_case_id"]
        export_url = f"/v1/reviews/{case_id}/export"
        assert client.get(export_url).status_code == 409
        assert client.get("/v1/reviews/absent/export").status_code == 404
        assert client.post("/v1/reviews/claim", json={"reviewer": "alice"}).status_code == 200

        correction = initial["extraction"]
        correction["fields"]["total"]["value"] = "10.00"
        correction["fields"]["total"]["confidence"] = 1.0
        resolve_url = f"/v1/reviews/{case_id}/resolve"
        mismatch = client.post(resolve_url, json={
            "reviewer": "alice", "correction": {**correction, "document_id": "different"},
        })
        assert mismatch.status_code == 409
        resolved = client.post(resolve_url, json={"reviewer": "alice", "correction": correction})
        exported = client.get(export_url)

    assert resolved.status_code == exported.status_code == 200
    assert exported.headers["cache-control"] == "no-store"
    assert exported.headers["content-disposition"].startswith("attachment;")
    assert exported.json()["status"] == "corrected"
    assert exported.json()["reviewer"] == "alice"
    assert exported.json()["extraction"] == resolved.json()["correction"]
    assert resolved.json()["prediction"]["fields"]["total"]["value"] is None


def test_api_launcher_preserves_environment_flags(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCINTEL_OCR_INVOICE_RULES", "true")
    monkeypatch.setenv("DOCINTEL_REQUIRE_CALIBRATION", "true")
    monkeypatch.delenv("DOCINTEL_CALIBRATION_PROFILE", raising=False)
    monkeypatch.setattr(sys, "argv", ["docintel-api", "--artifacts", str(tmp_path)])
    started = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.append(app))

    run_api()

    config = started[0].state.extraction_service.config
    assert config.artifacts_dir == tmp_path
    assert config.ocr_invoice_rules is True
    assert config.require_calibration is True
