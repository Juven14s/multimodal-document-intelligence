"""HTTP API for extraction and human-review workflow."""

from __future__ import annotations

import importlib.util
import shutil
import uuid
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from document_intelligence import __version__
from document_intelligence.api.models import (
    ClaimRequest,
    HealthResponse,
    ReadinessResponse,
    ResolveRequest,
)
from document_intelligence.api.settings import APISettings
from document_intelligence.observability import (
    InferenceTracer,
    JsonlTraceSink,
    MLflowTraceSink,
)
from document_intelligence.schemas import ExtractionMethod
from document_intelligence.service import (
    DocumentIntelligenceService,
    ExtractionResult,
    InferenceFailure,
    ServiceConfig,
)
from document_intelligence.uncertainty.calibration import CalibrationProfile
from document_intelligence.uncertainty.review_queue import (
    ReviewCase,
    ReviewedExtraction,
    ReviewQueue,
)


def _default_service(
    settings: APISettings,
    queue: ReviewQueue,
) -> DocumentIntelligenceService:
    sinks = [JsonlTraceSink(settings.artifacts_dir / "traces" / "inference.jsonl")]
    if settings.mlflow_tracking_uri:
        sinks.append(
            MLflowTraceSink(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=settings.mlflow_experiment,
            )
        )
    calibration_profile = None
    if settings.calibration_profile is not None:
        calibration_profile = CalibrationProfile.model_validate_json(
            settings.calibration_profile.read_text(encoding="utf-8")
        )
    return DocumentIntelligenceService(
        config=ServiceConfig(
            artifacts_dir=settings.artifacts_dir,
            max_upload_bytes=settings.max_upload_bytes,
            ocr_invoice_rules=settings.ocr_invoice_rules,
            require_calibration=settings.require_calibration,
        ),
        review_queue=queue,
        tracer=InferenceTracer(sinks),
        calibration_profile=calibration_profile,
    )


def create_app(
    *,
    service: DocumentIntelligenceService | None = None,
    review_queue: ReviewQueue | None = None,
    settings: APISettings | None = None,
) -> FastAPI:
    settings = settings or APISettings.from_environment()
    queue = review_queue or (service.review_queue if service else None)
    queue = queue or ReviewQueue(settings.review_database)
    extraction_service = service or _default_service(settings, queue)

    app = FastAPI(
        title="Multimodal Document Intelligence API",
        version=__version__,
        description="Extraction fiable avec validation métier et revue humaine.",
    )
    app.state.extraction_service = extraction_service
    app.state.review_queue = queue
    app.state.settings = settings

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/ready", response_model=ReadinessResponse, tags=["operations"])
    def ready() -> ReadinessResponse:
        methods = {
            ExtractionMethod.OCR_RULES.value: shutil.which("tesseract") is not None,
            ExtractionMethod.VLM_ZERO_SHOT.value: (
                importlib.util.find_spec("transformers") is not None
            ),
            ExtractionMethod.VLM_FEW_SHOT.value: bool(extraction_service.few_shot_examples),
        }
        return ReadinessResponse(ready=any(methods.values()), methods=methods)

    @app.post(
        "/v1/documents/extract",
        response_model=ExtractionResult,
        status_code=status.HTTP_201_CREATED,
        tags=["extraction"],
    )
    async def extract_document(
        file: Annotated[UploadFile, File(description="PDF or document image")],
        method: Annotated[ExtractionMethod, Form()] = ExtractionMethod.OCR_RULES,
        document_id: Annotated[str | None, Form(max_length=200)] = None,
    ) -> ExtractionResult:
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Upload exceeds {settings.max_upload_bytes} bytes",
            )
        try:
            return await run_in_threadpool(
                extraction_service.extract_bytes,
                content,
                filename=file.filename or "",
                method=method,
                document_id=document_id,
            )
        except InferenceFailure as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post(
        "/v1/reviews/claim",
        response_model=ReviewCase | None,
        tags=["human-review"],
    )
    def claim_review(payload: ClaimRequest) -> ReviewCase | None:
        return queue.claim_next(payload.reviewer.strip())

    @app.post(
        "/v1/reviews/{case_id}/resolve",
        response_model=ReviewCase,
        tags=["human-review"],
    )
    def resolve_review(case_id: str, payload: ResolveRequest) -> ReviewCase:
        try:
            return queue.resolve(
                case_id,
                payload.reviewer.strip(),
                correction=payload.correction,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Review case not found") from error
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get(
        "/v1/reviews/{case_id}/export", response_model=ReviewedExtraction, tags=["human-review"]
    )
    def export_review(case_id: str, response: Response) -> ReviewedExtraction:
        try:
            result = queue.reviewed_extraction(case_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Review case not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        download_name = f"review-{result.case_id}.json"
        response.headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
        response.headers["Cache-Control"] = "no-store"
        return result

    return app
