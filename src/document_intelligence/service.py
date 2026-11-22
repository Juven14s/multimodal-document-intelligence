"""Application service shared by HTTP, batch jobs, and future interfaces."""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.observability import InferenceTrace, InferenceTracer
from document_intelligence.ocr.engine import TesseractEngine
from document_intelligence.ocr.models import OCRDocument
from document_intelligence.ocr.rules import OCRRulesVersion, extract_document
from document_intelligence.preprocessing.models import PreprocessingConfig, PreprocessingResult
from document_intelligence.preprocessing.pipeline import (
    SUPPORTED_IMAGE_SUFFIXES,
    preprocess_document,
)
from document_intelligence.schemas import DocumentExtraction, ExtractionMethod, ValidationIssue
from document_intelligence.uncertainty.calibration import (
    CalibrationProfile,
    apply_calibration,
)
from document_intelligence.uncertainty.review_queue import ReviewQueue
from document_intelligence.vlm.backend import TransformersQwenBackend
from document_intelligence.vlm.examples import FewShotExample
from document_intelligence.vlm.pipeline import VLMBackend, run_few_shot, run_zero_shot
from document_intelligence.vlm.prompting import PromptStyle

SUPPORTED_SUFFIXES = {*SUPPORTED_IMAGE_SUFFIXES, ".pdf"}
SAFE_STEM = re.compile(r"[^a-zA-Z0-9._-]+")


class OCREngine(Protocol):
    def extract(self, image_paths: list[Path]) -> OCRDocument: ...


class InferenceFailure(RuntimeError):
    """The model ran but did not return a contract-valid extraction."""


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_dir: Path = Path("artifacts")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    vlm_prompt_style: PromptStyle = PromptStyle.JSON_SCHEMA
    ocr_invoice_rules: bool = False
    require_calibration: bool = False


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    document_id: str
    extraction: DocumentExtraction
    preprocessing: PreprocessingResult
    review_case_id: str | None = None


class DocumentIntelligenceService:
    def __init__(
        self,
        *,
        config: ServiceConfig | None = None,
        review_queue: ReviewQueue | None = None,
        tracer: InferenceTracer | None = None,
        ocr_engine: OCREngine | None = None,
        vlm_backend: VLMBackend | None = None,
        few_shot_examples: list[FewShotExample] | None = None,
        calibration_profile: CalibrationProfile | None = None,
    ) -> None:
        self.config = config or ServiceConfig()
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.review_queue = review_queue
        self.tracer = tracer
        self.ocr_engine = ocr_engine
        self.vlm_backend = vlm_backend
        self.few_shot_examples = few_shot_examples or []
        self.calibration_profile = calibration_profile

    @staticmethod
    def _safe_filename(filename: str) -> str:
        original = Path(filename or "").name
        suffix = Path(original).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document extension: {suffix or '<none>'}")
        stem = SAFE_STEM.sub("-", Path(original).stem).strip(".-") or "document"
        return f"{stem[:80]}{suffix}"

    def _trace(
        self,
        *,
        job_id: str,
        document_id: str,
        method: ExtractionMethod,
        model_name: str,
        latency_ms: int,
        page_count: int,
        extraction: DocumentExtraction | None,
        review_case_id: str | None = None,
        error: Exception | None = None,
    ) -> None:
        if self.tracer is None:
            return
        self.tracer.log(
            InferenceTrace(
                job_id=job_id,
                document_id=document_id,
                method=method,
                model_name=model_name,
                latency_ms=latency_ms,
                page_count=page_count,
                succeeded=extraction is not None and error is None,
                valid_json=extraction is not None,
                requires_human_review=(
                    extraction.requires_human_review if extraction is not None else True
                ),
                review_case_id=review_case_id,
                issue_codes=(
                    [issue.code for issue in extraction.validation_issues]
                    if extraction is not None
                    else []
                ),
                error_type=type(error).__name__ if error else None,
            ),
            extraction,
        )

    def extract_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        method: ExtractionMethod = ExtractionMethod.OCR_RULES,
        document_id: str | None = None,
    ) -> ExtractionResult:
        if not content:
            raise ValueError("Uploaded document is empty")
        if len(content) > self.config.max_upload_bytes:
            raise ValueError(
                f"Upload is {len(content)} bytes; maximum is {self.config.max_upload_bytes}"
            )
        if method not in {
            ExtractionMethod.OCR_RULES,
            ExtractionMethod.VLM_ZERO_SHOT,
            ExtractionMethod.VLM_FEW_SHOT,
        }:
            raise ValueError(f"Unsupported inference method: {method.value}")

        safe_filename = self._safe_filename(filename)
        job_id = str(uuid.uuid4())
        resolved_document_id = (
            document_id.strip() if document_id and document_id.strip() else job_id
        )
        job_dir = self.config.artifacts_dir / "jobs" / job_id
        source_path = job_dir / "input" / safe_filename
        source_path.parent.mkdir(parents=True, exist_ok=False)
        source_path.write_bytes(content)
        preprocessing = preprocess_document(
            source_path,
            job_dir / "pages",
            self.config.preprocessing,
        )
        image_paths = [page.output_path for page in preprocessing.pages]
        page_dir = image_paths[0].parent
        started = time.perf_counter()
        extraction: DocumentExtraction | None = None
        model_name = "tesseract" if method is ExtractionMethod.OCR_RULES else "qwen-vl"
        error: Exception | None = None
        try:
            if method is ExtractionMethod.OCR_RULES:
                engine = self.ocr_engine or TesseractEngine()
                extraction = extract_document(
                    engine.extract(image_paths),
                    document_id=resolved_document_id,
                    rules_version=(
                        OCRRulesVersion.INVOICE_V1 if self.config.ocr_invoice_rules
                        else OCRRulesVersion.V2
                    ),
                )
            else:
                backend = self.vlm_backend or TransformersQwenBackend()
                if method is ExtractionMethod.VLM_ZERO_SHOT:
                    vlm_result = run_zero_shot(
                        backend,
                        image_paths,
                        document_id=resolved_document_id,
                        prompt_style=self.config.vlm_prompt_style,
                    )
                else:
                    if not self.few_shot_examples:
                        raise ValueError("Few-shot inference requires a configured example catalog")
                    vlm_result = run_few_shot(
                        backend,
                        image_paths,
                        document_id=resolved_document_id,
                        examples=self.few_shot_examples,
                        prompt_style=self.config.vlm_prompt_style,
                    )
                (job_dir / "vlm-trace.json").write_text(
                    vlm_result.trace.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                model_name = vlm_result.trace.model_id
                extraction = vlm_result.extraction
                if extraction is None:
                    raise InferenceFailure(
                        vlm_result.trace.parse_error or "VLM returned an invalid extraction"
                    )

            matched_profile = False
            if self.calibration_profile is not None:
                inference_key = (
                    extraction.inference.method,
                    extraction.inference.model_name,
                    extraction.inference.prompt_version,
                )
                profile_key = (
                    self.calibration_profile.method,
                    self.calibration_profile.model_name,
                    self.calibration_profile.prompt_version,
                )
                if inference_key == profile_key:
                    extraction = apply_calibration(extraction, self.calibration_profile)
                    matched_profile = True
            if self.config.require_calibration and not matched_profile:
                extraction.validation_issues.append(ValidationIssue(
                    code="calibration_profile_missing",
                    message="No calibration profile matches this extraction",
                    blocking=True,
                ))
                extraction.requires_human_review = True

            review_case_id = None
            if extraction.requires_human_review and self.review_queue is not None:
                review_case_id = self.review_queue.enqueue(
                    extraction,
                    source_path=page_dir,
                ).case_id
            latency_ms = round((time.perf_counter() - started) * 1000)
            result = ExtractionResult(
                job_id=job_id,
                document_id=resolved_document_id,
                extraction=extraction,
                preprocessing=preprocessing,
                review_case_id=review_case_id,
            )
            (job_dir / "result.json").write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
            self._trace(
                job_id=job_id,
                document_id=resolved_document_id,
                method=method,
                model_name=extraction.inference.model_name,
                latency_ms=latency_ms,
                page_count=len(image_paths),
                extraction=extraction,
                review_case_id=review_case_id,
            )
            return result
        except Exception as caught:
            error = caught
            latency_ms = round((time.perf_counter() - started) * 1000)
            self._trace(
                job_id=job_id,
                document_id=resolved_document_id,
                method=method,
                model_name=model_name,
                latency_ms=latency_ms,
                page_count=len(image_paths),
                extraction=extraction,
                error=error,
            )
            raise
