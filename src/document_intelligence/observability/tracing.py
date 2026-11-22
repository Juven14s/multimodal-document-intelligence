"""Privacy-conscious operational traces with an optional MLflow mirror."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.schemas import DocumentExtraction, ExtractionMethod

LOGGER = logging.getLogger(__name__)


class InferenceTrace(BaseModel):
    """Operational metadata; extracted business values deliberately stay out of JSONL."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    job_id: str
    document_id: str
    method: ExtractionMethod
    model_name: str
    latency_ms: int = Field(ge=0)
    page_count: int = Field(ge=0)
    succeeded: bool
    valid_json: bool
    requires_human_review: bool = False
    review_case_id: str | None = None
    issue_codes: list[str] = Field(default_factory=list)
    error_type: str | None = None


class TraceSink(Protocol):
    def log(
        self,
        trace: InferenceTrace,
        extraction: DocumentExtraction | None = None,
    ) -> None: ...


class JsonlTraceSink:
    """Append one atomic, process-local trace line for each inference."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        trace: InferenceTrace,
        extraction: DocumentExtraction | None = None,
    ) -> None:
        del extraction
        line = trace.model_dump_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)


class MLflowTraceSink:
    """Lazy MLflow adapter so MLflow remains an optional runtime dependency."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str = "document-intelligence",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def log(
        self,
        trace: InferenceTrace,
        extraction: DocumentExtraction | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError('MLflow tracing requires: pip install -e ".[mlops]"') from error

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        tags = {
            "document_id": trace.document_id,
            "job_id": trace.job_id,
            "status": "succeeded" if trace.succeeded else "failed",
        }
        with mlflow.start_run(run_name=f"inference-{trace.job_id}", tags=tags):
            mlflow.log_params(
                {
                    "method": trace.method.value,
                    "model_name": trace.model_name,
                    "page_count": trace.page_count,
                }
            )
            mlflow.log_metrics(
                {
                    "latency_ms": trace.latency_ms,
                    "valid_json": float(trace.valid_json),
                    "requires_human_review": float(trace.requires_human_review),
                    "issue_count": float(len(trace.issue_codes)),
                }
            )
            mlflow.log_dict(trace.model_dump(mode="json"), "trace.json")
            if extraction is not None:
                mlflow.log_dict(
                    extraction.model_dump(mode="json"),
                    "extraction.json",
                )


class InferenceTracer:
    """Fan out a trace to local and remote sinks."""

    def __init__(self, sinks: Iterable[TraceSink], *, strict: bool = False) -> None:
        self.sinks = list(sinks)
        self.strict = strict

    def log(
        self,
        trace: InferenceTrace,
        extraction: DocumentExtraction | None = None,
    ) -> None:
        for sink in self.sinks:
            try:
                sink.log(trace, extraction)
            except Exception:
                if self.strict:
                    raise
                LOGGER.exception("Inference trace sink failed: %s", type(sink).__name__)
