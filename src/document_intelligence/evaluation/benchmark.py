"""Resumable batch benchmark over canonical dataset manifests."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from document_intelligence.data.io import read_records
from document_intelligence.data.models import CanonicalRecord, DatasetName, DatasetSplit
from document_intelligence.evaluation.metrics import BenchmarkResult, evaluate_predictions
from document_intelligence.observability import InferenceTracer, JsonlTraceSink
from document_intelligence.schemas import DocumentExtraction, ExtractionMethod
from document_intelligence.service import DocumentIntelligenceService, ServiceConfig
from document_intelligence.vlm.backend import QwenBackendConfig, TransformersQwenBackend
from document_intelligence.vlm.examples import (
    FewShotExample,
    SelectionContext,
    read_examples,
    select_examples,
)
from document_intelligence.vlm.prompting import PromptStyle

BENCHMARK_METHODS = (
    ExtractionMethod.OCR_RULES,
    ExtractionMethod.VLM_ZERO_SHOT,
    ExtractionMethod.VLM_FEW_SHOT,
)


class BenchmarkFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_id: str
    error_type: str
    message: str


class LatencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    mean: float = Field(ge=0.0)
    p50: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)


class BenchmarkExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset: DatasetName
    split: DatasetSplit
    method: ExtractionMethod
    started_at: datetime
    finished_at: datetime
    elapsed_ms: int = Field(ge=0)
    selected_documents: int = Field(ge=0)
    previously_completed: int = Field(ge=0)
    attempted: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    completed_documents: int = Field(ge=0)
    model_latency_ms: LatencySummary
    metrics: BenchmarkResult
    predictions_path: Path
    failures_path: Path


def _latency_summary(values: list[int]) -> LatencySummary:
    if not values:
        return LatencySummary(count=0, mean=0, p50=0, p95=0, minimum=0, maximum=0)
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
        return ordered[index]

    return LatencySummary(
        count=len(ordered),
        mean=statistics.fmean(ordered),
        p50=percentile(0.50),
        p95=percentile(0.95),
        minimum=ordered[0],
        maximum=ordered[-1],
    )


def _read_predictions(path: Path) -> tuple[list[DocumentExtraction], int]:
    predictions: list[DocumentExtraction] = []
    invalid = 0
    if not path.exists():
        return predictions, invalid
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                predictions.append(DocumentExtraction.model_validate_json(line))
            except (ValidationError, ValueError):
                invalid += 1
    return predictions, invalid


def _select_records(
    records: list[CanonicalRecord],
    split: DatasetSplit,
    limit: int | None,
) -> list[CanonicalRecord]:
    selected = [record for record in records if record.manifest.benchmark_split is split]
    return selected[:limit] if limit is not None else selected


def _image_path(record: CanonicalRecord, manifest_path: Path) -> Path:
    relative = record.manifest.image_path
    if relative is None:
        raise ValueError(f"Record {record.manifest.document_id} has no materialized image")
    portable = Path(str(relative).replace("\\", "/"))
    path = portable if portable.is_absolute() else manifest_path.parent / portable
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _few_shot_for_record(
    record: CanonicalRecord,
    catalog: list[FewShotExample],
    count: int,
) -> list[FewShotExample]:
    return select_examples(
        catalog,
        SelectionContext(
            document_id=record.manifest.document_id,
            dataset=record.manifest.dataset,
            image_sha256=record.manifest.image_sha256,
            image_width=record.manifest.image_width,
            image_height=record.manifest.image_height,
            desired_fields=record.annotation.annotated_fields,
        ),
        count=count,
    )


def run_benchmark(
    manifest_path: Path,
    *,
    predictions_path: Path,
    report_path: Path,
    failures_path: Path,
    artifacts_dir: Path,
    method: ExtractionMethod = ExtractionMethod.OCR_RULES,
    split: DatasetSplit = DatasetSplit.TEST,
    limit: int | None = None,
    resume: bool = True,
    service: DocumentIntelligenceService | None = None,
    few_shot_catalog: list[FewShotExample] | None = None,
    few_shot_count: int = 2,
) -> BenchmarkExecutionReport:
    records = read_records(manifest_path)
    selected = _select_records(records, split, limit)
    if not selected:
        raise ValueError(f"No records selected for split '{split.value}'")
    datasets = {record.manifest.dataset for record in selected}
    if len(datasets) != 1:
        raise ValueError("A benchmark run must contain exactly one dataset")
    dataset = next(iter(datasets))

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        predictions_path.write_text("", encoding="utf-8")
        failures_path.write_text("", encoding="utf-8")

    existing, _ = _read_predictions(predictions_path)
    selected_ids = {record.manifest.document_id for record in selected}
    completed_ids = {
        prediction.document_id for prediction in existing if prediction.document_id in selected_ids
    }
    previously_completed = len(completed_ids)
    run_service = service or DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=artifacts_dir / "inference"),
        tracer=InferenceTracer(
            [JsonlTraceSink(artifacts_dir / "traces" / "inference.jsonl")]
        ),
    )

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    succeeded = 0
    failed = 0
    for record in selected:
        document_id = record.manifest.document_id
        if document_id in completed_ids:
            continue
        try:
            image_path = _image_path(record, manifest_path)
            if method is ExtractionMethod.VLM_FEW_SHOT:
                if not few_shot_catalog:
                    raise ValueError("Few-shot method requires an example catalog")
                run_service.few_shot_examples = _few_shot_for_record(
                    record,
                    few_shot_catalog,
                    few_shot_count,
                )
            result = run_service.extract_bytes(
                image_path.read_bytes(),
                filename=image_path.name,
                method=method,
                document_id=document_id,
            )
            with predictions_path.open("a", encoding="utf-8") as stream:
                stream.write(result.extraction.model_dump_json() + "\n")
            completed_ids.add(document_id)
            succeeded += 1
        except Exception as error:
            failure = BenchmarkFailure(
                document_id=document_id,
                error_type=type(error).__name__,
                message=str(error),
            )
            with failures_path.open("a", encoding="utf-8") as stream:
                stream.write(failure.model_dump_json() + "\n")
            failed += 1

    predictions, invalid_final = _read_predictions(predictions_path)
    selected_predictions = [
        prediction for prediction in predictions if prediction.document_id in selected_ids
    ]
    predicted_ids = {prediction.document_id for prediction in selected_predictions}
    missing_predictions = len(selected_ids - predicted_ids)
    metrics = evaluate_predictions(
        selected,
        selected_predictions,
        invalid_prediction_count=invalid_final + missing_predictions,
    )
    latencies = [prediction.inference.latency_ms for prediction in selected_predictions]
    finished_at = datetime.now(UTC)
    report = BenchmarkExecutionReport(
        run_id=str(uuid.uuid4()),
        dataset=dataset,
        split=split,
        method=method,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
        selected_documents=len(selected),
        previously_completed=previously_completed,
        attempted=succeeded + failed,
        succeeded=succeeded,
        failed=failed,
        completed_documents=len(predicted_ids),
        model_latency_ms=_latency_summary(latencies),
        metrics=metrics,
        predictions_path=predictions_path,
        failures_path=failures_path,
    )
    payload = json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
    report_path.write_text(payload, encoding="utf-8")
    run_report = report_path.parent / "runs" / f"{report.run_id}.json"
    run_report.parent.mkdir(parents=True, exist_ok=True)
    run_report.write_text(payload, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--method", choices=[method.value for method in BENCHMARK_METHODS])
    parser.add_argument("--split", choices=[split.value for split in DatasetSplit], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks/latest"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--prompt-style",
        choices=[style.value for style in PromptStyle],
        default=PromptStyle.JSON_SCHEMA.value,
    )
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--few-shot-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    method = ExtractionMethod(args.method or ExtractionMethod.OCR_RULES.value)
    output_dir = args.output_dir
    backend = None
    if method in {ExtractionMethod.VLM_ZERO_SHOT, ExtractionMethod.VLM_FEW_SHOT}:
        backend = TransformersQwenBackend(
            QwenBackendConfig(
                model_id=args.model_id,
                max_new_tokens=args.max_new_tokens,
            )
        )
    service = DocumentIntelligenceService(
        config=ServiceConfig(
            artifacts_dir=output_dir / "inference",
            vlm_prompt_style=PromptStyle(args.prompt_style),
        ),
        tracer=InferenceTracer([JsonlTraceSink(output_dir / "traces" / "inference.jsonl")]),
        vlm_backend=backend,
    )
    catalog = read_examples(args.examples) if args.examples else None
    report = run_benchmark(
        args.manifest,
        predictions_path=output_dir / "predictions.jsonl",
        report_path=output_dir / "report.json",
        failures_path=output_dir / "failures.jsonl",
        artifacts_dir=output_dir,
        method=method,
        split=DatasetSplit(args.split),
        limit=args.limit,
        resume=not args.overwrite,
        service=service,
        few_shot_catalog=catalog,
        few_shot_count=args.few_shot_count,
    )
    print(
        f"{report.dataset.value}/{report.split.value}: "
        f"{report.succeeded} succeeded, {report.failed} failed; "
        f"report={output_dir / 'report.json'}"
    )


if __name__ == "__main__":
    main()
