from decimal import Decimal
from pathlib import Path

from PIL import Image
from product_fixtures import FakeOCREngine

from document_intelligence.data.io import write_records
from document_intelligence.data.models import (
    CanonicalAnnotation,
    CanonicalField,
    CanonicalRecord,
    DatasetName,
    DatasetSplit,
    ManifestEntry,
)
from document_intelligence.evaluation.benchmark import build_parser, run_benchmark
from document_intelligence.service import DocumentIntelligenceService, ServiceConfig


def make_record(document_id: str, image_path: Path) -> CanonicalRecord:
    annotated = {CanonicalField.MERCHANT_NAME, CanonicalField.TOTAL}
    return CanonicalRecord(
        manifest=ManifestEntry(
            document_id=document_id,
            dataset=DatasetName.SROIE,
            source_split="test",
            benchmark_split=DatasetSplit.TEST,
            source_key=document_id,
            image_path=image_path,
            image_width=800,
            image_height=1200,
            ocr_region_count=0,
            annotated_fields=annotated,
        ),
        annotation=CanonicalAnnotation(
            merchant_name="EXAMPLE CAFE",
            total=Decimal("10.00"),
            annotated_fields=annotated,
        ),
    )


def test_batch_benchmark_is_resumable_and_evaluates_predictions(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    for name in ("one.png", "two.png"):
        Image.new("RGB", (800, 1200), "white").save(images_dir / name)
    manifest = data_dir / "sroie.jsonl"
    write_records(
        manifest,
        [
            make_record("doc-1", Path("images/one.png")),
            make_record("doc-2", Path("images\\two.png")),
        ],
    )
    output = tmp_path / "benchmark"
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=output / "service"),
        ocr_engine=FakeOCREngine(),
    )
    kwargs = {
        "predictions_path": output / "predictions.jsonl",
        "report_path": output / "report.json",
        "failures_path": output / "failures.jsonl",
        "artifacts_dir": output,
        "service": service,
    }

    first = run_benchmark(manifest, **kwargs)
    second = run_benchmark(manifest, **kwargs)

    assert first.succeeded == 2
    assert first.completed_documents == 2
    assert first.metrics.prediction_coverage == 1.0
    assert first.metrics.fields["merchant_name"].f1 == 1.0
    assert first.metrics.fields["total"].exact_match == 1.0
    assert second.previously_completed == 2
    assert second.attempted == 0
    assert second.completed_documents == 2
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 2
    assert len(list((output / "runs").glob("*.json"))) == 2


def test_batch_benchmark_records_failures_without_stopping(tmp_path: Path) -> None:
    manifest = tmp_path / "sroie.jsonl"
    write_records(manifest, [make_record("missing", Path("missing.png"))])
    output = tmp_path / "benchmark"
    service = DocumentIntelligenceService(
        config=ServiceConfig(artifacts_dir=output / "service"),
        ocr_engine=FakeOCREngine(),
    )

    report = run_benchmark(
        manifest,
        predictions_path=output / "predictions.jsonl",
        report_path=output / "report.json",
        failures_path=output / "failures.jsonl",
        artifacts_dir=output,
        service=service,
    )

    assert report.failed == 1
    assert report.completed_documents == 0
    assert report.metrics.prediction_coverage == 0.0
    assert report.metrics.valid_json_rate == 0.0
    assert "FileNotFoundError" in (output / "failures.jsonl").read_text()


def test_benchmark_parser_accepts_generation_budget() -> None:
    args = build_parser().parse_args(["manifest.jsonl", "--max-new-tokens", "512"])

    assert args.max_new_tokens == 512
