import json
from pathlib import Path

import pytest

from document_intelligence.data.io import write_records
from document_intelligence.data.models import (
    CanonicalAnnotation,
    CanonicalField,
    CanonicalRecord,
    DatasetName,
    DatasetSplit,
    ManifestEntry,
)
from document_intelligence.finetuning.dataset import (
    example_from_record,
    export_sft_dataset,
)
from document_intelligence.vlm.models import VLMResponse


def make_record(document_id: str, split: DatasetSplit) -> CanonicalRecord:
    fields = {CanonicalField.MERCHANT_NAME, CanonicalField.TOTAL}
    return CanonicalRecord(
        manifest=ManifestEntry(
            document_id=document_id,
            dataset=DatasetName.SROIE,
            source_split=split.value,
            benchmark_split=split,
            source_key=document_id,
            image_path=Path(f"images/{document_id}.png"),
            image_width=100,
            image_height=200,
            image_sha256=("a" if split is DatasetSplit.TRAIN else "b") * 64,
            ocr_region_count=0,
            annotated_fields=fields,
        ),
        annotation=CanonicalAnnotation(
            merchant_name="Example Shop",
            total="12.30",
            annotated_fields=fields,
        ),
    )


def test_sft_export_is_portable_and_excludes_test_split(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    records = [
        make_record("train-1", DatasetSplit.TRAIN),
        make_record("validation-1", DatasetSplit.VALIDATION),
        make_record("test-1", DatasetSplit.TEST),
    ]
    for record in records:
        image_path = data_root / record.manifest.image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fixture-image")
    manifest = data_root / "sroie.jsonl"
    write_records(manifest, records)
    output_dir = data_root / "sft"

    report = export_sft_dataset(
        manifest,
        data_root=data_root,
        output_dir=output_dir,
    )

    assert report.train_examples == 1
    assert report.validation_examples == 1
    assert report.test_examples == 0
    train_payload = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
    assert train_payload["document_id"] == "train-1"
    assert train_payload["split"] == "train"
    image = train_payload["messages"][0]["content"][0]["image"]
    assert not Path(image).is_absolute()
    assistant = train_payload["messages"][1]["content"][0]["text"]
    assert VLMResponse.model_validate_json(assistant).fields.merchant_name.value == "Example Shop"
    assert "test-1" not in (output_dir / "train.jsonl").read_text(encoding="utf-8")
    assert "test-1" not in (output_dir / "validation.jsonl").read_text(encoding="utf-8")


def test_sft_example_rejects_test_record(tmp_path: Path) -> None:
    record = make_record("test-1", DatasetSplit.TEST)

    with pytest.raises(ValueError, match="Test records are forbidden"):
        example_from_record(
            record,
            data_root=tmp_path,
            output_path=tmp_path / "test.jsonl",
            prompt_style="compact",
        )
