from pathlib import Path

from document_intelligence.data.adapters import adapt_sroie_record
from document_intelligence.data.audit import audit_records
from document_intelligence.data.io import read_records, write_records
from document_intelligence.data.models import DatasetSplit


def make_record(key: str):
    return adapt_sroie_record(
        {
            "key": key,
            "image_size": {"width": 100, "height": 200},
            "entities": {"company": "Shop", "date": "01/01/2019", "total": "2.50"},
            "words": [],
            "bboxes": [],
        },
        source_split="train",
        index=0,
    )


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = [make_record("A"), make_record("B")]
    path = tmp_path / "records.jsonl"

    assert write_records(path, records) == 2
    assert read_records(path) == records


def test_audit_detects_duplicate_ids() -> None:
    record = make_record("A")
    audit = audit_records([record, record])

    assert audit.record_count == 2
    assert audit.duplicate_document_ids == ["sroie-A"]
    assert audit.annotated_field_counts["total"] == 2


def test_audit_detects_identical_images_under_different_ids() -> None:
    records = [make_record(key) for key in ("A", "B", "C", "D", "E", "F")]
    for record, digest, split in zip(
        records,
        ["a" * 64, "a" * 64, "b" * 64, "b" * 64, None, None],
        [DatasetSplit.TRAIN, DatasetSplit.TEST] + [DatasetSplit.TRAIN] * 4,
        strict=True,
    ):
        record.manifest.image_sha256 = digest
        record.manifest.benchmark_split = split

    audit = audit_records(records)

    assert audit.duplicate_document_ids == []
    assert audit.duplicate_image_groups == 2
    assert audit.images_shared_across_splits == {
        "a" * 64: {"test": ["sroie-B"], "train": ["sroie-A"]}
    }
