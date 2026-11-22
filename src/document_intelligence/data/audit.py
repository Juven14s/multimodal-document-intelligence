"""Dataset coverage and integrity summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.data.models import CanonicalRecord


class DatasetAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_count: int = Field(ge=0)
    records_by_dataset: dict[str, int]
    records_by_split: dict[str, int]
    annotated_field_counts: dict[str, int]
    duplicate_document_ids: list[str]
    records_without_total: list[str]
    duplicate_image_groups: int = Field(default=0, ge=0)
    images_shared_across_splits: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


def audit_records(records: Iterable[CanonicalRecord]) -> DatasetAudit:
    materialized = list(records)
    ids = Counter(record.manifest.document_id for record in materialized)
    datasets = Counter(record.manifest.dataset.value for record in materialized)
    splits = Counter(record.manifest.benchmark_split.value for record in materialized)
    fields = Counter(
        field.value
        for record in materialized
        for field in record.annotation.annotated_fields
    )
    images: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in materialized:
        entry = record.manifest
        if entry.image_sha256 is not None:
            images[entry.image_sha256].append((entry.benchmark_split.value, entry.document_id))

    shared_images = {}
    for digest, entries in sorted(images.items()):
        records_by_split: dict[str, list[str]] = defaultdict(list)
        for split, document_id in sorted(entries):
            records_by_split[split].append(document_id)
        if len(records_by_split) > 1:
            shared_images[digest] = dict(records_by_split)
    return DatasetAudit(
        record_count=len(materialized),
        records_by_dataset=dict(sorted(datasets.items())),
        records_by_split=dict(sorted(splits.items())),
        annotated_field_counts=dict(sorted(fields.items())),
        duplicate_document_ids=sorted(key for key, count in ids.items() if count > 1),
        records_without_total=sorted(
            record.manifest.document_id
            for record in materialized
            if record.annotation.total is None
        ),
        duplicate_image_groups=sum(len(entries) > 1 for entries in images.values()),
        images_shared_across_splits=shared_images,
    )
