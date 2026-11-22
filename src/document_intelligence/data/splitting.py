"""Stable benchmark split assignment that preserves official test sets."""

from __future__ import annotations

import hashlib

from document_intelligence.data.models import DatasetName, DatasetSplit


def assign_benchmark_split(
    dataset: DatasetName,
    source_split: str,
    document_id: str,
    *,
    sroie_validation_fraction: float = 0.15,
) -> DatasetSplit:
    normalized = source_split.lower()
    if normalized == "test":
        return DatasetSplit.TEST
    if normalized in {"validation", "valid", "dev"}:
        return DatasetSplit.VALIDATION
    if normalized != "train":
        raise ValueError(f"Unsupported source split: {source_split}")

    if dataset is not DatasetName.SROIE:
        return DatasetSplit.TRAIN
    if not 0.0 <= sroie_validation_fraction < 1.0:
        raise ValueError("sroie_validation_fraction must be in [0, 1)")

    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return (
        DatasetSplit.VALIDATION
        if bucket < sroie_validation_fraction
        else DatasetSplit.TRAIN
    )
