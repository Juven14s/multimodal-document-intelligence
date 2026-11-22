"""JSONL persistence for deterministic, inspectable dataset artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from document_intelligence.data.models import CanonicalRecord


def write_records(path: Path, records: Iterable[CanonicalRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            stream.write("\n")
            count += 1
    return count


def read_records(path: Path) -> list[CanonicalRecord]:
    with path.open("r", encoding="utf-8") as stream:
        return [CanonicalRecord.model_validate_json(line) for line in stream if line.strip()]
