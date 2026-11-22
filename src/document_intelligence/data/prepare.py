"""CLI for streaming source datasets into canonical JSONL manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from document_intelligence.data.adapters import adapt_cord_record, adapt_sroie_record
from document_intelligence.data.audit import audit_records
from document_intelligence.data.io import write_records
from document_intelligence.data.models import CanonicalRecord, DatasetName

DATASET_IDS = {
    DatasetName.CORD_V2: "naver-clova-ix/cord-v2",
    DatasetName.SROIE: "jsdnrs/ICDAR2019-SROIE",
}
SOURCE_SPLITS = {
    DatasetName.CORD_V2: ("train", "validation", "test"),
    DatasetName.SROIE: ("train", "test"),
}


def _load_stream(dataset: DatasetName, split: str) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError('Install data dependencies with: pip install -e ".[data]"') from error
    return load_dataset(DATASET_IDS[dataset], split=split, streaming=True)


def _convert(
    dataset: DatasetName,
    rows: Iterable[Mapping[str, Any]],
    split: str,
    limit: int | None,
    output_dir: Path,
) -> Iterator[CanonicalRecord]:
    adapter = adapt_cord_record if dataset is DatasetName.CORD_V2 else adapt_sroie_record
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        record = adapter(row, source_split=split, index=index)
        image = row.get("image")
        if image is None or not hasattr(image, "save"):
            raise ValueError(f"Record {record.manifest.document_id} has no decodable image")
        relative_path = (
            Path("images") / dataset.value / split / f"{record.manifest.document_id}.png"
        )
        absolute_path = output_dir / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        if not absolute_path.is_file():
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            image.save(absolute_path, format="PNG")
        digest = hashlib.sha256(absolute_path.read_bytes()).hexdigest()
        manifest = record.manifest.model_copy(
            update={"image_path": relative_path, "image_sha256": digest}
        )
        yield record.model_copy(update={"manifest": manifest})


def prepare_dataset(
    dataset: DatasetName,
    output_dir: Path,
    *,
    limit_per_split: int | None = None,
) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    for split in SOURCE_SPLITS[dataset]:
        records.extend(
            _convert(
                dataset,
                _load_stream(dataset, split),
                split,
                limit_per_split,
                output_dir,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_records(output_dir / f"{dataset.value}.jsonl", records)
    audit = audit_records(records)
    (output_dir / f"{dataset.value}_audit.json").write_text(
        json.dumps(audit.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=[dataset.value for dataset in DatasetName])
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit-per-split", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = prepare_dataset(
        DatasetName(args.dataset),
        args.output_dir,
        limit_per_split=args.limit_per_split,
    )
    print(f"Prepared {len(records)} records in {args.output_dir}")


if __name__ == "__main__":
    main()
