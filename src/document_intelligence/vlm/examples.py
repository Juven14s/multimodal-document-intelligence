"""Leakage-safe creation and deterministic selection of few-shot examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.data.io import read_records
from document_intelligence.data.models import (
    CanonicalField,
    CanonicalRecord,
    DatasetName,
    DatasetSplit,
)
from document_intelligence.schemas import DocumentType
from document_intelligence.vlm.models import (
    VLMField,
    VLMFields,
    VLMLineItem,
    VLMResponse,
)


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    dataset: DatasetName
    split: DatasetSplit
    image_paths: list[Path] = Field(min_length=1)
    image_sha256: str | None = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    annotated_fields: set[CanonicalField]
    response: VLMResponse


class SelectionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    dataset: DatasetName | None = None
    image_sha256: str | None = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    desired_fields: set[CanonicalField] = Field(default_factory=set)


def _label(value) -> VLMField:
    raw = str(value) if value is not None else None
    return VLMField(value=value, raw_text=raw, confidence=1.0)


def response_from_record(record: CanonicalRecord) -> VLMResponse:
    annotation = record.annotation
    return VLMResponse(
        document_type=DocumentType.RECEIPT,
        classification_confidence=1.0,
        fields=VLMFields(
            merchant_name=_label(annotation.merchant_name),
            merchant_address=_label(annotation.merchant_address),
            issue_date=_label(annotation.issue_date),
            currency=_label(annotation.currency),
            subtotal=_label(annotation.subtotal),
            tax=_label(annotation.tax),
            total=_label(annotation.total),
            line_items=[
                VLMLineItem(
                    description=_label(item.description),
                    quantity=_label(item.quantity),
                    unit_price=_label(item.unit_price),
                    amount=_label(item.amount),
                )
                for item in annotation.line_items
            ],
        ),
    )


def example_from_record(record: CanonicalRecord, data_root: Path) -> FewShotExample:
    if record.manifest.benchmark_split is not DatasetSplit.TRAIN:
        raise ValueError("Few-shot examples must come from the benchmark training split")
    if record.manifest.image_path is None:
        raise ValueError(f"Record {record.manifest.document_id} has no materialized image")
    return FewShotExample(
        document_id=record.manifest.document_id,
        dataset=record.manifest.dataset,
        split=record.manifest.benchmark_split,
        image_paths=[(data_root / record.manifest.image_path).resolve()],
        image_sha256=record.manifest.image_sha256,
        image_width=record.manifest.image_width,
        image_height=record.manifest.image_height,
        annotated_fields=record.annotation.annotated_fields,
        response=response_from_record(record),
    )


def select_examples(
    examples: Iterable[FewShotExample],
    context: SelectionContext,
    *,
    count: int = 2,
    seed: int = 42,
) -> list[FewShotExample]:
    if count < 0:
        raise ValueError("count must be non-negative")
    target_ratio = context.image_width / context.image_height
    ranked: list[tuple[float, str, FewShotExample]] = []
    for example in examples:
        if example.split is not DatasetSplit.TRAIN:
            continue
        if example.document_id == context.document_id:
            continue
        if (
            context.image_sha256
            and example.image_sha256
            and example.image_sha256 == context.image_sha256
        ):
            continue
        score = 0.0
        if context.dataset is not None and example.dataset is context.dataset:
            score += 2.0
        example_ratio = example.image_width / example.image_height
        score += 1.0 / (1.0 + abs(math.log(target_ratio / example_ratio)))
        if context.desired_fields:
            intersection = len(context.desired_fields & example.annotated_fields)
            union = len(context.desired_fields | example.annotated_fields)
            score += intersection / union if union else 0.0
        tie_break = hashlib.sha256(
            f"{seed}:{context.document_id}:{example.document_id}".encode()
        ).hexdigest()
        ranked.append((score, tie_break, example))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked[:count]]


def write_examples(path: Path, examples: Iterable[FewShotExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for example in examples:
            portable_paths: list[Path] = []
            for image_path in example.image_paths:
                try:
                    portable_paths.append(
                        Path(os.path.relpath(image_path.resolve(), path.parent.resolve()))
                    )
                except ValueError:
                    portable_paths.append(image_path.resolve())
            portable = example.model_copy(update={"image_paths": portable_paths})
            stream.write(json.dumps(portable.model_dump(mode="json"), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_examples(path: Path) -> list[FewShotExample]:
    with path.open("r", encoding="utf-8") as stream:
        examples = [FewShotExample.model_validate_json(line) for line in stream if line.strip()]
    return [
        example.model_copy(
            update={
                "image_paths": [
                    image_path
                    if image_path.is_absolute()
                    else (path.parent / image_path).resolve()
                    for image_path in example.image_paths
                ]
            }
        )
        for example in examples
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/fewshot.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = read_records(args.manifest)
    examples = [
        example_from_record(record, args.data_root)
        for record in records
        if record.manifest.benchmark_split is DatasetSplit.TRAIN
    ]
    count = write_examples(args.output, examples)
    print(f"Wrote {count} training examples to {args.output}")


if __name__ == "__main__":
    main()
