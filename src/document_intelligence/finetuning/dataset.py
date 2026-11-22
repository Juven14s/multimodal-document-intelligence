"""Export canonical records as portable multimodal chat JSONL for Qwen SFT/LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_intelligence.data.io import read_records
from document_intelligence.data.models import CanonicalRecord, DatasetSplit
from document_intelligence.vlm.examples import response_from_record
from document_intelligence.vlm.prompting import PromptStyle, build_extraction_prompt


class ChatContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image", "text"]
    image: Path | None = None
    text: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> ChatContent:
        if self.type == "image" and self.image is None:
            raise ValueError("Image content requires an image path")
        if self.type == "text" and not self.text:
            raise ValueError("Text content requires non-empty text")
        return self


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: list[ChatContent] = Field(min_length=1)


class SFTExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    split: DatasetSplit
    image_sha256: str | None = None
    messages: list[ChatMessage] = Field(min_length=2, max_length=2)


class SFTDatasetReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_style: PromptStyle
    train_examples: int = Field(ge=0)
    validation_examples: int = Field(ge=0)
    test_examples: int = Field(ge=0)
    train_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _portable_image_path(image_path: Path, output_path: Path) -> Path:
    try:
        return Path(os.path.relpath(image_path.resolve(), output_path.parent.resolve()))
    except ValueError:
        return image_path.resolve()


def example_from_record(
    record: CanonicalRecord,
    *,
    data_root: Path,
    output_path: Path,
    prompt_style: PromptStyle,
) -> SFTExample:
    if record.manifest.benchmark_split is DatasetSplit.TEST:
        raise ValueError("Test records are forbidden in fine-tuning datasets")
    if record.manifest.image_path is None:
        raise ValueError(f"Record {record.manifest.document_id} has no materialized image")
    image_path = data_root / record.manifest.image_path
    if not image_path.is_file():
        raise ValueError(f"Image not found for {record.manifest.document_id}: {image_path}")
    response = response_from_record(record).model_dump_json(exclude_none=False)
    return SFTExample(
        document_id=record.manifest.document_id,
        split=record.manifest.benchmark_split,
        image_sha256=record.manifest.image_sha256,
        messages=[
            ChatMessage(
                role="user",
                content=[
                    ChatContent(
                        type="image",
                        image=_portable_image_path(image_path, output_path),
                    ),
                    ChatContent(type="text", text=build_extraction_prompt(prompt_style)),
                ],
            ),
            ChatMessage(
                role="assistant",
                content=[ChatContent(type="text", text=response)],
            ),
        ],
    )


def _write_jsonl(path: Path, examples: list[SFTExample]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(example.model_dump_json() + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_sft_dataset(
    manifest: Path,
    *,
    data_root: Path,
    output_dir: Path,
    prompt_style: PromptStyle = PromptStyle.COMPACT,
) -> SFTDatasetReport:
    records = read_records(manifest)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    train = [
        example_from_record(
            record,
            data_root=data_root,
            output_path=train_path,
            prompt_style=prompt_style,
        )
        for record in records
        if record.manifest.benchmark_split is DatasetSplit.TRAIN
    ]
    validation = [
        example_from_record(
            record,
            data_root=data_root,
            output_path=validation_path,
            prompt_style=prompt_style,
        )
        for record in records
        if record.manifest.benchmark_split is DatasetSplit.VALIDATION
    ]
    report = SFTDatasetReport(
        prompt_style=prompt_style,
        train_examples=len(train),
        validation_examples=len(validation),
        test_examples=0,
        train_sha256=_write_jsonl(train_path, train),
        validation_sha256=_write_jsonl(validation_path, validation),
    )
    (output_dir / "dataset-report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-style",
        choices=[style.value for style in PromptStyle],
        default=PromptStyle.COMPACT.value,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = export_sft_dataset(
        args.manifest,
        data_root=args.data_root,
        output_dir=args.output_dir,
        prompt_style=PromptStyle(args.prompt_style),
    )
    print(
        f"Exported {report.train_examples} train and "
        f"{report.validation_examples} validation examples to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
