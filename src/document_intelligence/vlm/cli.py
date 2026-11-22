"""Run zero-shot structured extraction with a local Qwen-VL model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from document_intelligence.data.models import DatasetName
from document_intelligence.vlm.backend import QwenBackendConfig, TransformersQwenBackend
from document_intelligence.vlm.examples import SelectionContext, read_examples, select_examples
from document_intelligence.vlm.pipeline import run_few_shot, run_zero_shot
from document_intelligence.vlm.prompting import PromptStyle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/vlm"))
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument(
        "--prompt-style",
        choices=[style.value for style in PromptStyle],
        default=PromptStyle.JSON_SCHEMA.value,
    )
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--few-shot-count", type=int, default=2)
    parser.add_argument("--target-dataset", choices=[dataset.value for dataset in DatasetName])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = TransformersQwenBackend(
        QwenBackendConfig(model_id=args.model_id, max_new_tokens=args.max_new_tokens)
    )
    if args.examples:
        with Image.open(args.images[0]) as image:
            width, height = image.size
        context = SelectionContext(
            document_id=args.document_id,
            dataset=DatasetName(args.target_dataset) if args.target_dataset else None,
            image_sha256=_sha256(args.images[0]),
            image_width=width,
            image_height=height,
        )
        selected = select_examples(
            read_examples(args.examples),
            context,
            count=args.few_shot_count,
        )
        result = run_few_shot(
            backend,
            args.images,
            document_id=args.document_id,
            examples=selected,
            prompt_style=PromptStyle(args.prompt_style),
        )
    else:
        result = run_zero_shot(
            backend,
            args.images,
            document_id=args.document_id,
            prompt_style=PromptStyle(args.prompt_style),
        )
    output_dir = args.output_dir / args.document_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trace.json").write_text(
        json.dumps(result.trace.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if result.extraction is not None:
        (output_dir / "extraction.json").write_text(
            json.dumps(result.extraction.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(output_dir / "extraction.json")
    else:
        raise SystemExit(f"Invalid model output; see {output_dir / 'trace.json'}")


if __name__ == "__main__":
    main()
