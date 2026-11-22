"""Tesseract TSV adapter with normalized coordinates and confidence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.ocr.layout import reconstruct_lines
from document_intelligence.ocr.models import OCRDocument, OCRPage, OCRWord
from document_intelligence.schemas import BoundingBox


class TesseractConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = "eng"
    page_segmentation_mode: int = Field(default=4, ge=0, le=13)
    engine_mode: int = Field(default=1, ge=0, le=3)
    minimum_confidence: float = Field(default=0.20, ge=0.0, le=1.0)
    preserve_interword_spaces: bool = True


def words_from_tesseract_data(
    data: dict[str, list[Any]],
    *,
    page_number: int,
    width: int,
    height: int,
    minimum_confidence: float,
) -> list[OCRWord]:
    required = {
        "text",
        "conf",
        "left",
        "top",
        "width",
        "height",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Tesseract output is missing columns: {sorted(missing)}")

    words: list[OCRWord] = []
    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        try:
            confidence = float(data["conf"][index]) / 100.0
        except (TypeError, ValueError):
            continue
        if not text or confidence < minimum_confidence:
            continue
        left = max(0, int(data["left"][index]))
        top = max(0, int(data["top"][index]))
        right = min(width, left + max(0, int(data["width"][index])))
        bottom = min(height, top + max(0, int(data["height"][index])))
        if right <= left or bottom <= top:
            continue
        words.append(
            OCRWord(
                text=text,
                confidence=min(1.0, max(0.0, confidence)),
                box=BoundingBox(
                    page=page_number,
                    x0=left / width,
                    y0=top / height,
                    x1=right / width,
                    y1=bottom / height,
                ),
                block_number=int(data["block_num"][index]),
                paragraph_number=int(data["par_num"][index]),
                line_number=int(data["line_num"][index]),
                word_number=int(data["word_num"][index]),
            )
        )
    return words


class TesseractEngine:
    def __init__(self, config: TesseractConfig | None = None) -> None:
        self.config = config or TesseractConfig()

    def _runtime(self):
        import pytesseract

        try:
            version = str(pytesseract.get_tesseract_version())
        except pytesseract.TesseractNotFoundError as error:
            raise RuntimeError(
                "Tesseract executable was not found. Install Tesseract OCR and ensure "
                "the 'tesseract' command is available on PATH."
            ) from error
        return pytesseract, version

    def extract(
        self, image_paths: list[Path], *, raw_output_dir: Path | None = None
    ) -> OCRDocument:
        if not image_paths:
            raise ValueError("At least one page image is required")
        pytesseract, version = self._runtime()
        started = time.perf_counter()
        pages: list[OCRPage] = []
        preserve = int(self.config.preserve_interword_spaces)
        command_config = (
            f"--oem {self.config.engine_mode} --psm {self.config.page_segmentation_mode} "
            f"-c preserve_interword_spaces={preserve}"
        )
        if raw_output_dir is not None:
            raw_output_dir.mkdir(parents=True, exist_ok=True)
        for page_number, path in enumerate(image_paths, start=1):
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                data = pytesseract.image_to_data(
                    rgb,
                    lang=self.config.language,
                    config=command_config,
                    output_type=pytesseract.Output.DICT,
                )
                if raw_output_dir is not None:
                    (raw_output_dir / f"tesseract-{page_number:04d}.json").write_text(
                        json.dumps(data, ensure_ascii=False), encoding="utf-8"
                    )
                words = words_from_tesseract_data(
                    data,
                    page_number=page_number,
                    width=rgb.width,
                    height=rgb.height,
                    minimum_confidence=self.config.minimum_confidence,
                )
                pages.append(
                    OCRPage(
                        page_number=page_number,
                        image_path=path,
                        width=rgb.width,
                        height=rgb.height,
                        words=words,
                        lines=reconstruct_lines(words),
                    )
                )
        latency_ms = round((time.perf_counter() - started) * 1000)
        return OCRDocument(
            engine_name="tesseract",
            engine_version=version,
            language=self.config.language,
            latency_ms=latency_ms,
            pages=pages,
        )
