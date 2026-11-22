from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from document_intelligence.ocr.models import OCRDocument, OCRLine, OCRPage, OCRWord
from document_intelligence.schemas import BoundingBox


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (800, 1200), "white").save(stream, format="PNG")
    return stream.getvalue()


def _line(number: int, text: str, y0: float, y1: float) -> OCRLine:
    box = BoundingBox(page=1, x0=0.1, y0=y0, x1=0.9, y1=y1)
    word = OCRWord(
        text=text,
        confidence=0.95,
        box=box,
        block_number=1,
        paragraph_number=1,
        line_number=number,
        word_number=1,
    )
    return OCRLine(text=text, confidence=0.95, box=box, words=[word])


def receipt_ocr(*, include_total: bool = True) -> OCRDocument:
    lines = [
        _line(1, "EXAMPLE CAFE", 0.03, 0.06),
        _line(2, "RECEIPT NO: AB-1234", 0.09, 0.12),
        _line(3, "Coffee 10.00", 0.30, 0.34),
        _line(4, "SUBTOTAL EUR 10.00", 0.72, 0.76),
    ]
    if include_total:
        lines.append(_line(5, "GRAND TOTAL EUR 10.00", 0.85, 0.90))
    lines.append(_line(6, "CASH EUR 20.00", 0.92, 0.96))
    return OCRDocument(
        engine_name="fixture-ocr",
        engine_version="1.0",
        language="eng",
        latency_ms=2,
        pages=[
            OCRPage(
                page_number=1,
                image_path=Path("fixture.png"),
                width=800,
                height=1200,
                words=[word for line in lines for word in line.words],
                lines=lines,
            )
        ],
    )


class FakeOCREngine:
    def __init__(self, *, include_total: bool = True) -> None:
        self.include_total = include_total

    def extract(self, image_paths: list[Path]) -> OCRDocument:
        assert image_paths and all(path.is_file() for path in image_paths)
        return receipt_ocr(include_total=self.include_total)
