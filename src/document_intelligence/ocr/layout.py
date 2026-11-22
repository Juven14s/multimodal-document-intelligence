"""Reading-order reconstruction from word-level OCR output."""

from __future__ import annotations

from collections import defaultdict

from document_intelligence.ocr.models import OCRLine, OCRWord
from document_intelligence.schemas import BoundingBox


def reconstruct_lines(words: list[OCRWord]) -> list[OCRLine]:
    grouped: dict[tuple[int, int, int, int], list[OCRWord]] = defaultdict(list)
    for word in words:
        key = (
            word.box.page,
            word.block_number,
            word.paragraph_number,
            word.line_number,
        )
        grouped[key].append(word)

    lines: list[OCRLine] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda word: (word.box.x0, word.word_number))
        page = ordered[0].box.page
        box = BoundingBox(
            page=page,
            x0=min(word.box.x0 for word in ordered),
            y0=min(word.box.y0 for word in ordered),
            x1=max(word.box.x1 for word in ordered),
            y1=max(word.box.y1 for word in ordered),
        )
        confidence = sum(word.confidence for word in ordered) / len(ordered)
        lines.append(
            OCRLine(
                text=" ".join(word.text for word in ordered),
                confidence=round(confidence, 6),
                box=box,
                words=ordered,
            )
        )
    return sorted(lines, key=lambda line: (line.box.page, line.box.y0, line.box.x0))
