"""Engine-neutral OCR output contracts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.schemas import BoundingBox


class OCRWord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox
    block_number: int = Field(ge=0)
    paragraph_number: int = Field(ge=0)
    line_number: int = Field(ge=0)
    word_number: int = Field(ge=0)


class OCRLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox
    words: list[OCRWord] = Field(min_length=1)


class OCRPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    image_path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    words: list[OCRWord]
    lines: list[OCRLine]


class OCRDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_name: str
    engine_version: str | None = None
    language: str
    latency_ms: int = Field(ge=0)
    pages: list[OCRPage] = Field(min_length=1)

    @property
    def lines(self) -> list[OCRLine]:
        return [line for page in self.pages for line in page.lines]
