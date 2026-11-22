"""Configuration and reports emitted by the preprocessing pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MediaType(StrEnum):
    PDF = "application/pdf"
    IMAGE = "image"


class QualityIssue(StrEnum):
    LOW_RESOLUTION = "low_resolution"
    LOW_CONTRAST = "low_contrast"
    POSSIBLY_BLURRY = "possibly_blurry"
    DESKEW_UNCERTAIN = "deskew_uncertain"


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_dpi: int = Field(default=200, ge=72, le=600)
    max_pages: int = Field(default=20, ge=1, le=500)
    max_input_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    max_long_edge: int = Field(default=2400, ge=256, le=10000)
    min_short_edge_for_quality: int = Field(default=500, ge=1)
    auto_deskew: bool = True
    max_skew_degrees: float = Field(default=7.5, gt=0.0, le=15.0)
    deskew_step_degrees: float = Field(default=0.5, gt=0.0, le=2.0)
    min_deskew_degrees: float = Field(default=0.5, ge=0.0, le=5.0)
    min_contrast: float = Field(default=0.08, ge=0.0, le=1.0)
    min_sharpness: float = Field(default=0.025, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_deskew_range(self) -> PreprocessingConfig:
        if self.min_deskew_degrees >= self.max_skew_degrees:
            raise ValueError("min_deskew_degrees must be smaller than max_skew_degrees")
        return self


class Transformation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class PageQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contrast: float = Field(ge=0.0, le=1.0)
    sharpness: float = Field(ge=0.0, le=1.0)
    mean_brightness: float = Field(ge=0.0, le=1.0)
    issues: list[QualityIssue] = Field(default_factory=list)


class ProcessedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    output_path: Path
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    output_width: int = Field(gt=0)
    output_height: int = Field(gt=0)
    transformations: list[Transformation] = Field(default_factory=list)
    quality: PageQuality


class PreprocessingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_media_type: MediaType
    source_size_bytes: int = Field(ge=0)
    pages: list[ProcessedPage] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
