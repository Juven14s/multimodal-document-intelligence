"""Canonical representation independent from the source dataset."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.schemas import BoundingBox


class DatasetName(StrEnum):
    CORD_V2 = "cord_v2"
    SROIE = "sroie"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class CanonicalField(StrEnum):
    MERCHANT_NAME = "merchant_name"
    MERCHANT_ADDRESS = "merchant_address"
    ISSUE_DATE = "issue_date"
    CURRENCY = "currency"
    SUBTOTAL = "subtotal"
    TAX = "tax"
    TOTAL = "total"
    LINE_ITEMS = "line_items"


class CanonicalLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None


class CanonicalAnnotation(BaseModel):
    """Ground truth plus an explicit mask of fields that may be evaluated."""

    model_config = ConfigDict(extra="forbid")

    merchant_name: str | None = None
    merchant_address: str | None = None
    issue_date: date | None = None
    currency: str | None = None
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None
    line_items: list[CanonicalLineItem] = Field(default_factory=list)
    annotated_fields: set[CanonicalField] = Field(default_factory=set)


class OCRRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    box: BoundingBox


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    dataset: DatasetName
    source_split: str
    benchmark_split: DatasetSplit
    source_key: str
    image_path: Path | None = None
    annotation_path: Path | None = None
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ocr_region_count: int = Field(ge=0)
    annotated_fields: set[CanonicalField]


class CanonicalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: ManifestEntry
    annotation: CanonicalAnnotation
    ocr_regions: list[OCRRegion] = Field(default_factory=list)
