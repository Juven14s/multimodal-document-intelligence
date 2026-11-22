"""Canonical extraction contracts shared by OCR, VLM, API, and UI layers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentType(StrEnum):
    RECEIPT = "receipt"
    INVOICE = "invoice"
    CREDIT_NOTE = "credit_note"
    UNKNOWN = "unknown"


class ExtractionMethod(StrEnum):
    OCR_RULES = "ocr_rules"
    VLM_ZERO_SHOT = "vlm_zero_shot"
    VLM_FEW_SHOT = "vlm_few_shot"
    VLM_LORA = "vlm_lora"
    HUMAN = "human"


class BoundingBox(BaseModel):
    """Normalized coordinates relative to a page image."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_corners(self) -> BoundingBox:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("A bounding box must have a positive width and height")
        return self


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    box: BoundingBox | None = None


T = TypeVar("T")


class ExtractedField(BaseModel, Generic[T]):
    """A normalized value plus enough information to audit its origin."""

    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    raw_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: ExtractedField[str]
    quantity: ExtractedField[Decimal] | None = None
    unit_price: ExtractedField[Decimal] | None = None
    amount: ExtractedField[Decimal]


class DocumentFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merchant_name: ExtractedField[str]
    merchant_address: ExtractedField[str] | None = None
    document_number: ExtractedField[str] | None = None
    issue_date: ExtractedField[date] | None = None
    due_date: ExtractedField[date] | None = None
    currency: ExtractedField[str] | None = None
    subtotal: ExtractedField[Decimal] | None = None
    tax: ExtractedField[Decimal] | None = None
    total: ExtractedField[Decimal]
    balance_due: ExtractedField[Decimal] | None = None
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls, value: ExtractedField[str] | None
    ) -> ExtractedField[str] | None:
        if value is not None and value.value is not None:
            value.value = value.value.strip().upper()
            if len(value.value) != 3 or not value.value.isalpha():
                raise ValueError("Currency must be a three-letter ISO-style code")
        return value


class InferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: ExtractionMethod
    model_name: str
    model_version: str | None = None
    prompt_version: str | None = None
    latency_ms: int = Field(ge=0)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    field: str | None = None
    blocking: bool = False


class DocumentExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    document_id: str = Field(min_length=1)
    document_type: DocumentType
    classification_confidence: float = Field(ge=0.0, le=1.0)
    fields: DocumentFields
    inference: InferenceMetadata
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    requires_human_review: bool = False
