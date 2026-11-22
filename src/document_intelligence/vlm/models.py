"""Structured response and trace contracts for VLM inference."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.schemas import DocumentExtraction, DocumentType, ExtractionMethod

T = TypeVar("T")


class VLMField(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    value: T | None
    raw_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class VLMLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: VLMField[str]
    quantity: VLMField[Decimal] | None = None
    unit_price: VLMField[Decimal] | None = None
    amount: VLMField[Decimal]


class VLMFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merchant_name: VLMField[str]
    merchant_address: VLMField[str] | None = None
    document_number: VLMField[str] | None = None
    issue_date: VLMField[date] | None = None
    due_date: VLMField[date] | None = None
    currency: VLMField[str] | None = None
    subtotal: VLMField[Decimal] | None = None
    tax: VLMField[Decimal] | None = None
    total: VLMField[Decimal]
    line_items: list[VLMLineItem] = Field(default_factory=list)


class VLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    classification_confidence: float = Field(ge=0.0, le=1.0)
    fields: VLMFields


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    raw_response: str
    latency_ms: int = Field(ge=0)
    device: str
    peak_memory_mb: float | None = Field(default=None, ge=0.0)


class FewShotTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    image_paths: list[Path] = Field(min_length=1)
    assistant_response: str


class VLMTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    model_id: str
    strategy: ExtractionMethod
    prompt_version: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_paths: list[Path] = Field(min_length=1)
    latency_ms: int = Field(ge=0)
    device: str
    peak_memory_mb: float | None = Field(default=None, ge=0.0)
    raw_response: str
    json_syntax_valid: bool = False
    valid_json: bool
    parse_error: str | None = None
    example_document_ids: list[str] = Field(default_factory=list)


class VLMRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace: VLMTrace
    extraction: DocumentExtraction | None = None
