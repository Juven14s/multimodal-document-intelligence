"""HTTP-specific request and response contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.schemas import DocumentExtraction


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    methods: dict[str, bool]


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=1, max_length=100)


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=1, max_length=100)
    correction: DocumentExtraction | None = None
