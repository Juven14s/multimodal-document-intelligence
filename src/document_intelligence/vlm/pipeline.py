"""Zero-shot VLM orchestration, validation, and inference tracing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from document_intelligence.schemas import (
    DocumentExtraction,
    DocumentFields,
    Evidence,
    ExtractedField,
    ExtractionMethod,
    InferenceMetadata,
    LineItem,
)
from document_intelligence.validation import ReviewPolicy, validate_extraction
from document_intelligence.vlm.models import (
    FewShotTurn,
    GenerationResult,
    VLMField,
    VLMResponse,
    VLMRunResult,
    VLMTrace,
)
from document_intelligence.vlm.parsing import (
    VLMOutputError,
    extract_first_json_object,
    parse_vlm_response,
)
from document_intelligence.vlm.prompting import (
    PromptStyle,
    build_extraction_prompt,
    get_prompt_version,
)

if TYPE_CHECKING:
    from document_intelligence.vlm.examples import FewShotExample


class VLMBackend(Protocol):
    def generate(
        self,
        image_paths: list[Path],
        prompt: str,
        few_shot_turns: list[FewShotTurn] | None = None,
    ) -> GenerationResult: ...


def _field(field: VLMField | None) -> ExtractedField | None:
    if field is None:
        return None
    evidence = [Evidence(text=field.raw_text)] if field.raw_text else []
    return ExtractedField(
        value=field.value,
        raw_text=field.raw_text,
        confidence=field.confidence,
        evidence=evidence,
    )


def _to_extraction(
    parsed: VLMResponse,
    generation: GenerationResult,
    document_id: str,
    method: ExtractionMethod,
    prompt_version: str,
) -> DocumentExtraction:
    fields = parsed.fields
    extraction = DocumentExtraction(
        document_id=document_id,
        document_type=parsed.document_type,
        classification_confidence=parsed.classification_confidence,
        fields=DocumentFields(
            merchant_name=_field(fields.merchant_name),
            merchant_address=_field(fields.merchant_address),
            document_number=_field(fields.document_number),
            issue_date=_field(fields.issue_date),
            due_date=_field(fields.due_date),
            currency=_field(fields.currency),
            subtotal=_field(fields.subtotal),
            tax=_field(fields.tax),
            total=_field(fields.total),
            line_items=[
                LineItem(
                    description=_field(item.description),
                    quantity=_field(item.quantity),
                    unit_price=_field(item.unit_price),
                    amount=_field(item.amount),
                )
                for item in fields.line_items
            ],
        ),
        inference=InferenceMetadata(
            method=method,
            model_name=generation.model_id,
            prompt_version=prompt_version,
            latency_ms=generation.latency_ms,
        ),
    )
    return extraction


def _run(
    backend: VLMBackend,
    image_paths: list[Path],
    *,
    document_id: str,
    method: ExtractionMethod,
    prompt_version: str,
    prompt: str,
    few_shot_turns: list[FewShotTurn] | None = None,
    review_policy: ReviewPolicy | None = None,
) -> VLMRunResult:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    generation = backend.generate(image_paths, prompt, few_shot_turns)
    parse_error = None
    extraction = None
    try:
        json.loads(extract_first_json_object(generation.raw_response))
        json_syntax_valid = True
    except (VLMOutputError, json.JSONDecodeError):
        json_syntax_valid = False
    try:
        parsed = parse_vlm_response(generation.raw_response)
        extraction = validate_extraction(
            _to_extraction(parsed, generation, document_id, method, prompt_version), review_policy
        )
    except VLMOutputError as error:
        parse_error = str(error)
    trace = VLMTrace(
        document_id=document_id,
        model_id=generation.model_id,
        strategy=method,
        prompt_version=prompt_version,
        prompt_sha256=prompt_hash,
        image_paths=image_paths,
        latency_ms=generation.latency_ms,
        device=generation.device,
        peak_memory_mb=generation.peak_memory_mb,
        raw_response=generation.raw_response,
        json_syntax_valid=json_syntax_valid,
        valid_json=extraction is not None,
        parse_error=parse_error,
        example_document_ids=[turn.document_id for turn in few_shot_turns or []],
    )
    return VLMRunResult(trace=trace, extraction=extraction)


def run_zero_shot(
    backend: VLMBackend,
    image_paths: list[Path],
    *,
    document_id: str,
    prompt_style: PromptStyle = PromptStyle.JSON_SCHEMA,
    review_policy: ReviewPolicy | None = None,
) -> VLMRunResult:
    prompt = build_extraction_prompt(prompt_style)
    return _run(
        backend,
        image_paths,
        document_id=document_id,
        method=ExtractionMethod.VLM_ZERO_SHOT,
        prompt_version=get_prompt_version(prompt_style, few_shot=False),
        prompt=prompt,
        review_policy=review_policy,
    )


def run_few_shot(
    backend: VLMBackend,
    image_paths: list[Path],
    *,
    document_id: str,
    examples: list[FewShotExample],
    prompt_style: PromptStyle = PromptStyle.JSON_SCHEMA,
    review_policy: ReviewPolicy | None = None,
) -> VLMRunResult:
    from document_intelligence.vlm.examples import FewShotExample

    if not all(isinstance(example, FewShotExample) for example in examples):
        raise TypeError("examples must contain FewShotExample instances")
    turns = [
        FewShotTurn(
            document_id=example.document_id,
            image_paths=example.image_paths,
            assistant_response=json.dumps(
                example.response.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        for example in examples
    ]
    prompt = build_extraction_prompt(prompt_style)
    return _run(
        backend,
        image_paths,
        document_id=document_id,
        method=ExtractionMethod.VLM_FEW_SHOT,
        prompt_version=get_prompt_version(prompt_style, few_shot=True),
        prompt=prompt,
        few_shot_turns=turns,
        review_policy=review_policy,
    )
