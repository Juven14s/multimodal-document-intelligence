import json
from pathlib import Path

import pytest

from document_intelligence.data.models import DatasetName, DatasetSplit
from document_intelligence.schemas import ExtractionMethod
from document_intelligence.vlm.backend import build_chat_messages
from document_intelligence.vlm.examples import FewShotExample
from document_intelligence.vlm.models import FewShotTurn, GenerationResult, VLMResponse
from document_intelligence.vlm.parsing import VLMOutputError, parse_vlm_response
from document_intelligence.vlm.pipeline import run_few_shot, run_zero_shot
from document_intelligence.vlm.prompting import (
    PromptStyle,
    build_compact_prompt,
    build_zero_shot_prompt,
)


def valid_payload() -> dict:
    return {
        "document_type": "receipt",
        "classification_confidence": 0.96,
        "fields": {
            "merchant_name": {
                "value": "Example Cafe",
                "raw_text": "EXAMPLE CAFE",
                "confidence": 0.95,
            },
            "currency": {"value": "MYR", "raw_text": "RM", "confidence": 0.9},
            "subtotal": {"value": "10.00", "raw_text": "10.00", "confidence": 0.9},
            "tax": {"value": "0.60", "raw_text": "0.60", "confidence": 0.9},
            "total": {"value": "10.60", "raw_text": "10.60", "confidence": 0.95},
            "line_items": [
                {
                    "description": {
                        "value": "Coffee",
                        "raw_text": "Coffee",
                        "confidence": 0.9,
                    },
                    "amount": {"value": "10.00", "raw_text": "10.00", "confidence": 0.9},
                }
            ],
        },
    }


def test_parser_accepts_fenced_json_and_ignores_trailing_text() -> None:
    raw = f"```json\n{json.dumps(valid_payload())}\n```\nDone."

    parsed = parse_vlm_response(raw)

    assert str(parsed.fields.total.value) == "10.60"
    assert parsed.fields.merchant_name.value == "Example Cafe"


def test_parser_rejects_incomplete_or_schema_invalid_output() -> None:
    with pytest.raises(VLMOutputError, match="incomplete"):
        parse_vlm_response('{"document_type": "receipt"')
    with pytest.raises(VLMOutputError, match="Schema validation"):
        parse_vlm_response('{"document_type": "receipt"}')


def test_prompt_contains_abstention_and_json_schema() -> None:
    prompt = build_zero_shot_prompt()

    assert "Use null" in prompt
    assert "Never invent" in prompt
    assert '"classification_confidence"' in prompt


def test_compact_prompt_contains_complete_template_and_is_shorter() -> None:
    compact = build_compact_prompt()

    assert "JSON template" in compact
    assert '"merchant_name"' in compact
    assert '"line_items":[]' in compact
    assert len(compact) < len(build_zero_shot_prompt())


class FakeBackend:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, image_paths, prompt, few_shot_turns=None) -> GenerationResult:
        assert image_paths == [Path("page.png")]
        assert "JSON Schema" in prompt or "JSON template" in prompt
        self.few_shot_turns = few_shot_turns or []
        return GenerationResult(
            model_id="fake-qwen",
            raw_response=self.response,
            latency_ms=123,
            device="cpu",
        )


def test_zero_shot_pipeline_builds_validated_extraction_and_trace() -> None:
    result = run_zero_shot(
        FakeBackend(json.dumps(valid_payload())),
        [Path("page.png")],
        document_id="doc-1",
    )

    assert result.trace.valid_json is True
    assert result.trace.json_syntax_valid is True
    assert result.trace.parse_error is None
    assert result.extraction is not None
    assert result.extraction.inference.method is ExtractionMethod.VLM_ZERO_SHOT
    assert result.extraction.fields.total.evidence[0].text == "10.60"
    assert result.extraction.requires_human_review is False


def test_zero_shot_pipeline_preserves_invalid_output_in_trace() -> None:
    result = run_zero_shot(
        FakeBackend("I cannot read this receipt."),
        [Path("page.png")],
        document_id="doc-2",
    )

    assert result.extraction is None
    assert result.trace.valid_json is False
    assert result.trace.json_syntax_valid is False
    assert result.trace.parse_error == "Model response contains no JSON object"
    assert result.trace.raw_response == "I cannot read this receipt."


def test_chat_messages_place_examples_before_target() -> None:
    turn = FewShotTurn(
        document_id="train-1",
        image_paths=[Path("example.png")],
        assistant_response='{"answer":1}',
    )

    messages = build_chat_messages([Path("target.png")], "system", [turn])

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"][0]["text"] == '{"answer":1}'
    assert "return only the requested JSON object" in messages[-1]["content"][-1]["text"]


def test_compact_prompt_version_is_traced_separately() -> None:
    result = run_zero_shot(
        FakeBackend(json.dumps(valid_payload())),
        [Path("page.png")],
        document_id="doc-compact",
        prompt_style=PromptStyle.COMPACT,
    )

    assert result.trace.valid_json is True
    assert result.trace.prompt_version == "document-json-compact-v1"


def test_few_shot_pipeline_traces_examples_and_strategy() -> None:
    backend = FakeBackend(json.dumps(valid_payload()))
    example = FewShotExample(
        document_id="train-1",
        dataset=DatasetName.SROIE,
        split=DatasetSplit.TRAIN,
        image_paths=[Path("example.png")],
        image_width=100,
        image_height=200,
        annotated_fields=set(),
        response=VLMResponse.model_validate(valid_payload()),
    )

    result = run_few_shot(
        backend,
        [Path("page.png")],
        document_id="test-1",
        examples=[example],
    )

    assert result.extraction.inference.method is ExtractionMethod.VLM_FEW_SHOT
    assert result.trace.example_document_ids == ["train-1"]
    assert backend.few_shot_turns[0].document_id == "train-1"
