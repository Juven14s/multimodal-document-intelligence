"""Strict extraction of the first JSON object from model-generated text."""

from __future__ import annotations

import json

from pydantic import ValidationError

from document_intelligence.vlm.models import VLMResponse


class VLMOutputError(ValueError):
    """Raised when a model response is not valid structured output."""


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise VLMOutputError("Model response contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise VLMOutputError("Model response contains an incomplete JSON object")


def parse_vlm_response(text: str) -> VLMResponse:
    candidate = extract_first_json_object(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise VLMOutputError(f"Invalid JSON: {error.msg}") from error
    try:
        return VLMResponse.model_validate(payload)
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        raise VLMOutputError(f"Schema validation failed at {location}: {first['msg']}") from error
