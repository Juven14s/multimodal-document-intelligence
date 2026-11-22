"""Versioned zero-shot prompt derived from the response schema."""

from __future__ import annotations

import json
from enum import StrEnum

from document_intelligence.vlm.models import VLMResponse

ZERO_SHOT_PROMPT_VERSION = "qwen-document-json-v1"
FEW_SHOT_PROMPT_VERSION = "qwen-document-json-few-shot-v1"
COMPACT_ZERO_SHOT_PROMPT_VERSION = "document-json-compact-v1"
COMPACT_FEW_SHOT_PROMPT_VERSION = "document-json-compact-few-shot-v1"


class PromptStyle(StrEnum):
    JSON_SCHEMA = "json_schema"
    COMPACT = "compact"


def build_zero_shot_prompt() -> str:
    schema = json.dumps(VLMResponse.model_json_schema(), ensure_ascii=False, indent=2)
    return f"""You extract structured data from receipts and invoices.

Return exactly one JSON object and no Markdown, explanation, or code fence.
The object must validate against the JSON Schema below.

Rules:
- Read only information visible in the supplied document pages.
- Use null when a value is absent or unreadable. Never invent a value.
- Copy the visible source text into raw_text whenever a value is present.
- Dates must use YYYY-MM-DD.
- Currency must use a three-letter ISO code when identifiable; otherwise null.
- Decimal fields must contain numbers without currency symbols.
- confidence is a number from 0 to 1 reflecting visual certainty.
- Preserve line-item order across pages.
- Classify as receipt, invoice, credit_note, or unknown.

JSON Schema:
{schema}
"""


def build_compact_prompt() -> str:
    template = {
        "document_type": "receipt",
        "classification_confidence": 0.0,
        "fields": {
            "merchant_name": {"value": None, "raw_text": None, "confidence": 0.0},
            "merchant_address": None,
            "document_number": None,
            "issue_date": None,
            "due_date": None,
            "currency": None,
            "subtotal": None,
            "tax": None,
            "total": {"value": None, "raw_text": None, "confidence": 0.0},
            "line_items": [],
        },
    }
    payload = json.dumps(template, ensure_ascii=False, separators=(",", ":"))
    return f"""Read the supplied receipt or invoice and fill the JSON template below.

Output only one valid JSON object. Start with {{ and end with }}. Do not add Markdown or
explanations. Keep every key exactly as shown and do not add keys.

Rules:
- Use only visible document text; never guess.
- For an unreadable value use null and confidence 0.
- A present field is an object with value, exact visible raw_text, and confidence from 0 to 1.
- Dates use YYYY-MM-DD. Currency uses a three-letter ISO code. Amounts omit currency symbols.
- document_type is receipt, invoice, credit_note, or unknown.
- Keep line_items empty unless description and amount are both clearly readable.

JSON template:
{payload}
"""


def build_extraction_prompt(style: PromptStyle = PromptStyle.JSON_SCHEMA) -> str:
    if style is PromptStyle.COMPACT:
        return build_compact_prompt()
    return build_zero_shot_prompt()


def get_prompt_version(style: PromptStyle, *, few_shot: bool) -> str:
    if style is PromptStyle.COMPACT:
        return (
            COMPACT_FEW_SHOT_PROMPT_VERSION
            if few_shot
            else COMPACT_ZERO_SHOT_PROMPT_VERSION
        )
    return FEW_SHOT_PROMPT_VERSION if few_shot else ZERO_SHOT_PROMPT_VERSION
