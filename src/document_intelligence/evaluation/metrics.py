"""Masked exact-match and F1 metrics for heterogeneous receipt datasets."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.data.models import CanonicalField, CanonicalRecord
from document_intelligence.schemas import DocumentExtraction


class FieldMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    support: int = Field(ge=0)
    predicted: int = Field(ge=0)
    exact_matches: int = Field(ge=0)
    exact_match: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class SelectiveMetrics(BaseModel):
    """Document-level quality of auto-acceptance and human-review routing."""

    model_config = ConfigDict(extra="forbid")

    exact_documents: int = Field(ge=0)
    document_exact_match_rate: float = Field(ge=0.0, le=1.0)
    auto_accepted_documents: int = Field(ge=0)
    auto_accepted_exact_match_rate: float = Field(ge=0.0, le=1.0)
    reviewed_documents: int = Field(ge=0)
    review_precision: float = Field(ge=0.0, le=1.0)
    error_detection_recall: float = Field(ge=0.0, le=1.0)


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    prediction_coverage: float = Field(ge=0.0, le=1.0)
    valid_json_rate: float = Field(ge=0.0, le=1.0)
    human_review_rate: float = Field(ge=0.0, le=1.0)
    selective: SelectiveMetrics
    fields: dict[str, FieldMetrics]


def _normalize_text(value: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    text = "".join(character for character in decomposed if not unicodedata.combining(character))
    return re.findall(r"\w+", text, flags=re.UNICODE)


def _tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Decimal):
        normalized = value.normalize()
        return [format(normalized, "f")]
    if isinstance(value, date):
        return [value.isoformat()]
    return _normalize_text(str(value))


def _overlap(target_tokens: list[str], predicted_tokens: list[str]) -> int:
    target_counts = Counter(target_tokens)
    predicted_counts = Counter(predicted_tokens)
    return sum((target_counts & predicted_counts).values())


def _prediction_value(prediction: DocumentExtraction | None, field: CanonicalField) -> Any:
    if prediction is None:
        return None
    mapping = {
        CanonicalField.MERCHANT_NAME: prediction.fields.merchant_name,
        CanonicalField.MERCHANT_ADDRESS: prediction.fields.merchant_address,
        CanonicalField.ISSUE_DATE: prediction.fields.issue_date,
        CanonicalField.CURRENCY: prediction.fields.currency,
        CanonicalField.SUBTOTAL: prediction.fields.subtotal,
        CanonicalField.TAX: prediction.fields.tax,
        CanonicalField.TOTAL: prediction.fields.total,
    }
    extracted = mapping.get(field)
    return extracted.value if extracted is not None else None


def _target_value(record: CanonicalRecord, field: CanonicalField) -> Any:
    return getattr(record.annotation, field.value)


def _line_item_tokens(record: CanonicalRecord) -> list[str]:
    tokens: list[str] = []
    for item in record.annotation.line_items:
        description = " ".join(_tokens(item.description))
        amount = " ".join(_tokens(item.amount))
        tokens.append(f"{description}|{amount}")
    return tokens


def _predicted_line_item_tokens(prediction: DocumentExtraction | None) -> list[str]:
    if prediction is None:
        return []
    tokens: list[str] = []
    for item in prediction.fields.line_items:
        description = " ".join(_tokens(item.description.value))
        amount = " ".join(_tokens(item.amount.value))
        tokens.append(f"{description}|{amount}")
    return tokens


def _field_tokens(
    record: CanonicalRecord,
    prediction: DocumentExtraction | None,
    field: CanonicalField,
) -> tuple[list[str], list[str]]:
    if field is CanonicalField.LINE_ITEMS:
        return _line_item_tokens(record), _predicted_line_item_tokens(prediction)
    return _tokens(_target_value(record, field)), _tokens(_prediction_value(prediction, field))


def evaluate_predictions(
    records: Iterable[CanonicalRecord],
    predictions: Iterable[DocumentExtraction],
    *,
    invalid_prediction_count: int = 0,
) -> BenchmarkResult:
    materialized_records = list(records)
    materialized_predictions = list(predictions)
    by_id = {prediction.document_id: prediction for prediction in materialized_predictions}
    accumulators: dict[str, dict[str, int]] = defaultdict(
        lambda: {"support": 0, "predicted": 0, "exact": 0, "tp": 0, "target": 0, "pred": 0}
    )
    document_outcomes: list[tuple[bool, bool]] = []

    for record in materialized_records:
        prediction = by_id.get(record.manifest.document_id)
        document_exact = prediction is not None
        for field in record.annotation.annotated_fields:
            stats = accumulators[field.value]
            stats["support"] += 1
            target_tokens, predicted_tokens = _field_tokens(record, prediction, field)
            if predicted_tokens:
                stats["predicted"] += 1
            if target_tokens == predicted_tokens:
                stats["exact"] += 1
            else:
                document_exact = False
            stats["tp"] += _overlap(target_tokens, predicted_tokens)
            stats["target"] += len(target_tokens)
            stats["pred"] += len(predicted_tokens)
        requires_review = prediction is None or prediction.requires_human_review
        document_outcomes.append((document_exact, requires_review))

    field_results: dict[str, FieldMetrics] = {}
    for name, stats in sorted(accumulators.items()):
        support = stats["support"]
        precision = stats["tp"] / stats["pred"] if stats["pred"] else 0.0
        recall = stats["tp"] / stats["target"] if stats["target"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        field_results[name] = FieldMetrics(
            support=support,
            predicted=stats["predicted"],
            exact_matches=stats["exact"],
            exact_match=stats["exact"] / support if support else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    document_count = len(materialized_records)
    matched_predictions = sum(
        record.manifest.document_id in by_id for record in materialized_records
    )
    total_payloads = len(materialized_predictions) + invalid_prediction_count
    exact_count = sum(exact for exact, _ in document_outcomes)
    review_count = sum(reviewed for _, reviewed in document_outcomes)
    auto_outcomes = [exact for exact, reviewed in document_outcomes if not reviewed]
    reviewed_outcomes = [exact for exact, reviewed in document_outcomes if reviewed]
    error_count = document_count - exact_count
    reviewed_error_count = sum(not exact for exact in reviewed_outcomes)
    return BenchmarkResult(
        document_count=document_count,
        prediction_count=len(materialized_predictions),
        prediction_coverage=matched_predictions / document_count if document_count else 0.0,
        valid_json_rate=(
            len(materialized_predictions) / total_payloads if total_payloads else 0.0
        ),
        human_review_rate=review_count / document_count if document_count else 0.0,
        selective=SelectiveMetrics(
            exact_documents=exact_count,
            document_exact_match_rate=exact_count / document_count if document_count else 0.0,
            auto_accepted_documents=len(auto_outcomes),
            auto_accepted_exact_match_rate=(
                sum(auto_outcomes) / len(auto_outcomes) if auto_outcomes else 0.0
            ),
            reviewed_documents=len(reviewed_outcomes),
            review_precision=(
                reviewed_error_count / len(reviewed_outcomes) if reviewed_outcomes else 0.0
            ),
            error_detection_recall=(
                reviewed_error_count / error_count if error_count else 0.0
            ),
        ),
        fields=field_results,
    )
