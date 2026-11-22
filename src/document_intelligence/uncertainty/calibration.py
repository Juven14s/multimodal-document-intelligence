"""Validation-only confidence calibration and threshold selection."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.data.models import CanonicalField, CanonicalRecord, DatasetSplit
from document_intelligence.schemas import (
    DocumentExtraction,
    ExtractedField,
    ExtractionMethod,
    ValidationIssue,
)

SCALAR_FIELDS = (
    CanonicalField.MERCHANT_NAME,
    CanonicalField.MERCHANT_ADDRESS,
    CanonicalField.ISSUE_DATE,
    CanonicalField.CURRENCY,
    CanonicalField.SUBTOTAL,
    CanonicalField.TAX,
    CanonicalField.TOTAL,
)


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_auto_accept_accuracy: float = Field(default=0.95, gt=0.0, le=1.0)
    minimum_samples_per_field: int = Field(default=20, ge=1)
    minimum_accepted_per_field: int = Field(default=5, ge=1)
    reliability_bins: int = Field(default=10, ge=2, le=50)


class FieldCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    status: str
    threshold: float = Field(ge=0.0, le=1.0)
    accepted_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    accepted_accuracy: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    expected_calibration_error: float = Field(ge=0.0, le=1.0)


class CalibrationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    method: ExtractionMethod
    model_name: str
    prompt_version: str | None = None
    source_split: DatasetSplit = DatasetSplit.VALIDATION
    validation_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: CalibrationConfig
    fields: dict[str, FieldCalibration]


class _Observation(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    correct: bool


def _tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Decimal):
        return [format(value.normalize(), "f")]
    if isinstance(value, date):
        return [value.isoformat()]
    decomposed = unicodedata.normalize("NFKD", str(value)).casefold()
    normalized = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


def _prediction_field(
    prediction: DocumentExtraction, field: CanonicalField
) -> ExtractedField | None:
    mapping = {
        CanonicalField.MERCHANT_NAME: prediction.fields.merchant_name,
        CanonicalField.MERCHANT_ADDRESS: prediction.fields.merchant_address,
        CanonicalField.ISSUE_DATE: prediction.fields.issue_date,
        CanonicalField.CURRENCY: prediction.fields.currency,
        CanonicalField.SUBTOTAL: prediction.fields.subtotal,
        CanonicalField.TAX: prediction.fields.tax,
        CanonicalField.TOTAL: prediction.fields.total,
    }
    return mapping[field]


def _ece(observations: list[_Observation], bins: int) -> float:
    if not observations:
        return 0.0
    total = len(observations)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item in observations
            if lower <= item.confidence < upper
            or (index == bins - 1 and item.confidence == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(item.confidence for item in members) / len(members)
        accuracy = sum(item.correct for item in members) / len(members)
        error += len(members) / total * abs(accuracy - mean_confidence)
    return error


def _calibrate_field(
    observations: list[_Observation], config: CalibrationConfig
) -> FieldCalibration:
    count = len(observations)
    correct = sum(item.correct for item in observations)
    brier = (
        sum((item.confidence - float(item.correct)) ** 2 for item in observations) / count
        if count
        else 0.0
    )
    expected_calibration_error = _ece(observations, config.reliability_bins)
    if count < config.minimum_samples_per_field:
        return FieldCalibration(
            sample_count=count,
            correct_count=correct,
            status="insufficient_data",
            threshold=1.0,
            accepted_count=0,
            coverage=0.0,
            accepted_accuracy=0.0,
            brier_score=brier,
            expected_calibration_error=expected_calibration_error,
        )

    selected: tuple[float, list[_Observation], float] | None = None
    for threshold in sorted({item.confidence for item in observations}):
        accepted = [item for item in observations if item.confidence >= threshold]
        if len(accepted) < config.minimum_accepted_per_field:
            continue
        accuracy = sum(item.correct for item in accepted) / len(accepted)
        if accuracy >= config.target_auto_accept_accuracy:
            selected = (threshold, accepted, accuracy)
            break
    if selected is None:
        return FieldCalibration(
            sample_count=count,
            correct_count=correct,
            status="target_not_reached",
            threshold=1.0,
            accepted_count=0,
            coverage=0.0,
            accepted_accuracy=0.0,
            brier_score=brier,
            expected_calibration_error=expected_calibration_error,
        )
    threshold, accepted, accuracy = selected
    return FieldCalibration(
        sample_count=count,
        correct_count=correct,
        status="calibrated",
        threshold=threshold,
        accepted_count=len(accepted),
        coverage=len(accepted) / count,
        accepted_accuracy=accuracy,
        brier_score=brier,
        expected_calibration_error=expected_calibration_error,
    )


def calibrate_predictions(
    records: Iterable[CanonicalRecord],
    predictions: Iterable[DocumentExtraction],
    config: CalibrationConfig | None = None,
) -> CalibrationProfile:
    config = config or CalibrationConfig()
    materialized_records = list(records)
    if not materialized_records:
        raise ValueError("Calibration requires at least one validation record")
    invalid_splits = {
        record.manifest.benchmark_split
        for record in materialized_records
        if record.manifest.benchmark_split is not DatasetSplit.VALIDATION
    }
    if invalid_splits:
        raise ValueError(
            "Calibration accepts validation records only; test/train data is forbidden"
        )

    materialized_predictions = list(predictions)
    if not materialized_predictions:
        raise ValueError("Calibration requires at least one prediction")
    inference_keys = {
        (
            prediction.inference.method,
            prediction.inference.model_name,
            prediction.inference.prompt_version,
        )
        for prediction in materialized_predictions
    }
    if len(inference_keys) != 1:
        raise ValueError("Calibration predictions must use one method/model/prompt combination")
    method, model_name, prompt_version = inference_keys.pop()
    by_id = {prediction.document_id: prediction for prediction in materialized_predictions}
    observations: dict[str, list[_Observation]] = defaultdict(list)
    for record in materialized_records:
        prediction = by_id.get(record.manifest.document_id)
        if prediction is None:
            continue
        for field in SCALAR_FIELDS:
            if field not in record.annotation.annotated_fields:
                continue
            target = getattr(record.annotation, field.value)
            if target is None:
                continue
            extracted = _prediction_field(prediction, field)
            predicted_value = extracted.value if extracted is not None else None
            confidence = extracted.confidence if extracted is not None else 0.0
            observations[field.value].append(
                _Observation(
                    confidence=confidence,
                    correct=_tokens(target) == _tokens(predicted_value),
                )
            )

    document_ids = "\n".join(sorted(record.manifest.document_id for record in materialized_records))
    return CalibrationProfile(
        created_at=datetime.now(UTC),
        method=method,
        model_name=model_name,
        prompt_version=prompt_version,
        validation_set_sha256=hashlib.sha256(document_ids.encode()).hexdigest(),
        config=config,
        fields={
            field: _calibrate_field(field_observations, config)
            for field, field_observations in sorted(observations.items())
        },
    )


def apply_calibration(
    extraction: DocumentExtraction,
    profile: CalibrationProfile,
) -> DocumentExtraction:
    inference_key = (
        extraction.inference.method,
        extraction.inference.model_name,
        extraction.inference.prompt_version,
    )
    profile_key = (profile.method, profile.model_name, profile.prompt_version)
    if inference_key != profile_key:
        raise ValueError("Calibration profile does not match extraction method/model/prompt")

    issues = [
        issue
        for issue in extraction.validation_issues
        if not issue.code.startswith("calibration_")
    ]
    for name, field_profile in profile.fields.items():
        field = _prediction_field(extraction, CanonicalField(name))
        if field_profile.status != "calibrated":
            issues.append(
                ValidationIssue(
                    code="calibration_unavailable",
                    message=f"Reliable calibration is unavailable for '{name}'",
                    field=name,
                    blocking=True,
                )
            )
        elif field is None or field.value is None:
            issues.append(
                ValidationIssue(
                    code="calibration_missing_field",
                    message=f"Calibrated field '{name}' is missing",
                    field=name,
                    blocking=True,
                )
            )
        elif (
            field.confidence < field_profile.threshold
        ):
            issues.append(
                ValidationIssue(
                    code="calibration_low_confidence",
                    message=(
                        f"Confidence for '{name}' is below its calibrated threshold "
                        f"({field.confidence:.3f} < {field_profile.threshold:.3f})"
                    ),
                    field=name,
                    blocking=True,
                )
            )
    return extraction.model_copy(
        update={
            "validation_issues": issues,
            "requires_human_review": any(issue.blocking for issue in issues),
        }
    )
