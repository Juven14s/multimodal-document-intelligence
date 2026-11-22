"""Deterministic business checks and review-routing policy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from document_intelligence.schemas import (
    DocumentExtraction,
    DocumentType,
    ExtractedField,
    ValidationIssue,
)


@dataclass(frozen=True)
class ReviewPolicy:
    min_classification_confidence: float = 0.80
    min_required_field_confidence: float = 0.75
    amount_tolerance: Decimal = Decimal("0.02")


def _present_confidence(field: ExtractedField[object] | None) -> float | None:
    if field is None or field.value is None:
        return None
    return field.confidence


def validate_extraction(
    extraction: DocumentExtraction,
    policy: ReviewPolicy | None = None,
) -> DocumentExtraction:
    """Return a copy enriched with issues and an explicit review decision."""

    policy = policy or ReviewPolicy()
    issues: list[ValidationIssue] = []
    fields = extraction.fields

    if extraction.classification_confidence < policy.min_classification_confidence:
        issues.append(
            ValidationIssue(
                code="low_classification_confidence",
                message="Document classification confidence is below the review threshold",
                field="document_type",
                blocking=True,
            )
        )

    required = {
        "merchant_name": fields.merchant_name,
    }
    if (
        extraction.document_type is DocumentType.INVOICE
        and fields.total.value is None
        and fields.balance_due is not None
        and fields.balance_due.value is not None
    ):
        required["balance_due"] = fields.balance_due
    else:
        required["total"] = fields.total
    for name, field in required.items():
        confidence = _present_confidence(field)
        if confidence is None:
            issues.append(
                ValidationIssue(
                    code="missing_required_field",
                    message=f"Required field '{name}' is missing",
                    field=name,
                    blocking=True,
                )
            )
        elif confidence < policy.min_required_field_confidence:
            issues.append(
                ValidationIssue(
                    code="low_field_confidence",
                    message=f"Confidence for '{name}' is below the review threshold",
                    field=name,
                    blocking=True,
                )
            )

    subtotal = fields.subtotal.value if fields.subtotal else None
    tax = fields.tax.value if fields.tax else None
    total = fields.total.value
    if subtotal is not None and tax is not None and total is not None:
        difference = abs((subtotal + tax) - total)
        if difference > policy.amount_tolerance:
            issues.append(
                ValidationIssue(
                    code="inconsistent_total",
                    message="Subtotal plus tax does not match total",
                    field="total",
                    blocking=True,
                )
            )

    line_amounts = [item.amount.value for item in fields.line_items]
    if subtotal is not None and line_amounts and all(value is not None for value in line_amounts):
        line_sum = sum((value for value in line_amounts if value is not None), Decimal("0"))
        if abs(line_sum - subtotal) > policy.amount_tolerance:
            issues.append(
                ValidationIssue(
                    code="inconsistent_line_sum",
                    message="Sum of line amounts does not match subtotal",
                    field="line_items",
                    blocking=True,
                )
            )

    return extraction.model_copy(
        update={
            "validation_issues": issues,
            "requires_human_review": any(issue.blocking for issue in issues),
        }
    )
