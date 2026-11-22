"""Explainable receipt and invoice extraction rules over OCR lines."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import StrEnum

from document_intelligence.data.normalization import parse_date, parse_decimal
from document_intelligence.ocr.invoice_rules import INVOICE_HEADING, read_invoice_fields
from document_intelligence.ocr.models import OCRDocument, OCRLine
from document_intelligence.schemas import (
    DocumentExtraction,
    DocumentFields,
    DocumentType,
    Evidence,
    ExtractedField,
    ExtractionMethod,
    InferenceMetadata,
    LineItem,
    ValidationIssue,
)
from document_intelligence.validation import ReviewPolicy, validate_extraction

AMOUNT_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:RM|MYR|RP|IDR|USD|EUR|GBP|[$€£])?\s*"
    r"-?\d[\d]*(?:[.,:]\s*\d+)*",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[ -][A-Z]{3}[ -]\d{2,4})\b",
    re.IGNORECASE,
)
SUMMARY_WORDS = re.compile(
    r"\b(?:sub\s*total|grand\s*total|total|tax|vat|gst|tva|amount\s+due|balance\s+due)\b",
    re.IGNORECASE,
)
NON_PAYABLE_TOTAL = re.compile(
    r"\b(?:tax|sst|vat|tva)\s+total\b|"
    r"\btotal\s*\(?\s*excl(?:uding)?\.?\s+(?:gst|sst|tax|vat|tva)\b|"
    r"\d\s*%\s*(?:is\s+)?included\s+in\s+total\b",
    re.IGNORECASE,
)
PAYMENT_REFERENCE = re.compile(
    r"\b(?:card|account|acct)\s*(?:no\b|number\b|#)|"
    r"\b(?:approval|auth|batch|trace|terminal)\s*(?:no\b|number\b|code\b|id\b|#)|"
    r"(?<!\w)[*x]{3,}[\s-]*\d{2,}\b",
    re.IGNORECASE,
)


class OCRRulesVersion(StrEnum):
    V2 = "ocr-rules-v2"
    V3 = "ocr-rules-v3"
    INVOICE_V1 = "ocr-invoice-v1"


def _evidence(line: OCRLine) -> list[Evidence]:
    return [Evidence(text=line.text, box=line.box)]


def _empty_field() -> ExtractedField:
    return ExtractedField(value=None, confidence=0.0)


def _amounts(text: str, *, ignore_percentages: bool = False) -> list[tuple[str, Decimal]]:
    parsed: list[tuple[str, Decimal]] = []
    for match in AMOUNT_PATTERN.finditer(text):
        if ignore_percentages and re.match(r"[\s,.]*%", text[match.end() :]):
            continue
        raw = match.group(0).strip()
        normalized = re.sub(
            r"(?<=\d)([.,:])\s+(?=\d{2}\b)",
            lambda separator: "." if separator.group(1) == ":" else separator.group(1),
            raw,
        )
        normalized = re.sub(r"(?<=\d):(?=\d{2}\b)", ".", normalized)
        value = parse_decimal(normalized)
        if value is not None:
            parsed.append((raw, value))
    return parsed


def _labeled_amount(
    lines: list[OCRLine],
    labels: re.Pattern[str],
    *,
    exclusions: re.Pattern[str] | None = None,
    preferred: re.Pattern[str] | None = None,
    ignore_percentages: bool = False,
) -> ExtractedField[Decimal] | None:
    candidates: list[tuple[float, OCRLine, str, Decimal]] = []
    for line in lines:
        if not labels.search(line.text) or (exclusions and exclusions.search(line.text)):
            continue
        amounts = _amounts(line.text, ignore_percentages=ignore_percentages)
        if not amounts:
            continue
        raw, value = amounts[-1]
        label_bonus = 0.08 if re.search(r"grand|amount\s+due|balance\s+due", line.text, re.I) else 0
        if preferred and preferred.search(line.text):
            label_bonus += 0.12
        position_bonus = 0.05 * line.box.y0
        score = min(1.0, line.confidence * 0.9 + label_bonus + position_bonus)
        candidates.append((score, line, raw, value))
    if not candidates:
        return None
    score, line, raw, value = max(candidates, key=lambda item: item[0])
    return ExtractedField(
        value=value,
        raw_text=raw,
        confidence=score,
        evidence=_evidence(line),
    )


def _extract_date(lines: list[OCRLine]) -> ExtractedField[date] | None:
    candidates: list[tuple[float, OCRLine, str, date]] = []
    for line in lines:
        match = DATE_PATTERN.search(line.text)
        if not match:
            continue
        value = parse_date(match.group(0))
        if value is None:
            continue
        keyword_bonus = 0.08 if re.search(r"\bdate\b", line.text, re.I) else 0.0
        top_bonus = 0.05 * (1.0 - line.box.y0)
        score = min(1.0, line.confidence * 0.9 + keyword_bonus + top_bonus)
        candidates.append((score, line, match.group(0), value))
    if not candidates:
        return None
    score, line, raw, value = max(candidates, key=lambda item: item[0])
    return ExtractedField(
        value=value,
        raw_text=raw,
        confidence=score,
        evidence=_evidence(line),
    )


def _extract_document_number(lines: list[OCRLine]) -> ExtractedField[str] | None:
    pattern = re.compile(
        r"\b(?:invoice|receipt|document|doc|inv)\b\s*(?:no|number|#)?\s*[:#.]?\s*([A-Z0-9][A-Z0-9/-]{3,})",
        re.IGNORECASE,
    )
    for line in lines:
        match = pattern.search(line.text)
        if match:
            return ExtractedField(
                value=match.group(1).upper(),
                raw_text=match.group(1),
                confidence=min(1.0, line.confidence * 0.92),
                evidence=_evidence(line),
            )
    return None


def _extract_merchant(lines: list[OCRLine]) -> ExtractedField[str]:
    exclusions = re.compile(
        r"\b(?:receipt|invoice|tax|date|cashier|tel|fax|gst|no\.?|jalan|street|road)\b",
        re.IGNORECASE,
    )
    candidates: list[tuple[float, int, OCRLine]] = []
    business = re.compile(
        r"\b(?:sdn\.?\s*bhd\.?|s\s*/\s*b|enterprise|marketing|trading|restaurants?|"
        r"network|gallery|mart|hardware|bookstores?|stationery|pharmacy|confectionery|"
        r"kopitiam|caf[eé]|supermarket)\b",
        re.IGNORECASE,
    )
    address_like = re.compile(
        r"\b(?:jalan|jln|bandar|taman|lot|level|floor|selangor|johor|postcode)\b|\b\d{5}\b",
        re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        if line.box.page != 1 or line.box.y0 > 0.32 or exclusions.search(line.text):
            continue
        letters = sum(character.isalpha() for character in line.text)
        if letters < 3:
            continue
        length_bonus = min(0.18, letters / 150)
        top_bonus = max(0.0, 0.10 - index * 0.015)
        business_bonus = 0.22 if business.search(line.text) else 0.0
        address_penalty = 0.18 if address_like.search(line.text) else 0.0
        candidates.append(
            (
                min(
                    1.0,
                    line.confidence * 0.78
                    + length_bonus
                    + top_bonus
                    + business_bonus
                    - address_penalty,
                ),
                index,
                line,
            )
        )
    if not candidates:
        return _empty_field()
    score, _, line = max(candidates, key=lambda item: item[0])
    uppercase_alternatives = [
        candidate
        for candidate in candidates
        if candidate[2].text.isupper() and score - candidate[0] <= 0.07
    ]
    if uppercase_alternatives:
        score, _, line = min(uppercase_alternatives, key=lambda item: item[1])
    return ExtractedField(
        value=line.text.strip(),
        raw_text=line.text,
        confidence=score,
        evidence=_evidence(line),
    )


def _extract_address(
    lines: list[OCRLine],
    merchant: ExtractedField[str],
) -> ExtractedField[str] | None:
    if not merchant.evidence or merchant.evidence[0].box is None:
        return None
    merchant_box = merchant.evidence[0].box
    address_start = re.compile(
        r"(?:\b(?:no\.?|lot|level|km)\s*[:.#-]?\s*\d|\b(?:jalan|jln|road|street)\b|\b\d{5}\b)",
        re.IGNORECASE,
    )
    metadata = re.compile(
        r"\b(?:roc|registration|reg\.?\s*no|company\s+no|formerly|licensee)\b",
        re.IGNORECASE,
    )
    company_identifier = re.compile(r"^\W*\d{5,8}\s*[-–]\s*[A-Z]\W*$", re.IGNORECASE)
    boundary = re.compile(
        r"\b(?:tel|telephone|fax|email|gst|sst|tax\s+invoice|cash\s+bill|invoice|receipt|"
        r"credit\s+note|document\s+no|station|cashier|bill\s+date)\b",
        re.IGNORECASE,
    )
    selected: list[OCRLine] = []
    started = False
    for line in lines:
        if line.box.page != merchant_box.page or line.box.y0 <= merchant_box.y0:
            continue
        if line.box.y0 > 0.45:
            break
        if not started:
            if metadata.search(line.text) or company_identifier.search(line.text):
                continue
            if not address_start.search(line.text):
                continue
            started = True
        elif boundary.search(line.text) or metadata.search(line.text):
            break
        if boundary.search(line.text) or metadata.search(line.text):
            break
        selected.append(line)
        if len(selected) >= 5:
            break
    if not selected:
        return None
    text = ", ".join(line.text.strip(" ,") for line in selected)
    confidence = min(0.95, sum(line.confidence for line in selected) / len(selected) * 0.88)
    return ExtractedField(
        value=text,
        raw_text=text,
        confidence=confidence,
        evidence=[Evidence(text=line.text, box=line.box) for line in selected],
    )


def _extract_total(
    lines: list[OCRLine], *, filter_non_payable: bool
) -> ExtractedField[Decimal] | None:
    total_labels = re.compile(
        r"\b(?:grand\s+total|net\s+total|take\s*out\s+total|total\s+rounded|"
        r"total\s+sales|total\s+amount|amount\s+due|balance\s+due|total|tota|tot)\b",
        re.IGNORECASE,
    )
    total_exclusions = re.compile(
        r"\b(?:sub[-\s]*total|total\s+gst|gst\s+total|total\s+includes|"
        r"includes?\s+\d*%?\s*gst|total\s+item|total\s+(?:quantity|qty|points?|tax)|"
        r"(?:gst|tax|bst).*included\s+in\s+total|total\s+\d+\s*%\s*supplies|"
        r"accepted\s+total|(?:item|type).*\btotal|item\(s\)|change|discount)\b",
        re.IGNORECASE,
    )
    rounding_exclusions = re.compile(r"\b(?:gst\s+total|total\s+gst)\b", re.IGNORECASE)
    payment_exclusions = re.compile(r"\b(?:cashier|tendered|change|balance)\b", re.IGNORECASE)
    if filter_non_payable:
        total_exclusions = re.compile(
            rf"{total_exclusions.pattern}|{NON_PAYABLE_TOTAL.pattern}|{PAYMENT_REFERENCE.pattern}",
            re.IGNORECASE,
        )
        rounding_exclusions = total_exclusions
        payment_exclusions = re.compile(
            rf"{payment_exclusions.pattern}|{PAYMENT_REFERENCE.pattern}", re.IGNORECASE
        )
    rounding_label = re.compile(r"\broun(?:d|ding)?\s+adjustment\b", re.IGNORECASE)
    for index, line in enumerate(lines):
        if not rounding_label.search(line.text) or _amounts(
            line.text, ignore_percentages=filter_non_payable
        ):
            continue
        for follower in lines[index + 1 : index + 3]:
            if follower.box.page != line.box.page or follower.box.y0 - line.box.y1 > 0.06:
                break
            amounts = _amounts(follower.text, ignore_percentages=filter_non_payable)
            letters = sum(character.isalpha() for character in follower.text)
            if len(amounts) != 1 or letters > 3:
                continue
            raw, value = amounts[0]
            return ExtractedField(
                value=value,
                raw_text=raw,
                confidence=min(line.confidence, follower.confidence) * 0.85,
                evidence=[*_evidence(line), *_evidence(follower)],
            )

    pre_rounding_total = _labeled_amount(
        lines,
        re.compile(r"\btotal\s+(?:amt|amount)\b", re.IGNORECASE),
        exclusions=rounding_exclusions,
        ignore_percentages=filter_non_payable,
    )
    rounding = _labeled_amount(
        lines,
        rounding_label,
        ignore_percentages=filter_non_payable,
    )
    if (
        pre_rounding_total is not None
        and pre_rounding_total.value is not None
        and rounding is not None
        and rounding.value is not None
    ):
        value = pre_rounding_total.value + rounding.value
        raw = f"{pre_rounding_total.raw_text} + {rounding.raw_text}"
        return ExtractedField(
            value=value,
            raw_text=raw,
            confidence=min(pre_rounding_total.confidence, rounding.confidence) * 0.90,
            evidence=[*pre_rounding_total.evidence, *rounding.evidence],
        )

    total = _labeled_amount(
        lines,
        total_labels,
        exclusions=total_exclusions,
        preferred=re.compile(
            r"grand\s+total|net\s+total|take\s*out\s+total|total\s+rounded|"
            r"total\s+sales|total\s*\(\s*rm\s*\)|amount\s+due|balance\s+due",
            re.IGNORECASE,
        ),
        ignore_percentages=filter_non_payable,
    )
    if total is not None:
        return total
    if pre_rounding_total is not None:
        return pre_rounding_total

    has_unvalued_total_label = any(
        total_labels.search(line.text)
        and not total_exclusions.search(line.text)
        and not _amounts(line.text, ignore_percentages=filter_non_payable)
        for line in lines
    )
    if not has_unvalued_total_label:
        return None
    return _labeled_amount(
        lines,
        re.compile(r"\b(?:payment|cash|visa|master\s*card|card)\b", re.IGNORECASE),
        exclusions=payment_exclusions,
        preferred=re.compile(r"\b(?:visa|master\s*card|card)\b", re.IGNORECASE),
        ignore_percentages=filter_non_payable,
    )


def _infer_currency(lines: list[OCRLine]) -> ExtractedField[str] | None:
    mapping = [
        (re.compile(r"(?:\bRM\b|\bMYR\b)", re.I), "MYR"),
        (re.compile(r"(?:\bRP\b|\bIDR\b)", re.I), "IDR"),
        (re.compile(r"(?:€|\bEUR\b)", re.I), "EUR"),
        (re.compile(r"(?:£|\bGBP\b)", re.I), "GBP"),
        (re.compile(r"(?:\$|\bUSD\b)", re.I), "USD"),
    ]
    for line in reversed(lines):
        for pattern, currency in mapping:
            match = pattern.search(line.text)
            if match:
                return ExtractedField(
                    value=currency,
                    raw_text=match.group(0),
                    confidence=min(0.95, line.confidence * 0.9),
                    evidence=_evidence(line),
                )
    return None


def _classify(lines: list[OCRLine]) -> tuple[DocumentType, float]:
    text = "\n".join(line.text for line in lines).lower()
    if re.search(r"\bcredit\s+note\b", text):
        return DocumentType.CREDIT_NOTE, 0.96
    if re.search(r"\btax\s+invoice\b", text):
        return DocumentType.INVOICE, 0.94
    invoice_hits = len(re.findall(r"\b(?:tax\s+invoice|invoice|amount\s+due|due\s+date)\b", text))
    receipt_hits = len(re.findall(r"\b(?:receipt|cashier|change|cash)\b", text))
    if invoice_hits > receipt_hits and invoice_hits:
        return DocumentType.INVOICE, min(0.95, 0.74 + invoice_hits * 0.08)
    if receipt_hits:
        return DocumentType.RECEIPT, min(0.95, 0.78 + receipt_hits * 0.07)
    if re.search(r"\btotal\b", text):
        return DocumentType.RECEIPT, 0.58
    return DocumentType.UNKNOWN, 0.30


def _extract_line_items(lines: list[OCRLine]) -> list[LineItem]:
    ignored = re.compile(
        r"\b(?:date|invoice|receipt|cashier|change|cash|card|subtotal|total|tax|vat|gst|tva)\b",
        re.IGNORECASE,
    )
    items: list[LineItem] = []
    for line in lines:
        if line.box.y0 < 0.18 or ignored.search(line.text):
            continue
        amounts = _amounts(line.text)
        if not amounts:
            continue
        raw_amount, amount = amounts[-1]
        description = line.text[: line.text.rfind(raw_amount)].strip(" .:-")
        if len(description) < 2 or sum(char.isalpha() for char in description) < 2:
            continue
        confidence = min(0.90, line.confidence * 0.82)
        items.append(
            LineItem(
                description=ExtractedField(
                    value=description,
                    raw_text=description,
                    confidence=confidence,
                    evidence=_evidence(line),
                ),
                amount=ExtractedField(
                    value=amount,
                    raw_text=raw_amount,
                    confidence=confidence,
                    evidence=_evidence(line),
                ),
            )
        )
    return items


def extract_document(
    ocr: OCRDocument,
    *,
    document_id: str,
    review_policy: ReviewPolicy | None = None,
    rules_version: OCRRulesVersion = OCRRulesVersion.V2,
) -> DocumentExtraction:
    lines = ocr.lines
    document_type, classification_confidence = _classify(lines)
    if rules_version is OCRRulesVersion.INVOICE_V1 and any(
        INVOICE_HEADING.fullmatch(line.text) for line in lines
    ):
        document_type, classification_confidence = DocumentType.INVOICE, 0.94
    invoice = (
        read_invoice_fields(lines)
        if rules_version is OCRRulesVersion.INVOICE_V1 and document_type is DocumentType.INVOICE
        else None
    )
    if invoice is None and rules_version is OCRRulesVersion.INVOICE_V1:
        rules_version = OCRRulesVersion.V2
    subtotal = _labeled_amount(lines, re.compile(r"\bsub[-\s]*total\b", re.I))
    tax = _labeled_amount(lines, re.compile(r"\b(?:tax|vat|gst|tva)\b", re.I))
    total = (
        invoice.total if invoice
        else _extract_total(lines, filter_non_payable=rules_version is OCRRulesVersion.V3)
    )
    merchant = _extract_merchant(lines)
    extraction = DocumentExtraction(
        document_id=document_id,
        document_type=document_type,
        classification_confidence=classification_confidence,
        fields=DocumentFields(
            merchant_name=merchant,
            merchant_address=_extract_address(lines, merchant),
            document_number=_extract_document_number(lines),
            issue_date=invoice.issue_date if invoice else _extract_date(lines),
            due_date=invoice.due_date if invoice else None,
            currency=invoice.currency if invoice else _infer_currency(lines),
            subtotal=subtotal,
            tax=tax,
            total=total or _empty_field(),
            balance_due=invoice.balance_due if invoice else None,
            line_items=_extract_line_items(lines),
        ),
        inference=InferenceMetadata(
            method=ExtractionMethod.OCR_RULES,
            model_name=ocr.engine_name,
            model_version=ocr.engine_version,
            prompt_version=rules_version.value,
            latency_ms=ocr.latency_ms,
        ),
    )
    result = validate_extraction(extraction, review_policy)
    if invoice:
        result.validation_issues.append(ValidationIssue(
            code="invoice_profile_uncalibrated",
            message="Invoice rules require human review until independently calibrated",
            blocking=True,
        ))
        result.requires_human_review = True
    return result
