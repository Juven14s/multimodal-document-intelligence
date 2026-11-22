"""Read explicitly labelled invoice totals, balances, currencies and dates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TypeVar

from document_intelligence.data.normalization import parse_date, parse_decimal
from document_intelligence.ocr.models import OCRLine
from document_intelligence.schemas import Evidence, ExtractedField

INVOICE_HEADING = re.compile(r"^\s*facture\s*$", re.IGNORECASE)
GROSS_LABEL = re.compile(
    r"^\s*(?:invoice\s+total|grand\s+total|total(?:\s+(?:ttc|amount|"
    r"incl(?:uding)?\.?\s*(?:vat|gst|tax|tva)))?)\b\s*[:=]?\s*",
    re.IGNORECASE,
)
BALANCE_LABEL = re.compile(
    r"^\s*(?:balance\s+due|amount\s+due|total\s+due|outstanding\s+balance|"
    r"remaining\s+balance|solde\s+[aà]\s+payer|net\s+[aà]\s+payer|solde)\b\s*[:=]?\s*",
    re.IGNORECASE,
)
CURRENCIES = r"EUR|GBP|USD|CAD|AUD|CHF|MYR|IDR|SGD|JPY|RM|RP|€|£|\$"
CURRENCY_LABEL = re.compile(rf"\b(?:currency|devise)\s*[:=]?\s*({CURRENCIES})(?!\w)", re.I)
MONEY = re.compile(
    rf"(?:(?P<left>{CURRENCIES})\s*|\((?P<wrapped>{CURRENCIES})\)\s*)?"
    r"(?P<number>-?(?:\d{1,3}(?:[,.\s]\d{3})+(?:[,.]\s*\d{1,2})?|"
    r"\d+(?:[,.]\s*\d{1,2})?))"
    rf"(?:\s*(?P<right>{CURRENCIES}))?",
    re.IGNORECASE,
)
ISSUE = r"invoice\s+date|date\s+of\s+issue|issue\s+date|date\s+de\s+facture"
DUE = (
    r"payment\s+due(?:\s+date)?|due\s+(?:date|by|on)|pay\s+by|"
    r"date\s+d['’][eé]ch[eé]ance|[eé]ch[eé]ance"
)
DATE_LABEL = re.compile(rf"\b(?:(?P<due>{DUE})|(?P<issue>{ISSUE}))\b", re.I)
SIMPLE_DATE_LABEL = re.compile(r"^\s*date\s*[:=]", re.I)
DATE_TOKEN = re.compile(
    r"(?<![\w/-])(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{1,2}[ -][A-Z]{3}[ -]\d{2,4})(?![\w/-])",
    re.I,
)

T = TypeVar("T", Decimal, date, str)


@dataclass(frozen=True)
class InvoiceFields:
    total: ExtractedField[Decimal] | None
    balance_due: ExtractedField[Decimal] | None
    issue_date: ExtractedField[date] | None
    due_date: ExtractedField[date] | None
    currency: ExtractedField[str] | None


def _unique(candidates: dict[T, ExtractedField[T]]) -> ExtractedField[T] | None:
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _save(
    candidates: dict[T, ExtractedField[T]], value: T, raw: str, lines: list[OCRLine]
) -> None:
    confidence = min(line.confidence for line in lines)
    if value in candidates and candidates[value].confidence >= confidence:
        return
    candidates[value] = ExtractedField(
        value=value, raw_text=raw, confidence=confidence,
        evidence=[Evidence(text=line.text, box=line.box) for line in lines],
    )


def _aligned(label: OCRLine, amount: OCRLine) -> bool:
    first, second = label.box, amount.box
    if first.page != second.page or first.x1 > second.x0:
        return False
    overlap = min(first.y1, second.y1) - max(first.y0, second.y0)
    return overlap >= 0.5 * min(first.y1 - first.y0, second.y1 - second.y0)


def _amount_for_label(
    lines: list[OCRLine], label: re.Pattern[str]
) -> ExtractedField[Decimal] | None:
    candidates: dict[Decimal, ExtractedField[Decimal]] = {}
    standalone = [line for line in lines if MONEY.fullmatch(line.text.strip())]
    for line in lines:
        match = label.match(line.text)
        if match is None:
            continue
        suffix = line.text[match.end() :].strip()
        if suffix:
            amounts = [(suffix, [line])]
        else:
            amounts = [
                (amount.text.strip(), [line, amount])
                for amount in standalone if _aligned(line, amount)
            ]
        for raw, evidence in amounts:
            money = MONEY.fullmatch(raw)
            if money is None:
                continue
            value = parse_decimal(money["number"])
            if value is not None:
                _save(candidates, value, raw, evidence)
    return _unique(candidates)


def _parse_invoice_date(raw: str) -> date | None:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None
    return parse_date(raw)


def _dates(lines: list[OCRLine]) -> tuple[ExtractedField[date] | None, ExtractedField[date] | None]:
    issue: dict[date, ExtractedField[date]] = {}
    due: dict[date, ExtractedField[date]] = {}
    for line in lines:
        labels = list(DATE_LABEL.finditer(line.text))
        generic = SIMPLE_DATE_LABEL.match(line.text)
        spans = [
            (due if label["due"] else issue, line.text[label.end() : following.start()])
            for label, following in zip(labels, labels[1:], strict=False)
        ]
        if labels:
            last = labels[-1]
            spans.append((due if last["due"] else issue, line.text[last.end() :]))
        elif generic:
            spans.append((issue, line.text[generic.end() :]))
        for candidates, span in spans:
            for token in DATE_TOKEN.finditer(span):
                raw = token.group(0)
                value = _parse_invoice_date(raw)
                if value is not None:
                    _save(candidates, value, raw, [line])
    return _unique(issue), _unique(due)


def _currency_code(raw: str) -> str | None:
    token = raw.upper()
    return {"RM": "MYR", "RP": "IDR", "€": "EUR", "£": "GBP", "$": None}.get(token, token)


def _currency(
    lines: list[OCRLine], amounts: list[ExtractedField[Decimal] | None]
) -> ExtractedField[str] | None:
    explicit: dict[str, ExtractedField[str]] = {}
    for line in lines:
        for match in CURRENCY_LABEL.finditer(line.text):
            raw = match.group(1)
            code = _currency_code(raw)
            if code:
                _save(explicit, code, raw, [line])
    if explicit:
        return _unique(explicit)

    attached: dict[str, ExtractedField[str]] = {}
    for amount in amounts:
        if amount is None or amount.raw_text is None:
            continue
        match = MONEY.fullmatch(amount.raw_text)
        if match is None:
            continue
        for raw in (match["left"], match["wrapped"], match["right"]):
            code = _currency_code(raw) if raw else None
            if code and raw:
                attached[code] = ExtractedField(
                    value=code, raw_text=raw, confidence=amount.confidence,
                    evidence=amount.evidence,
                )
    return _unique(attached)


def read_invoice_fields(lines: list[OCRLine]) -> InvoiceFields:
    total = _amount_for_label(lines, GROSS_LABEL)
    balance = _amount_for_label(lines, BALANCE_LABEL)
    issue, due = _dates(lines)
    return InvoiceFields(total, balance, issue, due, _currency(lines, [total, balance]))
