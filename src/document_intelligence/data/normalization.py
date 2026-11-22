"""Conservative normalizers for heterogeneous receipt annotations."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation


def normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = " ".join(text.split()).strip()
    return text or None


def parse_decimal(value: object) -> Decimal | None:
    """Parse common receipt numbers without guessing an exchange rate or currency."""

    text = normalize_text(value)
    if text is None:
        return None

    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    cleaned = re.sub(r"[^0-9,.-]", "", text).lstrip("-")
    if not cleaned or not any(character.isdigit() for character in cleaned):
        return None

    comma = cleaned.rfind(",")
    dot = cleaned.rfind(".")
    separator_index = max(comma, dot)
    if separator_index >= 0:
        trailing_digits = len(cleaned) - separator_index - 1
        if trailing_digits in {1, 2}:
            integer = re.sub(r"[,.]", "", cleaned[:separator_index]) or "0"
            fraction = cleaned[separator_index + 1 :]
            normalized = f"{integer}.{fraction}"
        else:
            normalized = re.sub(r"[,.]", "", cleaned)
    else:
        normalized = cleaned

    try:
        amount = Decimal(normalized)
    except InvalidOperation:
        return None
    return -amount if negative else amount


def parse_quantity(value: object) -> Decimal | None:
    text = normalize_text(value)
    if text is None:
        return None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    return parse_decimal(match.group(0)) if match else None


_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def parse_date(value: object) -> date | None:
    text = normalize_text(value)
    if text is None:
        return None
    text = text.upper()

    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text)
    if numeric:
        day, month, year = (int(part) for part in numeric.groups())
    else:
        named = re.search(r"\b(\d{1,2})[\s-]+([A-Z]{3})[\s-]+(\d{2,4})\b", text)
        if not named or named.group(2) not in _MONTHS:
            return None
        day = int(named.group(1))
        month = _MONTHS[named.group(2)]
        year = int(named.group(3))

    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None
