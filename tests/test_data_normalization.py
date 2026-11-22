from datetime import date
from decimal import Decimal

import pytest

from document_intelligence.data.normalization import parse_date, parse_decimal, parse_quantity


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RM 31.00", Decimal("31.00")),
        ("75,000", Decimal("75000")),
        ("43.636", Decimal("43636")),
        ("1.234,56", Decimal("1234.56")),
        ("-7,800", Decimal("-7800")),
        (None, None),
    ],
)
def test_parse_decimal(raw, expected) -> None:
    assert parse_decimal(raw) == expected


def test_parse_quantity_ignores_unit_markers() -> None:
    assert parse_quantity("x1") == Decimal("1")
    assert parse_quantity("2 x") == Decimal("2")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("25/12/2018 8:13:39 PM", date(2018, 12, 25)),
        ("14-02-2018", date(2018, 2, 14)),
        ("30 DEC 17", date(2017, 12, 30)),
        ("not a date", None),
    ],
)
def test_parse_date(raw, expected) -> None:
    assert parse_date(raw) == expected
