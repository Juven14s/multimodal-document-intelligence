import json
from datetime import date
from decimal import Decimal

import pytest

from document_intelligence.data.adapters import adapt_cord_record, adapt_sroie_record
from document_intelligence.data.models import CanonicalField, DatasetSplit


def test_cord_adapter_handles_single_menu_mapping_and_indonesian_amounts() -> None:
    ground_truth = {
        "gt_parse": {
            "menu": {"nm": "BASO BIHUN", "unitprice": "43.636", "cnt": "1", "price": "43.636"},
            "sub_total": {"subtotal_price": "43.636", "tax_price": "4.364"},
            "total": {"total_price": "48.000"},
        },
        "meta": {"image_id": 4, "image_size": {"width": 864, "height": 1296}},
        "valid_line": [
            {
                "words": [
                    {
                        "text": "BASO BIHUN",
                        "quad": {
                            "x1": 86,
                            "y1": 129,
                            "x2": 432,
                            "y2": 129,
                            "x3": 432,
                            "y3": 259,
                            "x4": 86,
                            "y4": 259,
                        },
                    }
                ]
            }
        ],
    }

    result = adapt_cord_record(
        {"ground_truth": json.dumps(ground_truth)}, source_split="train", index=0
    )

    assert result.manifest.document_id == "cord-v2-train-4"
    assert result.annotation.total == Decimal("48000")
    assert result.annotation.line_items[0].unit_price == Decimal("43636")
    assert result.manifest.ocr_region_count == 1
    assert result.ocr_regions[0].box.x0 == pytest.approx(86 / 864)
    assert result.annotation.annotated_fields == {
        CanonicalField.LINE_ITEMS,
        CanonicalField.SUBTOTAL,
        CanonicalField.TAX,
        CanonicalField.TOTAL,
    }


def test_sroie_adapter_normalizes_entities_and_boxes() -> None:
    row = {
        "key": "X001",
        "image_size": {"width": 100, "height": 200},
        "entities": {
            "company": "  ABC   TRADING ",
            "date": "30 DEC 17",
            "address": None,
            "total": "RM 31.00",
        },
        "words": ["ABC TRADING", "31.00"],
        "bboxes": [[10, 20, 90, 40], [50, 160, 90, 180]],
    }

    result = adapt_sroie_record(row, source_split="train", index=0)

    assert result.annotation.merchant_name == "ABC TRADING"
    assert result.annotation.issue_date == date(2017, 12, 30)
    assert result.annotation.total == Decimal("31.00")
    assert result.manifest.ocr_region_count == 2
    assert result.ocr_regions[0].box.model_dump() == {
        "page": 1,
        "x0": 0.1,
        "y0": 0.1,
        "x1": 0.9,
        "y1": 0.2,
    }
    assert result.annotation.annotated_fields == {
        CanonicalField.MERCHANT_NAME,
        CanonicalField.ISSUE_DATE,
        CanonicalField.TOTAL,
    }


def test_sroie_adapter_rejects_misaligned_ocr_arrays() -> None:
    row = {
        "key": "X001",
        "image_size": {"width": 100, "height": 200},
        "entities": {},
        "words": ["one"],
        "bboxes": [],
    }

    with pytest.raises(ValueError, match="same length"):
        adapt_sroie_record(row, source_split="test", index=0)


def test_sroie_adapter_skips_degenerate_and_malformed_ocr_boxes() -> None:
    row = {
        "key": "X001",
        "image_size": {"width": 100, "height": 200},
        "entities": {},
        "words": ["valid", "zero width", "clamped", "malformed", "not numeric"],
        "bboxes": [
            [10, 20, 90, 40],
            [10, 20, 10, 40],
            [-10, 20, 0, 40],
            [10, 20],
            ["x", 20, 90, 40],
        ],
    }

    result = adapt_sroie_record(row, source_split="test", index=0)

    assert result.manifest.ocr_region_count == 1
    assert [region.text for region in result.ocr_regions] == ["valid"]


def test_official_test_split_is_preserved() -> None:
    row = {
        "key": "X001",
        "image_size": {"width": 100, "height": 200},
        "entities": {},
        "words": [],
        "bboxes": [],
    }

    result = adapt_sroie_record(row, source_split="test", index=0)

    assert result.manifest.benchmark_split is DatasetSplit.TEST
