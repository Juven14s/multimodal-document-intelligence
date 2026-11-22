from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from document_intelligence.ocr.models import OCRDocument, OCRLine, OCRPage, OCRWord
from document_intelligence.ocr.rules import OCRRulesVersion, extract_document
from document_intelligence.ocr.total_fallback import apply_second_pass
from document_intelligence.ocr.total_selection import select_total_proposal
from document_intelligence.schemas import BoundingBox, DocumentType


def make_line(number: int, text: str, y0: float, y1: float, confidence: float = 0.95):
    box = BoundingBox(page=1, x0=0.1, y0=y0, x1=0.9, y1=y1)
    word = OCRWord(
        text=text,
        confidence=confidence,
        box=box,
        block_number=1,
        paragraph_number=1,
        line_number=number,
        word_number=1,
    )
    return OCRLine(text=text, confidence=confidence, box=box, words=[word])


def make_receipt() -> OCRDocument:
    lines = [
        make_line(1, "EXAMPLE CAFE SDN BHD", 0.03, 0.06),
        make_line(2, "RECEIPT NO: AB-1234", 0.09, 0.12),
        make_line(3, "DATE: 25/12/2018", 0.14, 0.17),
        make_line(4, "Coffee 8.00", 0.30, 0.34),
        make_line(5, "Cake 2.00", 0.36, 0.40),
        make_line(6, "SUBTOTAL RM 10.00", 0.72, 0.76),
        make_line(7, "GST RM 0.60", 0.78, 0.82),
        make_line(8, "GRAND TOTAL RM 10.60", 0.85, 0.90),
        make_line(9, "CASH RM 20.00", 0.92, 0.95),
    ]
    return OCRDocument(
        engine_name="fixture-ocr",
        engine_version="1.0",
        language="eng",
        latency_ms=20,
        pages=[
            OCRPage(
                page_number=1,
                image_path=Path("receipt.png"),
                width=1000,
                height=2000,
                words=[word for line in lines for word in line.words],
                lines=lines,
            )
        ],
    )


def make_document(lines: list[OCRLine]) -> OCRDocument:
    return OCRDocument(
        engine_name="fixture-ocr",
        engine_version="1.0",
        language="eng",
        latency_ms=20,
        pages=[
            OCRPage(
                page_number=1,
                image_path=Path("document.png"),
                width=1000,
                height=2000,
                words=[word for line in lines for word in line.words],
                lines=lines,
            )
        ],
    )


def test_rule_baseline_extracts_receipt_fields_with_evidence() -> None:
    result = extract_document(make_receipt(), document_id="receipt-001")

    assert result.document_type is DocumentType.RECEIPT
    assert result.fields.merchant_name.value == "EXAMPLE CAFE SDN BHD"
    assert result.fields.document_number.value == "AB-1234"
    assert result.fields.issue_date.value.isoformat() == "2018-12-25"
    assert result.fields.currency.value == "MYR"
    assert str(result.fields.subtotal.value) == "10.00"
    assert str(result.fields.tax.value) == "0.60"
    assert str(result.fields.total.value) == "10.60"
    assert [item.description.value for item in result.fields.line_items] == ["Coffee", "Cake"]
    assert result.fields.total.evidence[0].text == "GRAND TOTAL RM 10.60"
    assert result.requires_human_review is False


def test_missing_total_routes_document_to_review() -> None:
    ocr = make_receipt()
    ocr.pages[0].lines = ocr.pages[0].lines[:5]

    result = extract_document(ocr, document_id="receipt-002")

    assert result.fields.total.value is None
    assert result.requires_human_review is True
    assert "missing_required_field" in {issue.code for issue in result.validation_issues}


def test_cash_tendered_is_not_used_without_an_empty_total_label() -> None:
    ocr = make_receipt()
    ocr.pages[0].lines = [
        line for line in ocr.pages[0].lines if "GRAND TOTAL" not in line.text
    ]

    result = extract_document(ocr, document_id="receipt-cash-only")

    assert result.fields.total.value is None
    assert result.requires_human_review is True


def test_extracts_business_name_address_and_card_total() -> None:
    lines = [
        make_line(1, "tan chay yee", 0.02, 0.04),
        make_line(2, "OJC MARKETING SDN BHD", 0.05, 0.08),
        make_line(3, "ROC NO: 123456-X", 0.09, 0.11),
        make_line(4, "NO. 12, JALAN MAWAR 3", 0.12, 0.15),
        make_line(5, "TAMAN MAWAR", 0.16, 0.18),
        make_line(6, "43000 KAJANG, SELANGOR", 0.19, 0.22),
        make_line(7, "TEL: 03-12345678", 0.23, 0.25),
        make_line(8, "TAX INVOICE", 0.27, 0.30),
        make_line(9, "TOTAL:", 0.75, 0.78),
        make_line(10, "VISA CARD 170.00", 0.80, 0.84),
    ]

    result = extract_document(make_document(lines), document_id="invoice-001")

    assert result.document_type is DocumentType.INVOICE
    assert result.fields.merchant_name.value == "OJC MARKETING SDN BHD"
    assert result.fields.merchant_address is not None
    assert result.fields.merchant_address.value == (
        "NO. 12, JALAN MAWAR 3, TAMAN MAWAR, 43000 KAJANG, SELANGOR"
    )
    assert result.fields.document_number is None
    assert "ROC" not in result.fields.merchant_address.value
    assert "TEL" not in result.fields.merchant_address.value
    assert str(result.fields.total.value) == "170.00"
    assert "low_classification_confidence" not in {
        issue.code for issue in result.validation_issues
    }


@pytest.mark.parametrize(
    ("summary_lines", "expected"),
    [
        (["Total GST (RM) 24.69", "Total (RM): 436.20"], "436.20"),
        (
            ["TakeOut Total (incl GST) 38.90", "TOTAL INCLUDES 6% GST 2.20"],
            "38.90",
        ),
        (["Total Sales inclusive of GST: 7.30", "Total item(s): 2"], "7.30"),
        (["TOT’ : RM 278.80", "CAS1, : RM 278.80"], "278.80"),
        (["TOTA: RM 278.80", "CAS1, : RM 278.80"], "278.80"),
        (["Total Amt incl. GST @ 6%: 6.36", "Roun Adjustment: -0.01"], "6.35"),
        (["Total GST @ 6%: 6.36", "Roun Adjustment:", "al 6.35"], "6.35"),
        (["Total Sales RM 65. 20"], "65.20"),
        (["Total ~ 92:80"], "92.80"),
        (["SUB-TOTAL 88.97", "TOTAL 88.95"], "88.95"),
        (["TOTAL:", "TOTAL QUANTITY 4", "CASH RM 48.00"], "48.00"),
        (["TOTAL:", "GST @6% INCLUDED IN TOTAL RM 1.48", "CASH RM 26.10"], "26.10"),
    ],
)
def test_total_disambiguation(summary_lines: list[str], expected: str) -> None:
    lines = [
        make_line(1, "EXAMPLE TRADING SDN BHD", 0.03, 0.06),
        make_line(2, "RECEIPT", 0.10, 0.13),
        *[
            make_line(index + 3, text, 0.70 + index * 0.06, 0.74 + index * 0.06)
            for index, text in enumerate(summary_lines)
        ],
    ]

    result = extract_document(make_document(lines), document_id="total-case")

    assert str(result.fields.total.value) == expected


@pytest.mark.parametrize(
    ("summary_lines", "expected"),
    [
        (["TOTAL 48.00", "TAX TOTAL: 2.72"], "48.00"),
        (["TOTAL 6.00", "Total (Excluding GST): 5.66"], "6.00"),
        (["TOTAL 15.00", "XGST @ 6% INCLUDED IN TOTAL"], "15.00"),
        (["NET TOTAL 67.85 6, %"], "67.85"),
        (["TOTAL:", "Card No 46 XXXxXXXxX 1819"], None),
        (["TOTAL:", "VISA CARD 170.00", "Card No 46 XXXxXXXxX 1819"], "170.00"),
    ],
)
def test_v3_total_filters_tax_percentages_and_card_references(
    summary_lines: list[str], expected: str | None
) -> None:
    ocr = make_document([
        make_line(index, text, 0.70 + index * 0.06, 0.74 + index * 0.06)
        for index, text in enumerate(summary_lines)
    ])

    result = extract_document(ocr, document_id="total-v3-case", rules_version=OCRRulesVersion.V3)

    value = result.fields.total.value
    assert (str(value) if value is not None else None) == expected
    assert result.inference.prompt_version == "ocr-rules-v3"
    if expected is None:
        assert result.requires_human_review is True


def test_second_pass_recovers_spatial_total_and_requires_human_review() -> None:
    primary = extract_document(
        make_document([make_line(1, "EXAMPLE CAFE SDN BHD", 0.03, 0.06)]),
        document_id="fallback",
        rules_version=OCRRulesVersion.V3,
    )
    label = make_line(2, "GRAND TOTAL", 0.76, 0.80)
    label.box = label.box.model_copy(update={"x1": 0.35})
    amount = make_line(3, "RM 48.00", 0.77, 0.80)
    amount.box = amount.box.model_copy(update={"x0": 0.66})

    result = apply_second_pass(primary, make_document([label, amount]))

    assert primary.fields.total.value is None
    assert result.fields.total.value == Decimal("48.00")
    assert [item.text for item in result.fields.total.evidence] == ["GRAND TOTAL", "RM 48.00"]
    assert result.requires_human_review
    assert any(
        issue.code == "total_from_second_ocr_pass" and issue.blocking
        for issue in result.validation_issues
    )
    assert not any(
        issue.code == "missing_required_field" and issue.field == "total"
        for issue in result.validation_issues
    )


@pytest.mark.parametrize(
    "alternate_lines",
    [
        ["Tax total 12.30", "Credit card 98.00"],
        ["TOTAL 12.30", "GRAND TOTAL 98.00"],
        ["TOTAL 6 %", "TOTAL ITEMS 12.30"],
    ],
)
def test_second_pass_abstains_for_invalid_or_competing_totals(alternate_lines: list[str]) -> None:
    primary = extract_document(
        make_document([make_line(1, "EXAMPLE CAFE SDN BHD", 0.03, 0.06)]),
        document_id="fallback-abstention",
    )
    alternate = make_document([
        make_line(index, text, 0.74 + 0.05 * index, 0.78 + 0.05 * index)
        for index, text in enumerate(alternate_lines)
    ])

    assert apply_second_pass(primary, alternate) == primary


def test_second_pass_preserves_existing_totals() -> None:
    primary = extract_document(make_receipt(), document_id="no-fallback")
    alternate = make_document([make_line(1, "TOTAL 98.00", 0.74, 0.78)])

    assert apply_second_pass(primary, alternate) == primary


def test_selective_total_uses_a_unique_payable_amount() -> None:
    ocr = make_receipt()
    extraction = extract_document(ocr, document_id="selective-total")
    proposal = select_total_proposal(extraction, ocr)

    assert proposal is not None and proposal.value == Decimal("10.60")
    assert proposal.evidence[0].text == "GRAND TOTAL RM 10.60"

    ambiguous = make_document([
        *ocr.lines,
        make_line(10, "TOTAL 16.00", 0.96, 0.99),
    ])
    assert select_total_proposal(extraction, ambiguous) is None


def test_selective_total_rejects_tax_supply_line_and_amount_inconsistency() -> None:
    supplies = make_document([make_line(1, "Total supplies: 23.38", 0.75, 0.79)])
    extraction = extract_document(supplies, document_id="supplies")
    assert extraction.fields.total.value == Decimal("23.38")
    assert select_total_proposal(extraction, supplies) is None

    ocr = make_receipt()
    lines = [
        make_line(7, "GST RM 1.60", 0.78, 0.82) if "GST" in line.text else line
        for line in ocr.lines
    ]
    inconsistent = make_document(lines)
    extraction = extract_document(inconsistent, document_id="inconsistent")
    assert extraction.fields.total.value == Decimal("10.60")
    assert select_total_proposal(extraction, inconsistent) is None


def test_invoice_profile_separates_gross_balance_currency_and_dates() -> None:
    lines = [
        make_line(1, "REDBRIDGE MAINTENANCE SERVICES", 0.03, 0.06),
        make_line(2, "TAX INVOICE", 0.11, 0.14),
        make_line(3, "Invoice No. RM-204 Currency EUR", 0.19, 0.22),
        make_line(4, "Invoice date: 14/11/2026 Due by: 2026-12-14", 0.25, 0.28),
        make_line(5, "Subtotal 250.00", 0.55, 0.59),
        make_line(6, "Tax 50.00", 0.61, 0.65),
        make_line(7, "Invoice total 300.00", 0.66, 0.70),
        make_line(8, "Balance due 200.00", 0.72, 0.76),
    ]
    result = extract_document(
        make_document(lines), document_id="partial-payment",
        rules_version=OCRRulesVersion.INVOICE_V1,
    )
    fields = result.fields

    assert result.inference.prompt_version == "ocr-invoice-v1"
    assert fields.total.value == Decimal("300.00")
    assert fields.total.evidence[0].text == "Invoice total 300.00"
    assert fields.balance_due.value == Decimal("200.00")
    assert fields.balance_due.evidence[0].text == "Balance due 200.00"
    assert fields.currency.value == "EUR"
    assert fields.issue_date.value == date(2026, 11, 14)
    assert fields.due_date.value == date(2026, 12, 14)
    assert result.requires_human_review


def test_invoice_profile_leaves_total_and_currency_blank_without_unique_evidence() -> None:
    lines = [
        make_line(1, "COASTLINE STORAGE SERVICES", 0.03, 0.06),
        make_line(2, "TAX INVOICE", 0.11, 0.14),
        make_line(3, "Currency EUR Currency GBP", 0.21, 0.24),
        make_line(4, "Amount due 84.00", 0.60, 0.64),
    ]
    result = extract_document(
        make_document(lines), document_id="only-balance", rules_version=OCRRulesVersion.INVOICE_V1
    )

    assert result.fields.total.value is None
    assert result.fields.balance_due.value == Decimal("84.00")
    assert result.fields.currency is None
    assert result.requires_human_review


def test_invoice_profile_reads_french_invoice_labels() -> None:
    lines = [
        make_line(1, "ATELIER DU BELVEDERE", 0.03, 0.06),
        make_line(2, "FACTURE", 0.11, 0.14),
        make_line(3, "Date de facture: 20/10/2026", 0.21, 0.24),
        make_line(4, "Date d'échéance: 25/11/2026", 0.26, 0.30),
        make_line(5, "Total HT 1 000,00", 0.60, 0.64),
        make_line(6, "Total TTC 1 200,00 EUR", 0.66, 0.70),
        make_line(7, "Solde à payer 1000,00 EUR", 0.73, 0.77),
    ]
    result = extract_document(
        make_document(lines), document_id="invoice-fr", rules_version=OCRRulesVersion.INVOICE_V1
    )

    assert result.document_type is DocumentType.INVOICE
    assert result.fields.total.value == Decimal("1200.00")
    assert result.fields.balance_due.value == Decimal("1000.00")
    assert result.fields.currency.value == "EUR"
    assert result.fields.issue_date.value == date(2026, 10, 20)
    assert result.fields.due_date.value == date(2026, 11, 25)


def test_invoice_profile_preserves_the_receipt_rules_and_version() -> None:
    ocr = make_receipt()
    invoice_enabled = extract_document(
        ocr, document_id="receipt", rules_version=OCRRulesVersion.INVOICE_V1
    )

    assert invoice_enabled == extract_document(ocr, document_id="receipt")


def test_prefers_near_tied_uppercase_brand_on_first_line() -> None:
    lines = [
        make_line(1, "PETRON BKT LANJAN SB", 0.02, 0.05),
        make_line(2, "ALSERKAM ENTERPRISE", 0.06, 0.09),
        make_line(3, "KM 456.4 BKT LANJAN UTARA", 0.11, 0.14),
        make_line(4, "TAX INVOICE", 0.25, 0.28),
        make_line(5, "Total RM inc.GST: 4.90", 0.80, 0.84),
    ]

    result = extract_document(make_document(lines), document_id="petron")

    assert result.fields.merchant_name.value == "PETRON BKT LANJAN SB"


def test_merchant_search_is_not_limited_to_first_eight_lines() -> None:
    lines = [
        make_line(1, "TAN WOON YANN", 0.02, 0.04),
        make_line(2, "WELCOME", 0.05, 0.07),
        make_line(3, "LOT 12", 0.08, 0.10),
        make_line(4, "BANDAR BARU", 0.11, 0.13),
        make_line(5, "TAMAN MAWAR", 0.14, 0.16),
        make_line(6, "43000 KAJANG", 0.17, 0.19),
        make_line(7, "SELANGOR", 0.20, 0.21),
        make_line(8, "CUSTOMER COPY", 0.22, 0.23),
        make_line(9, "MR. D.I.Y. (M) SDN BHD", 0.24, 0.27),
        make_line(10, "TOTAL RM 10.00", 0.80, 0.84),
    ]

    result = extract_document(make_document(lines), document_id="merchant-after-eight")

    assert result.fields.merchant_name.value == "MR. D.I.Y. (M) SDN BHD"


def test_skips_company_history_before_address_and_stops_at_registration() -> None:
    lines = [
        make_line(1, "GERBANG ALAF RESTAURANTS SDN BHD", 0.03, 0.06),
        make_line(2, "(65351-M)", 0.08, 0.10),
        make_line(3, "formerly known as", 0.11, 0.13),
        make_line(4, "Golden Arches Restaurants Sdn Bhd", 0.14, 0.16),
        make_line(5, "Licensee of McDonald's", 0.17, 0.19),
        make_line(6, "Level 6, Bangunan TH, Damansara Uptown3", 0.20, 0.23),
        make_line(7, "No. 3, Jalan SS21/39", 0.24, 0.27),
        make_line(8, "47400 Petaling Jaya, Selangor", 0.28, 0.31),
        make_line(9, "ROC No: 12345-X", 0.32, 0.34),
        make_line(10, "TAX INVOICE", 0.40, 0.43),
        make_line(11, "TOTAL RM 38.90", 0.80, 0.84),
    ]

    result = extract_document(make_document(lines), document_id="address-history")

    assert result.fields.merchant_address is not None
    assert result.fields.merchant_address.value == (
        "Level 6, Bangunan TH, Damansara Uptown3, No. 3, Jalan SS21/39, "
        "47400 Petaling Jaya, Selangor"
    )
