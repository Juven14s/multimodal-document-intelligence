"""Generate four synthetic, inspectable invoice fixtures for the product demo."""

from __future__ import annotations

import argparse
import io
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pymupdf

INVOICES = (
    {
        "id": "invoice-office",
        "merchant": "NORTHBRIDGE OFFICE SUPPLIES",
        "number": "NB-101",
        "date": "2026-11-13",
        "due": "2026-12-13",
        "currency": "EUR",
        "items": (("Archive boxes", "2", "25.00"), ("Desk organisers", "1", "50.00")),
        "tax": "20.00",
    },
    {
        "id": "invoice-part-payment",
        "merchant": "REDBRIDGE MAINTENANCE SERVICES",
        "number": "RM-204",
        "date": "2026-11-14",
        "due": "2026-12-14",
        "currency": "EUR",
        "items": (("Quarterly equipment service", "1", "200.00"), ("Service kit", "1", "50.00")),
        "tax": "50.00",
        "paid": "100.00",
    },
    {
        "id": "invoice-scan",
        "merchant": "WESTFIELD PRINT STUDIO",
        "number": "WP-308",
        "date": "2026-11-15",
        "due": "2026-12-15",
        "currency": "GBP",
        "items": (("Workshop manuals", "3", "25.00"), ("Cover prints", "1", "15.00")),
        "tax": "18.00",
        "scan": True,
    },
    {
        "id": "invoice-multipage",
        "merchant": "MAPLE RIDGE WORKSPACE SERVICES",
        "number": "MW-412",
        "date": "2026-11-16",
        "due": "2026-12-16",
        "currency": "EUR",
        "items": (
            ("Workspace inspection", "2", "75.00"), ("Site access setup", "1", "50.00"),
            ("Safety labels", "5", "10.00"), ("Follow-up visit", "1", "100.00"),
        ),
        "tax": "70.00",
        "page_break": 2,
    },
)

HOLDOUT = (
    {
        "id": "invoice-holdout-vat",
        "merchant": "CEDAR VALLEY ELECTRICAL",
        "number": "CV-713",
        "date": "2026-10-17", "due": "2026-11-22", "currency": "GBP",
        "items": (("Lighting installation", "2", "90.00"), ("Safety check", "1", "40.00")),
        "tax": "44.00", "paid": "80.00",
        "labels": {"total": "Total incl. VAT", "balance": "Outstanding balance", "due": "Due date"},
    },
    {
        "id": "invoice-holdout-balance-only",
        "merchant": "COASTLINE STORAGE SERVICES",
        "number": "CS-820",
        "date": "2026-10-18", "due": "2026-11-23", "currency": "EUR",
        "items": (("Storage renewal", "1", "95.00"),), "tax": "19.00",
        "balance_only": "84.00",
        "labels": {"balance": "Amount due", "due": "Pay by"},
    },
    {
        "id": "invoice-holdout-iso",
        "merchant": "ORCHARD TECHNICAL SUPPORT",
        "number": "EUR-921",
        "date": "2026-10-19", "due": "2026-11-24", "currency": "USD",
        "items": (("Remote support", "4", "70.00"), ("Setup review", "1", "60.00")),
        "tax": "34.00", "iso_dates": True,
        "labels": {"issue": "Date of issue", "due": "Due by", "total": "Grand total"},
    },
    {
        "id": "invoice-holdout-fr",
        "merchant": "ATELIER DU BELVEDERE",
        "number": "AB-731",
        "date": "2026-10-20", "due": "2026-11-25", "currency": "EUR",
        "items": (("Entretien du mobilier", "2", "160.00"),),
        "tax": "64.00", "paid": "60.00", "scan": True,
        "labels": {
            "title": "FACTURE", "number": "Facture no", "issue": "Date de facture",
            "due": "Date d'échéance", "currency": "Devise", "subtotal": "Total HT",
            "tax": "TVA", "total": "Total TTC", "paid": "Acompte", "balance": "Solde à payer",
        },
    },
    {
        "id": "invoice-holdout-currency-conflict",
        "merchant": "HILLSIDE DOCUMENT SERVICES",
        "number": "HD-842",
        "date": "2026-10-21", "due": "2026-11-26", "currency": "EUR",
        "second_currency": "GBP",
        "items": (("Archive preparation", "1", "70.00"), ("Digitisation", "2", "30.00")),
        "tax": "26.00", "page_break": 1,
        "labels": {"total": "Total amount", "due": "Due date"},
    },
)


def _money(value: Decimal) -> str:
    return f"{value:.2f}"


def _display_date(value: str) -> str:
    return date.fromisoformat(value).strftime("%d/%m/%Y")


def _generate(invoice: dict, output: Path) -> dict:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    width, height = A4
    labels = {
        "title": "TAX INVOICE", "number": "Invoice No.", "issue": "Invoice date",
        "due": "Payment due", "currency": "Currency", "subtotal": "Subtotal", "tax": "Tax",
        "total": "Invoice total", "paid": "Amount paid", "balance": "Balance due",
        **invoice.get("labels", {}),
    }
    left, right = 52, width - 52
    ink, muted = HexColor("#132e36"), HexColor("#475d64")
    rule, tint = HexColor("#b6cacc"), HexColor("#eff5f5")
    items = [
        {
            "description": name, "quantity": quantity, "unit_price": unit,
            "amount": _money(Decimal(quantity) * Decimal(unit)),
        }
        for name, quantity, unit in invoice["items"]
    ]
    subtotal = sum(Decimal(item["amount"]) for item in items)
    tax = Decimal(invoice["tax"])
    total = subtotal + tax
    paid = Decimal(invoice["paid"]) if "paid" in invoice else None
    balance_only = "balance_only" in invoice
    balance = Decimal(invoice["balance_only"]) if balance_only else (
        total - paid if paid is not None else None
    )
    if "page_break" in invoice:
        index = invoice["page_break"]
        pages = [items[:index], items[index:]]
    else:
        pages = [items]

    buffer = io.BytesIO()
    pdf = Canvas(buffer, pagesize=A4, invariant=1)
    pdf.setTitle(f"Synthetic invoice {invoice['number']}")
    for number, page_items in enumerate(pages, start=1):
        pdf.setFillColor(ink)
        pdf.setFont("Helvetica-Bold", 17)
        pdf.drawString(left, height - 67, invoice["merchant"])
        pdf.setFont("Helvetica-Bold", 26)
        pdf.drawString(left, height - 110, labels["title"])
        pdf.setStrokeColor(rule)
        pdf.line(left, height - 123, right, height - 123)
        pdf.setFillColor(muted)
        pdf.setFont("Helvetica", 12)
        pdf.drawString(left, height - 151, f"{labels['number']} {invoice['number']}")
        date_display = str if invoice.get("iso_dates") else _display_date
        pdf.drawString(left, height - 175, f"{labels['issue']}: {date_display(invoice['date'])}")
        pdf.drawString(left, height - 199, f"{labels['due']}: {date_display(invoice['due'])}")
        pdf.drawRightString(right, height - 151, f"{labels['currency']} {invoice['currency']}")
        if "second_currency" in invoice:
            pdf.drawRightString(right, height - 175, f"Currency {invoice['second_currency']}")

        y = height - 259
        pdf.setFillColor(tint)
        pdf.rect(left - 8, y - 10, right - left + 16, 28, fill=1, stroke=0)
        pdf.setFillColor(ink)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(left, y, "DESCRIPTION")
        pdf.drawRightString(right - 150, y, "QTY")
        pdf.drawRightString(right - 72, y, "UNIT PRICE")
        pdf.drawRightString(right, y, "AMOUNT")
        pdf.setFont("Helvetica", 12)
        for item in page_items:
            y -= 35
            pdf.drawString(left, y, item["description"])
            pdf.drawRightString(right - 150, y, item["quantity"])
            pdf.drawRightString(right - 72, y, item["unit_price"])
            pdf.drawRightString(right, y, item["amount"])
            pdf.line(left, y - 12, right, y - 12)

        if number == len(pages):
            y -= 73
            amounts = [] if balance_only else [
                ("subtotal", subtotal), ("tax", tax), ("total", total),
            ]
            if not balance_only and paid is not None:
                amounts.append(("paid", paid))
            if balance is not None:
                amounts.append(("balance", balance))
            for role, value in amounts:
                pdf.setFont("Helvetica-Bold" if role == "total" else "Helvetica", 13)
                pdf.drawString(right - 226, y, labels[role])
                pdf.drawRightString(right, y, _money(value))
                y -= 30
        else:
            pdf.setFont("Helvetica", 11)
            pdf.drawString(left, y - 43, "Continues on the next page")

        pdf.setFillColor(muted)
        pdf.line(left, 75, right, 75)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(left, 57, "SYNTHETIC EXAMPLE - NOT FOR PAYMENT")
        pdf.drawRightString(right, 57, f"Page {number} of {len(pages)}")
        pdf.showPage()
    pdf.save()

    name = f"{invoice['id']}.pdf"
    if invoice.get("scan"):
        with pymupdf.open(stream=buffer.getvalue(), filetype="pdf") as source:
            png = source[0].get_pixmap(dpi=180, alpha=False).tobytes("png")
        scanned = Canvas(str(output / name), pagesize=A4, invariant=1)
        scanned.setTitle(f"Synthetic raster invoice {invoice['number']}")
        scanned.drawImage(ImageReader(io.BytesIO(png)), 0, 0, width, height)
        scanned.showPage()
        scanned.save()
    else:
        (output / name).write_bytes(buffer.getvalue())

    return {
        "id": invoice["id"], "file": name, "pages": len(pages),
        "document_type": "invoice", "raster_only": bool(invoice.get("scan")),
        "fields": {
            "merchant_name": invoice["merchant"], "document_number": invoice["number"],
            "issue_date": invoice["date"], "due_date": invoice["due"],
            "currency": None if "second_currency" in invoice else invoice["currency"],
            "subtotal": None if balance_only else _money(subtotal),
            "tax": None if balance_only else _money(tax),
            "total": None if balance_only else _money(total),
            "balance_due": _money(balance) if balance is not None else None,
        },
        "line_items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/pdf/invoice-demo"))
    parser.add_argument("--suite", choices=("demo", "holdout"), default="demo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    invoices = HOLDOUT if args.suite == "holdout" else INVOICES
    references = [_generate(invoice, args.output) for invoice in invoices]
    (args.output / "references.json").write_text(
        json.dumps(references, indent=2), encoding="utf-8"
    )
    print(f"Generated {len(references)} synthetic invoices in {args.output}")


if __name__ == "__main__":
    main()
