"""Check a fresh local stack from invoice upload to reviewed JSON export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from document_intelligence.service import ExtractionResult
from document_intelligence.ui.workbench import FIELD_NAMES, build_human_correction
from document_intelligence.uncertainty.review_queue import ReviewCase, ReviewedExtraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8010")
    parser.add_argument("--fixtures", type=Path, default=Path("output/pdf/invoice-demo"))
    args = parser.parse_args()

    with httpx.Client(base_url=args.api, timeout=45) as client:
        ready = client.get("/ready").raise_for_status().json()
        assert ready["methods"]["ocr_rules"], "Tesseract is unavailable"
        refs = json.loads((args.fixtures / "references.json").read_text(encoding="utf-8"))
        ref = next(row for row in refs if row["id"] == "invoice-part-payment")
        with (args.fixtures / ref["file"]).open("rb") as pdf:
            upload = client.post(
                "/v1/documents/extract",
                files={"file": (ref["file"], pdf, "application/pdf")},
                data={"document_id": "release-smoke-invoice", "method": "ocr_rules"},
            ).raise_for_status()
        result = ExtractionResult.model_validate(upload.json())
        case_id = result.review_case_id
        assert upload.status_code == 201 and case_id is not None
        assert result.extraction.inference.prompt_version == "ocr-invoice-v1"
        assert "calibration_profile_missing" in {
            issue.code for issue in result.extraction.validation_issues
        }
        fields = result.extraction.model_dump(mode="json")["fields"]
        for key in ("total", "balance_due", "currency", "issue_date", "due_date"):
            assert fields[key]["value"] == ref["fields"][key]
            assert fields[key]["evidence"], f"Missing OCR evidence for {key}"
        assert client.get(f"/v1/reviews/{case_id}/export").status_code == 409

        reviewer = "release-smoke"
        claim = client.post("/v1/reviews/claim", json={"reviewer": reviewer}).raise_for_status()
        case = ReviewCase.model_validate(claim.json())
        assert case.case_id == case_id, "Run the smoke check on an isolated review database"
        correction = build_human_correction(
            case.prediction,
            reviewer=reviewer,
            document_type="invoice",
            field_rows=[[key, ref["fields"].get(key, "")] for key in FIELD_NAMES],
            line_item_rows=[
                [item[key] for key in ("description", "quantity", "unit_price", "amount")]
                for item in ref["line_items"]
            ],
        )
        resolve = client.post(f"/v1/reviews/{case_id}/resolve", json={
            "reviewer": reviewer, "correction": correction.model_dump(mode="json"),
        }).raise_for_status()
        assert ReviewCase.model_validate(resolve.json()).prediction == result.extraction
        export = client.get(f"/v1/reviews/{case_id}/export").raise_for_status()
        assert export.headers["content-disposition"].startswith("attachment;")
        final = ReviewedExtraction.model_validate(export.json())
        assert final.status == "corrected" and final.extraction == correction
    print("Release smoke passed: invoice upload, actual OCR, review and JSON export")


if __name__ == "__main__":
    main()
