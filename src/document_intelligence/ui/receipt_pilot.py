"""Local timed comparison of manual and assisted entry for receipt totals."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from document_intelligence.data.io import read_records
from document_intelligence.data.models import DatasetSplit
from document_intelligence.evaluation.benchmark import _image_path, _read_predictions
from document_intelligence.ocr.total_selection import POLICY_ID

MANIFEST = Path("data/processed/sroie-full/sroie.jsonl")
BENCHMARKS = Path("artifacts/benchmarks")
STUDY_ID = "sroie-ocr-v2-total-pilot-v1"
SELECTIVE_STUDY_ID = "sroie-selective-total-pilot-v1"
REPRESENTATIVE_STUDY_ID = "sroie-selective-total-representative-v1"
REPRESENTATIVE_PAIR_COUNT = 20
SELECTIVE_MANIFEST = MANIFEST.with_name("sroie-selective-test.jsonl")
SELECTIVE_REPORT = BENCHMARKS / "sroie-selective-test/report.json"


@dataclass(frozen=True)
class Receipt:
    document_id: str
    image: Path
    reference: Decimal
    proposal: Decimal | None
    evidence: tuple[str, ...] = ()
    image_sha256: str | None = None


class Answer(BaseModel):
    index: int = Field(ge=0)
    amount: str | None = Field(max_length=32)
    elapsed_ms: int = Field(ge=1, le=3_600_000)
    interrupted: bool = False


def select_receipts(manifest: Path, benchmarks: Path) -> list[tuple[Receipt, Receipt]]:
    records = read_records(manifest)
    predictions, _ = _read_predictions(benchmarks / "sroie-ocr-full-v2/predictions.jsonl")
    by_id = {prediction.document_id: prediction for prediction in predictions}
    seen_ids = {
        prediction.document_id
        for name in ("sroie-ocr-smoke", "sroie-ocr-improved")
        for prediction in _read_predictions(benchmarks / name / "predictions.jsonl")[0]
    }
    seen_hashes = {
        record.manifest.image_sha256 for record in records
        if record.manifest.benchmark_split is not DatasetSplit.TEST
        or record.manifest.document_id in seen_ids
    }
    groups: dict[str, list[Receipt]] = {"exact": [], "wrong": [], "absent": []}
    for record in records:
        entry = record.manifest
        expected = record.annotation.total
        prediction = by_id.get(entry.document_id)
        if (
            entry.benchmark_split is not DatasetSplit.TEST
            or entry.document_id in seen_ids
            or entry.image_sha256 is None
            or entry.image_sha256 in seen_hashes
            or expected is None
            or prediction is None
            or prediction.inference.prompt_version != "ocr-rules-v2"
        ):
            continue
        seen_hashes.add(entry.image_sha256)
        total = prediction.fields.total
        outcome = (
            "absent" if total.value is None else "exact" if total.value == expected else "wrong"
        )
        groups[outcome].append(Receipt(
            entry.document_id, _image_path(record, manifest), expected, total.value,
            tuple(item.text for item in total.evidence if item.text),
            entry.image_sha256,
        ))

    rng = random.Random(STUDY_ID)
    pairs = []
    for outcome, count in (("exact", 5), ("wrong", 3), ("absent", 2)):
        choices = rng.sample(sorted(groups[outcome], key=lambda item: item.document_id), 2 * count)
        pairs.extend(zip(choices[:count], choices[count:], strict=True))
    _check_images(pairs)
    return pairs


def _check_images(pairs: list[tuple[Receipt, Receipt]]) -> None:
    for receipt in (receipt for pair in pairs for receipt in pair):
        with receipt.image.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != receipt.image_sha256:
            raise ValueError(f"Image checksum mismatch: {receipt.image}")


def _sample_filtered_rows(
    rows: list[dict], seed: str, shown_pairs: int, blank_pairs: int
) -> list[tuple[dict, dict]]:
    rng = random.Random(seed)
    pairs = []
    for has_proposal, count in ((True, shown_pairs), (False, blank_pairs)):
        pool = sorted(
            (row for row in rows if (row["proposed_total"] is not None) == has_proposal),
            key=lambda row: row["document_id"],
        )
        selected = rng.sample(pool, count * 2)
        pairs.extend(zip(selected[:count], selected[count:], strict=True))
    return pairs


def select_filtered_receipts(
    manifest: Path, report_path: Path, *, representative: bool = False
) -> list[tuple[Receipt, Receipt]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if report["policy"] != POLICY_ID or report["manifest_sha256"] != manifest_sha256:
        raise ValueError("Selective report does not match the study policy and manifest")
    records = {record.manifest.document_id: record for record in read_records(manifest)}
    rows = report["receipts"]
    if len({row["document_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate receipts in the selective report")
    row_pairs = _sample_filtered_rows(rows, SELECTIVE_STUDY_ID, 4, 6)
    if representative:
        seen_hashes = {row["image_sha256"] for pair in row_pairs for row in pair}
        unseen = [row for row in rows if row["image_sha256"] not in seen_hashes]
        coverage = sum(row["proposed_total"] is not None for row in rows) / len(rows)
        shown = round(REPRESENTATIVE_PAIR_COUNT * coverage)
        row_pairs = _sample_filtered_rows(
            unseen, REPRESENTATIVE_STUDY_ID, shown, REPRESENTATIVE_PAIR_COUNT - shown
        )

    def make_receipt(row: dict) -> Receipt:
        record = records[row["document_id"]]
        if row["image_sha256"] != record.manifest.image_sha256:
            raise ValueError(f"Report image hash mismatch: {row['document_id']}")
        expected = record.annotation.total
        if expected is None:
            raise ValueError(f"Missing reference total: {row['document_id']}")
        proposed = row["proposed_total"]
        return Receipt(
            row["document_id"], _image_path(record, manifest), expected,
            Decimal(proposed) if proposed is not None else None, tuple(row["evidence"]),
            row["image_sha256"],
        )

    pairs = [(make_receipt(first), make_receipt(second)) for first, second in row_pairs]
    _check_images(pairs)
    return pairs


def _parse_amount(text: str) -> Decimal:
    cleaned = "".join(text.split())
    if not re.fullmatch(r"-?\d{1,12}(?:[.,]\d{1,2})?", cleaned):
        raise HTTPException(422, "Saisis un montant comme 48,00, ou indique qu’il est illisible.")
    return Decimal(cleaned.replace(",", "."))


def _summarize(observations: list[dict]) -> dict:
    summary = {}
    for mode in ("manual", "assisted"):
        rows = [row for row in observations if row["mode"] == mode]
        times = [row["elapsed_ms"] for row in rows if not row["interrupted"]]
        summary[mode] = {
            "count": len(rows),
            "exact": sum(row["outcome"] == "exact" for row in rows),
            "wrong": sum(row["outcome"] == "wrong" for row in rows),
            "unreadable": sum(row["outcome"] == "unreadable" for row in rows),
            "timed": len(times),
            "median_seconds": round(median(times) / 1000, 2) if times else None,
            "total_seconds": round(sum(times) / 1000, 2),
        }
    return summary


def create_app(
    pairs: list[tuple[Receipt, Receipt]], output: Path, *, study_id: str = STUDY_ID
) -> FastAPI:
    if not pairs:
        raise ValueError("At least one pair of receipts is required")
    receipts = {receipt.document_id: receipt for pair in pairs for receipt in pair}
    app = FastAPI(title="Essai de saisie des reçus", docs_url=None, redoc_url=None)
    lock = threading.Lock()

    def read_session(session_id: UUID) -> dict:
        path = output / f"{session_id}.json"
        if not path.is_file():
            raise HTTPException(404, "Session introuvable.")
        session = json.loads(path.read_text(encoding="utf-8"))
        if session["study"] != study_id:
            raise HTTPException(404, "Session introuvable pour cet essai.")
        return session

    def save_session(session_id: UUID, session: dict) -> None:
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{session_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def state(session_id: UUID, session: dict) -> dict:
        index = len(session["answers"])
        size = len(session["trials"])
        result = {"session_id": str(session_id), "size": size}
        if index == size:
            return {**result, "summary": _summarize(session["answers"])}
        trial = session["trials"][index]
        receipt = receipts[trial["document_id"]]
        result.update(
            index=index, mode=trial["mode"], image_url=f"/sessions/{session_id}/image/{index}"
        )
        if trial["mode"] == "assisted":
            result["proposal"] = (
                format(receipt.proposal, "f") if receipt.proposal is not None else None
            )
            result["evidence"] = receipt.evidence
        return result

    @app.get("/", include_in_schema=False)
    def front_page() -> HTMLResponse:
        template = Path(__file__).with_name("receipt_pilot.html").read_text(encoding="utf-8")
        html = template.replace('id="receipt-count">20', f'id="receipt-count">{2 * len(pairs)}')
        return HTMLResponse(html)

    @app.post("/sessions")
    def start() -> dict:
        rng = random.SystemRandom()
        blocks = []
        for pair_index, (first, second) in enumerate(pairs):
            manual, assisted = (first, second) if rng.randrange(2) else (second, first)
            blocks.append([
                {"document_id": manual.document_id, "mode": "manual", "pair": pair_index},
                {"document_id": assisted.document_id, "mode": "assisted", "pair": pair_index},
            ])
        if study_id == REPRESENTATIVE_STUDY_ID:
            rng.shuffle(blocks)
            for block in blocks:
                rng.shuffle(block)
        trials = [trial for block in blocks for trial in block]
        if study_id != REPRESENTATIVE_STUDY_ID:
            rng.shuffle(trials)
        session_id = uuid4()
        session = {"study": study_id, "trials": trials, "answers": []}
        with lock:
            save_session(session_id, session)
        return state(session_id, session)

    @app.get("/sessions/{session_id}")
    def resume(session_id: UUID) -> dict:
        return state(session_id, read_session(session_id))

    @app.get("/sessions/{session_id}/image/{index}")
    def image(session_id: UUID, index: int) -> FileResponse:
        session = read_session(session_id)
        if index != len(session["answers"]) or not 0 <= index < len(session["trials"]):
            raise HTTPException(404)
        path = receipts[session["trials"][index]["document_id"]].image
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/sessions/{session_id}/answers")
    def answer(session_id: UUID, body: Answer) -> dict:
        with lock:
            session = read_session(session_id)
            position = len(session["answers"])
            if body.index < position:
                return state(session_id, session)
            if body.index != position or position == len(session["trials"]):
                raise HTTPException(409, "Ce reçu n’est pas actif.")
            trial = session["trials"][position]
            receipt = receipts[trial["document_id"]]
            amount = _parse_amount(body.amount) if body.amount is not None else None
            outcome = "unreadable"
            if amount is not None:
                outcome = "exact" if amount == receipt.reference else "wrong"
            session["answers"].append({
                **trial, "elapsed_ms": body.elapsed_ms, "interrupted": body.interrupted,
                "amount": str(amount) if amount is not None else None, "outcome": outcome,
                "shown_proposal": (
                    format(receipt.proposal, "f")
                    if trial["mode"] == "assisted" and receipt.proposal is not None else None
                ),
            })
            if len(session["answers"]) == len(session["trials"]):
                session["summary"] = _summarize(session["answers"])
            save_session(session_id, session)
        return state(session_id, session)

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=("original", "selective", "representative"), default="original"
    )
    parser.add_argument("--port", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path("artifacts/receipt-pilot")
    if args.study == "original":
        pairs = select_receipts(MANIFEST, BENCHMARKS)
        output = args.output or root / "sessions"
        app = create_app(pairs, output)
        port = args.port or 7871
    else:
        representative = args.study == "representative"
        pairs = select_filtered_receipts(
            SELECTIVE_MANIFEST, SELECTIVE_REPORT, representative=representative
        )
        output = args.output or root / args.study / "sessions"
        study = REPRESENTATIVE_STUDY_ID if representative else SELECTIVE_STUDY_ID
        app = create_app(pairs, output, study_id=study)
        port = args.port or (7873 if representative else 7872)
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
