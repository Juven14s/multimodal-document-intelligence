"""SQLite-backed human-review queue that preserves original predictions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.schemas import DocumentExtraction
from document_intelligence.validation import validate_extraction


class ReviewStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    CORRECTED = "corrected"


class ReviewCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    document_id: str
    status: ReviewStatus
    priority: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    source_path: Path | None = None
    assigned_to: str | None = None
    reasons: list[str]
    prediction: DocumentExtraction
    correction: DocumentExtraction | None = None


class ReviewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str
    actor: str | None
    created_at: datetime
    payload: dict = Field(default_factory=dict)


class ReviewedExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    document_id: str
    status: ReviewStatus
    reviewer: str | None
    reviewed_at: datetime
    extraction: DocumentExtraction


class ReviewQueue:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_cases (
                    case_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_path TEXT,
                    assigned_to TEXT,
                    reasons_json TEXT NOT NULL,
                    prediction_json TEXT NOT NULL,
                    correction_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_review_queue
                    ON review_cases(status, priority DESC, created_at ASC);
                CREATE TABLE IF NOT EXISTS review_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES review_cases(case_id)
                );
                """
            )

    @staticmethod
    def _case(row: sqlite3.Row) -> ReviewCase:
        return ReviewCase(
            case_id=row["case_id"],
            document_id=row["document_id"],
            status=row["status"],
            priority=row["priority"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            source_path=row["source_path"],
            assigned_to=row["assigned_to"],
            reasons=json.loads(row["reasons_json"]),
            prediction=DocumentExtraction.model_validate_json(row["prediction_json"]),
            correction=(
                DocumentExtraction.model_validate_json(row["correction_json"])
                if row["correction_json"]
                else None
            ),
        )

    def _event(
        self,
        connection: sqlite3.Connection,
        case_id: str,
        event_type: str,
        actor: str | None,
        payload: dict | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO review_events(case_id, event_type, actor, created_at, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                case_id,
                event_type,
                actor,
                datetime.now(UTC).isoformat(),
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )

    def enqueue(
        self,
        extraction: DocumentExtraction,
        *,
        source_path: Path | None = None,
        priority: int | None = None,
    ) -> ReviewCase:
        if not extraction.requires_human_review:
            raise ValueError("Only extractions requiring human review can be enqueued")
        reasons = [issue.code for issue in extraction.validation_issues if issue.blocking]
        priority = priority if priority is not None else 10 * len(reasons)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM review_cases WHERE document_id = ? "
                "AND status IN (?, ?) ORDER BY created_at LIMIT 1",
                (
                    extraction.document_id,
                    ReviewStatus.PENDING.value,
                    ReviewStatus.IN_PROGRESS.value,
                ),
            ).fetchone()
            if existing:
                return self._case(existing)
            case_id = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO review_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    extraction.document_id,
                    ReviewStatus.PENDING.value,
                    priority,
                    now,
                    now,
                    str(source_path) if source_path else None,
                    None,
                    json.dumps(reasons),
                    extraction.model_dump_json(),
                    None,
                ),
            )
            self._event(connection, case_id, "enqueued", None, {"reasons": reasons})
            row = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            return self._case(row)

    def claim_next(self, reviewer: str) -> ReviewCase | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM review_cases WHERE status = ? "
                "ORDER BY priority DESC, created_at ASC LIMIT 1",
                (ReviewStatus.PENDING.value,),
            ).fetchone()
            if row is None:
                return None
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "UPDATE review_cases SET status = ?, assigned_to = ?, updated_at = ? "
                "WHERE case_id = ?",
                (ReviewStatus.IN_PROGRESS.value, reviewer, now, row["case_id"]),
            )
            self._event(connection, row["case_id"], "claimed", reviewer)
            updated = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (row["case_id"],)
            ).fetchone()
            return self._case(updated)

    def resolve(
        self,
        case_id: str,
        reviewer: str,
        *,
        correction: DocumentExtraction | None = None,
    ) -> ReviewCase:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row is None:
                raise KeyError(case_id)
            if row["status"] != ReviewStatus.IN_PROGRESS.value:
                raise ValueError("Review case must be in progress before resolution")
            if row["assigned_to"] != reviewer:
                raise PermissionError("Review case is assigned to another reviewer")
            if correction is not None:
                if correction.document_id != row["document_id"]:
                    raise ValueError("Correction document ID differs from the review case")
                data_errors = {
                    "missing_required_field", "inconsistent_total", "inconsistent_line_sum",
                }
                if any(
                    issue.code in data_errors
                    for issue in validate_extraction(correction).validation_issues
                ):
                    raise ValueError("Correction contains missing or inconsistent business fields")
            status = ReviewStatus.CORRECTED if correction else ReviewStatus.APPROVED
            now = datetime.now(UTC).isoformat()
            correction_json = correction.model_dump_json() if correction else None
            connection.execute(
                "UPDATE review_cases SET status = ?, correction_json = ?, updated_at = ? "
                "WHERE case_id = ?",
                (status.value, correction_json, now, case_id),
            )
            self._event(connection, case_id, status.value, reviewer)
            updated = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            return self._case(updated)

    def reviewed_extraction(self, case_id: str) -> ReviewedExtraction:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            raise KeyError(case_id)
        case = self._case(row)
        if case.status not in {ReviewStatus.APPROVED, ReviewStatus.CORRECTED}:
            raise ValueError("The review must be completed before export")
        return ReviewedExtraction(
            case_id=case.case_id,
            document_id=case.document_id,
            status=case.status,
            reviewer=case.assigned_to,
            reviewed_at=case.updated_at,
            extraction=case.correction or case.prediction,
        )

    def history(self, case_id: str) -> list[ReviewEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_type, actor, created_at, payload_json FROM review_events "
                "WHERE case_id = ? ORDER BY event_id",
                (case_id,),
            ).fetchall()
        return [
            ReviewEvent(
                event_type=row["event_type"],
                actor=row["actor"],
                created_at=row["created_at"],
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        ]
