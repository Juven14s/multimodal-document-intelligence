"""Small dependency-free runtime settings object."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class APISettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_dir: Path = Path("artifacts")
    review_database: Path = Path("artifacts/review/review.sqlite3")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "document-intelligence"
    calibration_profile: Path | None = None
    ocr_invoice_rules: bool = False
    require_calibration: bool = False

    @classmethod
    def from_environment(cls) -> APISettings:
        artifacts = Path(os.getenv("DOCINTEL_ARTIFACTS_DIR", "artifacts"))
        return cls(
            artifacts_dir=artifacts,
            review_database=Path(
                os.getenv(
                    "DOCINTEL_REVIEW_DATABASE",
                    str(artifacts / "review" / "review.sqlite3"),
                )
            ),
            max_upload_bytes=int(
                os.getenv("DOCINTEL_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))
            ),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI") or None,
            mlflow_experiment=os.getenv(
                "DOCINTEL_MLFLOW_EXPERIMENT",
                "document-intelligence",
            ),
            calibration_profile=(
                Path(value)
                if (value := os.getenv("DOCINTEL_CALIBRATION_PROFILE"))
                else None
            ),
            ocr_invoice_rules=os.getenv("DOCINTEL_OCR_INVOICE_RULES", "false"),
            require_calibration=os.getenv("DOCINTEL_REQUIRE_CALIBRATION", "false"),
        )
