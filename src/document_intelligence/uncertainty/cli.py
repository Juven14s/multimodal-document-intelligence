"""Fit a validation-only confidence calibration profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_intelligence.data.io import read_records
from document_intelligence.data.models import DatasetSplit
from document_intelligence.evaluation.cli import _read_predictions
from document_intelligence.uncertainty.calibration import (
    CalibrationConfig,
    calibrate_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--minimum-samples", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions, invalid = _read_predictions(args.predictions)
    if invalid:
        raise SystemExit(f"Calibration aborted: {invalid} invalid prediction payload(s)")
    validation_records = [
        record
        for record in read_records(args.ground_truth)
        if record.manifest.benchmark_split is DatasetSplit.VALIDATION
    ]
    profile = calibrate_predictions(
        validation_records,
        predictions,
        CalibrationConfig(
            target_auto_accept_accuracy=args.target_accuracy,
            minimum_samples_per_field=args.minimum_samples,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
