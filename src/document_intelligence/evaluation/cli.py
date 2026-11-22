"""Evaluate JSONL predictions against a canonical dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from document_intelligence.data.io import read_records
from document_intelligence.data.models import DatasetSplit
from document_intelligence.evaluation.metrics import evaluate_predictions
from document_intelligence.schemas import DocumentExtraction
from document_intelligence.uncertainty.calibration import (
    CalibrationProfile,
    apply_calibration,
)


def _read_predictions(path: Path) -> tuple[list[DocumentExtraction], int]:
    predictions: list[DocumentExtraction] = []
    invalid = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                predictions.append(DocumentExtraction.model_validate_json(line))
            except (ValidationError, ValueError):
                invalid += 1
    return predictions, invalid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=[split.value for split in DatasetSplit])
    parser.add_argument("--calibration-profile", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = read_records(args.ground_truth)
    if args.split:
        split = DatasetSplit(args.split)
        records = [
            record for record in records if record.manifest.benchmark_split is split
        ]
    predictions, invalid = _read_predictions(args.predictions)
    if args.calibration_profile:
        profile = CalibrationProfile.model_validate_json(
            args.calibration_profile.read_text(encoding="utf-8")
        )
        predictions = [apply_calibration(prediction, profile) for prediction in predictions]
    result = evaluate_predictions(records, predictions, invalid_prediction_count=invalid)
    payload = json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
