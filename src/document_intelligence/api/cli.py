"""Production-oriented local API launcher."""

from __future__ import annotations

import argparse
from pathlib import Path

from document_intelligence.api.app import create_app
from document_intelligence.api.settings import APISettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--review-database", type=Path)
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--calibration-profile", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError('API dependencies require: pip install -e ".[api]"') from error
    environment = APISettings.from_environment()
    artifacts = args.artifacts or environment.artifacts_dir
    if args.review_database:
        review_database = args.review_database
    elif args.artifacts:
        review_database = artifacts / "review" / "review.sqlite3"
    else:
        review_database = environment.review_database
    settings = environment.model_copy(update={
        "artifacts_dir": artifacts,
        "review_database": review_database,
        "mlflow_tracking_uri": args.mlflow_tracking_uri or environment.mlflow_tracking_uri,
        "calibration_profile": args.calibration_profile or environment.calibration_profile,
    })
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
