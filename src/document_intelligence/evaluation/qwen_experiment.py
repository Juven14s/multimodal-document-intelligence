"""Run the reproducible three-strategy Qwen VLM experiment on a GPU host."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from document_intelligence.evaluation.benchmark import (
    BenchmarkExecutionReport,
    run_benchmark,
)
from document_intelligence.observability import InferenceTracer, JsonlTraceSink
from document_intelligence.schemas import ExtractionMethod
from document_intelligence.service import DocumentIntelligenceService, ServiceConfig
from document_intelligence.vlm.backend import QwenBackendConfig, TransformersQwenBackend
from document_intelligence.vlm.examples import read_examples
from document_intelligence.vlm.prompting import PromptStyle


class HardwareSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_version: str
    torch_version: str
    transformers_version: str
    cuda_available: bool
    cuda_version: str | None
    device_count: int
    device_names: list[str]
    device_memory_mb: list[int]


class StrategyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    method: ExtractionMethod
    prompt_style: PromptStyle
    report_path: Path
    report: BenchmarkExecutionReport


class QwenExperimentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    started_at: datetime
    finished_at: datetime
    model_id: str
    manifest: Path
    examples: Path
    limit: int
    hardware: HardwareSnapshot
    strategies: list[StrategyResult]


def validate_accelerator(*, cuda_available: bool, allow_cpu: bool) -> None:
    if not cuda_available and not allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. Run this experiment on a GPU host or pass --allow-cpu "
            "only for a deliberately slow smoke test."
        )


def hardware_snapshot() -> HardwareSnapshot:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('Install VLM dependencies with: pip install -e ".[vlm]"') from error

    device_count = torch.cuda.device_count()
    return HardwareSnapshot(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=importlib.metadata.version("transformers"),
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        device_count=device_count,
        device_names=[torch.cuda.get_device_name(index) for index in range(device_count)],
        device_memory_mb=[
            round(torch.cuda.get_device_properties(index).total_memory / (1024 * 1024))
            for index in range(device_count)
        ],
    )


def _strategy_specs() -> list[tuple[str, ExtractionMethod, PromptStyle]]:
    return [
        ("zero-shot-schema", ExtractionMethod.VLM_ZERO_SHOT, PromptStyle.JSON_SCHEMA),
        ("zero-shot-compact", ExtractionMethod.VLM_ZERO_SHOT, PromptStyle.COMPACT),
        ("few-shot-compact", ExtractionMethod.VLM_FEW_SHOT, PromptStyle.COMPACT),
    ]


def run_experiment(
    manifest: Path,
    *,
    examples: Path,
    output_dir: Path,
    model_id: str,
    limit: int,
    few_shot_count: int,
    max_new_tokens: int,
    overwrite: bool,
    allow_cpu: bool,
) -> QwenExperimentReport:
    hardware = hardware_snapshot()
    validate_accelerator(cuda_available=hardware.cuda_available, allow_cpu=allow_cpu)
    catalog = read_examples(examples)
    backend = TransformersQwenBackend(
        QwenBackendConfig(model_id=model_id, max_new_tokens=max_new_tokens)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    results: list[StrategyResult] = []

    for name, method, prompt_style in _strategy_specs():
        strategy_dir = output_dir / name
        service = DocumentIntelligenceService(
            config=ServiceConfig(
                artifacts_dir=strategy_dir / "inference",
                vlm_prompt_style=prompt_style,
            ),
            tracer=InferenceTracer(
                [JsonlTraceSink(strategy_dir / "traces" / "inference.jsonl")]
            ),
            vlm_backend=backend,
            few_shot_examples=catalog if method is ExtractionMethod.VLM_FEW_SHOT else None,
        )
        report_path = strategy_dir / "report.json"
        report = run_benchmark(
            manifest,
            predictions_path=strategy_dir / "predictions.jsonl",
            report_path=report_path,
            failures_path=strategy_dir / "failures.jsonl",
            artifacts_dir=strategy_dir,
            method=method,
            limit=limit,
            resume=not overwrite,
            service=service,
            few_shot_catalog=catalog,
            few_shot_count=few_shot_count,
        )
        results.append(
            StrategyResult(
                name=name,
                method=method,
                prompt_style=prompt_style,
                report_path=report_path,
                report=report,
            )
        )

    experiment = QwenExperimentReport(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        model_id=model_id,
        manifest=manifest,
        examples=examples,
        limit=limit,
        hardware=hardware,
        strategies=results,
    )
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen-experiment"))
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--few-shot-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_experiment(
            args.manifest,
            examples=args.examples,
            output_dir=args.output_dir,
            model_id=args.model_id,
            limit=args.limit,
            few_shot_count=args.few_shot_count,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
            allow_cpu=args.allow_cpu,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    print(
        f"Completed {len(report.strategies)} Qwen strategies; "
        f"report={args.output_dir / 'experiment.json'}"
    )


if __name__ == "__main__":
    main()
