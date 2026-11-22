import pytest

from document_intelligence.evaluation.qwen_experiment import (
    _strategy_specs,
    validate_accelerator,
)
from document_intelligence.schemas import ExtractionMethod
from document_intelligence.vlm.prompting import PromptStyle


def test_gpu_experiment_refuses_accidental_cpu_execution() -> None:
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        validate_accelerator(cuda_available=False, allow_cpu=False)

    validate_accelerator(cuda_available=False, allow_cpu=True)
    validate_accelerator(cuda_available=True, allow_cpu=False)


def test_gpu_experiment_compares_expected_strategies() -> None:
    assert _strategy_specs() == [
        ("zero-shot-schema", ExtractionMethod.VLM_ZERO_SHOT, PromptStyle.JSON_SCHEMA),
        ("zero-shot-compact", ExtractionMethod.VLM_ZERO_SHOT, PromptStyle.COMPACT),
        ("few-shot-compact", ExtractionMethod.VLM_FEW_SHOT, PromptStyle.COMPACT),
    ]
