"""Lazy Transformers backend compatible with Qwen image-text models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from document_intelligence.vlm.models import FewShotTurn, GenerationResult


class QwenBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    device_map: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = Field(default=1536, ge=64, le=8192)
    min_pixels: int = Field(default=256 * 28 * 28, ge=28 * 28)
    max_pixels: int = Field(default=1280 * 28 * 28, ge=28 * 28)


def build_chat_messages(
    image_paths: list[Path],
    system_prompt: str,
    few_shot_turns: list[FewShotTurn] | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]
    for turn in few_shot_turns or []:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        *[
                            {"type": "image", "image": str(path.resolve())}
                            for path in turn.image_paths
                        ],
                        {"type": "text", "text": "Extract this example document."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn.assistant_response}],
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": [
                *[
                    {"type": "image", "image": str(path.resolve())}
                    for path in image_paths
                ],
                {
                    "type": "text",
                    "text": (
                        "Extract this target document. Follow the system instructions and "
                        "return only the requested JSON object."
                    ),
                },
            ],
        }
    )
    return messages


class TransformersQwenBackend:
    def __init__(self, config: QwenBackendConfig | None = None) -> None:
        self.config = config or QwenBackendConfig()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._torch, self._model, self._processor
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError('Install VLM dependencies with: pip install -e ".[vlm]"') from error

        processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            dtype=self.config.dtype,
            device_map=self.config.device_map,
        )
        self._torch = torch
        self._model = model
        self._processor = processor
        return torch, model, processor

    def generate(
        self,
        image_paths: list[Path],
        prompt: str,
        few_shot_turns: list[FewShotTurn] | None = None,
    ) -> GenerationResult:
        if not image_paths:
            raise ValueError("At least one document image is required")
        example_paths = [
            path for turn in few_shot_turns or [] for path in turn.image_paths
        ]
        for path in [*image_paths, *example_paths]:
            if not path.is_file():
                raise FileNotFoundError(path)
        torch, model, processor = self._load()
        messages = build_chat_messages(image_paths, prompt, few_shot_turns)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        latency_ms = round((time.perf_counter() - started) * 1000)
        input_length = inputs["input_ids"].shape[1]
        trimmed = [sequence[input_length:] for sequence in generated]
        raw_response = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        peak_memory = None
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return GenerationResult(
            model_id=self.config.model_id,
            raw_response=raw_response,
            latency_ms=latency_ms,
            device=str(model.device),
            peak_memory_mb=peak_memory,
        )
