"""Small-angle deskew based on horizontal projection profiles."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def estimate_deskew_angle(
    image: Image.Image,
    *,
    max_degrees: float = 7.5,
    step_degrees: float = 0.5,
    analysis_long_edge: int = 1000,
) -> float | None:
    """Return the correction angle, or ``None`` when there is too little ink."""

    gray = ImageOps.grayscale(image)
    if max(gray.size) > analysis_long_edge:
        scale = analysis_long_edge / max(gray.size)
        gray = gray.resize(
            (max(1, round(gray.width * scale)), max(1, round(gray.height * scale))),
            Image.Resampling.BILINEAR,
        )
    array = np.asarray(gray, dtype=np.uint8)
    threshold = min(220, int(np.percentile(array, 55)))
    mask = array < threshold
    ink_ratio = float(mask.mean())
    if ink_ratio < 0.002 or ink_ratio > 0.60:
        return None

    binary = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    angles = np.arange(-max_degrees, max_degrees + step_degrees / 2, step_degrees)
    best_angle = 0.0
    best_score = float("-inf")
    for angle in angles:
        rotated = binary.rotate(
            float(angle),
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=0,
        )
        projection = np.asarray(rotated, dtype=np.float32).sum(axis=1)
        score = float(np.square(np.diff(projection)).sum())
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return round(best_angle, 3)


def apply_deskew(image: Image.Image, angle: float) -> Image.Image:
    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor="white",
    )
