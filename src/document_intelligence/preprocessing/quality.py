"""Lightweight image-quality signals used for routing and diagnostics."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from document_intelligence.preprocessing.models import (
    PageQuality,
    PreprocessingConfig,
    QualityIssue,
)


def assess_quality(image: Image.Image, config: PreprocessingConfig) -> PageQuality:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0
    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))
    horizontal = float(np.mean(np.abs(np.diff(gray, axis=1)))) if gray.shape[1] > 1 else 0.0
    vertical = float(np.mean(np.abs(np.diff(gray, axis=0)))) if gray.shape[0] > 1 else 0.0
    sharpness = min(1.0, (horizontal + vertical) / 2.0)

    issues: list[QualityIssue] = []
    if min(image.size) < config.min_short_edge_for_quality:
        issues.append(QualityIssue.LOW_RESOLUTION)
    if contrast < config.min_contrast:
        issues.append(QualityIssue.LOW_CONTRAST)
    if sharpness < config.min_sharpness:
        issues.append(QualityIssue.POSSIBLY_BLURRY)
    return PageQuality(
        contrast=round(contrast, 6),
        sharpness=round(sharpness, 6),
        mean_brightness=round(brightness, 6),
        issues=issues,
    )
