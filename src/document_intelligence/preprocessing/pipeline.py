"""Document loading, rendering, normalization, and quality reporting."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from document_intelligence.preprocessing.deskew import apply_deskew, estimate_deskew_angle
from document_intelligence.preprocessing.models import (
    MediaType,
    PreprocessingConfig,
    PreprocessingResult,
    ProcessedPage,
    QualityIssue,
    Transformation,
)
from document_intelligence.preprocessing.quality import assess_quality

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pdf(path: Path, config: PreprocessingConfig) -> Iterator[Image.Image]:
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError('PDF support requires: pip install -e ".[ocr]"') from error

    with pymupdf.open(path) as document:
        if document.page_count == 0:
            raise ValueError("PDF contains no pages")
        if document.page_count > config.max_pages:
            raise ValueError(
                f"PDF has {document.page_count} pages; configured maximum is {config.max_pages}"
            )
        for page in document:
            pixmap = page.get_pixmap(
                dpi=config.pdf_dpi,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image.info["_docintel_pdf_rotation"] = int(page.rotation)
            yield image


def _load_image(path: Path, config: PreprocessingConfig) -> Iterator[Image.Image]:
    try:
        with Image.open(path) as source:
            frames = min(getattr(source, "n_frames", 1), config.max_pages + 1)
            if frames > config.max_pages:
                raise ValueError(
                    f"Image has {frames} frames; configured maximum is {config.max_pages}"
                )
            for frame in ImageSequence.Iterator(source):
                copied = frame.copy()
                orientation = int(copied.getexif().get(274, 1))
                original_size = copied.size
                oriented = ImageOps.exif_transpose(copied)
                oriented.info["_docintel_exif_orientation"] = orientation
                oriented.info["_docintel_original_size"] = original_size
                yield oriented
    except UnidentifiedImageError as error:
        raise ValueError(f"Unsupported or corrupt image: {path}") from error


def _resize(image: Image.Image, max_long_edge: int) -> tuple[Image.Image, float | None]:
    long_edge = max(image.size)
    if long_edge <= max_long_edge:
        return image, None
    scale = max_long_edge / long_edge
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


def _process_page(
    page: Image.Image,
    page_number: int,
    output_path: Path,
    config: PreprocessingConfig,
) -> ProcessedPage:
    original_width, original_height = page.info.get("_docintel_original_size", page.size)
    transformations: list[Transformation] = []
    exif_orientation = int(page.info.get("_docintel_exif_orientation", 1))
    if exif_orientation != 1:
        transformations.append(
            Transformation(
                name="exif_transpose",
                parameters={"orientation": exif_orientation},
            )
        )
    pdf_rotation = int(page.info.get("_docintel_pdf_rotation", 0))
    if pdf_rotation:
        transformations.append(
            Transformation(
                name="pdf_page_rotation",
                parameters={"degrees": pdf_rotation},
            )
        )
    image = page.convert("RGB")
    if page.mode != "RGB":
        transformations.append(
            Transformation(name="convert_color", parameters={"from": page.mode, "to": "RGB"})
        )

    uncertain_deskew = False
    if config.auto_deskew:
        angle = estimate_deskew_angle(
            image,
            max_degrees=config.max_skew_degrees,
            step_degrees=config.deskew_step_degrees,
        )
        if angle is not None and abs(angle) >= config.min_deskew_degrees:
            at_search_boundary = abs(angle) >= (
                config.max_skew_degrees - config.deskew_step_degrees / 2
            )
            if at_search_boundary:
                uncertain_deskew = True
            else:
                image = apply_deskew(image, angle)
                transformations.append(
                    Transformation(name="deskew", parameters={"angle_degrees": angle})
                )

    image, resize_scale = _resize(image, config.max_long_edge)
    if resize_scale is not None:
        transformations.append(
            Transformation(name="resize", parameters={"scale": round(resize_scale, 6)})
        )

    quality = assess_quality(image, config)
    if uncertain_deskew and QualityIssue.DESKEW_UNCERTAIN not in quality.issues:
        quality.issues.append(QualityIssue.DESKEW_UNCERTAIN)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return ProcessedPage(
        page_number=page_number,
        output_path=output_path,
        output_sha256=_sha256(output_path),
        original_width=original_width,
        original_height=original_height,
        output_width=image.width,
        output_height=image.height,
        transformations=transformations,
        quality=quality,
    )


def preprocess_document(
    source_path: Path,
    output_dir: Path,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    config = config or PreprocessingConfig()
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_size = source_path.stat().st_size
    if source_size > config.max_input_bytes:
        raise ValueError(
            f"Input is {source_size} bytes; configured maximum is {config.max_input_bytes}"
        )

    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        media_type = MediaType.PDF
        pages = _load_pdf(source_path, config)
    elif suffix in SUPPORTED_IMAGE_SUFFIXES:
        media_type = MediaType.IMAGE
        pages = _load_image(source_path, config)
    else:
        raise ValueError(f"Unsupported document extension: {suffix or '<none>'}")

    source_hash = _sha256(source_path)
    document_dir = output_dir / f"{source_path.stem}-{source_hash[:12]}"
    processed = [
        _process_page(
            page,
            page_number,
            document_dir / f"page-{page_number:04d}.png",
            config,
        )
        for page_number, page in enumerate(pages, start=1)
    ]
    if not processed:
        raise ValueError("Document contains no processable pages")
    return PreprocessingResult(
        source_path=source_path,
        source_sha256=source_hash,
        source_media_type=media_type,
        source_size_bytes=source_size,
        pages=processed,
    )
