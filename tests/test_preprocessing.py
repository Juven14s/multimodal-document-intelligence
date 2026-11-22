from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from document_intelligence.preprocessing.deskew import estimate_deskew_angle
from document_intelligence.preprocessing.models import (
    MediaType,
    PreprocessingConfig,
    QualityIssue,
)
from document_intelligence.preprocessing.pipeline import preprocess_document


def test_preprocess_image_resizes_and_writes_reportable_page(tmp_path: Path) -> None:
    source = tmp_path / "receipt.png"
    Image.new("RGB", (1000, 3000), "white").save(source)
    config = PreprocessingConfig(
        auto_deskew=False,
        max_long_edge=1200,
        min_short_edge_for_quality=500,
    )

    result = preprocess_document(source, tmp_path / "output", config)

    assert result.source_media_type is MediaType.IMAGE
    assert result.pages[0].output_width == 400
    assert result.pages[0].output_height == 1200
    assert result.pages[0].output_path.is_file()
    assert len(result.pages[0].output_sha256) == 64
    assert [item.name for item in result.pages[0].transformations] == ["resize"]
    assert QualityIssue.LOW_RESOLUTION in result.pages[0].quality.issues


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    source = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (20, 40), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)

    result = preprocess_document(
        source,
        tmp_path / "output",
        PreprocessingConfig(auto_deskew=False),
    )

    assert (result.pages[0].output_width, result.pages[0].output_height) == (40, 20)
    assert result.pages[0].original_width == 20
    assert result.pages[0].original_height == 40
    assert result.pages[0].transformations[0].model_dump() == {
        "name": "exif_transpose",
        "parameters": {"orientation": 6},
    }


def test_projection_profile_estimates_small_skew() -> None:
    image = Image.new("L", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    for y in range(100, 700, 40):
        draw.line((80, y, 520, y), fill="black", width=4)
    skewed = image.rotate(4.0, expand=True, fillcolor="white")

    correction = estimate_deskew_angle(skewed, max_degrees=7, step_degrees=0.5)

    assert correction is not None
    assert correction == pytest.approx(-4.0, abs=0.75)


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "receipt.txt"
    source.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document extension"):
        preprocess_document(source, tmp_path / "output")


def test_config_rejects_incoherent_deskew_range() -> None:
    with pytest.raises(ValidationError):
        PreprocessingConfig(min_deskew_degrees=5, max_skew_degrees=5)


def test_pdf_is_rendered_and_page_rotation_is_traced(tmp_path: Path) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    source = tmp_path / "rotated.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=144, height=72)
        page.insert_text((20, 40), "Receipt total 12.00")
        page.set_rotation(90)
        document.save(source)

    result = preprocess_document(
        source,
        tmp_path / "output",
        PreprocessingConfig(pdf_dpi=144, auto_deskew=False),
    )

    assert result.source_media_type is MediaType.PDF
    assert len(result.pages) == 1
    assert result.pages[0].output_path.is_file()
    assert result.pages[0].output_width == 144
    assert result.pages[0].output_height == 288
    assert result.pages[0].transformations[0].model_dump() == {
        "name": "pdf_page_rotation",
        "parameters": {"degrees": 90},
    }
