import json
from pathlib import Path

from PIL import Image

from document_intelligence.ocr.engine import TesseractEngine, words_from_tesseract_data
from document_intelligence.ocr.layout import reconstruct_lines


def test_tesseract_tsv_is_filtered_normalized_and_reconstructed() -> None:
    data = {
        "text": ["TOTAL", "12.30", "noise", ""],
        "conf": ["95.0", "90.0", "5.0", "-1"],
        "left": [10, 70, 0, 0],
        "top": [80, 80, 0, 0],
        "width": [50, 25, 1, 0],
        "height": [10, 10, 1, 0],
        "block_num": [1, 1, 2, 0],
        "par_num": [1, 1, 1, 0],
        "line_num": [1, 1, 1, 0],
        "word_num": [1, 2, 1, 0],
    }

    words = words_from_tesseract_data(
        data,
        page_number=1,
        width=100,
        height=100,
        minimum_confidence=0.20,
    )
    lines = reconstruct_lines(words)

    assert [word.text for word in words] == ["TOTAL", "12.30"]
    assert words[0].box.x0 == 0.1
    assert lines[0].text == "TOTAL 12.30"
    assert lines[0].confidence == 0.925
    assert lines[0].box.x1 == 0.95


def test_engine_builds_document_from_tsv_runtime(tmp_path: Path, monkeypatch) -> None:
    page = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(page)
    data = {
        "text": ["TOTAL", "12.30"],
        "conf": ["95", "90"],
        "left": [10, 70],
        "top": [80, 80],
        "width": [50, 25],
        "height": [10, 10],
        "block_num": [1, 1],
        "par_num": [1, 1],
        "line_num": [1, 1],
        "word_num": [1, 2],
    }

    class FakeOutput:
        DICT = "dict"

    class FakeRuntime:
        Output = FakeOutput

        @staticmethod
        def image_to_data(image, **kwargs):
            assert image.size == (100, 100)
            assert kwargs["output_type"] == "dict"
            return data

    engine = TesseractEngine()
    monkeypatch.setattr(engine, "_runtime", lambda: (FakeRuntime, "5-test"))

    result = engine.extract([page], raw_output_dir=tmp_path / "raw")

    assert result.engine_version == "5-test"
    assert result.pages[0].lines[0].text == "TOTAL 12.30"
    raw = (tmp_path / "raw" / "tesseract-0001.json").read_text(encoding="utf-8")
    assert json.loads(raw) == data
