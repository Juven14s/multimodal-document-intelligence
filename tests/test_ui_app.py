from pathlib import Path

from document_intelligence.ui.app import create_app


def test_gradio_application_instantiates(tmp_path: Path) -> None:
    app = create_app(tmp_path / "review.sqlite3")

    assert app.title == "Document Intelligence — Human Review"
    assert len(app.blocks) >= 10
