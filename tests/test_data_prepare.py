import json
from pathlib import Path

from PIL import Image

from document_intelligence.data import prepare
from document_intelligence.data.io import read_records
from document_intelligence.data.models import DatasetName


def test_prepare_dataset_materializes_images_manifest_and_audit(
    tmp_path: Path, monkeypatch
) -> None:
    row = {
        "key": "X001",
        "image": Image.new("RGB", (20, 30), "white"),
        "image_size": {"width": 20, "height": 30},
        "entities": {"company": "Shop", "date": "01/01/2019", "total": "2.50"},
        "words": [],
        "bboxes": [],
    }
    monkeypatch.setitem(prepare.SOURCE_SPLITS, DatasetName.SROIE, ("train",))
    monkeypatch.setattr(prepare, "_load_stream", lambda dataset, split: [row])

    records = prepare.prepare_dataset(DatasetName.SROIE, tmp_path)

    assert len(records) == 1
    manifest_path = tmp_path / "sroie.jsonl"
    saved = read_records(manifest_path)[0]
    assert saved.manifest.image_sha256 is not None
    image_path = tmp_path / saved.manifest.image_path
    assert image_path.is_file()
    original_bytes = image_path.read_bytes()
    audit = json.loads((tmp_path / "sroie_audit.json").read_text(encoding="utf-8"))
    assert audit["record_count"] == 1

    row["image"] = Image.new("RGB", (20, 30), "black")
    prepare.prepare_dataset(DatasetName.SROIE, tmp_path)

    assert image_path.read_bytes() == original_bytes
