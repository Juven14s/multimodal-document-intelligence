import json
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from document_intelligence.ui.receipt_pilot import REPRESENTATIVE_STUDY_ID, Receipt, create_app


def test_pilot_hides_references_and_compares_both_modes(tmp_path: Path) -> None:
    image = tmp_path / "receipt.png"
    Image.new("RGB", (50, 120), "white").save(image)
    pairs = [(
        Receipt("doc-1", image, Decimal("48.00"), Decimal("50.00"), ("TOTAL 50.00",)),
        Receipt("doc-2", image, Decimal("48.00"), Decimal("50.00"), ("TOTAL 50.00",)),
    )]
    output = tmp_path / "sessions"
    with TestClient(create_app(pairs, output)) as client:
        front_page = client.get("/")
        assert front_page.status_code == 200
        assert "Commencer l’essai" in front_page.text
        state = client.post("/sessions").json()
        session_url = f"/sessions/{state['session_id']}"
        assert client.get(f"{session_url}/image/1").status_code == 404
        for index in range(2):
            assert state["index"] == index
            assert len(client.get(state["image_url"]).content) > 0
            assert "reference" not in state and "summary" not in state
            assisted = state["mode"] == "assisted"
            if assisted:
                assert state["proposal"] == "50.00"
                assert state["evidence"] == ["TOTAL 50.00"]
            else:
                assert "proposal" not in state and "evidence" not in state
            payload = {"index": index, "amount": "48,00", "elapsed_ms": 1250}
            answers_url = f"{session_url}/answers"
            invalid = client.post(answers_url, json={**payload, "amount": "?"})
            assert invalid.status_code == 422
            response = client.post(answers_url, json={**payload, "interrupted": assisted})
            assert response.status_code == 200
            state = response.json()
            assert client.post(answers_url, json=payload).json() == state

        summary = state["summary"]
        assert summary["manual"]["exact"] == summary["assisted"]["exact"] == 1
        assert summary["manual"]["median_seconds"] == 1.25
        assert summary["assisted"]["median_seconds"] is None
        saved = json.loads(next(output.glob("*.json")).read_text(encoding="utf-8"))
        assert len(saved["answers"]) == 2
        assert saved["summary"] == summary


def test_representative_pilot_balances_modes_throughout_the_session(tmp_path: Path) -> None:
    pairs = [
        (
            Receipt(f"doc-{index}-a", tmp_path / "a.png", Decimal("1.00"), None),
            Receipt(f"doc-{index}-b", tmp_path / "b.png", Decimal("1.00"), None),
        )
        for index in range(3)
    ]
    output = tmp_path / "sessions"
    with TestClient(create_app(pairs, output, study_id=REPRESENTATIVE_STUDY_ID)) as client:
        assert 'id="receipt-count">6</span>' in client.get("/").text
        assert client.post("/sessions").json()["size"] == 6

    saved = json.loads(next(output.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["study"] == REPRESENTATIVE_STUDY_ID
    assert len({trial["document_id"] for trial in saved["trials"]}) == 6
    for first, second in zip(saved["trials"][::2], saved["trials"][1::2], strict=True):
        assert first["pair"] == second["pair"]
        assert {first["mode"], second["mode"]} == {"manual", "assisted"}
