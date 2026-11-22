"""Gradio application for evidence review and human correction."""

from __future__ import annotations

import argparse
from pathlib import Path

from document_intelligence.schemas import DocumentType
from document_intelligence.ui.workbench import FIELD_NAMES, ReviewWorkbench
from document_intelligence.uncertainty.review_queue import ReviewQueue


def create_app(database_path: Path):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError('Install UI dependencies with: pip install -e ".[ui]"') from error

    workbench = ReviewWorkbench(ReviewQueue(database_path))

    def claim_case(reviewer: str):
        try:
            view = workbench.claim(reviewer)
        except Exception as error:  # noqa: BLE001 - surfaced as a UI status
            return [], "", "receipt", [], [], {}, f"❌ {error}"
        if view is None:
            return [], "", "receipt", [], [], {}, "✅ Aucun dossier en attente."
        state = {"case_id": view.case_id, "prediction_json": view.prediction_json}
        return (
            view.image_paths,
            view.metadata_markdown,
            view.document_type.value,
            view.field_rows,
            view.line_item_rows,
            state,
            f"Dossier `{view.case_id}` attribué à **{reviewer.strip()}**.",
        )

    def approve_case(reviewer: str, state: dict):
        if not state or not state.get("case_id"):
            return "❌ Aucun dossier actif.", state
        try:
            workbench.approve(state["case_id"], reviewer)
        except Exception as error:  # noqa: BLE001 - surfaced as a UI status
            return f"❌ {error}", state
        return "✅ Prédiction approuvée. Chargez le dossier suivant.", {}

    def correct_case(
        reviewer: str,
        state: dict,
        document_type: str,
        field_rows,
        line_item_rows,
    ):
        if not state or not state.get("case_id"):
            return "❌ Aucun dossier actif.", state
        try:
            workbench.correct(
                state["case_id"],
                reviewer,
                state["prediction_json"],
                document_type,
                field_rows,
                line_item_rows,
            )
        except Exception as error:  # noqa: BLE001 - surfaced as a UI status
            return f"❌ {error}", state
        return "✅ Correction enregistrée. Chargez le dossier suivant.", {}

    with gr.Blocks(title="Document Intelligence — Human Review", fill_width=True) as app:
        gr.Markdown(
            "# Validation humaine\n"
            "Inspectez les preuves, corrigez uniquement les valeurs nécessaires, puis validez."
        )
        state = gr.State({})
        with gr.Row():
            reviewer = gr.Textbox(label="Reviewer", placeholder="Votre nom", scale=2)
            next_button = gr.Button("Charger le prochain dossier", variant="primary", scale=1)
        status = gr.Markdown("En attente d’un reviewer.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gallery = gr.Gallery(
                    label="Pages du document",
                    object_fit="contain",
                    columns=1,
                    height="75vh",
                    preview=True,
                )
            with gr.Column(scale=5):
                metadata = gr.Markdown()
                document_type = gr.Dropdown(
                    choices=[item.value for item in DocumentType],
                    value=DocumentType.RECEIPT.value,
                    label="Type de document",
                )
                fields = gr.Dataframe(
                    headers=["field", "value", "confidence", "raw_text"],
                    datatype=["str", "str", "number", "str"],
                    type="array",
                    interactive=True,
                    static_columns=[0, 2, 3],
                    row_count=len(FIELD_NAMES),
                    label="Champs extraits — seule la colonne value est modifiable",
                    wrap=True,
                )
                line_items = gr.Dataframe(
                    headers=["description", "quantity", "unit_price", "amount"],
                    datatype=["str", "str", "str", "str"],
                    type="array",
                    interactive=True,
                    row_count=5,
                    label="Lignes d’achat",
                )
                with gr.Row():
                    approve_button = gr.Button("Approuver sans modification")
                    correct_button = gr.Button("Enregistrer la correction", variant="primary")

        next_button.click(
            claim_case,
            inputs=[reviewer],
            outputs=[
                gallery,
                metadata,
                document_type,
                fields,
                line_items,
                state,
                status,
            ],
            api_visibility="private",
        )
        approve_button.click(
            approve_case,
            inputs=[reviewer, state],
            outputs=[status, state],
            api_visibility="private",
        )
        correct_button.click(
            correct_case,
            inputs=[reviewer, state, document_type, fields, line_items],
            outputs=[status, state],
            api_visibility="private",
        )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("artifacts/review/review.sqlite3"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    create_app(args.database).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
