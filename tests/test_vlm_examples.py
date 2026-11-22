from pathlib import Path

from document_intelligence.data.models import (
    CanonicalAnnotation,
    CanonicalField,
    CanonicalRecord,
    DatasetName,
    DatasetSplit,
    ManifestEntry,
)
from document_intelligence.vlm.examples import (
    FewShotExample,
    SelectionContext,
    example_from_record,
    read_examples,
    select_examples,
    write_examples,
)
from document_intelligence.vlm.models import VLMField, VLMFields, VLMResponse


def make_record(document_id: str = "train-1") -> CanonicalRecord:
    fields = {CanonicalField.MERCHANT_NAME, CanonicalField.TOTAL}
    return CanonicalRecord(
        manifest=ManifestEntry(
            document_id=document_id,
            dataset=DatasetName.SROIE,
            source_split="train",
            benchmark_split=DatasetSplit.TRAIN,
            source_key=document_id,
            image_path=Path(f"images/{document_id}.png"),
            image_width=100,
            image_height=200,
            image_sha256="a" * 64,
            ocr_region_count=0,
            annotated_fields=fields,
        ),
        annotation=CanonicalAnnotation(
            merchant_name="Shop",
            total="12.30",
            annotated_fields=fields,
        ),
    )


def make_example(document_id: str, split: DatasetSplit, sha: str) -> FewShotExample:
    return FewShotExample(
        document_id=document_id,
        dataset=DatasetName.SROIE,
        split=split,
        image_paths=[Path(f"{document_id}.png")],
        image_sha256=sha,
        image_width=100,
        image_height=200,
        annotated_fields={CanonicalField.TOTAL},
        response=VLMResponse(
            document_type="receipt",
            classification_confidence=1.0,
            fields=VLMFields(
                merchant_name=VLMField(value="Shop", confidence=1.0),
                total=VLMField(value="12.30", confidence=1.0),
            ),
        ),
    )


def test_example_is_built_only_from_training_record() -> None:
    example = example_from_record(make_record(), Path("data/processed"))

    assert example.split is DatasetSplit.TRAIN
    assert example.response.fields.merchant_name.value == "Shop"
    assert str(example.response.fields.total.value) == "12.30"
    assert example.image_paths[0].is_absolute()


def test_selection_excludes_target_duplicate_hash_and_non_train() -> None:
    examples = [
        make_example("target", DatasetSplit.TRAIN, "1" * 64),
        make_example("duplicate", DatasetSplit.TRAIN, "a" * 64),
        make_example("test-leak", DatasetSplit.TEST, "2" * 64),
        make_example("eligible-a", DatasetSplit.TRAIN, "3" * 64),
        make_example("eligible-b", DatasetSplit.TRAIN, "4" * 64),
    ]
    context = SelectionContext(
        document_id="target",
        dataset=DatasetName.SROIE,
        image_sha256="a" * 64,
        image_width=100,
        image_height=200,
        desired_fields={CanonicalField.TOTAL},
    )

    first = select_examples(examples, context, count=2, seed=7)
    second = select_examples(examples, context, count=2, seed=7)

    assert [example.document_id for example in first] == [
        example.document_id for example in second
    ]
    assert {example.document_id for example in first} == {"eligible-a", "eligible-b"}


def test_catalog_serializes_portable_paths_and_resolves_them_on_read(tmp_path: Path) -> None:
    example = example_from_record(make_record(), tmp_path / "dataset")
    catalog = tmp_path / "catalog" / "fewshot.jsonl"

    write_examples(catalog, [example])
    raw = catalog.read_text(encoding="utf-8")
    loaded = read_examples(catalog)[0]

    assert str(example.image_paths[0]) not in raw
    assert loaded.image_paths[0] == example.image_paths[0]
    assert loaded.image_paths[0].is_absolute()
