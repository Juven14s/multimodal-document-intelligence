"""Pure adapters for Hugging Face CORD v2 and SROIE records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from document_intelligence.data.models import (
    CanonicalAnnotation,
    CanonicalField,
    CanonicalLineItem,
    CanonicalRecord,
    DatasetName,
    ManifestEntry,
    OCRRegion,
)
from document_intelligence.data.normalization import (
    normalize_text,
    parse_date,
    parse_decimal,
    parse_quantity,
)
from document_intelligence.data.splitting import assign_benchmark_split
from document_intelligence.schemas import BoundingBox


def _as_list(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        return value
    raise ValueError("Expected a mapping or a list of mappings")


def _manifest(
    *,
    dataset: DatasetName,
    document_id: str,
    source_key: str,
    source_split: str,
    width: int,
    height: int,
    annotated_fields: set[CanonicalField],
    ocr_region_count: int,
    image_path: Path | None,
) -> ManifestEntry:
    return ManifestEntry(
        document_id=document_id,
        dataset=dataset,
        source_split=source_split,
        benchmark_split=assign_benchmark_split(dataset, source_split, document_id),
        source_key=source_key,
        image_path=image_path,
        image_width=width,
        image_height=height,
        ocr_region_count=ocr_region_count,
        annotated_fields=annotated_fields,
    )


def _cord_ocr_regions(
    ground_truth: Mapping[str, Any], width: int, height: int
) -> list[OCRRegion]:
    regions: list[OCRRegion] = []
    for line in ground_truth.get("valid_line") or []:
        for word in line.get("words") or []:
            text = normalize_text(word.get("text"))
            quad = word.get("quad") or {}
            xs = [float(quad[key]) for key in ("x1", "x2", "x3", "x4") if key in quad]
            ys = [float(quad[key]) for key in ("y1", "y2", "y3", "y4") if key in quad]
            if not text or len(xs) != 4 or len(ys) != 4:
                continue
            x0, x1 = max(0.0, min(xs) / width), min(1.0, max(xs) / width)
            y0, y1 = max(0.0, min(ys) / height), min(1.0, max(ys) / height)
            if x1 <= x0 or y1 <= y0:
                continue
            regions.append(
                OCRRegion(
                    text=text,
                    box=BoundingBox(page=1, x0=x0, y0=y0, x1=x1, y1=y1),
                )
            )
    return regions


def adapt_cord_record(
    record: Mapping[str, Any],
    *,
    source_split: str,
    index: int,
    image_path: Path | None = None,
) -> CanonicalRecord:
    raw_ground_truth = record.get("ground_truth")
    ground_truth = (
        json.loads(raw_ground_truth) if isinstance(raw_ground_truth, str) else raw_ground_truth
    )
    if not isinstance(ground_truth, Mapping):
        raise ValueError("CORD record must contain a JSON ground_truth mapping")

    meta = ground_truth.get("meta") or {}
    image_size = meta.get("image_size") or {}
    width = int(image_size.get("width") or getattr(record.get("image"), "width", 0))
    height = int(image_size.get("height") or getattr(record.get("image"), "height", 0))
    image_id = str(meta.get("image_id", index))
    document_id = f"cord-v2-{source_split}-{image_id}"
    parsed = ground_truth.get("gt_parse") or {}
    if not isinstance(parsed, Mapping):
        raise ValueError("CORD gt_parse must be a mapping")

    items: list[CanonicalLineItem] = []
    for item in _as_list(parsed.get("menu")):
        items.append(
            CanonicalLineItem(
                description=normalize_text(item.get("nm")),
                quantity=parse_quantity(item.get("cnt")),
                unit_price=parse_decimal(item.get("unitprice")),
                amount=parse_decimal(item.get("price") or item.get("itemsubtotal")),
            )
        )

    subtotal_data = parsed.get("sub_total") or {}
    total_data = parsed.get("total") or {}
    annotated_fields: set[CanonicalField] = set()
    if "menu" in parsed:
        annotated_fields.add(CanonicalField.LINE_ITEMS)
    if "subtotal_price" in subtotal_data:
        annotated_fields.add(CanonicalField.SUBTOTAL)
    if "tax_price" in subtotal_data:
        annotated_fields.add(CanonicalField.TAX)
    if "total_price" in total_data:
        annotated_fields.add(CanonicalField.TOTAL)

    annotation = CanonicalAnnotation(
        subtotal=parse_decimal(subtotal_data.get("subtotal_price")),
        tax=parse_decimal(subtotal_data.get("tax_price")),
        total=parse_decimal(total_data.get("total_price")),
        line_items=items,
        annotated_fields=annotated_fields,
    )
    ocr_regions = _cord_ocr_regions(ground_truth, width, height)
    manifest = _manifest(
        dataset=DatasetName.CORD_V2,
        document_id=document_id,
        source_key=image_id,
        source_split=source_split,
        width=width,
        height=height,
        annotated_fields=annotated_fields,
        ocr_region_count=len(ocr_regions),
        image_path=image_path,
    )
    return CanonicalRecord(
        manifest=manifest,
        annotation=annotation,
        ocr_regions=ocr_regions,
    )


def adapt_sroie_record(
    record: Mapping[str, Any],
    *,
    source_split: str,
    index: int,
    image_path: Path | None = None,
) -> CanonicalRecord:
    source_key = str(record.get("key") or index)
    document_id = f"sroie-{source_key}"
    image_size = record.get("image_size") or {}
    width = int(image_size.get("width") or getattr(record.get("image"), "width", 0))
    height = int(image_size.get("height") or getattr(record.get("image"), "height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("SROIE record must contain positive image dimensions")

    entities = record.get("entities") or {}
    if not isinstance(entities, Mapping):
        raise ValueError("SROIE entities must be a mapping")
    field_map = {
        "company": CanonicalField.MERCHANT_NAME,
        "address": CanonicalField.MERCHANT_ADDRESS,
        "date": CanonicalField.ISSUE_DATE,
        "total": CanonicalField.TOTAL,
    }
    normalized_entities = {
        "company": normalize_text(entities.get("company")),
        "address": normalize_text(entities.get("address")),
        "date": parse_date(entities.get("date")),
        "total": parse_decimal(entities.get("total")),
    }
    annotated_fields = {
        target
        for source, target in field_map.items()
        if normalized_entities[source] is not None
    }

    words = record.get("words") or []
    boxes = record.get("bboxes") or []
    if len(words) != len(boxes):
        raise ValueError("SROIE words and bboxes must have the same length")
    ocr_regions: list[OCRRegion] = []
    for word, box in zip(words, boxes, strict=True):
        text = normalize_text(word)
        if not text or not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        try:
            x0 = max(0.0, min(1.0, float(box[0]) / width))
            y0 = max(0.0, min(1.0, float(box[1]) / height))
            x1 = max(0.0, min(1.0, float(box[2]) / width))
            y1 = max(0.0, min(1.0, float(box[3]) / height))
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        ocr_regions.append(
            OCRRegion(
                text=text,
                box=BoundingBox(page=1, x0=x0, y0=y0, x1=x1, y1=y1),
            )
        )

    annotation = CanonicalAnnotation(
        merchant_name=normalized_entities["company"],
        merchant_address=normalized_entities["address"],
        issue_date=normalized_entities["date"],
        total=normalized_entities["total"],
        annotated_fields=annotated_fields,
    )
    manifest = _manifest(
        dataset=DatasetName.SROIE,
        document_id=document_id,
        source_key=source_key,
        source_split=source_split,
        width=width,
        height=height,
        annotated_fields=annotated_fields,
        ocr_region_count=len(ocr_regions),
        image_path=image_path,
    )
    return CanonicalRecord(
        manifest=manifest,
        annotation=annotation,
        ocr_regions=ocr_regions,
    )
