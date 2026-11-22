"""OCR engine adapters and deterministic extraction baseline."""

from document_intelligence.ocr.engine import TesseractConfig, TesseractEngine
from document_intelligence.ocr.rules import extract_document

__all__ = ["TesseractConfig", "TesseractEngine", "extract_document"]
