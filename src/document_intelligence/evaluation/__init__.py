"""Field-level evaluation shared by OCR and VLM approaches."""

from document_intelligence.evaluation.metrics import BenchmarkResult, evaluate_predictions

__all__ = ["BenchmarkResult", "evaluate_predictions"]
