from pathlib import Path

from document_intelligence.observability import InferenceTrace, MLflowTraceSink
from document_intelligence.schemas import ExtractionMethod


class _Run:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class FakeMLflow:
    def __init__(self) -> None:
        self.calls = []

    def set_tracking_uri(self, value):
        self.calls.append(("uri", value))

    def set_experiment(self, value):
        self.calls.append(("experiment", value))

    def start_run(self, **kwargs):
        self.calls.append(("run", kwargs))
        return _Run()

    def log_params(self, value):
        self.calls.append(("params", value))

    def log_metrics(self, value):
        self.calls.append(("metrics", value))

    def log_dict(self, value, path):
        self.calls.append(("dict", path, value))


def test_mlflow_sink_logs_params_metrics_and_trace(monkeypatch, tmp_path: Path) -> None:
    del tmp_path
    fake = FakeMLflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    trace = InferenceTrace(
        job_id="job-1",
        document_id="doc-1",
        method=ExtractionMethod.OCR_RULES,
        model_name="fixture-ocr",
        latency_ms=12,
        page_count=1,
        succeeded=True,
        valid_json=True,
    )

    MLflowTraceSink(tracking_uri="file:./mlruns").log(trace)

    assert ("uri", "file:./mlruns") in fake.calls
    assert any(call[0] == "metrics" for call in fake.calls)
    assert any(call[:2] == ("dict", "trace.json") for call in fake.calls)
