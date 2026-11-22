FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN touch README.md \
    && mkdir -p src/document_intelligence \
    && touch src/document_intelligence/__init__.py \
    && python -m pip install --upgrade pip \
    && python -m pip install ".[api,ocr,ui,mlops]" \
    && rm -rf src
COPY src ./src
RUN python -m pip install --no-deps .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app/artifacts
USER appuser

EXPOSE 8000
CMD ["docintel-api", "--host", "0.0.0.0", "--port", "8000"]
