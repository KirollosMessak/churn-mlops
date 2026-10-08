# Production image for the prediction API
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models/champion \
    PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

# Dependencies first so this layer is cached when only code changes
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY configs/ configs/
COPY src/ src/
COPY models/champion/ models/champion/

# Run as a non-root user
RUN useradd --create-home appuser && mkdir -p logs && chown appuser logs
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
