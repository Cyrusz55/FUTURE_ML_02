FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

COPY apps/ ./apps/
COPY frontend/ ./frontend/
COPY machine_learning/ ./machine_learning/
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from apps.model_loader import get_model; from machine_learning.machine_learning import MODEL_PATH, load_model; get_model(); assert load_model(MODEL_PATH) is not None"

RUN useradd --uid 10001 --create-home app \
    && mkdir -p /app/model "${HF_HOME}" \
    && chown -R app:app /app "${HF_HOME}"

USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json, os, urllib.request; port=os.getenv('PORT', '8000'); data=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/v1/health', timeout=4)); raise SystemExit(0 if data.get('model_loaded') is True else 1)"

CMD exec uvicorn apps.main:app --host 0.0.0.0 --port ${PORT:-8000}
